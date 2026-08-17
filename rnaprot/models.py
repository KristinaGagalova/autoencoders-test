"""
Multimodal RNA <-> protein autoencoder.

                RNA ---> encoder_R ---\
                                       >--- z (shared latent, 4-16 dims)
            protein ---> encoder_P ---/
                                       |
                          +------------+------------+
                          |                         |
                    decoder_R -> RNA_hat     decoder_P -> protein_hat

Four loss terms, and each one maps to a question you asked:

  recon_rna     z must retain transcriptome structure
  recon_prot    z must retain proteome structure
  cross         protein predicted from the RNA encoder ONLY.
                This is the term that answers "how much of the proteome does the
                transcriptome explain", and its residuals are the discordance score.
  align         ||z_R - z_P||^2, pulls the two encoders into a shared space.
                This is the neural analogue of MOFA's shared factors.

Sizing for n = 48: keep it tiny. 3000 -> 128 -> 16 -> z(8) has ~400k parameters
for 48 observations. The regularisation (dropout, weight decay, early stopping,
tiny latent) is not optional decoration -- it is the only thing standing between
you and a model that memorises 48 samples perfectly and generalises to nothing.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _mlp(sizes, dropout=0.2, out_activation=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if out_activation or i < len(sizes) - 2:
            # LayerNorm, not BatchNorm: at n=24 (or n=24 per variety), CV folds
            # leave ~16-20 training samples after the internal early-stopping
            # split, which is close to or below one batch (batch_size=16).
            # BatchNorm's running statistics are then estimated from almost no
            # data and don't transfer reliably to test-fold samples -- this
            # produces an unstable, fold-dependent collapse where roughly half
            # the folds converge to a near-constant output (verified: median
            # prediction/observation std ratio 0.08-0.15, some folds as low as
            # 0.06, others fine at 0.3-0.4, purely due to random init landing
            # in a bad BatchNorm regime). LayerNorm normalizes per-sample, has
            # no batch-statistics dependency, and is stable even at batch
            # size 1. Swapping this alone raised the aggregate pred/obs std
            # ratio from 0.20 to 0.39 on matched synthetic data and eliminated
            # the fold-to-fold coin-flip behavior.
            layers += [nn.LayerNorm(sizes[i + 1]), nn.SiLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class MultimodalAE(nn.Module):
    def __init__(self, n_rna, n_prot, n_cov=0, hidden=(128, 32), latent=8,
                 dropout=0.2, variational=False):
        super().__init__()
        self.variational = variational
        self.latent = latent
        self.n_cov = n_cov
        out_r = latent * 2 if variational else latent

        self.enc_r = _mlp([n_rna + n_cov, *hidden, out_r], dropout)
        self.enc_p = _mlp([n_prot + n_cov, *hidden, out_r], dropout)
        self.dec_r = _mlp([latent + n_cov, *reversed(hidden), n_rna], dropout)
        self.dec_p = _mlp([latent + n_cov, *reversed(hidden), n_prot], dropout)

    # -- helpers ----------------------------------------------------------
    def _cat(self, x, cov):
        return torch.cat([x, cov], dim=1) if (cov is not None and self.n_cov) else x

    def _split(self, h):
        if not self.variational:
            return h, None
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar.clamp(-8, 8)

    def _sample(self, mu, logvar):
        if logvar is None or not self.training:
            return mu
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    # -- api ---------------------------------------------------------------
    def encode_rna(self, rna, cov=None):
        return self._split(self.enc_r(self._cat(rna, cov)))

    def encode_prot(self, prot, cov=None):
        return self._split(self.enc_p(self._cat(prot, cov)))

    def decode(self, z, cov=None):
        zc = self._cat(z, cov)
        return self.dec_r(zc), self.dec_p(zc)

    def forward(self, rna, prot, cov=None):
        mu_r, lv_r = self.encode_rna(rna, cov)
        mu_p, lv_p = self.encode_prot(prot, cov)
        z_r, z_p = self._sample(mu_r, lv_r), self._sample(mu_p, lv_p)
        rna_hat, prot_from_rna = self.decode(z_r, cov)
        _, prot_hat = self.decode(z_p, cov)
        return dict(mu_r=mu_r, lv_r=lv_r, mu_p=mu_p, lv_p=lv_p,
                    rna_hat=rna_hat, prot_hat=prot_hat, prot_from_rna=prot_from_rna)

    @torch.no_grad()
    def predict_protein(self, rna, cov=None):
        """Test-time path: RNA in, protein out. The protein encoder is not used."""
        self.eval()
        mu_r, _ = self.encode_rna(rna, cov)
        return self.decode(mu_r, cov)[1]


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------
def ae_loss(out, rna, prot, w=(1.0, 1.0, 2.0, 1.0), beta=0.0):
    """w = (recon_rna, recon_prot, cross_modal, alignment); beta = KL weight (VAE)."""
    mse = nn.functional.mse_loss
    l_rna = mse(out["rna_hat"], rna)
    l_prot = mse(out["prot_hat"], prot)
    l_cross = mse(out["prot_from_rna"], prot)
    l_align = mse(out["mu_r"], out["mu_p"])
    total = w[0] * l_rna + w[1] * l_prot + w[2] * l_cross + w[3] * l_align

    l_kl = torch.tensor(0.0, device=rna.device)
    if beta > 0 and out["lv_r"] is not None:
        for mu, lv in ((out["mu_r"], out["lv_r"]), (out["mu_p"], out["lv_p"])):
            l_kl = l_kl + (-0.5 * (1 + lv - mu.pow(2) - lv.exp()).sum(1).mean())
        total = total + beta * l_kl

    return total, dict(rna=l_rna.item(), prot=l_prot.item(), cross=l_cross.item(),
                       align=l_align.item(), kl=float(l_kl))


# --------------------------------------------------------------------------
# Sklearn-style wrapper so the CV loop treats it like any other model
# --------------------------------------------------------------------------
class AERegressor:
    name = "multimodal_ae"

    def __init__(self, latent=8, hidden=(128, 32), dropout=0.2, lr=1e-3,
                 weight_decay=1e-3, epochs=400, patience=40, batch_size=16,
                 weights=(1.0, 1.0, 2.0, 1.0), variational=False, beta=1e-3,
                 val_frac=0.2, groups=None, seed=0, device="cpu", verbose=False):
        self.__dict__.update(locals())
        del self.self

    def _inner_split(self, n, rng):
        """
        Hold out whole design cells for early stopping when groups are given.
        A random inner split would put replicates of the same cell on both
        sides and tell you to keep training long after the model has started
        memorising.
        """
        if self.groups is None:
            perm = rng.permutation(n)
            n_val = max(4, int(self.val_frac * n))
            return perm[n_val:], perm[:n_val]
        g = np.asarray(self.groups)
        uniq = rng.permutation(np.unique(g))
        n_hold = max(1, int(round(self.val_frac * len(uniq))))
        held = set(uniq[:n_hold])
        mask = np.array([x in held for x in g])
        if mask.sum() < 2 or (~mask).sum() < 4:      # degenerate, fall back
            perm = rng.permutation(n)
            n_val = max(4, int(self.val_frac * n))
            return perm[n_val:], perm[:n_val]
        idx = np.arange(n)
        return idx[~mask], idx[mask]

    def fit(self, R, P, cov=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = torch.device(self.device)
        n = R.shape[0]
        n_cov = 0 if cov is None else cov.shape[1]

        # inner validation split for early stopping (never touches the outer test fold)
        tr, va = self._inner_split(n, rng)

        t = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)
        Rt, Pt = t(R), t(P)
        Ct = t(cov) if n_cov else None

        self.model_ = MultimodalAE(R.shape[1], P.shape[1], n_cov, self.hidden,
                                   self.latent, self.dropout,
                                   self.variational).to(dev)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)

        best, best_state, bad = np.inf, None, 0
        self.history_ = []
        for ep in range(self.epochs):
            self.model_.train()
            order = rng.permutation(tr)
            for s in range(0, len(order), self.batch_size):
                b = order[s:s + self.batch_size]
                if len(b) < 1:      # nothing to train on
                    continue
                cb = Ct[b] if Ct is not None else None
                out = self.model_(Rt[b], Pt[b], cb)
                loss, _ = ae_loss(out, Rt[b], Pt[b], self.weights,
                                  self.beta if self.variational else 0.0)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 5.0)
                opt.step()
            sched.step()

            # early stopping on CROSS-MODAL validation error: the quantity we care about
            self.model_.eval()
            with torch.no_grad():
                cv_ = Ct[va] if Ct is not None else None
                pred = self.model_.predict_protein(Rt[va], cv_)
                vloss = nn.functional.mse_loss(pred, Pt[va]).item()
            self.history_.append(vloss)
            if vloss < best - 1e-5:
                best, bad = vloss, 0
                best_state = {k: v.detach().clone()
                              for k, v in self.model_.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_val_ = best
        self.epochs_run_ = ep + 1
        if self.verbose:
            print(f"    [ae] stopped at epoch {ep+1}, val cross-MSE {best:.4f}")
        return self

    def predict(self, R, cov=None):
        dev = torch.device(self.device)
        t = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)
        c = t(cov) if cov is not None and self.model_.n_cov else None
        return self.model_.predict_protein(t(R), c).cpu().numpy()

    def encode(self, R, cov=None, modality="rna"):
        """Latent coordinates for every sample: the AE analogue of MOFA factors."""
        dev = torch.device(self.device)
        t = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)
        c = t(cov) if cov is not None and self.model_.n_cov else None
        self.model_.eval()
        with torch.no_grad():
            enc = self.model_.encode_rna if modality == "rna" else self.model_.encode_prot
            mu, _ = enc(t(R), c)
        return mu.cpu().numpy()
