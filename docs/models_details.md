# Model Evaluation and Dimensionality-Reduction Methods

This analysis compares several approaches for predicting **protein abundance from RNA expression data**. Because omics datasets typically contain many more molecular features than biological samples, dimensionality reduction and regularisation are used to reduce overfitting.

## R² — Coefficient of Determination

The coefficient of determination (**R²**) measures how well predicted protein abundances agree with the observed protein abundances.

Conceptually:

[
R^2 = 1 - \frac{\text{prediction error}}{\text{baseline error}}
]

where the baseline corresponds to predicting the average protein abundance.

Interpretation:

* **R² = 1** — perfect prediction.
* **R² = 0** — the model performs approximately as well as predicting the mean.
* **R² < 0** — the model performs worse than predicting the mean.

In this analysis, R² is calculated for protein predictions on samples that were not used to train the model. It therefore provides a measure of **out-of-sample predictive performance**.

When the median R² across proteins is reported, it represents the prediction performance of the typical protein rather than a single global R² for the entire dataset.

---

## Ridge Regression

**Ridge regression** is a regularised form of linear regression that is particularly useful for high-dimensional omics datasets.

A standard linear model can be represented as:

[
y = X\beta + \epsilon
]

where:

* (X) represents RNA expression measurements,
* (y) represents protein abundance,
* (\beta) represents regression coefficients.

Ridge regression adds an **L2 penalty** that discourages very large regression coefficients:

[
\text{Loss} =
\text{prediction error}
+
\lambda \sum_j \beta_j^2
]

The parameter (\lambda) controls the strength of regularisation.

### Why ridge regression is useful for omics

Omics datasets commonly have:

* many more features than samples,
* strongly correlated genes,
* substantial biological and technical noise.

Ridge regression helps stabilise the model by shrinking regression coefficients toward zero.

Unlike LASSO, ridge regression normally does **not remove features completely**. Instead, correlated features can share the predictive signal.

### Interpretation in RNA → protein modelling

Ridge regression asks:

> Can protein abundance be predicted using a linear combination of RNA expression measurements?

A **cognate ridge model** considers the corresponding RNA transcript when predicting a protein and can therefore be used as a simple biological baseline for testing the relationship between transcript and protein abundance.

---

## PCA — Principal Component Analysis

**Principal Component Analysis (PCA)** is an **unsupervised dimensionality-reduction method**.

PCA identifies combinations of features that explain the largest amount of variation in the predictor dataset.

For RNA expression:

[
RNA \rightarrow PC_1, PC_2, PC_3, \ldots
]

The first principal component explains the largest amount of RNA variation, the second explains the next largest amount, and so on.

Importantly, PCA only considers the RNA data when constructing the components.

It does **not** use protein abundance information.

### PCA for RNA → protein prediction

A typical workflow is:

[
RNA
\rightarrow
PCA
\rightarrow
Principal\ Components
\rightarrow
Regression
\rightarrow
Protein
]

For example, thousands of RNA features can be reduced to a small number of principal components, which are subsequently used in a regression model such as ridge regression.

PCA therefore asks:

> What are the major patterns of variation in the RNA dataset?

A limitation is that the largest sources of RNA variation are not necessarily the RNA patterns that are most informative for protein abundance.

---

## PLS — Partial Least Squares

**Partial Least Squares (PLS)** is a **supervised dimensionality-reduction and regression method**.

Unlike PCA, PLS considers both the predictor and response datasets when constructing its latent components.

For this analysis:

[
X = RNA
]

and

[
Y = Protein
]

PLS identifies RNA components that have strong covariance with protein abundance.

Conceptually:

[
RNA
\rightarrow
PLS\ components
\leftrightarrow
Protein
]

PLS therefore asks:

> Which major patterns in RNA expression are most informative for explaining protein variation?

Because protein abundance contributes to the construction of the latent components, PLS is directly optimised for the RNA → protein relationship.

---

## PCA versus PLS

| Method                                         | PCA                          | PLS                            |
| ---------------------------------------------- | ---------------------------- | ------------------------------ |
| Full name                                      | Principal Component Analysis | Partial Least Squares          |
| Type                                           | Unsupervised                 | Supervised                     |
| Uses RNA data                                  | Yes                          | Yes                            |
| Uses protein data when constructing components | No                           | Yes                            |
| Main objective                                 | Explain RNA variance         | Explain RNA–protein covariance |
| Components optimised for protein prediction    | No                           | Yes                            |
| Useful for dimensionality reduction            | Yes                          | Yes                            |
| Suitable for high-dimensional omics            | Yes                          | Yes                            |

The fundamental difference is therefore:

> **PCA finds the strongest patterns in RNA variation, whereas PLS finds RNA patterns that are most strongly related to protein variation.**

For example, a very strong source of RNA variation may have little relationship with protein abundance. PCA may capture this variation as its first principal component because it explains substantial RNA variance.

PLS may instead prioritise a weaker RNA pattern if that pattern is strongly associated with protein abundance.

---

## PCA + Ridge Regression

PCA and ridge regression can be combined:

[
RNA
\rightarrow
PCA
\rightarrow
Reduced\ RNA\ representation
\rightarrow
Ridge
\rightarrow
Protein
]

PCA first reduces thousands of correlated RNA features into a small number of components.

Ridge regression then uses these components to predict protein abundance.

This provides a relatively simple linear benchmark for comparison with more complex models.

---

## Autoencoder

An **autoencoder** is a neural-network approach that learns a lower-dimensional representation of high-dimensional data.

In the RNA–protein model, RNA and protein measurements are compressed into a small **latent representation**.

Conceptually:

[
RNA
\rightarrow
Latent\ space
\rightarrow
Protein
]

Unlike PCA and standard PLS, an autoencoder can learn **non-linear relationships** between molecular measurements.

The purpose of including the autoencoder is therefore to test whether a nonlinear latent representation captures RNA–protein relationships that cannot be adequately represented by simpler linear approaches.

However, the additional flexibility also increases the risk of overfitting, particularly when the number of biological samples is small.

---

## Role of the Different Models

The different models provide increasingly complex ways of testing the RNA–protein relationship:

| Model             | Biological/statistical question                                                   |
| ----------------- | --------------------------------------------------------------------------------- |
| Mean predictor    | Can the model outperform simply predicting average protein abundance?             |
| Design-only model | Can treatment and timepoint alone explain protein abundance?                      |
| Cognate ridge     | Does the RNA transcript corresponding to a protein predict that protein?          |
| PCA + ridge       | Do the major global patterns of RNA variation predict proteins?                   |
| PLS               | Are there RNA expression patterns specifically associated with protein variation? |
| Autoencoder       | Can nonlinear latent RNA patterns improve protein prediction?                     |

Together, these models allow increasingly complex methods to be compared against simple baselines.

A complex model such as an autoencoder is most informative when it consistently predicts held-out protein measurements better than simpler alternatives such as ridge regression, PCA + ridge, or PLS.
