## Multi-Model Ensemble with Implicit Overfitting Correction

A weighted multi-model ensemble predictor reliably performs in the upper-bound of candidate models.

Thus, model choice is replaced with an _endogenised_ framework of model-combnation that 

(i) prioritizes high-performers, and 

(ii) penalizes models that overfit at the farm-level 

The result is a *localised and precision-weighted* ensemble predictor, developed using,

1. **Candidate models:** Random Forest, Gradient Boosting and Support Vector Regression.

2. **Prediction Tensor:** Point-predictors across candidate models and cross-validation folds.

For the $i$-th farm, the weighted prediction across $m$ models is, 

$$ \hat{y}^{p}_i = \frac{1}{K}\sum_{k=1}^K \sum_{m=1}^M \omega^{*m}_i \hat{y}^{m,k}_i $$  

where, $\omega^{*m}_i = \frac{\omega_m}{\sigma_i^m}$

And furthemore, 
$$\omega_m = e^{\kappa \cdot \left (R_{m,k}^2 - R_{\text{max},k}^2 \right)}$$

where $\sigma^m_i$ is the within-model dispersion, measured as either:

$$\sigma^m_{i} = \sqrt{\frac{1}{K} \sum_{k=1}^K \left( \hat{y}_{i,m,k} - \overline{\hat{y}}_{i,m} \right)^2}$$

or

$$\sigma^m_i = \hat{y}_{i,m,\text{max}} - \hat{y}_{i,m,\text{min}}$$

The ensemble predictor is _sufficiently_  <ins>SANE</ins> if it aggregates inductive signals, NOT model noise. 

#### Stability Ratio 

$$ \boldsymbol{\phi_i}=\frac{\overline{\sigma^m_{i}}}{\overline{\sigma^k_{i}}} $$
	
$$\overline{\sigma^m_{i}} = \frac{1}{K} \sum_{k=1}^K \sqrt{ \frac{1}{M} \sum_{m=1}^M \left( \hat{y}_{i,m,k} - \overline{\hat{y}}_{i,k} \right)^2  } $$
$$\overline{\sigma^k_{i}}=\frac{1}{M} \sum_{m=1}^M \sqrt{\frac{1}{K} \sum_{k=1}^K \left( \hat{y}_{i,m,k} - \overline{\hat{y}}_{i,m} \right)^2 } $$

The predictor $\hat{y}^{p}_i$ is _sufficiently reliable_ if:

$$\overline{\sigma^m_{i}} > \overline{\sigma^k_{i}} \implies \boldsymbol{\phi_i} > 1$$

