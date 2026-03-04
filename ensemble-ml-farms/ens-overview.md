# Multi-Model Ensemble with Implicit Overfitting Correction

In high-heterogeneity environments, like Indian agriculture, a low error in a single model can be a case of "lucky fold" or "lucky sample". The Ensemble Consensus Predictor employs a Frequentist "Nothing is True" approach towards model selection and suggests an "Everything is Permitted" Consensus. The variation in the multi-model and multi-fold prediction tensor is used to isolate signals from noise. Thus, model choice is replaced with an _endogenised_ framework of model-combnation that 

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

### Stability Ratio 

$$ \boldsymbol{\phi_i}=\frac{\overline{\sigma^m_{i}}}{\overline{\sigma^k_{i}}} $$
	
$$\overline{\sigma^m_{i}} = \frac{1}{K} \sum_{k=1}^K \sqrt{ \frac{1}{M} \sum_{m=1}^M \left( \hat{y}_{i,m,k} - \overline{\hat{y}}_{i,k} \right)^2  } $$
$$\overline{\sigma^k_{i}}=\frac{1}{M} \sum_{m=1}^M \sqrt{\frac{1}{K} \sum_{k=1}^K \left( \hat{y}_{i,m,k} - \overline{\hat{y}}_{i,m} \right)^2 } $$

The predictor $\hat{y}^{p}_i$ is _sufficiently reliable_ if:

$$\overline{\sigma^m_{i}} > \overline{\sigma^k_{i}} \implies \boldsymbol{\phi_i} > 1$$

Approximately 98% farm predictions satisfy the "sanity check".

### 1. Ensemble Perfomance:

Sample Data is India-NSSO 2019 Situation of Agriculture Survey, with about 22,196 rice and 13,518 wheat farms. 

The multi-model predictor achieves ($R^2=$) 56% and 59% for predicting out-of-sample yields for rice and wheat farms, respectively.


![image-find](https://github.com/VolkaneicFox/public-kane.io/blob/e0945847321fd624ff61e54dda003bfd4f4d2530/ensemble-ml-farms/ensemble_rice_fit.png)*Ensemble Predictions of Rice Yields*

![image-find](https://github.com/VolkaneicFox/public-kane.io/blob/e0945847321fd624ff61e54dda003bfd4f4d2530/ensemble-ml-farms/ensemble_wheat_fit.png)*Ensemble Predictions of Wheat Yields*

### 2. Identifying Risky Farms:

The exposure of $i$-th farm to extreme yield outcomes, for a given prediction environment, can be identified using $\delta_i$. Farms with larger $\delta_i$ exhibit wide conditional yield ranges relative to the ensemble predictions and are therefore classified as higher-risk. For example, the top-decile $\delta_i$ farms represent the highest tail-risk exposure and can be prioritized for agriculture risk management. 

The formula given below utilizes the **quantile-loss function** of XGBoost to compute the 5th $(q_{05})$ and 95th $(q_{95})$ conditional prediction quantiles for each farm.

$$\delta_i = \frac{q_{95,i}-q_{05,i}}{\sigma_{\hat{y}^*}}$$

### <ins>Motivation behind the Ensemble</ins>

Endogenous model weighting no longer rewards "lucky folds" or "lucky samples" for individual models, instead, a localised-precision weighted approach avoids model selection risk, reduces local overfitting and captures local tail risk. This repository demonstrates that in high-heterogeneous environments, a robust estimator is not one just one with low global error, but rather one with a high stability ratio. 

## Industry Applications:
Important applications of such a local-level risk identification framework is in insurance. The feature matrix is a combination of farm and climate characteristics that can be used to anchor median expectations from the point predictor and use $\delta$ to identify the riskiest farms. On the flip-side, it also reveals the combinations of features that make prediction difficult for the candidate models. 


