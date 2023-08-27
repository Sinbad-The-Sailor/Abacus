<div align="center">

# Abacus

</div>

> Automatic sequential investment decisions.
> Utilizing realistic market simulations to solve portfolio optimization and risk management.
>
> Currently only working for domestic stocks. Work in progress.


<div align="center">

![GitHub](https://img.shields.io/github/license/sinbad-the-sailor/abacus?color=%23002D5A&style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/sinbad-the-sailor/abacus?color=%23002D5A&style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/sinbad-the-sailor/abacus?color=%23002D5A&style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat-square&logo=docker&logoColor=white)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/Sinbad-the-sailor/abacus/test.yaml?color=%23002D5A&style=flat-square)


</div>

### **Table of Contents**
- [Getting Started](#getting-started)
- [Architecture](#architecture)
- [Portfolio Optimization](#portfolio-optimization)
- [Forecasting](#forecasting)
    - [Equity Models](#equity-model)
    - [Vine Copula Dependency Structure](#vine-copula-dependency-strucutre)
- [Risk Management](#risk-management)
- [References](#references)

### **Getting Started**

### **Architecture**

### **Portfolio Optimization**

For portfolio optimzation there are two paragimes which utilizes the simulation tensor. Stochastic Programming (SP) and Model Predictive Control (MPC). The main difference being SP utilizing all scenarios for one time period, while MPC considers the average scenario over multiple time periods. The implemented models yield different results and can be modified with additional constraints and asset classes to suit any investor. 

##### 1. Maximize Expected Utility Domestic Stocks (Stochastic Programming)
$$
\mbox{max} \quad \mathbb{E}\Big[ U\Big( x_1^{\text{cash}} +  p_{1}^{\text{mid}} \boldsymbol{x}_{1}^{\text{hold}}  \Big)   \Big]
$$

$$
\boldsymbol{x}_1^{\text{hold}} = \boldsymbol{h}_0^{\text{hold}} + \boldsymbol{x}_0^{\text{buy}} - \boldsymbol{x}_0^{\text{sell}}
$$

$$
x_1^{\text{cash}} = \big(h_0^{\text{cash}} +  p_0^{\text{mid}}(\boldsymbol{x}_0^{\text{buy}} - \boldsymbol{x}_0^{\text{sell}})   \big) e^{r \Delta t}
$$

$$
\boldsymbol{x}_0^{\text{buy}}, \boldsymbol{x}_0^{\text{sell}}, \boldsymbol{x}_1^{\text{hold}} \geq \boldsymbol{0}
$$

$$
x_1^{\text{cash}} \geq 0
$$


### **Forecasting**

##### **Equity Model**
Stocks and other equities are model using a modified version of a discretized Merton-Jump-Diffusion SDE. Volatility is updated using a GJR-GARCH(1,1) model, and innovations within the model have a generalized Student's t distribution

$$
r_t = \mu + \sigma_t \xi_t + \kappa \sigma_t \sqrt{\zeta_t} \varepsilon_t
$$

$$
\sigma_t^2 = \beta_0 + \beta_1 r_{t-1}^2 + \beta_3 1_{ \{r_{t-1}<0 \} }r_{t-1}^2  + \beta_2 \sigma_{t-1}^2
$$

$$
\beta_0, \beta_1, \beta_2 \geq 0, \quad \beta_1 + \beta_2 + \frac{1}{2}\beta_3 \leq 1, \quad \beta_2 + \beta_3 \geq 0
$$

$$
\kappa \in \mathbb{R},\quad \mu \in \mathbb{R}, \quad \lambda_J > 0 
$$

$$
\xi_t \overset{\textrm{iid}}{\sim} \textrm{\textbf{t}}_{\nu}(0,1),\quad \zeta_t \overset{\textrm{iid}}{\sim} \textrm{\textbf{Po}}(\lambda_J),\quad \varepsilon_t \overset{\textrm{iid}}{\sim} \textrm{\textbf{N}}(0,1).
$$

The generalized Student's t distribution is given as the following

$$
f_{\Xi}(\xi) = \frac{\Gamma(\frac{\nu + 1}{2})}{\sqrt{(\nu-2) \pi}\Gamma(\frac{\nu}{2})}\Big(1 + \frac{\xi^2}{\nu - 2} \Big)^{-\frac{\nu + 1}{2}}.
$$

The conditional return distribution follow a so-called Student-Poisson-Mixture which is implemented. The distribution is showcased below

$$
 f_{r_t|\mathcal{F}_{t-1}}(r) =  \frac{\Gamma(\frac{\nu+1}{2})}{\sigma_t\sqrt{\pi(\nu-2)}\Gamma(\frac{\nu}{2})}\Bigg[e^{-\lambda}\Big(1 + \frac{(r - \mu)^2}{\sigma_t^2(\nu-2)} \Big)^{-\frac{\nu+1}{2}}
$$

$$
+\sum_{k=1}^{\infty}\frac{\lambda^k}{k!}e^{-\lambda}\frac{1}{\kappa \sigma_t \sqrt{2\pi k}} \int_{\mathbb{R}} \Big(1 + \frac{(s - \mu)^2}{\sigma_t^2(\nu-2)} \Big)^{-\frac{\nu+1}{2}}e^{-\frac{1}{2}(\frac{r-s}{\sigma_t \kappa \sqrt{k}})^2}ds \Bigg].
$$

##### **Vine Copula Dependency Strucutre**
Vine copulas are derived from the so-called pair-copula-construction (PCC). This is done by decomposing a multivariate distribution (pdf), and repetedly applying Sklar's theorem. One can describe the pair construction using the graph theoreical concept of Vines.

Example of 5-dimensional vine coupla (PCC).
<p align="center">
<img width="403" alt="image" src="https://user-images.githubusercontent.com/62723280/169716182-d73f6456-3f21-4074-b24c-bc94de7272f0.png">
</p>

### **Risk Management**

### **References** 
---------------------------------------

#### Model Predictive Control for Multi-Period Portfolio Optimization.

* S. Boyd, E. Busseti, S. Diamond, R. N. Kahn, K. Koh, P. Nystrup, J. Speth. Multi-Period Trading via Convex Optimization. Foundations and Trends in Optimization, vol. 3, no. 1, pp. 1–76, 2016.
* Oprisor R, Kwon R. Multi-Period Portfolio Optimization with Investor Views under Regime Switching. Journal of Risk and Financial Management. 2021; 14(1):3. https://doi.org/10.3390/jrfm14010003
* Fremlin, S. (2019). Online intra-day portfolio optimization using regime based models (Dissertation). Retrieved from http://lup.lub.lu.se/student-papers/record/8972097

#### Stochastic Programming for Portfolio Optimization.

* G. Cornuejols, R. Tütüncü. Optimization Methods in Finance. Optimization Methods in Finance. Mathematics, Finance and Risk. Cambridge University Press. 2007.

#### AR, MA, and ARMA Time Series Models.

* Hamilton, J. D. (1994). Time Series Analysis. Princeton University Press. 

#### Nonlinear Autoregressive model with Neural Networks.

* Benrhmach G, Namir K, Namir A, Bouyaghroumni J. (2020). "Nonlinear Autoregressive Neural Network and Extended Kalman Filters for Prediction of Financial Time Series". Journal of Applied Mathematics. vol 2020. Article ID 5057801. https://doi.org/10.1155/2020/5057801

#### Variance Stabalizing and Preconditioning for GARCH/GJR-GARCH models.

* Zumbach, G. (2000). The Pitfalls in Fitting Garch(1,1) Processes. In: Dunis, C.L. (eds) Advances in Quantitative Asset Management. Studies in Computational Finance, vol 1. Springer, Boston, MA. https://doi.org/10.1007/978-1-4615-4389-3_8
* Sundström, D. (2017). Automatized GARCH parameter estimation (Dissertation). Retrieved from http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-213725

#### Copulas, Copula based Forecasting and Vine Copulas.

* Roger, N. (2006). An Introduction to Copulas (Springer Series in Statistics). Springer-Verlag, Berlin, Heidelberg.
* Simard, C. & Rémillard, B. (2015). Forecasting time series with multivariate copulas. Dependence Modeling, 3(1). https://doi.org/10.1515/demo-2015-0005
* Czado, C. (2019). Analyzing Dependent Data with Vine Copulas: A Practical Guide With R. Lecture Notes in Statistics. Springer International Publishing.

#### Extreme Value Theory and Financial Risk Management.

* Avdulaj, K. (2011). The Extreme Value Theory as a Tool to Measure Market Risk, IES Working Paper, No. 26/2011, Charles University in Prague, Institute of Economic Studies (IES), Prague. http://hdl.handle.net/10419/83384
* McNeil A, Frey R, Embrechts, P. (2005). Quantitative Risk Management: Concepts, Techniques, and Tools. 
* Hull, J. (2018). Risk management and financial institutions. Wiley Finance Series. Wiley.
