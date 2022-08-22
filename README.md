# Abacus
<div align="center">

![GitHub](https://img.shields.io/github/license/sinbad-the-sailor/abacus?color=%23002D5A&style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/sinbad-the-sailor/abacus?color=%23002D5A&style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/sinbad-the-sailor/abacus?color=%23002D5A&style=flat-square)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/sinbad-the-sailor/abacus/Tests?color=%23002D5A&style=flat-square)

</div>


Sequential investment decisions.

Currently only working as a risk tool for ETF portfolios and scenario generation.

### **Equity Model**
Stocks and other equities are model using a modified version of a discretized Merton-Jump-Diffusion SDE. Volatility is updated using a GJR-GARCH(1,1) model, and innovations within the model have a generalized Student's t distribution.

<p align="center">
  <img width="399" alt="model" src="https://user-images.githubusercontent.com/62723280/164225698-01cfa241-a2d1-4570-bb11-f771db8e966a.png">
</p>

The generalized Student's t distribution is given as the following.

<p align="center">
<img width="399" alt="image" src="https://user-images.githubusercontent.com/62723280/164237572-0abf1f4b-aadc-43ba-b1d2-b8f7219acfe5.png">
</p>

The conditional return distribution follow a so-called Student-Poisson-Mixture which is implemented. The distribution is showcased below.

<p align="center">
<img width="559" alt="image" src="https://user-images.githubusercontent.com/62723280/164237856-57a466a2-0a65-4e48-bfb3-5c6181605706.png">
</p>

### **Vine Copula Dependency Strucutre**
Vine copulas are derived from the so-called pair-copula-construction (PCC). This is done by decomposing a multivariate distribution (pdf), and repetedly applying Sklar's theorem. One can describe the pair construction using the graph theoreical concept of Vines.

Example of 5-dimensional vine coupla (PCC).
<p align="center">
<img width="403" alt="image" src="https://user-images.githubusercontent.com/62723280/169716182-d73f6456-3f21-4074-b24c-bc94de7272f0.png">
</p>

### **References**
---

#### Model Predictive Control for Multi-Period Portfolio Optimization.

* S. Boyd, E. Busseti, S. Diamond, R. N. Kahn, K. Koh, P. Nystrup, J. Speth. Multi-Period Trading via Convex Optimization. Foundations and Trends in Optimization, vol. 3, no. 1, pp. 1–76, 2016.
* Oprisor R, Kwon R. Multi-Period Portfolio Optimization with Investor Views under Regime Switching. Journal of Risk and Financial Management. 2021; 14(1):3. https://doi.org/10.3390/jrfm14010003

#### Variance Stabalizing and Preconditioning for GARCH/GJR-GARCH models.

* Zumbach, G. (2000). The Pitfalls in Fitting Garch(1,1) Processes. In: Dunis, C.L. (eds) Advances in Quantitative Asset Management. Studies in Computational Finance, vol 1. Springer, Boston, MA. https://doi.org/10.1007/978-1-4615-4389-3_8
* Sundström, D. (2017). Automatized GARCH parameter estimation (Dissertation). Retrieved from http://urn.kb.se/resolve?urn=urn:nbn:se:kth:diva-213725
