# Abacus 

Sequential investment decisions. 

Currently only working as risk utilites for portfolios and scenario generation.

### **Equitie Model**
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
