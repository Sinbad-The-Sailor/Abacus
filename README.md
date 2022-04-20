# Abacus Asset Modelling

Modelling future prices of assets including:

### **Equities**

---
Stocks and other equities are model using a modified version of a discretized Merton-Jump-Diffusion SDE. Volatility is updated using a GJR-GARCH(1,1) model, and innovations within the model have a generalized Student's t distribution given below.

<p align="center">
  <img width="399" alt="model" src="https://user-images.githubusercontent.com/62723280/164225698-01cfa241-a2d1-4570-bb11-f771db8e966a.png">
</p>

<p align="center">
<img width="399" alt="image" src="https://user-images.githubusercontent.com/62723280/164237572-0abf1f4b-aadc-43ba-b1d2-b8f7219acfe5.png">
</p>

The conditional return distribution follow a so-called Student-Poisson-Mixture which is implemented. The distribution is showcased below.

<p align="center">
<img width="559" alt="image" src="https://user-images.githubusercontent.com/62723280/164237856-57a466a2-0a65-4e48-bfb3-5c6181605706.png">
</p>

### **Fixed Income**

---
### **Commodities**

---
### **Foreign Exchange**

---
