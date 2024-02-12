problem spMaximizeUtility;

set assets;

param dt > 0;
param risk_free_rate;
param gamma;
param inital_cash > 0;
param number_of_scenarios > 0 integer;
param number_of_assets > 0 integer;
param inital_holdings {assets};
param inital_prices {assets};
param prices {1..number_of_scenarios, assets};

var x_buy {assets};
var x_sell {assets};

maximize OBJECTIVE:
    sum{i in 1..number_of_scenarios} 1/number_of_scenarios * ((inital_cash + sum{j in assets} (inital_prices[j] * (x_sell[j] - x_buy[j]))) * exp(risk_free_rate * dt) +  sum{j in assets} (prices[i, j] * (inital_holdings[j] + x_buy[j] - x_sell[j]))) ** gamma / gamma
;

subject to SHORTING_CONSTRAINT {j in assets}:
    inital_holdings[j] + x_buy[j] - x_sell[j] >= 0
;

subject to LEVERAGE_CONSTRAINT:
    inital_cash + sum{j in assets} ( (x_sell[j] - x_buy[j]) *  inital_prices[j]) >= 0
;

subject to FEASIBLE_CONSTRAINT_BUY {j in assets}:
    x_buy[j] >= 0
;

subject to FEASIBLE_CONSTRAINT_SELL {j in assets}:
    x_sell[j] >= 0
;
