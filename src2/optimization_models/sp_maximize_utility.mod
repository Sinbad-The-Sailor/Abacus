problem spMaximizeUtility;

set assets;

param dt > 0;
param gamma <= 1;
param risk_free_rate;
param inital_cash > 0;
param number_of_scenarios >0 integer;
param number_of_assets > 0 integer;

param inital_holdings {assets};
param prices {1..number_of_scenarios, 1..number_of_assets};

var x_buy {assets};
var x_sell {assets};

maximize Objective:
    sum{i in 1..number_of_scenarios} (((inital_cash + (sum{j in assets} (x_sell[j] - x_buy[j]) * prices[i, j] )) * exp(risk_free_rate * dt)) + sum{j in assets} prices[i, j] * (inital_holdings[j] + x_buy[j] - x_sell[j])) ** gamma / gamma
;

subject to SHORTING_CONSTRAINT {i in 1..number_of_scenarios, j in assets}:
    inital_holdings[j] + x_buy[j] - x_sell[j] >= 0
;

subject to LEVERAGE_CONSTRAINT {i in 1..number_of_scenarios}:
    inital_cash + (sum{j in assets} (x_sell[j] - x_buy[j] *  prices[i, j])) * exp(risk_free_rate * dt) >= 0
;

subject to FEASIBLE_CONSTRAINT_BUY {j in assets}:
    x_buy[j] >= 0
;

subject to FEASIBLE_CONSTRAINT_SELL {j in assets}:
    x_sell[j] >= 0
;
