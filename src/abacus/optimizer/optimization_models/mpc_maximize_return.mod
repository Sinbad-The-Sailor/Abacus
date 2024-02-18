problem mpcMaximizeReturn;

set assets;

param gamma;
param inital_weights {assets};
param number_of_time_steps > 0 integer;
param returns {1..number_of_time_steps, assets};
param covariance {assets, assets};
param l1_penalty {assets};
param l2_penalty {assets};

var weights {0..number_of_time_steps, assets};

maximize OBJECTIVE:
    sum{t in 1..number_of_time_steps} (sum {a in assets} (returns[t,a] * weights[t,a])
    - gamma * sum{a in assets, b in assets} weights[t, a] * covariance[a , b] * weights [t, b]
    - sum{a in assets} ( l1_penalty[a] * abs(weights[t, a] - weights[t-1, a]) + l2_penalty[a] * (weights[t, a] - weights[t-1, a]) **2))
;

subject to WEIGHT_CONSTRAINT_SUM {t in 1..number_of_time_steps}:
    sum{a in assets} weights[t, a] = 1
;

subject to WEGIHT_CONSTRAINT {t in 1..number_of_time_steps, a in assets}:
   weights[t, a] <= 1
;

subject to WEGIHT_CONSTRAINT_NON_NEGATIVE {t in 1..number_of_time_steps, a in assets}:
    weights[t, a] >= 0
;

subject to INITIAL_WEIGHTS {a in assets}:
    weights[0, a] = inital_weights[a]
;
