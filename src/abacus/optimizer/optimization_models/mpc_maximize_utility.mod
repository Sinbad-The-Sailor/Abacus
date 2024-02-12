problem mpcMaximizeUtility;

set assets;

param gamma;
param inital_weights {assets};
param number_of_time_steps > 0 integer;
param returns {1..number_of_time_steps, assets};

var weights {0..number_of_time_steps, assets};

maximize OBJECTIVE:
    (sum{t in 1..number_of_time_steps} sum {a in assets} (returns[t,a] * weights[t,a])) ** gamma / gamma
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
