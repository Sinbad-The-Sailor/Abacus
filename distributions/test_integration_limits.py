import time

from scipy.integrate import quad
from student_poisson_mixture import spm

params = {1: (0, 1, 0.09, 0.05, 4),
          2: (0.1, 0.50, 0.35, 0.15, 5),
          3: (-0.3, 0.75, 0.55, 0.55, 10),
          4: (0.05, 1.1, 0.75, 0.75, 16),
          5: (0, 0.10, 0.11, 0.15, 5)}
lower_limit = [1, 2, 3, 4, 5]

for i in params.keys():
    t = time.time()
    corr = val = quad(spm.pdf, -25, 1, args=params[i])[0]
    elapsed = time.time() - t
    print(f'time elapsed corr: {elapsed}')
    for limit in lower_limit:
        t = time.time()
        val = quad(spm.pdf, -limit, 1, args=params[i])[0]
        elapsed = time.time() - t
        err = abs(val-corr)
        print(f'Lower limit: {limit} yields value: {val} with error: {err} with time elapsed {elapsed}')
        