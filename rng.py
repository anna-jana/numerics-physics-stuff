from __future__ import print_function, division

import numpy as np
from itertools import islice

def linear_congruential(x, upper_limit=100, factor=21, increment=7):
    while True:
        yield x
        x = (factor*x + increment) % upper_limit

def multiply_with_carry(x, c, upper_limit=100, factor=6):
    while True:
        yield x
        helper = factor*x + c
        x = helper % upper_limit
        c = helper // upper_limit

def test(make_gen, num_samples=100, sample_length=5):
    tests = [list(islice(make_gen(seed), sample_length)) for seed in range(1, num_samples)]
    corr = np.corrcoef(np.array(tests))
    return corr

print("linear_congruential:\n", test(linear_congruential))
print("multiply_with_carry:\n", test(lambda seed: multiply_with_carry(seed, 50)))
