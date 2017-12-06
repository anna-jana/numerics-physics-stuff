import numpy as np

# Haskell:
# primes n = n : filter ((/= 0) . (`mod` n)) (primes (n + 1))

def primes(n):
    nums = np.arange(1, n)
    return nums[np.sum(nums % nums[:,None] == 0, axis=0) == 2]
