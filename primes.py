import numpy as np

# Julia:
# primes(n) = (1:n)[2 .== sum([a % b == 0 for a=1:n, b=1:n], 2)]

# Haskell:
# primes n = n : filter ((/= 0) . (`mod` n)) (primes (n + 1))

def primes(n):
    nums = np.arange(1, n)
    return nums[2 == np.sum(np.array([[a % b == 0 for a in nums] for b in nums]), axis=0)]
