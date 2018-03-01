import numpy as np
import matplotlib.pyplot as plt

def make_spiral(n):
    spiral = np.array([[1]])
    for _ in range(4 * n):
        spiral = np.rot90(np.vstack([np.arange(spiral[0,0], spiral[0,0] + spiral.shape[1]) + 1, spiral]))
    return spiral

def primes_up_to(n):
    nums = np.arange(1,n+1)
    return nums[2 == np.sum(0 == nums % nums[:,None], axis=0)]

def make_ulam_spiral(n):
    return np.any(make_spiral(n)[:,:,None] == primes_up_to((2 * n + 1)**2)[None,None,:], axis=2)

if __name__ == "__main__":
    plt.imshow(make_ulam_spiral(50), cmap="Greys")
    plt.show()
