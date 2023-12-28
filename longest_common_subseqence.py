from __future__ import print_function, division
import numpy as np

def lcs_rec(s1, s2, n=None, m=None):
    if n is None and m is None:
        n = len(s1) - 1
        m = len(s2) - 1
    if n < 0 or m < 0:
        return  ""
    if s1[n] == s2[m]:
        rec_str = lcs_rec(s1, s2, n - 1, m - 1)
        return rec_str + s1[n]
    return max(lcs_rec(s1, s2, n - 1, m), lcs_rec(s1, s2, n, m - 1), key=len)

def lcs_dy_prog(s1, s2):
    table = np.zeros((len(s1), len(s2)), dtype="int")
    def lookup(i, j):
        if i < 0 or j < 0:
            return 0
        else:
            return table[i, j]
    # find length of the lcs
    for i, c1 in enumerate(s1):
        for j, c2 in enumerate(s2):
            if c1 == c2:
                table[i, j] = lookup(i - 1, j - 1) + 1
            else:
                table[i, j] = max(lookup(i - 1, j), lookup(i, j - 1))
    # backtrac to find lcs (not unique)
    i = len(s1) - 1
    j = len(s2) - 1
    res = ""
    while i >= 0 and j >= 0:
        if s1[i] == s2[j]:
            res += s1[i]
            i -= 1
            j -= 1
        else:
            if lookup(i - 1,j) > lookup(i, j - 1):
                i -= 1
            else:
                j -= 1
    res = res[::-1]
    return res



s1 = "XMJYAUZ"
s2 = "MZJAWXU"
print("recursive:", lcs_rec(s1,s2))
print("dynamic programming:", lcs_dy_prog(s1,s2))
