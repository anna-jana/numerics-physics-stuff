# are category 1 and 2 independent?

import numpy as np
from scipy.stats import chi2, chi2_contingency

# axis 0 (rows) = category 1, axis 1 (cols) = category 2
data = np.array([[90, 60, 104, 95],
                 [30, 50, 51, 20],
                 [30, 40, 45, 35]])

# total number for each category 1
total_each1 = data.sum(axis=1)
total_each2 = data.sum(axis=0)

# total number of samples for each of category 1 and 2
total = np.sum(data)

# what do we expect for each sample in category 1 and 2 based on data
expected_each = total_each1[:, None] * total_each2[None, :] / total

# terms for chi^2
test_for_each_sample = (data - expected_each)**2 / expected_each

# chi^2 value
test_stat = np.sum(test_for_each_sample)

# number of degrees of freedom
ndofs = (data.shape[0] - 1) * (data.shape[1] - 1)

# p-value
p = 1 - chi2.cdf(test_stat, ndofs)

# confirm with scipy
p_scipy = chi2_contingency(data).pvalue

print(f"{p = } vs {p_scipy =}")
