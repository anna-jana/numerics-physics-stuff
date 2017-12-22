
# coding: utf-8

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import sys


########################### Aufgabe 1 ###########################
def sign(x):
    if x == 0:
        return 1
    return x/abs(x)

def QR(A):
    A = A.copy()
    m,n = A.shape
    U = np.zeros((m,n))
    for k in range(n):
        # berechnung der k-ten householder matrix
        A_k_max = np.max(np.abs(A[k:, k]))
        alpha = 0
        for i in range(k, m):
            U[i, k] = A[i, k]/A_k_max
            alpha += np.abs(U[i, k])**2
        alpha = np.sqrt(alpha)
        beta_k = 1/(alpha*(alpha + np.abs(U[k,k])))
        U[k,k] += sign(A[k,k]) * alpha
        # multiplikation der k-ten householder matrix
        A[k,k] = -sign(A[k,k]) * alpha * A_k_max
        for i in range(k + 1, m):
            A[i,k] = 0
        for j in range(k + 1, n):
            s = beta_k * np.sum(U[k:, k] * A[k:, j])
            for i in range(k, m):
                A[i,j] -= s*U[i,k]
    # A ist jetzt R
    # berechnung von Q
    Q = np.eye(m)
    for k in range(n):
        u_k = U[:, k]
        H_k = np.eye(m) - 2/np.dot(u_k, u_k) * np.dot(u_k[:, None], u_k[None, :])
        Q = np.dot(Q, H_k)
    return Q, A

# check QR
for i in range(1, 4):
    print("checking QR decomposition algorithm with example %d ..." % i, end="")
    sys.stdout.flush()
    tol = 1e-10
    A = np.loadtxt("{}.qr".format(i), skiprows=4)
    Q, R = QR(A)
    assert np.sum((A - np.dot(Q, R))**2) < tol, "Q*R != A"
    assert np.sum((np.dot(Q.T, Q) - np.eye(Q.shape[0]))**2) < tol, "Q is not orthogonal"
    assert all(R[i,j] == 0 for j in range(R.shape[1]) for i in range(j + 1, R.shape[0])), "R ist not upper triangular"
    print(" ok")
print("all tests passed")

########################## Aufgabe 2 #########################
def lineare_regression(a, b):
    n = len(a)
    A = np.array([[a[i] for i in range(n)], [1 for i in range(n)]])
    A = A.T
    Q, R = QR(A)
    b_ = np.dot(Q.T, b)
    R_hut = R[0:2,0:2]
    x = np.dot(np.linalg.inv(R_hut), b_[0:2])
    err = np.sum(((x[0]*a + x[1]) - b)**2)
    return x, err

data = np.loadtxt("1.ls", skiprows=5)
a = data[0, :]
b = data[1, :]

print("linear regression")
plt.subplot(2, 1, 1)
plt.title("Linear Regresssion")
plt.plot(a, b, "xr", label="Daten")
fit, err = lineare_regression(a, b)
print("fit parameter", fit)
print("fit error", err)
x = np.linspace(min(a), max(a), 300)
plt.plot(x, fit[0]*x + fit[1], "-k", label="Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Linearer Fit Test")
plt.grid()
plt.legend(loc=2)


########################### Aufgabe 3 ###########################
def multi_linear_regression(x_data, y_data, cost=5):
    cache = {}
    def multi_linear_regression_rec(x, y):
        tup_x = tuple(x)
        if tup_x in cache:
            return cache[tup_x]
        if x.size == 2 or x.size == 3:
            parameter, err = lineare_regression(x, y)
            return [(x, parameter)], err
        parameter, lowest_error = lineare_regression(x, y)
        best_fit = [(x, parameter)]
        for i in range(2, x.size - 1):
            seg_x = x[:i]
            seg_y = y[:i]
            rest_x = x[i:]
            rest_y = y[i:]
            parameter, seg_err = lineare_regression(seg_x, seg_y)
            rest_fit, rest_err = multi_linear_regression_rec(rest_x, rest_y)
            total_err = seg_err + rest_err + cost
            if total_err < lowest_error:
                lowest_error = total_err
                best_fit = [(seg_x, parameter)] + rest_fit
        ans = best_fit, lowest_error
        cache[tup_x] = ans
        return ans
    ans = multi_linear_regression_rec(x_data, y_data)
    return ans

print("linear regression with multiple segements")
segments, total_error = multi_linear_regression(a, b, cost=5)
print("fitted segments", segments)
print("total error with costs", total_error)
plt.subplot(2, 1, 2)
plt.title("Linear Regression with multiple segements")
plt.plot(a, b, "rx", label="Data")
for i, (x_domain, parameter) in enumerate(segments):
    y = parameter[0]*x_domain + parameter[1]
    plt.plot(x_domain, y, "k-", label="%d fitted segment" % i)
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend(loc=2)
plt.ylim(0, 30)

plt.tight_layout()
plt.show()

