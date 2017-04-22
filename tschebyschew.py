import numpy as np
import matplotlib.pyplot as plt

def tschebyschow(n):
    T1 = np.zeros(n + 1)
    T1[0] = 1.0
    T2 = np.zeros(n + 1)
    T2[1] = 1.0
    T = [T1, T2]
    for i in range(2, n + 1):
        T.append(2*np.roll(T[i - 1], 1) - T[i - 2])
    return T

def poly_to_fn(poly):
    return lambda x: np.sum(poly*x**np.arange(poly.size))

def poly_to_string(poly):
    ans = ""
    first = True # first term with coeff. != 0?
    for i in range(poly.size - 1, -1, -1):
        # only display term with a coeff. != 0
        # but display a 0 if all terms are 0
        if poly[i] != 0.0 or i == 0 and first:
            if poly[i] > 0.0:
                if not first: # we don't need a + if we are in the first term
                    ans += " + "
            else:
                # in the first term we use a - without a sign
                # e.g. -x^2 - 2
                if first:
                    ans += "-"
                else:
                    ans += " - "
            # we don't want to display a coeff. with 0 decimals as e.g. 1.0 but as 1
            # and we already dealt with the sign
            if round(poly[i]) == poly[i]:
                val = abs(int(round(poly[i])))
            else:
                val = abs(poly[i])
            # display the constant term with only its value
            if i == 0:
                ans += str(val)
            # ommit the exponent for the linear term (x instead of x^1)
            elif i == 1:
                # ommit the coeff. if the coeff is 1
                if val == 1:
                    ans += "x"
                else:
                    ans += "{}*x".format(val)
            else:
                # ommit the coeff. if the coeff is 1
                if val == 1:
                    ans += "x^{}".format(i)
                else:
                    ans += "{}*x^{}".format(val, i)
            first = False # we had a term != 0
    return ans

n = 5

for i, p in enumerate(tschebyschow(n)):
    xs = list(np.linspace(-1, 1, 100))
    f = poly_to_fn(p)
    #plt.plot(xs, map(f, xs), label="n = {}".format(i))
    plt.plot(xs, map(f, xs), label=poly_to_string(p))
    # plt.plot(xs, map(f, xs))

plt.legend()
plt.title("Tschebyschow Polynomials")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

