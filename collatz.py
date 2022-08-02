import matplotlib.pyplot as plt

def collatz(x):
    n = 0
    while x != 1:
        if x % 2 == 0:
            x //= 2
        else:
            x = 3*x + 1
        n += 1
    return n

xs = range(1, 10000 + 1)
cs = [collatz(x) for x in xs]
plt.plot(xs, cs, ".", ms=2)
plt.title("collatz sequence")
plt.xlabel("initial value")
plt.ylabel("number of iterations until 1 is reached")
plt.show()

