from __future__ import division, print_function
from math import *

std_point_num = 1000

def get_step_size(a, b, n):
    return (b - a)/n

def riemann_sum_left(f, a, b, n=std_point_num):
    h = get_step_size(a, b, n)
    return h*sum(f(a + i*h) for i in range(n))

def riemann_sum_right(f, a, b, n=std_point_num):
    h = get_step_size(a, b, n)
    return h*sum(f(a + i*h) for i in range(1, n + 1))

def upper_sum(f, a, b, n=std_point_num):
    h = get_step_size(a, b, n)
    return h*sum(max(f(a + h*i), f(a + h*(i + 1))) for i in range(n))

def lower_sum(f, a, b, n=std_point_num):
    h = get_step_size(a, b, n)
    return h*sum(min(f(a + h*i), f(a + h*(i + 1))) for i in range(n))

def trapez(f, a, b, n=std_point_num):
    # sum h*(f(a + (i + 1)*h) + f(a + i*h))/2 for i = 0,n-1
    # h*((f(a) + f(b))/2 + sum f(a + i*h) for i = 1,n-1)
    h = get_step_size(a, b, n)
    return h*(sum(f(a + i*h) for i in range(1, n)) + (f(a) + f(b))/2)

def midpoint(f, a, b, n=std_point_num):
    # sum h*f((a + ih + a + h(i + 1))/2) for i = 0..n-1
    # h*sum f(a + hi + h/2) for i = 0..n-1
    # h*sum f(a + h(i + 1/2)) for i = 0..n-1
    h = get_step_size(a, b, n)
    return h*sum(f(a + h*(i + 0.5)) for i in range(n))

def simpson(f, a, b, n=std_point_num):
    # h/2 * sum f(a + i*h)/3 + 4*f(a + hi + h/2)/3 + f(a + (i + 1)*h)/3
    # h/2 * sum f(a + i*h)/3 + 4*f(a + h(i + 1/2))/3 + f(a + (i + 1)*h)/3
    h = get_step_size(a, b, n)
    return h/2 * sum(f(a + i*h)/3 + 4*f(a + h*(i + 0.5))/3 + f(a + (i + 1)*h)/3 for i in range(n))

# TODO: romberg

def test(integrator):
    print("Testing", integrator.func_name + ":\n")

    def test_case(f, desc, solution, a=0, b=10):
        print("Test case:               ", desc)
        print("Bounds:                  ", (a, b))
        print("Correct solution:        ", solution)
        my_solution = integrator(f, a, b)
        print("Result of the integrator:", my_solution)
        abs_error = solution - my_solution
        print("Absolute error:          ", abs_error)
        if solution == 0.0:
            print("Relative error:           no relative error with solution = 0")
        else:
            rel_error = abs_error / solution
            print("Relative error:          ", str(100*rel_error) + "%")
        print("")

    test_case(lambda x: 1, "1", 10)
    test_case(lambda x: x, "x", 50)
    test_case(lambda x: x**2, "x^2", 1/3*10**3)
    test_case(lambda x: sin(x), "sin(x)", 0, b=2*pi)
    F = lambda x: -x**5/5 + 5*x**4/4
    test_case(lambda x: -x**4 + 5*x**3, "-x^4 + 5x^3", F(20) - F(5), a=5, b=20)
