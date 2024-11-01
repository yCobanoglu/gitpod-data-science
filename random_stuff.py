import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


def f1(x, n):
    return np.cos(2* np.pi * n * x)

def f_an(x, n):
    return 2*(1-2*x)**2 * np.cos(2* np.pi * n * x)

def f_bn(x, n):
    return 2*(1-2*x)**2 * np.sin(2* np.pi * n * x)

def f(x, n):
    return 8*x**2* np.sin(2* np.pi * n * x)



xs = np.linspace(0, 1, 1000)
n = 10
integral = np.trapz(f_bn(xs, n), xs)
print("Integral", integral)
#plt.figure()
#plt.plot(xs, f(xs, 1))
#plt.plot(xs, f(xs, 10))
#plt.plot(xs, f(xs, 100))
#plt.plot(xs, f1(xs, 500))
#plt.show()