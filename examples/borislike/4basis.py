import numpy as np
import matplotlib.pyplot as plt

def f(x, x0, x1, c1, c2, c3, c4, c5):
    L = x1 - x0
    t = (x - x0) / L

    # basis functions on [0,1]
    # b(0)=1 b'(0)=0 b(1)=0 b'(1)=0 int(b1,0,1)=0
    b1 = 1 - 18*t**2 + 32*t**3 - 15*t**4
    # b(0)=0 b'(0)=1 b(1)=0 b'(1)=0 int(b1,0,1)=0
    b2 = t - 4.5*t**2 + 6*t**3 - 2.5*t**4
    # b(0)=0 b'(0)=0 b(1)=1 b'(1)=0 int(b1,0,1)=0
    b3 = -12*t**2 + 28*t**3 - 15*t**4
    # b(0)=0 b'(0)=0 b(1)=0 b'(1)=1 int(b1,0,1)=0
    b4 = 1.5*t**2 - 4*t**3 + 2.5*t**4
    # b(0)=0 b'(0)=0 b(1)=0 b'(1)=0 int(b1,0,1)=1
    b5 = 30*t**2*(1 - t)**2

    # combine with correct scaling for derivatives/integral
    return (
        c1 * b1 +
        L * c2 * b2 +
        c3 * b3 +
        L * c4 * b4 +
        (c5 / L) * b5
    )

# Example usage
x0, x1 = 0.0, 1.0
x = np.linspace(x0, x1, 200)

# Choose arbitrary coefficients:
# f(0)=0, f'(0)=1, f(1)=2, f'(1)=0, integral=0.5
y = f(x, x0, x1, c1=0, c2=1, c3=2, c4=0, c5=0.5)

plt.figure(figsize=(7,4))
plt.plot(x, y, label="f(x)")
plt.title("Polynomial with endpoint and integral control")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend()

import sympy as sp
x_sp=sp.var('x',real=True)
c1,c2,c3,c4,c5=sp.var('c1,c2,c3,c4,c5',real=True)
a,b=sp.var('a,b',real=True)
print(f(x,a,b,c1,c2,c3,c4,c5).subs({x:a}).simplify())
print(f(x,a,b,c1,c2,c3,c4,c5).diff(x).subs({x:a}).simplify())
print(f(x,a,b,c1,c2,c3,c4,c5).subs({x:b}).simplify())
print(f(x,a,b,c1,c2,c3,c4,c5).diff(x).subs({x:b}).simplify())
print(f(x,a,b,c1,c2,c3,c4,c5).integrate((x,a,b)).simplify())


