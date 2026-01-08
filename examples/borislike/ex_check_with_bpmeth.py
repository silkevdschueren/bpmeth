import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt

from cubicmagnet import CubicMagnet
import bpmeth
import time


def comp_to_sympy(comp):
    import sympy

    s = sympy.symbols("s")
    out = []
    for row in comp:
        pol = row[0]
        for n, v in enumerate(row[1:]):
            pol += v * s ** (n + 1)
        out.append(pol)
    bs = out[0]
    b = out[1::2]
    a = out[2::2]
    return bs, b, a


p0c=1e9
c = 299_792_458.0
comp = np.zeros((9, 4), dtype=np.float64)
comp[0, 0] = 0
comp[0, 1] = 0.0
comp[1, 0] = 0.0
comp[1, 1] = 0.1
comp[1, 2] = 0.0
comp[3, 0] = 0.0
comp[3, 1] = 0.0
bs, b, a = comp_to_sympy(comp)
h = "0.0"
length = 0.0001

#A_magnet = bpmeth.GeneralVectorPotential(hs=h, bs=bs, b=b, a=a)
A_magnet = bpmeth.GeneralVectorPotential(hs=h, b=[b[0],b[1]], nphi=5)
H_magnet = bpmeth.Hamiltonian(length=length, curv=float(h), vectp=A_magnet)
B_magnet = CubicMagnet(comp=comp*p0c/c, length=length, ds=0.0001, h=float(h), s_start=0.0)  # order 5, determined in import of cubicmagnet.py

part = xt.Particles(
    x=0.1,
    y=0.2,
    px=0,
    py=0,
    delta=0,
    zeta=0,
    s=0,
    mass0=938272088.16,
    q0=1.0,
    p0c=p0c,
)

p1 = part.copy()
sol = H_magnet.track(p1, return_sol=True, ivp_opt={"rtol": 1e-12, "atol": 1e-14})
p2 = part.copy()
t1 = time.time()
B_magnet.track(p2)
t2 = time.time()

print(f"Max difference on x: {np.max(np.abs(p1.x - p2.x))}")
print(f"Max difference on y: {np.max(np.abs(p1.y - p2.y))}")
print(f"Max difference on px: {np.max(np.abs(p1.px - p2.px))}")
print(f"Max difference on py: {np.max(np.abs(p1.py - p2.py))}")

fig, ax = plt.subplots()
B_magnet.plot_x(part, ax=ax)
ax.plot(sol[0]['t']+part.s,sol[0]['y'][0], label='bpmeth', linestyle='dashed')
plt.legend()
plt.show()
