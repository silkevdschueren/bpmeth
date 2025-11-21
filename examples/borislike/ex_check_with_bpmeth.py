import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt

from cubicmagnet import CubicMagnet
import bpmeth


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
comp[0, 0] = 1
comp[0, 1] = 1.0
comp[1, 0] = 1.0
comp[1, 2] = 0.1
comp[2, 0] = 1.0
comp[2, 1] = 0.1
bs, b, a = comp_to_sympy(comp)
h = "0.0"
length = 0.5

A_magnet = bpmeth.GeneralVectorPotential(hs=h, bs=bs, b=b, a=a)
H_magnet = bpmeth.Hamiltonian(length=length, curv=float(h), vectp=A_magnet)
B_magnet = CubicMagnet(comp=comp*p0c/c, length=length, ds=0.01, h=float(h), s_start=0.0)

part = xt.Particles(
    x=0.1,
    y=0.2,
    px=0.3,
    py=0.2,
    delta=0.1,
    zeta=0.1,
    s=0.1,
    mass0=938272088.16,
    q0=1.0,
    p0c=p0c,
)

p1 = part.copy()
sol = H_magnet.track(p1, return_sol=True, ivp_opt={"rtol": 1e-10, "atol": 1e-12})
p2 = part.copy()
B_magnet.track(p2)

print(f"Max difference on x: {np.max(np.abs(p1.x - p2.x))}")
print(f"Max difference on y: {np.max(np.abs(p1.y - p2.y))}")
print(f"Max difference on px: {np.max(np.abs(p1.px - p2.px))}")
print(f"Max difference on py: {np.max(np.abs(p1.py - p2.py))}")


B_magnet.plot_x(part)
plt.plot(sol[0]['t']+0.1,sol[0]['y'][0])