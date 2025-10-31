import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
from curvedboris import *
import time


def track_xsuite(
    k0=0.1,
    h=0.1,
    l=10,
    ds=0.1,
    mass0=938272088.16,
    q0=1,
    p0c=1e9,
    x=0.0,
    y=0.0,
    px=0.0,
    py=0.0,
    delta=0.0,
    zeta=0.0,
):
    nstep = int(l / ds)
    bend = xt.Bend(length=ds, k0=k0, h=h, model="bend-kick-bend")
    ln = xt.Line([bend] * nstep)
    ln.set_particle_ref("p", p0c=p0c, mass0=mass0, q0=q0)
    part = ln.build_particles(x=x, px=px, y=y, py=py, delta=delta, zeta=zeta)
    ln.track(part, turn_by_turn_monitor="ONE_TURN_EBE")
    res = ln.record_last_track
    return res

# Field model
def efield(x, y, s, t, h, pars):
    return 0, 0, 0

def bfield(x, y, s, t, h, pars):
    B0 = pars[0]
    return 0, B0, 0

efield = njit(efield, cache=True)
bfield = njit(bfield, cache=True)

def track(
    k0=0.1,
    h=0.1,
    l=10,
    ds=0.1,
    mass0=938272088.16,
    q0=1,
    p0c=1e9,
    x=0.0,
    y=0.0,
    px=0.0,
    py=0.0,
    delta=0.0,
    zeta=0.0,
):
    # Physical constants
    c = 299_792_458.0  # speed of light [m/s]
    qe = +1.602176634e-19  # charge [C]
    q = qe * q0
    m = mass0 * qe / c**2  # proton mass [kg]
    p0_SI = (p0c * qe) / c  # eV/c -> kg·m/s
    B0 = (k0 * p0_SI) / q  # from k0 = q B0 / p0

    epars = np.array([0.0], dtype=np.float64)
    bpars = np.array([B0], dtype=np.float64)
    # ---- Build initial state (with momentum offset delta) ----
    px0 = px * p0_SI
    py0 = py * p0_SI
    p_tot = p0_SI * (1.0 + delta)
    pT2 = px0**2 + py0**2

    beta_gamma = p_tot / (m * c)
    gamma = np.sqrt(1.0 + beta_gamma**2)
    e0 = gamma * m * c * c
    beta0_gamma0 = p0_SI / (m * c)
    gamma0 = np.sqrt(1.0 + beta0_gamma0**2)
    beta0 = beta0_gamma0 / gamma0
    t0 = zeta / beta0 / c

    st0 = make_state(
        s=0.0, x=x, y=y, t=t0, px=px0, py=py0, e=e0, h=h, m=m, c=c
    )

    # ---- Integrate over the requested magnet length ----
    st = time.time()
    out = integrate_numba(st0, (0.0, l), ds, h, efield, bfield, epars, bpars, m, q, c)
    print(f"{l/ds=} {(time.time()-st)/(l/ds)}")
    out["zeta"] = out["s"] - beta0 * c * out["t"]

    return out


l = 10.0
h = 0.1
k0 = 0.1
ds = 0.01
x = 0.1
px = 0.0
delta = 0.1
y = 0.1
py = 0.1
res0 = track_xsuite(l=l, h=h, k0=k0, ds=ds, x=x, px=px, delta=delta, y=y, py=py)
res1 = track(l=l, h=h, k0=k0, ds=ds, x=x, px=px, delta=delta, y=y, py=py)

#plt.plot(res0.s[0],res1['x']-res0.x[0])
#plt.plot(res0.s[0],res0.y[0])
#plt.plot(res0.s[0],res1['y'])
#plt.plot(res0.s[0],res0.zeta[0])
#plt.plot(res0.s[0],res1['zeta'])


plt.plot(res0.s[0],res1['x']-res0.x[0])
plt.plot(res0.s[0],res1['y']-res0.y[0])
plt.plot(res0.s[0],res1['zeta']-res0.zeta[0])

