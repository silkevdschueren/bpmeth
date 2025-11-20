import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import numba

from cubicmagnet import CubicMagnet


def track_xsuite(
    part,
    k1=0.1,
    length=10,
    kicks=40
):
    mag = xt.Quadrupole(
        length=length, k1=k1, model="drift-kick-drift-exact", integrator="yoshida4",
        num_multipole_kicks=kicks,
    )
    ln = xt.Line([mag])
    ln.set_particle_ref(part.p0c, mass0=part.mass0, q0=part.q0)
    part = part.copy()
    mag.track(part)
    return part


def track_boris(
    part,
    k1=0.1,
    length=10,
    ds=0.1,
):
    comp = np.zeros((9, 4), dtype=np.float64)
    c = 299_792_458.0
    comp[3, 0] = k1 * part.p0c[0] / part.q0 / c  # from k0 = q B0 / p0
    cubic_mag = CubicMagnet(comp, length=length, ds=ds, h=0.0)
    part = part.copy()
    cubic_mag.track(part)
    return part,cubic_mag


k1 = 0.001
ds = 0.01
length = 10
part = xt.Particles(
    x=np.array([0.1], dtype=np.float64),
    y=np.array([0.0], dtype=np.float64),
    px=np.array([0.1], dtype=np.float64),
    py=np.array([0.1], dtype=np.float64),
    delta=np.array([0.1], dtype=np.float64),
    zeta=np.array([0.1], dtype=np.float64),
    s=np.array([1.0], dtype=np.float64),
    mass0=938272088.16,
    q0=1.0,
    p0c=1e9,
)


part_xsuite = track_xsuite(part, k1=k1, length=length,kicks=2*7+1)
part_boris,mag = track_boris(part, k1=k1, length=length, ds=ds)

for aa in ["x", "y", "px", "py", "delta", "zeta", "s"]:
    vv1 = getattr(part_xsuite, aa)
    vv2 = getattr(part_boris, aa)
    print(f"{aa}: xsuite={vv1}, boris={vv2}, diff={vv1-vv2}")


