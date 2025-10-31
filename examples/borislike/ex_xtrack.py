import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import numba

from curvedboris import make_state, integrate_numba_vect_final
from cubic_bp import bfield, afield


@numba.njit(cache=True)
def efield(x, y, s, t, h, pars):
    return 0, 0, 0


class CubicMagnet:
    is_thick = True

    def __init__(self, comp, length, ds, h):
        self.comp = comp
        self.length = length
        self.ds = ds
        self.h = h

    def track(self, part):
        c = 299_792_458.0
        qe = +1.602176634e-19
        q = qe * part.q0
        m = part.mass0 * qe / c**2
        p0_SI = (part.p0c * qe) / c
        epars = np.array([0.0], dtype=np.float64)
        px_ini = (part.px - part.ax) * p0_SI
        py_ini = (part.py - part.ay) * p0_SI
        p_ini = p0_SI * (1.0 + part.delta)
        beta_gamma_ini = p_ini / (m * c)
        gamma_ini = np.sqrt(1.0 + beta_gamma_ini**2)
        e_ini = gamma_ini * m * c * c
        t_ini = -part.zeta / part.beta0 / c

        state = make_state(
            s=part.s,
            x=part.x,
            y=part.y,
            t=t_ini,
            px=px_ini,
            py=py_ini,
            e=e_ini,
            h=self.h,
            m=m,
            c=c,
        ).T
        out = integrate_numba_vect_final(
            state,
            (part.s[0], part.s[0] + self.length),
            self.ds,
            self.h,
            efield,
            bfield,
            epars,
            self.comp,
            m,
            q,
            c,
        )

        part.x = out["x"]
        part.y = out["y"]
        part.s = out["s"]
        ax, ay, _ = afield(part.x, part.y, part.s, 0, self.h, self.comp)
        ax = ax * q / p0_SI
        ay = ay * q / p0_SI
        part.px = out["px"] / p0_SI + ax
        part.py = out["py"] / p0_SI + ay
        part.zeta = self.length - out["t"] * part.beta0 * c
        p = np.sqrt(out["px"] ** 2 + out["py"] ** 2 + out["ps"] ** 2) / p0_SI
        part.delta = p  - 1.0
        part.ax = ax
        part.ay = ay


def track_xsuite(
    part,
    k0=0.1,
    h=0.1,
    length=10,
):
    bend = xt.Bend(length=length, k0=k0, h=h, model="bend-kick-bend")
    ln = xt.Line([bend])
    ln.set_particle_ref(part.p0c, mass0=part.mass0, q0=part.q0)
    part = part.copy()
    ln.track(part)
    return part


def track_boris(
    part,
    k0=0.1,
    h=0.1,
    length=10,
    ds=0.1,
):
    comp = np.zeros((9, 4), dtype=np.float64)
    c = 299_792_458.0
    comp[1, 0] = k0 * part.p0c[0] / part.q0 / c  # from k0 = q B0 / p0
    cubic_mag = CubicMagnet(comp, length=length, ds=ds, h=h)
    part = part.copy()
    cubic_mag.track(part)
    return part


k0 = 0.1
h = 0.1
length = 10
ds = 0.1
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

part_xsuite = track_xsuite(part, k0=k0, h=h, length=length)
part_boris = track_boris(part, k0=k0, h=h, length=length)

for aa in ['x','y','px','py','delta','zeta','s']:
    vv1=getattr(part_xsuite,aa)
    vv2=getattr(part_boris,aa)
    print(f"{aa}: xsuite={vv1}, boris={vv2}, diff={vv1-vv2}")


def mk_line_boris(
    k0=0.1,
    h=0.1,
    length=10,
    ds=0.1,
    p0c=1e9,
    mass0=938272088.16,
    q0=1,
):
    comp = np.zeros((9, 4), dtype=np.float64)
    c = 299_792_458.0
    comp[1, 0] = k0 * part.p0c[0] / part.q0 / c  # from k0 = q B0 / p0
    cubic_mag = CubicMagnet(comp, length=length, ds=ds, h=h)
    line=xt.Line([cubic_mag])
    line.set_particle_ref(p0c, mass0=mass0, q0=q0)
    return line

line=mk_line_boris(ds=0.001)
line.twiss(betx=1,bety=1,include_collective=True)
rmat= line.compute_one_turn_matrix_finite_differences(particle_on_co=line.particle_ref,include_collective=True)['R_matrix']
np.prod(np.linalg.eigvals(rmat))