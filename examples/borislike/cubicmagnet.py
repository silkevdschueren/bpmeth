import numpy as np
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
        px_ini = part.kin_px * p0_SI
        py_ini = part.kin_py * p0_SI
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
        part.delta = p - 1.0
        part.ax = ax
        part.ay = ay
        assert part.kin_px==out["px"] / p0_SI
        assert part.kin_py==out["py"] / p0_SI


