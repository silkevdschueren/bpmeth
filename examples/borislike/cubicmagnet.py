import numpy as np
import numba

from curvedboris import make_state, integrate_numba_vect_final
# from cubic_bp import bfield, afield
from ba_fields_2_3_h_nphi5 import bfield, afield


def select_fields(nphi):
    module_name = f"ba_fields_2_3_h_nphi{nphi}"
    mod = importlib.import_module(module_name)

    # assign into *this* module's global namespace
    g = sys.modules[__name__].__dict__
    g['bfield'] = mod.bfield
    g['afield'] = mod.afield


@numba.njit(cache=True)
def efield(x, y, s, t, h, pars):
    return 0, 0, 0


import matplotlib.pyplot as plt


class CubicMagnet:
    is_thick = True

    def __init__(self, comp, length, ds, h, s_start=0.0):
        self.comp = comp
        self.length = length
        self.ds = length / np.ceil(length / ds)
        self.h = h
        self.s_start = s_start

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
            s=np.full(part.x.shape, self.s_start),
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
            (self.s_start, self.s_start + self.length),
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

        ax, ay, _ = afield(out["x"], out["y"], out["s"], 0, self.h, self.comp)
        part.x = out["x"]
        part.y = out["y"]
        part.s += self.length
        assert np.allclose(out["s"] - self.s_start, self.length)
        ax = ax * q / p0_SI
        ay = ay * q / p0_SI
        part.px = out["px"] / p0_SI + ax
        part.py = out["py"] / p0_SI + ay
        part.zeta = self.length - out["t"] * part.beta0 * c
        p = np.sqrt(out["px"] ** 2 + out["py"] ** 2 + out["ps"] ** 2) / p0_SI
        part.delta = p - 1.0
        part.ax = ax
        part.ay = ay
        assert np.allclose(part.kin_px, out["px"] / p0_SI)
        assert np.allclose(part.kin_py, out["py"] / p0_SI)

    def track_step_by_step(self, part):
        tpart = part.copy()
        self.track(tpart)  # final points for check
        steps = np.ceil(self.length / self.ds)
        mag = CubicMagnet(
            self.comp, length=self.ds, ds=self.ds, h=self.h, s_start=self.s_start
        )
        out = [part.copy()]
        for _ in range(int(steps)):
            mag.track(part)
            out.append(part.copy())
            mag.s_start += self.ds

        assert np.allclose(tpart.x, part.x)
        assert np.allclose(tpart.y, part.y)
        assert np.allclose(tpart.zeta, part.zeta)
        assert np.allclose(tpart.s, part.s)

        return out

    def plot_x(self, part, ax=None):
        out = self.track_step_by_step(part.copy())
        s_vals = [p.s[0] for p in out]
        x_vals = [p.x[0] for p in out]
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(s_vals, x_vals, label="x vs s")
        ax.set_xlabel("s (m)")
        ax.set_ylabel("x (m)")
        ax.set_title("Particle Trajectory in Cubic Magnet")
        plt.legend()
        plt.grid()

    def plot_y(self, part):
        out = self.track_step_by_step(part.copy())
        s_vals = [p.s[0] for p in out]
        y_vals = [p.y[0] for p in out]
        plt.plot(s_vals, y_vals, label="y vs s", color="orange")
        plt.xlabel("s (m)")
        plt.ylabel("y (m)")
        plt.title("Particle Trajectory in Cubic Magnet")
        plt.legend()
        plt.grid()
        plt.show()
