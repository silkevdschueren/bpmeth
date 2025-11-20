import numpy as np
import numba

from curvedboris import make_state, integrate_numba_vect_final
from cubic_bp import bfield, afield

@numba.njit(cache=True)
def efield(x, y, s, t, h, pars):
    return 0, 0, 0
        
import matplotlib.pyplot as plt

class CubicMagnet:
    is_thick = True

    def __init__(self, comp, length, ds, h, s0=0.0):
        self.comp = comp
        self.length = length
        self.ds = length / np.ceil(length / ds)
        self.h = h
        self.s0 = s0

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
            s=np.full(part.x.shape, self.s0),
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
            (self.s0, self.s0 + self.length),
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
        part.s += self.length
        assert np.allclose(out["s"] - self.s0, self.length)
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

    def track_step_by_step(self, part):
        steps=np.ceil(self.length / self.ds)
        mag=CubicMagnet(self.comp,length=self.ds,ds=self.ds,h=self.h)
        out=[part.copy()]
        for _ in range(int(steps)):
            mag.track(part)
            out.append(part.copy())
            mag.s0+=self.ds

        return out
    
    def plot_x(self, part):
        out=self.track_step_by_step(part.copy())
        s_vals=[p.s[0] for p in out]
        x_vals=[p.x[0] for p in out]
        plt.plot(s_vals,x_vals,label="x vs s")
        plt.xlabel("s (m)")
        plt.ylabel("x (m)")
        plt.title("Particle Trajectory in Cubic Magnet")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_y(self, part):
        out=self.track_step_by_step(part.copy())
        s_vals=[p.s[0] for p in out]
        y_vals=[p.y[0] for p in out]
        plt.plot(s_vals,y_vals,label="y vs s",color="orange")
        plt.xlabel("s (m)")
        plt.ylabel("y (m)")
        plt.title("Particle Trajectory in Cubic Magnet")
        plt.legend()
        plt.grid()
        plt.show()


