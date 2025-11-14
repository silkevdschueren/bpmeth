import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import numba

from cubicmagnet import CubicMagnet


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
part_boris = track_boris(part, k0=k0, h=h, length=length, ds=ds)

for aa in ["x", "y", "px", "py", "delta", "zeta", "s"]:
    vv1 = getattr(part_xsuite, aa)
    vv2 = getattr(part_boris, aa)
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
    line = xt.Line([cubic_mag])
    line.set_particle_ref(p0c, mass0=mass0, q0=q0)
    return line


plt.figure()
for step in (10.0) ** -np.arange(2, 8):
    steps_r_matrix = {
        "dx": step,
        "dpx": step,
        "dy": step,
        "dpy": step,
        "dzeta": step,
        "ddelta": step,
    }
    err = []
    for ds in np.logspace(-1, -5, 13):
        line = mk_line_boris(ds=ds)
        line.build_tracker()
        rmat = line.compute_one_turn_matrix_finite_differences(
            particle_on_co=line.particle_ref,
            include_collective=True,
            steps_r_matrix=steps_r_matrix,
        )["R_matrix"]
        err.append((ds, abs(1 - np.prod(np.linalg.eigvals(rmat)))))

    err = np.array(err)
    plt.loglog(err[:, 0], np.abs(err[:, 1]), "-o", label=f"step={step}")

plt.xlabel("ds [m]")
plt.ylabel("Symplectic error")
plt.title("Symplectic error vs ds for one turn matrix with Boris-like integrator")
plt.grid()
plt.show()


plt.figure()
for step in (10.0) ** -np.arange(2, 8):
    steps_r_matrix = {
        "dx": step,
        "dpx": step,
        "dy": step,
        "dpy": step,
        "dzeta": step,
        "ddelta": step,
    }
    err = []
    for ds in np.logspace(-1, -3, 5):
        line = xt.Line(
            [xt.Bend(length=ds, k0=0.1, h=0.1, model="bend-kick-bend")] * int(10 / ds)
        )
        line.build_tracker()
        line.particle_ref = xt.Particles(
            mass0=938272088.16,
            q0=1.0,
            p0c=1e9,
        )
        rmat = line.compute_one_turn_matrix_finite_differences(
            particle_on_co=line.particle_ref,
            include_collective=True,
            steps_r_matrix=steps_r_matrix,
        )["R_matrix"]
        err.append((ds, abs(1 - np.prod(np.linalg.eigvals(rmat)))))
    err = np.array(err)
    plt.loglog(err[:, 0], np.abs(err[:, 1]), "-o", label=f"step={step}")

plt.xlabel("ds [m]")
plt.ylabel("Symplectic error")
plt.title("Symplectic error vs ds for one turn matrix with Sbend integrator")
plt.grid()
plt.legend()
plt.show()
