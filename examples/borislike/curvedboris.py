import numpy as np
import matplotlib.pyplot as plt
from numba import njit

clight = 299_792_458.0  # m/s
qe = +1.602176634e-19  # [C]


@njit(cache=True)
def _kin_half_drift_jit(state, ds, h, m, c):
    # st: array length 9
    s, x, y, t, px, py, ps, e, g = state
    x += 0.5 * ds * g * (px / ps)
    y += 0.5 * ds * g * (py / ps)
    t += 0.5 * ds * g * (e / (c * c * ps))
    s += 0.5 * ds
    g = 1.0 + h * x
    state[0], state[1], state[2], state[3], state[8] = s, x, y, t, g


@njit(cache=True)
def _kin_half_drift2_jit(state, ds, h, m, c):
    # st: array length 9
    s, x, y, t, px, py, ps, e, g = state
    a = px / ps
    g_in = 1.0 + h * x
    Delta = 0.5 * ds
    ahD = a * h * Delta
    em1 = np.expm1(ahD)
    x_out = (1.0 + em1) * (x + 1.0 / h) - 1.0 / h
    I = (g_in / a) * em1
    y_out = y + (py / ps) * I/h
    t_out = t + (e / (c * c * ps)) * I/h
    #print(I/h, 0.5 * ds * g)

    state[0] = s + Delta
    state[1] = x_out
    #state[2] = y + 0.5 * ds * g * (py / ps)  # to be-recheceked
    #state[3] = t + 0.5 * ds * g * (e / (c * c * ps))  # to be-recheceked
    state[2] = y_out
    state[3] = t_out
    state[8] = 1.0 + h * x_out


@njit(cache=True)
def _kin_half_drift3_jit(state, ds, h, m, c):
    """
    Exact half-step kinematic drift with curvature h for Frenet–Serret frame.

    State layout (as used previously):
      st[0]=s   (independent variable; NOT changed here)
      st[1]=x
      st[2]=y
      st[3]=t
      st[4]=px
      st[5]=py
      st[6]=ps
      st[7]=e
      st[8]=g   (=1+h*x)  -- will be refreshed

    This integrates, with momenta and e frozen:
        x' = g * (px/ps),   y' = g * (py/ps),   t' = g * (e/(c^2 ps)),  g=1+h x,
    over a half-step Δ = ds/2, exactly in x and with the correct ∫g ds
    for y and t. Volume is preserved by the overall palindromic composition.
    """
    # unpack
    x = state[1]
    y = state[2]
    t = state[3]
    px = state[4]
    py = state[5]
    ps = state[6]
    e = state[7]

    # half-step
    Delta = 0.5 * ds

    # Guard: ps should not be zero (turning point). If it is extremely small,
    # treat the drift as zero-length to avoid blow-ups.
    eps_ps = 1e-18
    if np.abs(ps) < eps_ps or Delta == 0.0:
        # nothing to do except keep g consistent
        state[8] = 1.0 + h * x
        return

    # a = px/ps is constant within the K block
    a = px / ps
    g_in = 1.0 + h * x

    # We need x_out and I = ∫_0^Δ g(s) ds with g(s)=1+h x(s).
    # Exact solution with constant a:
    #   x(s) = exp(a h s) * (x_in + 1/h) - 1/h
    #   I    = (g_in / a) * (exp(a h Δ) - 1)      (if a != 0)
    #
    # Use numerically stable branches for small |a h Δ|.
    ahD = a * h * Delta

    if np.abs(ahD) < 1e-8:
        # series expansions to avoid loss of significance
        # exp(ahD) - 1 ≈ ahD + 1/2 (ahD)^2
        em1 = ahD + 0.5 * ahD * ahD
        # x_out ≈ x_in + Δ * a * g_in + 1/2 Δ^2 * a^2 * h * g_in
        x_out = x + Delta * a * g_in + 0.5 * (Delta * Delta) * (a * a * h) * g_in
        if np.abs(a) < 1e-16:
            # when a ≈ 0, x is (near) constant and ∫g ds ≈ Δ g_in
            I = Delta * g_in
        else:
            I = (g_in / a) * em1
    else:
        # general exact formulas
        # exp(a h Δ) = em1 + 1
        em1 = np.expm1(ahD)
        # x_out = exp(ahD) * (x + 1/h) - 1/h
        # handle h=0 separately to avoid division by zero
        if np.abs(h) < 1e-18:
            # straight line case (should be caught by small ahD branch, but keep safe)
            x_out = x + Delta * a * g_in
        else:
            x_out = (1.0 + em1) * (x + 1.0 / h) - 1.0 / h

        # I = (g_in/a) * (exp(ahD) - 1) = (g_in/a) * em1
        if np.abs(a) < 1e-16:
            I = Delta * g_in  # limit a->0
        else:
            I = (g_in / a) * em1

    # Update y and t using the same integral of g
    y_out = y + (py / ps) * I/h
    t_out = t + (e / (c * c * ps)) * I/h

    # write back & refresh g
    state[1] = x_out
    state[2] = y_out
    state[3] = t_out
    state[8] = 1.0 + h * x_out


@njit(cache=True)
def _geom_half_rot_jit(state, ds, h):
    px = state[4]
    ps = state[6]
    theta = 0.5 * h * ds
    ct = np.cos(theta)
    st = np.sin(theta)
    px_new = ct * px + st * ps
    ps_new = -st * px + ct * ps
    state[4] = px_new
    state[6] = ps_new


# gain one order of magnitude
@njit(cache=True)
def _em_full_step_jit2(st, ds, h, q, m, c, E_func, B_func, E_pars, B_pars):
    # unpack midpoint state (after K(½), G(½))
    x, y, s, t = st[1], st[2], st[0], st[3]
    px, py, ps, e, g = st[4], st[5], st[6], st[7], st[8]

    # fields at midpoint (already correct)
    Ex, Ey, Es = E_func(x, y, s, t, h, E_pars)
    Bx, By, Bs = B_func(x, y, s, t, h, B_pars)

    # -------- predictor: use entry ps for a trial step to estimate ps_{1/2}
    d_tau0 = (g / ps) * ds
    # do exactly your current EM map but writing into locals (p0,p1,p2,ee)
    p0, p1, p2, ee = px, py, ps, e

    # half energy
    ee = ee + 0.25 * q * d_tau0 * (p0 * Ex + p1 * Ey + p2 * Es)
    coef = 0.25 * q * d_tau0 * (ee / (c * c))
    # half E kick
    p0 += coef * Ex
    p1 += coef * Ey
    p2 += coef * Es
    # magnetic rotation (Boris)
    tx = 0.25 * q * d_tau0 * Bx
    ty = 0.25 * q * d_tau0 * By
    tz = 0.25 * q * d_tau0 * Bs
    t2 = tx * tx + ty * ty + tz * tz
    sx = 2.0 * tx / (1.0 + t2)
    sy = 2.0 * ty / (1.0 + t2)
    sz = 2.0 * tz / (1.0 + t2)
    ppx = p0 + (p1 * tz - p2 * ty)
    ppy = p1 + (p2 * tx - p0 * tz)
    ppz = p2 + (p0 * ty - p1 * tx)
    p0 = p0 + (ppy * sz - ppz * sy)
    p1 = p1 + (ppz * sx - ppx * sz)
    p2 = p2 + (ppx * sy - ppy * sx)
    # second half E kick
    p0 += coef * Ex
    p1 += coef * Ey
    p2 += coef * Es
    # second half energy
    ee = ee + 0.25 * q * d_tau0 * (p0 * Ex + p1 * Ey + p2 * Es)

    ## make half rotation
    # theta = 0.5 * h * ds
    # ct = np.cos(theta)
    # sthe = np.sin(theta)
    # px_new = ct * p0 + sthe * p2
    # ps_new = -sthe * p0 + ct * p2
    ## estimate x
    # g_new = 1 + h * x_new

    # estimate midpoint ps from the trial result (no projection inside EM)
    # ps_mid = 0.5 * (ps + p2)
    # print(g-g_new)
    d_tau2 = (g / p2) * ds

    # -------- corrector: recompute d_tau at midpoint and redo EM once
    # d_tau = (d_tau0+d_tau2)/2
    # pmid=1/(0.5*(1/ps+1/p2))
    d_tau = d_tau2

    # reset to entry state
    p0, p1, p2, ee = px, py, ps, e

    # now perform your current EM map once with d_tau (identical code as above)
    ee = ee + 0.5 * q * d_tau * (p0 * Ex + p1 * Ey + p2 * Es)
    coef = 0.5 * q * d_tau * (ee / (c * c))
    p0 += coef * Ex
    p1 += coef * Ey
    p2 += coef * Es
    tx = 0.5 * q * d_tau * Bx
    ty = 0.5 * q * d_tau * By
    tz = 0.5 * q * d_tau * Bs
    t2 = tx * tx + ty * ty + tz * tz
    sx = 2.0 * tx / (1.0 + t2)
    sy = 2.0 * ty / (1.0 + t2)
    sz = 2.0 * tz / (1.0 + t2)
    ppx = p0 + (p1 * tz - p2 * ty)
    ppy = p1 + (p2 * tx - p0 * tz)
    ppz = p2 + (p0 * ty - p1 * tx)
    p0 = p0 + (ppy * sz - ppz * sy)
    p1 = p1 + (ppz * sx - ppx * sz)
    p2 = p2 + (ppx * sy - ppy * sx)
    p0 += coef * Ex
    p1 += coef * Ey
    p2 += coef * Es
    ee = ee + 0.5 * q * d_tau * (p0 * Ex + p1 * Ey + p2 * Es)

    # write back
    st[4], st[5], st[6], st[7] = p0, p1, p2, ee


@njit(cache=True)
def _em_full_step_jit(state, ds, h, q, m, c, E_func, B_func, E_pars, B_pars):
    # Unpack
    x = state[1]
    y = state[2]
    s = state[0]
    t = state[3]
    px = state[4]
    py = state[5]
    ps = state[6]
    e = state[7]
    g = state[8]
    if ps == 0.0:
        raise ValueError("ps -> 0 during EM step")
    d_tau = (g / ps) * ds
    Ex, Ey, Es = E_func(x, y, s, t, h, E_pars)
    Bx, By, Bs = B_func(x, y, s, t, h, B_pars)
    # pack
    p0 = px
    p1 = py
    p2 = ps
    # half energy
    e = e + 0.5 * q * d_tau * (p0 * Ex + p1 * Ey + p2 * Es)
    e_half = e
    # half E kick
    coef = 0.5 * q * d_tau * (e_half / (c * c))
    p0 = p0 + coef * Ex
    p1 = p1 + coef * Ey
    p2 = p2 + coef * Es
    # magnetic rotation
    tx = 0.5 * q * d_tau * Bx
    ty = 0.5 * q * d_tau * By
    tz = 0.5 * q * d_tau * Bs
    t2 = tx * tx + ty * ty + tz * tz
    sx = 2.0 * tx / (1.0 + t2)
    sy = 2.0 * ty / (1.0 + t2)
    sz = 2.0 * tz / (1.0 + t2)
    # p' = p + p x t
    ppx = p0 + (p1 * tz - p2 * ty)
    ppy = p1 + (p2 * tx - p0 * tz)
    ppz = p2 + (p0 * ty - p1 * tx)
    # p  = p' + p' x s
    p0 = p0 + (ppy * sz - ppz * sy)
    p1 = p1 + (ppz * sx - ppx * sz)
    p2 = p2 + (ppx * sy - ppy * sx)
    # second half E kick
    p0 = p0 + coef * Ex
    p1 = p1 + coef * Ey
    p2 = p2 + coef * Es
    # second half energy
    e = e + 0.5 * q * d_tau * (p0 * Ex + p1 * Ey + p2 * Es)
    # write back
    state[4] = p0
    state[5] = p1
    state[6] = p2
    state[7] = e
    # g depends on x only; unchanged here


@njit(cache=True)
def _step_jit(state, ds, h, m, q, c, E_func, B_func, E_pars, B_pars):
    _kin_half_drift2_jit(state, ds, h, m, c)
    _geom_half_rot_jit(state, ds, h)
    _em_full_step_jit2(state, ds, h, q, m, c, E_func, B_func, E_pars, B_pars)
    _geom_half_rot_jit(state, ds, h)
    _kin_half_drift_jit(state, ds, h, m, c)


@njit(cache=True)
def _integrate_jit(state_array, s0, s1, ds, h, m, q, c, E_func, B_func, E_pars, B_pars):
    n = int(np.ceil((s1 - s0) / ds))
    ds_eff = (s1 - s0) / n
    S = np.empty(n + 1)
    X = np.empty(n + 1)
    Y = np.empty(n + 1)
    T = np.empty(n + 1)
    PX = np.empty(n + 1)
    PY = np.empty(n + 1)
    PS = np.empty(n + 1)
    EE = np.empty(n + 1)
    G = np.empty(n + 1)
    for i in range(n + 1):
        S[i] = state_array[0]
        X[i] = state_array[1]
        Y[i] = state_array[2]
        T[i] = state_array[3]
        PX[i] = state_array[4]
        PY[i] = state_array[5]
        PS[i] = state_array[6]
        EE[i] = state_array[7]
        G[i] = state_array[8]
        if i < n:
            _step_jit(state_array, ds_eff, h, m, q, c, E_func, B_func, E_pars, B_pars)
    return S, X, Y, T, PX, PY, PS, EE, G


# --- 4th-order (Yoshida) composition over s ---------------------------------


@njit(cache=True)
def _step4_jit(st, ds, h, m, q, c, E_func, B_func, E_pars, B_pars):
    # Yoshida (1990): 4th-order from 2nd-order kernel
    # w0, w1 are constants; sum is 1, composition is palindromic.
    two_13 = 2.0 ** (1.0 / 3.0)
    w1 = 1.0 / (2.0 - two_13)
    w0 = -two_13 / (2.0 - two_13)

    _step_jit(st, w1 * ds, h, m, q, c, E_func, B_func, E_pars, B_pars)
    _step_jit(st, w0 * ds, h, m, q, c, E_func, B_func, E_pars, B_pars)
    _step_jit(st, w1 * ds, h, m, q, c, E_func, B_func, E_pars, B_pars)


@njit(cache=True)
def _integrate4_jit(st_arr, s0, s1, ds, h, m, q, c, E_func, B_func, E_pars, B_pars):
    # same interface as _integrate_jit but uses 4th-order steps
    n = int(np.ceil((s1 - s0) / ds))
    ds_eff = (s1 - s0) / n

    S = np.empty(n + 1)
    X = np.empty(n + 1)
    Y = np.empty(n + 1)
    T = np.empty(n + 1)
    PX = np.empty(n + 1)
    PY = np.empty(n + 1)
    PS = np.empty(n + 1)
    EE = np.empty(n + 1)
    G = np.empty(n + 1)

    for i in range(n + 1):
        S[i] = st_arr[0]
        X[i] = st_arr[1]
        Y[i] = st_arr[2]
        T[i] = st_arr[3]
        PX[i] = st_arr[4]
        PY[i] = st_arr[5]
        PS[i] = st_arr[6]
        EE[i] = st_arr[7]
        G[i] = st_arr[8]
        if i < n:
            _step4_jit(st_arr, ds_eff, h, m, q, c, E_func, B_func, E_pars, B_pars)

    return S, X, Y, T, PX, PY, PS, EE, G


@njit(cache=True)
def _integrate4_jit_vect_final(
    st_arr, s0, s1, ds, h, m, q, c, E_func, B_func, E_pars, B_pars
):
    """vectorized version over multiple particles only returning final states"""
    n_particles = st_arr.shape[0]
    final_states = np.empty_like(st_arr)

    total_ds = s1 - s0
    n_steps = int(np.ceil(total_ds / ds))

    if n_steps <= 0:
        for ip in range(n_particles):
            for comp in range(9):
                final_states[ip, comp] = st_arr[ip, comp]
        return final_states

    ds_eff = total_ds / n_steps

    tmp_state = np.empty(9)
    for ip in range(n_particles):
        for comp in range(9):
            tmp_state[comp] = st_arr[ip, comp]

        for _ in range(n_steps):
            _step4_jit(tmp_state, ds_eff, h, m, q, c, E_func, B_func, E_pars, B_pars)

        for comp in range(9):
            final_states[ip, comp] = tmp_state[comp]

    return final_states


def make_state(s, x, y, t, px, py, e, h, m, c):
    ps = np.sqrt((e / c) ** 2 - (m * c) ** 2 - px**2 - py**2)
    g = 1.0 + h * x
    return np.array([s, x, y, t, px, py, ps, e, g], dtype=np.float64)


def integrate_numba(
    state_array, s_span, ds, h, efield, bfield, epars, bpars, m, q, c=clight
):
    S, X, Y, T, PX, PY, PS, EE, G = _integrate4_jit(
        state_array, s_span[0], s_span[1], ds, h, m, q, c, efield, bfield, epars, bpars
    )
    print("_integrate_jit signatures:", len(_integrate_jit.signatures))
    return {
        "s": S,
        "x": X,
        "y": Y,
        "t": T,
        "px": PX,
        "py": PY,
        "ps": PS,
        "e": EE,
        "g": G,
    }

def integrate_numba_vect_final(
    state_array, s_span, ds, h, efield, bfield, epars, bpars, m, q, c=clight
):
    final_states = _integrate4_jit_vect_final(
        state_array, s_span[0], s_span[1], ds, h, m, q, c, efield, bfield, epars, bpars
    )
    return final_states


def plot_zx(h, s, x, lbl="Particle R(s)", ref=True, ax=None):
    ch = np.cos(h * s)
    sh = np.sin(h * s)
    X_ref = (ch - 1.0) / h
    Z_ref = sh / h
    X = X_ref + x * ch
    Z = Z_ref + x * sh

    if ax is None:
        _, ax = plt.subplots()
    if ref:
        ax.plot(Z_ref, X_ref, linestyle="--", label="Reference R0(s)")
    ax.plot(Z, X, label=lbl)
    ax.set_xlabel("Z [m]")
    ax.set_ylabel("X [m]")
    ax.set_aspect("equal")
    ax.set_title("Physical-space trajectory (X–Z plane)")
    ax.legend()
    return ax


def plot_sy(s, y, lbl="Particle R(s)", ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(s, y, label=lbl)
    ax.set_xlabel("y [m]")
    ax.set_ylabel("s [m]")
    ax.set_title("Vertical trajectory")
    ax.legend()
    return ax


def plot_szeta(s, zeta, lbl="Particle R(s)", ax=None):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(s, zeta, label=lbl)
    ax.set_xlabel("zeta [m]")
    ax.set_ylabel("s [m]")
    ax.set_title("Longitudinal trajectory")
    ax.legend()
    return ax
