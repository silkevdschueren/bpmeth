import bpmeth
import sympy as sp
import numba
import numpy as np

"""
fieldder[0] -> bs
fieldder[1] -> b1
fieldder[2] -> a1
fieldder[3] -> b2
fieldder[4] -> a2
fieldder[5] -> b3
fieldder[6] -> a3
fieldder[7] -> b4
fieldder[8] -> a4
"""

comps = []


def mk_s_poly(cpmidx, sorder):
    ss = [sp.Symbol(f"fieldder[{cpmidx},{ii}]", real=True) for ii in range(sorder + 1)]
    s = sp.var("s", real=True)
    return sum(ss[i] * s**i for i in range(len(ss)))


def mk_fieldder_sp(sorder, ab_order):
    comps = [mk_s_poly(0, sorder)]
    for ii in range(ab_order * 2):
        comps.append(mk_s_poly(ii + 1, sorder))
    return comps


def mk_field(ab_order=4, sorder=3, h=True, out=None):
    fd = mk_fieldder_sp(sorder, ab_order)
    if h:
        h = sp.var("h", real=True)
    else:
        h = "0"
    b = fd[1 : ab_order * 2 + 1 : 2]
    a = fd[2 : ab_order * 2 + 1 : 2]
    vp = bpmeth.GeneralVectorPotential(bs=fd[0], b=b, a=a, hs=h)
    Bx_sp, By_sp, Bs_sp = vp.get_Bfield(lambdify=False)
    Ax_sp, Ay_sp, As_sp = vp.get_A(lambdify=False)
    # pp=lambda ex:ex.expand().simplify()
    # Bx_sp,By_sp,Bs_sp=map(pp,(Bx_sp,By_sp,Bs_sp))
    # Ax_sp,Ay_sp,As_sp=map(pp,(Ax_sp,Ay_sp,As_sp))
    if out is None:
        return Bx_sp, By_sp, Bs_sp
    elif out is numba:
        Bx = numba.njit(eval(f"lambda x,y,s,h,t,fieldder: {Bx_sp}"))
        By = numba.njit(eval(f"lambda x,y,s,h,t,fieldder: {By_sp}"))
        Bs = numba.njit(eval(f"lambda x,y,s,h,t,fieldder: {Bs_sp}"))
        Ax = numba.njit(eval(f"lambda x,y,s,h,t,fieldder: {Ax_sp}"))
        Ay = numba.njit(eval(f"lambda x,y,s,h,t,fieldder: {Ay_sp}"))
        As = numba.njit(eval(f"lambda x,y,s,h,t,fieldder: {As_sp}"))
        comp = np.zeros((ab_order * 2 + 1, sorder + 1))
        return (Bx, By, Bs), (Ax, Ay, As), comp
    elif out is str:
        print(Bx_sp)
        print(By_sp)
        print(Bs_sp)
    elif out is str:
        print(Bx_sp)
        print(By_sp)
        print(Bs_sp)
    else:
        src = ["import numba"]
        src.append("@numba.njit(cache=True)")
        src.append(f"def bfield(x,y,s,h,t,fieldder):")
        src.append(f"  return {Bx_sp},{By_sp},{Bs_sp}")
        src.append("@numba.njit(cache=True)")
        src.append(f"def afield(x,y,s,h,t,fieldder):")
        src.append(f"  return {Ax_sp},{Ay_sp},{As_sp}")
        src = "\n".join(src)
        open(out, "w").write(src)
        print(out)
        comp = np.zeros((ab_order * 2 + 1, sorder + 1))
        return comp


if __name__ == "__main__":
    for ab in [1, 2, 3, 4]:
        for sorder in [3, 4, 5]:
            for h in True, False:
                if h:
                    fn = f"ba_fields_{ab}_{sorder}_h.py"
                else:
                    fn = f"ba_fields_{ab}_{sorder}.py"
                mk_field(ab_order=ab, sorder=sorder, h=h, out=fn)
