import numpy as np
import matplotlib.pyplot as plt
import math
import sympy as sp
import bpmeth
import mpmath

"""
Analysis of a ring with a single fringe field
"""

def K0ggtanh(b1, a, L):
    """    
    K0 fringe field integral for a tanh times the gap height squared.

    :param b1: fringe field coefficient.
    :param a: fringe field range.
    :param L: integral around the fringe field ranges from -L to L.
        (In theory the value of K0gg should not depend on L, otherwise L is chosen too small.)
    """

    return b1* ( L**2/2 - L*a/2*np.log((1+np.exp(-2*L/a))*(1+np.exp(2*L/a))) \
        + a**2/4*(mpmath.polylog(2, -np.exp(-2*L/a))-mpmath.polylog(2, -np.exp(2*L/a))) )

######################
# Fringe field specs #
######################

npart = 10
part = np.zeros((4, npart))
part[0] = np.linspace(0, 0.5, npart)
part[2] = np.linspace(0, 0.5, npart)
Qx = 10.112
Qxdec = Qx%1
Qy = 10.252
Qydec = Qy%1
nturns = 1000
xlims=[-1,1] 
ylims=[-1,1]

b1 = 0.1
aa = 1
Kg = aa/2
b1shape = f"(tanh(s/{aa})+1)/2"
s = sp.symbols("s")
b1sym = eval(b1shape, sp.__dict__, {"s": s})
length = 5
K0gg = K0ggtanh(b1, aa, length/2)


######################
# Fringe field maps  #
######################

print("Calculating thin fringe")
line_thinfringe = bpmeth.Line4d([bpmeth.Phase4d(Qx, Qy), bpmeth.ThinNumericalFringe(b1, b1shape, length=length, nphi=5)])
o_thinfringe = line_thinfringe.track(part, num_turns=nturns)
# o_thinfringe.plot_xpx(xlims=xlims, ylims=ylims)
# o_thinfringe.plot_ypy(xlims=xlims, ylims=ylims)

print("Calculating forest fringe")
line_forest = bpmeth.Line4d([bpmeth.Phase4d(Qx, Qy), bpmeth.ForestFringe(b1, Kg)])
o_forest = line_forest.track(part, num_turns=nturns)
# o_forest.plot_xpx(xlims=xlims, ylims=ylims)
# o_forest.plot_ypy(xlims=xlims, ylims=ylims)

print("Calculating forest fringe with closed orbit")
line_forest_co = bpmeth.Line4d([bpmeth.Phase4d(Qx, Qy), bpmeth.ForestFringe(b1, Kg, K0gg, closedorbit=True)])
o_forest_co = line_forest_co.track(part, num_turns=nturns)
# o_forest_co.plot_xpx(xlims=xlims, ylims=ylims)
# o_forest_co.plot_ypy(xlims=xlims, ylims=ylims)


######################
# Normal forms       #
######################

print("Calculating normal forms")

nsvals = 50
ds = length/(nsvals-1)
svals = np.arange(-length/2, length/2+ds, ds)

h = np.zeros((5,5,5,5,nsvals), dtype=complex)

# First term Hamiltonian: closed orbit distortion
h[1,0,0,0] = [1/2 * b1sym.subs({s:sval}).evalf() * ds for sval in svals]
h[0,1,0,0] = [1/2 * b1sym.subs({s:sval}).evalf() * ds for sval in svals]

# Second term Hamiltonian: "sextupole-like"
h[1,0,2,0] = [-1/16 * 1j * b1sym.diff(s).subs({s:sval}).evalf() * ds for sval in svals] 
h[0,1,2,0] = [ 1/16 * 1j * b1sym.diff(s).subs({s:sval}).evalf() * ds for sval in svals] 
h[1,0,0,2] = [-1/16 * 1j * b1sym.diff(s).subs({s:sval}).evalf() * ds for sval in svals] 
h[0,1,0,2] = [ 1/16 * 1j * b1sym.diff(s).subs({s:sval}).evalf() * ds for sval in svals] 
h[1,0,1,1] = [-1/16 * 2 * 1j * b1sym.diff(s).subs({s:sval}).evalf() * ds for sval in svals] 
h[0,1,1,1] = [ 1/16 * 2 * 1j * b1sym.diff(s).subs({s:sval}).evalf() * ds for sval in svals]

h[1,0,4,0] = [1j / 32 * 1/ 24 * b1sym.diff(s, 3).subs({s:sval}).evalf() * ds for sval in svals] 
h[0,1,4,0] = [1j / 32 * 1/ 24 * b1sym.diff(s, 3).subs({s:sval}).evalf() * ds for sval in svals]
h[1,0,3,1] = [1j / 32 * 1/ 6 * b1sym.diff(s, 2).subs({s:sval}).evalf() * ds for sval in svals]
h[0,1,3,1] = [1j / 32 * 1/ 6 * b1sym.diff(s, 2).subs({s:sval}).evalf() * ds for sval in svals]
h[1,0,2,2] = [1j / 32 * 1/ 4 * b1sym.diff(s, 1).subs({s:sval}).evalf() * ds for sval in svals]
h[0,1,2,2] = [1j / 32 * 1/ 4 * b1sym.diff(s, 1).subs({s:sval}).evalf() * ds for sval in svals]
h[1,0,1,3] = [1j / 32 * 1/ 6 * b1sym.diff(s, 2).subs({s:sval}).evalf() * ds for sval in svals]
h[0,1,1,3] = [1j / 32 * 1/ 6 * b1sym.diff(s, 2).subs({s:sval}).evalf() * ds for sval in svals]
h[1,0,0,4] = [1j / 32 * 1/ 24 * b1sym.diff(s, 3).subs({s:sval}).evalf() * ds for sval in svals]
h[0,1,0,4] = [1j / 32 * 1/ 24 * b1sym.diff(s, 3).subs({s:sval}).evalf() * ds for sval in svals]

# Third term Hamiltonian
h[0,0,4,0] = [3/16 / 24 * b1sym.diff(s).subs({s:sval}).evalf()**2 * ds for sval in svals]
h[0,0,3,1] = [3/16 / 6 *  b1sym.diff(s).subs({s:sval}).evalf()**2 * ds for sval in svals]
h[0,0,1,3] = [3/16 / 6 *  b1sym.diff(s).subs({s:sval}).evalf()**2 * ds for sval in svals]
h[0,0,0,4] = [3/16 / 24 * b1sym.diff(s).subs({s:sval}).evalf()**2 * ds for sval in svals]
# Detuning term: not included in normal forms??
h[0,0,2,2] = [3/16 / 4 *  b1sym.diff(s).subs({s:sval}).evalf()**2 * ds for sval in svals]


# Phase advance: beta=1 so the phase advance is equal to the s position
# Fringe field is situated at the end of the lattice
phi_x = (svals + Qx) % Qx
phi_y = (svals + Qy) % Qy

frin_normalforms = bpmeth.NormalForms4d(h, phi_x, phi_y, Qx, Qy, nturns)
o_normalforms = frin_normalforms.calc_coords(part)


######################
# Plot spectra       #
######################

index=1
padding=nturns
log=True
unwrap=False


figx, axx = plt.subplots(2)
plt.title('Spectrum x')

o_thinfringe.plot_spectrum_x(index, ax=axx[0], padding=padding, 
                             label="Thin", log=log, plot_phase=True, unwrap=unwrap, ax_phase=axx[1], 
                             color="orange")
o_forest.plot_spectrum_x(index, ax=axx[0], padding=padding, 
                         label="Forest", log=log, plot_phase=True, unwrap=unwrap, ax_phase=axx[1], 
                         color="green")
o_forest_co.plot_spectrum_x(index, ax=axx[0], padding=padding, 
                            label="Forest with closed orbit", log=log, plot_phase=True, unwrap=unwrap, ax_phase=axx[1], 
                            color="blue")
o_normalforms.plot_spectrum_x(index, ax=axx[0], padding=padding,
                              label="Normal forms", log=log, plot_phase=True, unwrap=unwrap, ax_phase=axx[1], 
                              color="purple")

sl_x = [(0,0), (0, -2), (0, 2), (0, -4), (0, 4)]
for xx, yy in sl_x:
    val = xx*Qxdec + yy*Qydec
    axx[0].vlines((val + 0.5) % 1 - 0.5, -9, -7, color="red")
    axx[0].text((val + 0.5) % 1 - 0.5, -11, f"({xx},{yy})", color="red",
                horizontalalignment="center", verticalalignment="center")

axx[0].set_xlim(-0.6, 0.6)
axx[1].set_xlim(-0.6, 0.6)
axx[0].set_ylabel("Amplitude")
axx[1].set_ylabel("Phase")


figy, axy = plt.subplots(2)
plt.title('Spectrum y')

o_thinfringe.plot_spectrum_y(index, ax=axy[0], padding=padding, 
                             label="Thin", log=log, plot_phase=True, unwrap=unwrap, ax_phase=axy[1], 
                             color="orange")
o_forest.plot_spectrum_y(index, ax=axy[0], padding=padding, 
                         label="Forest", log=log, plot_phase=True, unwrap=unwrap, ax_phase=axy[1], 
                         color="green")
o_forest_co.plot_spectrum_y(index, ax=axy[0], padding=padding, 
                            label="Forest with closed orbit", log=log, plot_phase=True, unwrap=unwrap, ax_phase=axy[1], 
                            color="blue")
o_normalforms.plot_spectrum_y(index, ax=axy[0], padding=padding,
                              label="Normal forms", log=log, plot_phase=True, unwrap=unwrap, ax_phase=axy[1], 
                              color="purple")

sl_y = [(-1, -1), (1, -1), (-1, 1), (1, 1), (-1, -3), (1, -3), (-1, 3), (1, 3), (0, 3), (0, 1), (0, -1), (0, -3)]
for xx, yy in sl_y:
    val = xx*Qxdec + yy*Qydec
    axy[0].vlines((val + 0.5) % 1 - 0.5, -7, -5, color="red")
    axy[0].text((val + 0.5) % 1 - 0.5, -9, f"({xx},{yy})", color="red", 
                horizontalalignment="center", verticalalignment="center")

axy[0].set_xlim(-0.6, 0.6)
axy[1].set_xlim(-0.6, 0.6)
axy[0].set_ylabel("Amplitude")
axy[1].set_ylabel("Phase")


