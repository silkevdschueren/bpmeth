
import numpy as np
import matplotlib.pyplot as plt
import math
import bpmeth

"""
Analysis of a ring with a single sextupole
"""

npart = 50
part = np.zeros((4, npart))
part[0] = np.linspace(0, 0.9, npart)
Qx = 0.332
Qy = 0.221
nturns = 2**14
xlims=[-1,1] 
ylims=[-1,1]
b3 = 0.1

# Solution with kicks
line_kick = bpmeth.Line4d([bpmeth.Phase4d(Qx, Qy), bpmeth.Kick_x(b3, 2)])
o_kick = line_kick.track(part, num_turns=nturns)
o_kick.plot_xpx(xlims=xlims, ylims=ylims)

line_sext = bpmeth.Line4d([bpmeth.Phase4d(Qx, Qy), bpmeth.Sextupole(b3)])
o_sext = line_sext.track(part, num_turns=nturns)
o_sext.plot_xpx(xlims=xlims, ylims=ylims)

for i,tune in enumerate(np.arange(0.332,0.334,0.0001)):
   npart = 100
   part = np.zeros((4, npart))
   part[0] = np.linspace(0, 1, npart)
   line_sext = bpmeth.Line4d([bpmeth.Phase4d(tune, Qy), bpmeth.Sextupole(b3)])
   o_kick = line_sext.track(part, num_turns=nturns)
   o_kick.plot_xpx(xlims=xlims, ylims=ylims, savepath=f"tune_{i}.png")

# Solution with normal forms
h = np.zeros((4,4,4,4))
h[3,0,0,0] = b3/math.factorial(3)/8
h[0,3,0,0] = b3/math.factorial(3)/8
h[1,2,0,0] = 3*b3/math.factorial(2)/8
h[2,1,0,0] = 3*b3/math.factorial(2)/8
h[1,0,2,0] = -3*b3/math.factorial(3)/8
h[0,1,2,0] = -3*b3/math.factorial(3)/8
h[1,0,0,2] = -3*b3/math.factorial(3)/8
h[0,1,0,2] = -3*b3/math.factorial(3)/8
h[1,0,1,1] = -6*b3/math.factorial(3)/8
h[0,1,1,1] = -6*b3/math.factorial(3)/8

sext4d = bpmeth.NormalForms4d(h, 2*np.pi*Qx, 2*np.pi*Qy, Qx, Qy, nturns)
o_norm = sext4d.calc_coords(part)
o_norm.plot_xpx(xlims=xlims, ylims=ylims)

# Solution with numerical integration
line_num = bpmeth.Line4d([bpmeth.Phase4d(Qx, Qy), bpmeth.NumericalSextupole(-100, b3/100)])
o_num = line_num.track(part, num_turns=nturns)
o_num.plot_xpx(xlims=xlims, ylims=ylims)

index=95
o_kick.plot_spectrum_x(index)
o_norm.plot_spectrum_x(index)
o_num.plot_spectrum_x(index)
