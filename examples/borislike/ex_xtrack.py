import numpy as np
import matplotlib.pyplot as plt
import xtrack as xt
import time

from curvedboris import *
from cubic_bp import bfield, afield

@njit(cache=True)
def efield(x, y, s, t, h, pars):
    return 0, 0, 0

class CubicMagnet:
    is_thick=True
    def __init__(self,comp,length,ds,h):
        self.comp=comp
        self.length=length
        self.ds=ds
        self.h=h

    def track(self,part):
        c = 299_792_458.0
        qe = +1.602176634e-19
        q= qe * part.q0
        m = part.mass0 * qe / c**2
        p0_SI = (part.p0c * qe) / c
        epars = np.array([0.0], dtype=np.float64)
        px0 = (part.px-part.ax) * p0_SI
        py0 = (part.py-part.ay) * p0_SI


        ax,ay,_=afield(x,y,s,0,self.h,self.comp)

        pass





