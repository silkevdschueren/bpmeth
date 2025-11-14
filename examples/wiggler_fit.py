import numpy as np
import matplotlib.pyplot as plt

import bpmeth

def fit_segment(ia,ib,x,y,yp,order=3):
    x0=[x[ia],x[ib]]
    y0=[y[ia],y[ib]]
    yp0=[yp[ia],yp[ib]]
    xd=x[ia:ib+1]
    yd=y[ia:ib+1]
    pol=bpmeth.poly_fit.poly_fit(order,xd,yd,x0,y0,x0,yp0)
    return pol

def plot_fit(ia,ib,x,y,pol,data=False):
    xd=x[ia:ib+1]
    yd=y[ia:ib+1]
    yf=bpmeth.poly_fit.poly_val(pol,xd)
    if data:
       plt.plot(xd,yd,label='data')
    plt.plot(xd,yf,label=bpmeth.poly_fit.poly_print(pol))


x,y,z,bx,by,bz=np.loadtxt('example_data/Knot_map_test.txt').T
idx=(x==0)&(y==0)
x0,y0,z0,a1,b1,bs=x[idx],y[idx],z[idx],bx[idx],by[idx],bz[idx]
dz=np.ones_like(z0)*np.diff(z0)[0]
a1p=np.diff(a1,prepend=a1[0])/dz
b1p=np.diff(b1,prepend=b1[0])/dz


step=10
ii=np.arange(0,150,step)
ia,ib=ii[0],ii[-1]+step
plt.plot(z0[ia:ib+1],b1[ia:ib+1],lw=5,color='k',label='data')

for ia,ib in zip(ii,ii+step):
   pol=fit_segment(ia,ib,z0,b1,b1p,3)
   plot_fit(ia,ib,z0,b1,pol)

