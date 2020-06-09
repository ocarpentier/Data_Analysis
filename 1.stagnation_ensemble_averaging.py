import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from scipy.interpolate import Rbf

def rbf2d1(x, y, z, x_new, y_new):
    try:
        """
        Optimal values
            x=-10:      500, 0.02
            x=10:       500, 0.02
            y=0:        500, 0.02
            z=0:        500, 0.02
        """
        rbf = Rbf(x, y, z, epsilon=500, smooth=0.02)
    except:
        return False
    z_new = rbf(x_new, y_new)
    return z_new

def rbf2d2(x, y, z, x_new, y_new):
    try:
        """
        Optimal values
            x=-10:      500, 0.02
            x=10:       500, 0.02
            y=0:        500, 0.02
            z=0:        500, 0.02
        """
        rbf = Rbf(x, y, z, epsilon=2, smooth=0.02)
    except:
        return False
    z_new = rbf(x_new, y_new)
    return z_new

def poly2(x, y, z, x_new, y_new):
    A = np.array([x*0+1, x, y, x*y, x**2, y**2, x**2*y, y**2*x, y**2*x**2]).T
    B = z
    coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
    a, b, c, d, e, f, g, h, i = coeff
    z_new = a + b*x_new + c*y_new + d*x_new*y_new + e*x_new**2 + f*y_new**2 + g*x_new**2*y_new + h*y_new**2*x_new + i*y_new**2*x_new**2
    return z_new


flst = os.listdir('C:/Users/oscar/OneDrive/Bureaublad/TU Delft/BSc 2de jaar/project/Test analysis and simulation/bins/z=0_wo_outliers')
k = 0
lst1 = []
lst2 = []
for i,xi in enumerate(flst) :
    fname = xi
    xi = xi[0:-5]
    xi = xi.split('_')

    if len(xi)<3:
        pass
    elif (xi[1]) == '0' and (xi[2]) == '0':

        df = pd.read_excel(f'C:/Users/oscar/OneDrive/Bureaublad/TU Delft/BSc 2de jaar/project/Test analysis and simulation/bins/z=0_wo_outliers/{fname}',skip_header=1)
        velarr = np.array(df)
        xavg = float(xi[0][6::])*-1
        lst1.append(np.array([xavg * 10 + 5, rbf2d1(velarr[:, 0], velarr[:, 1], velarr[:, 3], xavg * 10 + 5, 0)]))

# interpolating the bins
lst1 = np.array(lst1)

xx = np.linspace(-200,-70,200)
yy  =np.zeros(200)
uu1 = rbf2d2(lst1[:,0],np.zeros(len(lst1)),lst1[:,1],xx,yy)
uu2 = poly2(lst1[:,0],np.zeros(len(lst1)),lst1[:,1],xx,yy)

# potential flow
dfp = pd.read_excel('PotentialData.xlsx')
up = (np.array(dfp.iloc[:,16])[1:-2])
uplst = []
for i in range(len(up)):
    if (i-1)%3==0:
        uplst.append(up[i])
xxp = np.linspace(-200,200,len(uplst))
print(uplst)

#plotting
plt.plot([-200,-187,-173,-160,-147,-133,-120,-107,-93,-80],uplst[0:10],'o-',label='Potential flow')
plt.plot(lst1[:, 0], lst1[:, 1], 'v',label='Bin points')
plt.plot(xx,uu1,label='RBF interpolation method')
plt.plot(xx,uu2,label='Least squares interpolation method')
plt.grid(True)
plt.title('Stagnation Line Analysis',fontsize=20)
plt.xlabel('X [m]',fontsize=20)
plt.ylabel('u [m/s]',fontsize=20)
plt.legend(fontsize=20,loc='lower left',borderpad=2)

plt.show()





