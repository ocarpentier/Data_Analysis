import pandas as pd
import numpy as np
import re
import sys

from os import listdir
from scipy import interpolate
from scipy.interpolate import Rbf
import os.path

import matplotlib.pyplot as plt

from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

from mpl_plotter_mpl_plotting_methods import MatPlotLibPublicationPlotter as mplPlotter

"""
Uncertainty of mean estimates
"""
data_analysis = os.path.dirname(__file__)
data_path = os.path.join(data_analysis, 'data')
img_path = os.path.join(data_analysis, 'images')
bin_path = os.path.join(data_path, 'bins')
uncertainty_path = os.path.join(data_path, 'uncertainty_mean_estimate')
ens_avg_path = os.path.join(data_path, 'velocity_ensemble_averaged')

def remove_nan():
    un = pd.read_excel(
        os.path.join(uncertainty_path, 'ErrorEst.xlsx'),
        index_col=0)
    un.columns = ['u', 'v', 'w']
    un = un[un['u'] != ' nan']
    un.to_csv(
        os.path.join(uncertainty_path, 'ErrorEst.csv'))

# remove_nan()

un = pd.read_csv(
    os.path.join(uncertainty_path, 'ErrorEst.csv'),
    index_col=0)

"""
RBF interpolation
    epsilon = 1
    smooth = 0.03
"""
def rbf2d(x, y, z, x_new, y_new):
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

def poly2(x, y, z, x_new, y_new):
    A = np.array([x*0+1, x, y, x*y, x**2, y**2, x**2*y, y**2*x, y**2*x**2]).T
    B = z
    coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
    a, b, c, d, e, f, g, h, i = coeff
    z_new = a + b*x_new + c*y_new + d*x_new*y_new + e*x_new**2 + f*y_new**2 + g*x_new**2*y_new + h*y_new**2*x_new + i*y_new**2*x_new**2
    return z_new

"""
 """
def interpolate_all(fill, plane, version, quirk, filenamestart, var, f):
    if quirk == 'xten' or quirk == 'xminusten':
        array_shape = (30, 30)
        k_top = 15
    else:
        array_shape = (30, 40)
        k_top = 20
    u_mosaic = np.empty(array_shape)
    v_mosaic = np.empty(array_shape)
    w_mosaic = np.empty(array_shape)
    i = 0
    for k in range(-k_top, k_top+1):
        for l in range(-15, 16):
            if quirk == 'xten' or quirk == 'xminusten':
                file = '{}{}_{}_{}.xlsx'.format(quirk, var, k, l)
            if quirk == 'yzero':
                file = '{}{}_{}_{}.xlsx'.format(quirk, k, var, l)
            if quirk == 'zzero':
                file = '{}{}_{}_{}.xlsx'.format(quirk, k, l, var)

            if os.path.isfile(os.path.join(bin_path, '{}_wo_outliers\{}'.format(
                              plane, file))):

                df = pd.read_excel(os.path.join(bin_path, '{}_wo_outliers\{}'.format(
                                   plane, file)))
                df.columns = ['x', 'y', 'z', 'u', 'v', 'w']
                x = df['x']
                y = df['y']
                z = df['z']
                u = df['u']
                v = df['v']
                w = df['w']

                # Interpolation setup
                loc_x, loc_y, loc_z = re.findall(r"-?\d+", file)
                loc_x = 10 * int(loc_x)
                loc_y = 10 * int(loc_y)
                loc_z = 10 * int(loc_z)  # In mm

                xx = loc_x+5
                yy = loc_y+5
                zz = loc_z+5
                if quirk == 'xten' or quirk == 'xminusten':
                    x1 = y
                    x2 = z
                    x_new = yy
                    y_new = zz
                if quirk == 'yzero':
                    x1 = x
                    x2 = z
                    x_new = xx
                    y_new = zz
                if quirk == 'zzero':
                    x1 = x
                    x2 = y
                    x_new = xx
                    y_new = yy

                # Uncertainty of mean estimates
                index = file[filenamestart:-5] + "'"

                """
                Average u
                """
                if un.loc[index][0] > 0.01:
                    # Interpolation
                    uu = f(x=x1, y=x2, z=u, x_new=x_new, y_new=y_new)

                    if uu is False:
                        print('{}: u interpolation not converged'.format(file))
                        uu = fill
                    else:
                        uu = uu.item()

                else:
                    uu = fill

                """
                Average v
                """
                if un.loc[index][1] > 0.01:
                    # Interpolation
                    vv = f(x=x1, y=x2, z=v, x_new=x_new, y_new=y_new)

                    if vv is False:
                        print('{}: u interpolation not converged'.format(file))
                        vv = fill
                    else:
                        vv = vv.item()

                else:
                    vv = fill

                """
                Average w
                """
                if un.loc[index][2] > 0.01:
                    # Interpolation
                    ww = f(x=x1, y=x2, z=w, x_new=x_new, y_new=y_new)

                    if ww is False:
                        print('{}: u interpolation not converged'.format(file))
                        ww = fill
                    else:
                        ww = ww.item()
                else:
                    ww = fill

            else:
                uu = fill
                vv = fill
                ww = fill

            if k < k_top and l < 15:
                u_mosaic[l+15][k+k_top] = uu
                v_mosaic[l+15][k+k_top] = vv
                w_mosaic[l+15][k+k_top] = ww

        print(i)
        i = i + 1

    names = ['u', 'v', 'w']
    msics = [u_mosaic, v_mosaic, w_mosaic]
    for comp in range(3):
        np.savetxt(os.path.join(ens_avg_path, r'{}\{}_{}.txt'.format(plane, names[comp], version)), msics[comp])

    return u_mosaic, v_mosaic, w_mosaic

def find_real_extremes(mosaic):
    df = pd.DataFrame(mosaic)
    df = df.loc[:, (df != 0).any(axis=0)]
    min = df.min().min()
    max = df.max().max()
    return max, min

"""
Ensemble averaging plot setup
"""

fill = 0
plane = 'x=-10'
versions = ['rbf', 'polynomial']
version = versions[0]
unified_color = True
shrink = 0.69
cbtit_y = -5
surface = False
save = False

if version == 'rbf':
    f = rbf2d
if version == 'polynomial':
    f = poly2

if plane == 'z=0':
    quirk = 'zzero'
if plane == 'y=0':
    quirk = 'yzero'
if plane == 'x=10':
    quirk = 'xten'
if plane == 'x=-10':
    quirk = 'xminusten'

filenamestart = len(quirk)

try:
    u_mosaic = np.loadtxt(os.path.join(ens_avg_path, '{}\{}_{}.txt'.format(plane, 'u', version)))
    v_mosaic = np.loadtxt(os.path.join(ens_avg_path, '{}\{}_{}.txt'.format(plane, 'v', version)))
    w_mosaic = np.loadtxt(os.path.join(ens_avg_path, '{}\{}_{}.txt'.format(plane, 'w', version)))
except:
    u_mosaic, v_mosaic, w_mosaic = interpolate_all(fill=fill, plane=plane, version=version, quirk=quirk, filenamestart=filenamestart, var=re.findall(r'-?\d+', plane)[0], f=f)


