import pandas as pd
import numpy as np
import re
import sys

from os import listdir
from scipy import interpolate
from scipy.interpolate import Rbf
import os.path
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from matplotlib.patches import Circle
from matplotlib.patches import Rectangle

from mpl_plotter_mpl_plotting_methods import MatPlotLibPublicationPlotter as mplPlotter
from mpl_plotter_colormaps import ColorMaps

"""
Uncertainty of mean estimates
"""


def remove_nan():
    un = pd.read_excel(
        r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\uncertainty_mean_estimates\ErrorEst.xlsx',
        index_col=0)
    un.columns = ['u', 'v', 'w']
    un = un[un['u'] != ' nan']
    un.to_csv(
        r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\uncertainty_mean_estimates\ErrorEst.csv')

# remove_nan()

un = pd.read_csv(
    r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\uncertainty_mean_estimates\ErrorEst.csv',
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
Interpolation and ensembling
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

            if os.path.isfile(
                    r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\{}\{}_wo_outliers\{}'.format(
                            plane, plane, file)):

                df = pd.read_excel(
                    r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\{}\{}_wo_outliers\{}'.format(
                        plane, plane, file))
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
        np.savetxt(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_ensemble_averaged\{}\{}_{}.txt'.format(plane, names[comp], version), msics[comp])

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
plane = 'y=0'
versions = ['1.0', 'polynomial']
version = versions[0]
unified_color = True
shrink = 0.69
cbtit_y = -5

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
    u_mosaic = np.loadtxt(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_ensemble_averaged\{}\{}_{}.txt'.format(plane, 'u', version))
    v_mosaic = np.loadtxt(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_ensemble_averaged\{}\{}_{}.txt'.format(plane, 'v', version))
    w_mosaic = np.loadtxt(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_ensemble_averaged\{}\{}_{}.txt'.format(plane, 'w', version))
except:
    interpolate_all(fill=fill, plane=plane, version=version, quirk=quirk, filenamestart=filenamestart, var=re.findall(r'-?\d+', plane)[0], f=f)

"""
Plot
"""

# Subplot setup
fig = mplPlotter(light=True).setup2d(figsize=(20, 5))

y_ticks = 8
x_ticks = 9 if plane == 'y=0' or plane == 'z=0' else 8
degree = 2
tsize=20
axsize = 20
pad = 15
tit_y = 1.05
cbtit_size = 15
fillsphere = True
aspect = 1
save = False
if plane == 'z=0' or plane == 'y=0':
    x_bounds = [0, 40]
    y_bounds = [0, 30]
    if version == 'polynomial':
        values = True
else:
    x_bounds = [0, 30]
    y_bounds = [0, 30]
    if version == 'polynomial' and plane == 'x=-10':
        values = False

if unified_color is True:
    values = True

if values is True:
    if plane == 'x=-10':
        actualmax1 = 12
        actualmin1 = 5

        actualmax2 = 2
        actualmin2 = -2

        actualmax3 = 2.5
        actualmin3 = -2.5
    if plane == 'x=10':
        actualmax1 = 15
        actualmin1 = -6

        actualmax2 = 6
        actualmin2 = -6

        actualmax3 = 5
        actualmin3 = -5
    if plane == 'y=0' or plane == 'z=0':
        actualmax1 = 14.5
        actualmin1 = -3

        actualmax2 = 6
        actualmin2 = -6

        actualmax3 = 6
        actualmin3 = -6
else:
    actualmax1 = None
    actualmin1 = None

    actualmax2 = None
    actualmin2 = None

    actualmax3 = None
    actualmin3 = None

"""
Potential flow
"""
U = 10  # [m/s]
a = 0.075  # [m]

z = np.linspace(-20, 20, 100)
r = np.linspace(-15, 15, 100)
z, r = np.meshgrid(z, r)

u = 10 * (((27) / (128000 * ((r/100) * (r/100) + (z/100) * (z/100)) ** 1.5)) + 1) - ((81 * (z/100) * (z/100)) / (12800 * ((z/100) * (z/100) + (r/100) * (r/100)) ** 2.5))
v = - ((81 * (z/100) * (r/100)) / (12800 * ((z/100) * (z/100) + (r/100) * (r/100)) ** 2.5))

ax = mplPlotter(light=True, fig=fig, shape_and_position=131).streamlines2d(x=z, y=r, u=u, v=v,
                                                                           color=u, density=1,
                                                                           resize_axes=True, yresize_pad=0, xresize_pad=0,
                                                                           cmap='RdBu_r', color_bar=True,
                                                                           more_subplots_left=True,
                                                                           plot_title='$y=0$: potential flow',
                                                                           shrink=0.5,
                                                                           cb_vmin=-2,
                                                                           cb_vmax=13,
                                                                           title_size=tsize,
                                                                           title_y=tit_y,
                                                                           cb_top_title_x=-1,
                                                                           x_tick_number=x_ticks,
                                                                           y_tick_number=y_ticks,
                                                                           x_ticklabels=[-20, 20],
                                                                           y_ticklabels=[-15, 15],
                                                                           cb_top_title=True,
                                                                           cb_top_title_pad=cbtit_y,
                                                                           cb_title='{} $[m/s]$'.format('u'),
                                                                           cb_title_weight='bold',
                                                                           cb_top_title_y=1.1,
                                                                           cb_title_size=cbtit_size,
                                                                           yaxis_labelpad=pad + 10,
                                                                           xaxis_label_size=axsize,
                                                                           yaxis_label_size=axsize,
                                                                           xaxis_bold=True,
                                                                           yaxis_bold=True,
                                                                           x_label='x $[cm]$', y_label='z $[cm]$',
                                                                           )

sphere_loc = (0, 0)

sphere = Circle(sphere_loc, 7.5, facecolor=(4/255, 15/255, 115/255), edgecolor=None, lw=2, fill=fillsphere, zorder=2)

ax.add_patch(sphere)

"""
Experimental
"""

x = np.linspace(-20, 20, 40)
y = np.linspace(-15, 15, 30)
x, y = np.meshgrid(x, y)

u = u_mosaic
v = v_mosaic
w = w_mosaic

mask = np.logical_or(u_mosaic != 0, u_mosaic != 0)

rule = u

ax = mplPlotter(light=True, fig=fig, shape_and_position=132).streamlines2d(x=x, y=y, u=u, v=v,
                                                                           color=u, density=5,
                                                                           resize_axes=True, yresize_pad=0, xresize_pad=0,
                                                                           cmap='RdBu_r', color_bar=True,
                                                                           more_subplots_left=True,
                                                                           plot_title='$y=0$: ensemble averaged flow',
                                                                           shrink=0.5,
                                                                           cb_vmin=-2,
                                                                           cb_vmax=13,
                                                                           title_size=tsize,
                                                                           title_y=tit_y,
                                                                           cb_top_title_x=-1,
                                                                           x_tick_number=x_ticks,
                                                                           y_tick_number=y_ticks,
                                                                           cb_top_title=True,
                                                                           cb_top_title_pad=cbtit_y,
                                                                           # cb_title='{} $[m/s]$'.format('u'),
                                                                           cb_title_weight='bold',
                                                                           cb_top_title_y=1.1,
                                                                           cb_title_size=cbtit_size,
                                                                           x_label=None, y_label=None,
                                                                           )

sphere_loc = (0, 0)

sphere = Circle(sphere_loc, 7.5, facecolor=(4/255, 15/255, 115/255), edgecolor=None, lw=2, fill=fillsphere, zorder=2)
ax.add_patch(sphere)

"""
High vorticity
"""

RdBu = cm.get_cmap('gray', 256)
gist_stern = cm.get_cmap('hot', 256)
newcolors = RdBu(np.linspace(0, 1, 256))
highlight = gist_stern(np.linspace(0, 1, 256))
newcolors[15:65, :] = highlight[45:95]
newcmp = ListedColormap(newcolors)

rule = u

ax = mplPlotter(light=True, fig=fig, shape_and_position=133).streamlines2d(x=x, y=y, u=u, v=v,
                                                                           color=rule, density=5,
                                                                           resize_axes=True, yresize_pad=0, xresize_pad=0,
                                                                           cmap=newcmp, color_bar=True,
                                                                           more_subplots_left=True,
                                                                           plot_title='$y=0$: high vorticity areas',
                                                                           shrink=0.5,
                                                                           cb_vmin=-2,
                                                                           cb_vmax=13,
                                                                           title_size=tsize,
                                                                           title_y=tit_y,
                                                                           cb_top_title_x=-1,
                                                                           x_tick_number=x_ticks,
                                                                           y_tick_number=y_ticks,
                                                                           cb_top_title=True,
                                                                           cb_top_title_pad=cbtit_y,
                                                                           # cb_title='{} $[m/s]$'.format('u'),
                                                                           cb_title_weight='bold',
                                                                           cb_top_title_y=1.1,
                                                                           cb_title_size=cbtit_size,
                                                                           x_label=None, y_label=None,
                                                                           )

sphere_loc = (0, 0)

sphere = Circle(sphere_loc, 7.5, facecolor=(4/255, 15/255, 115/255), edgecolor=None, lw=2, fill=fillsphere, zorder=2)
ax.add_patch(sphere)

plt.subplots_adjust(left=0.075, right=0.975, wspace=0.2, top=1.2, bottom=-0.2)
save = False
if save is True:
    plt.savefig(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\images\EA_PF_quiver.png',
                dpi=150)

plt.show()
