import pandas as pd
import numpy as np
import re
import sys

from matplotlib import pyplot as plt
from os import listdir
from scipy import interpolate
from scipy.interpolate import Rbf
import os.path

from mpl_plotting_methods import MatPlotLibPublicationPlotter as mplPlotter
from colormaps import ColorMaps

"""
Uncertainty of mean estimates
"""
def remove_nan():
    un = pd.read_excel(
                r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\uncertainty_mean_estimates\ErrorEst.xlsx',
                index_col=0)
    un.columns = ['u', 'v', 'w']
    un = un[un['u'] != ' nan']
    un.to_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\uncertainty_mean_estimates\ErrorEst.csv')

# remove_nan()

un = pd.read_csv(
                r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\uncertainty_mean_estimates\ErrorEst.csv',
                index_col=0)

"""
Interpolation functions
"""
def poly2(x, y, z, x_new, y_new):
    A = np.array([x*0+1, x, y, x*y, x**2, y**2, x**2*y, y**2*x, y**2*x**2]).T
    B = z
    coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
    a, b, c, d, e, f, g, h, i = coeff
    z_new = a + b*x_new + c*y_new + d*x_new*y_new + e*x_new**2 + f*y_new**2 + g*x_new**2*y_new + h*y_new**2*x_new + i*y_new**2*x_new**2
    return z_new

def poly4(x, y, z, x_new, y_new):
    A = np.array([x*0+1, x, y, x*y, x**2, x**2*y, x**2*y**2, y**2*x, y**2]).T
    B = z
    coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
    a, b, c, d, e, f, g, h, i = coeff
    z_new = a + b*x + c*y + d*x*y + e*x**2 + f*x**2*y + g*x**2*y**2 + h*y**2*x + i*y**2
    return z_new

def bicubic_fit(x, y, z, x_new, y_new):
    # Doesn't work with scattered data
    f_scipy = interpolate.interp2d(x, y, z, kind='cubic')
    z_new = f_scipy(x_new, y_new)
    return z_new

def rbf2d(x, y, z, x_new, y_new, epsilon, smooth, function):
    try:
        rbf = Rbf(x, y, z, epsilon=epsilon, smooth=smooth, function=function)
    except:
        return False
    z_new = rbf(x_new, y_new)
    return z_new

"""
Loop
"""

def interpolate_all(fill, epsilon, smooth, smooth2, comp, plane, threshold, function, epsilon2=None):
    br = False
    for k in range(-15, 16):
        i = 0
        for l in range(-15, 16):
            file = 'xminusten{}_{}_{}.xlsx'.format(-10, k, l)
            j = 0
            if os.path.isfile(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\{}\{}_wo_outliers\{}'.format(plane, plane, file)):
                df = pd.read_excel(
                        r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\{}\{}_wo_outliers\{}'.format(plane, plane, file))
                df.columns = ['x', 'y', 'z', 'u', 'v', 'w']
                x = df['x']
                y = df['y']
                z = df['z']
                u = df['u']
                v = df['v']
                w = df['w']

                # Interpolation setup
                loc_x, loc_y, loc_z = re.findall(r"-?\d+", file)
                loc_x = 10*int(loc_x)
                loc_y = 10*int(loc_y)
                loc_z = 10*int(loc_z)  # In mm

                yy = np.linspace(loc_y, loc_y + 10, 100)
                zz = np.linspace(loc_z, loc_z + 10, 100)
                yym, zzm = np.meshgrid(yy, zz)

                # Uncertainty of mean estimates
                index = file[9:-5]+"'"

                """
                Average u
                """
                if un.loc[index][0] > 0.05 and len(u.index) > threshold:
                    print('u interpolation')
                    print(u.size)
                    if comp == 'u':
                        br = True
                    # Interpolation
                    uu = poly2(x=y, y=z, z=u, x_new=yym, y_new=zzm)
                    uuu = rbf2d(x=y, y=z, z=u, x_new=yym, y_new=zzm, epsilon=epsilon, smooth=smooth, function=function)
                    uuuu = rbf2d(x=y, y=z, z=u, x_new=yym, y_new=zzm, epsilon=epsilon2 if not isinstance(epsilon2, type(None)) else epsilon, smooth=smooth2, function=function)

                    if uu is False or uuu is False or uuuu is False:
                        print('{}: u interpolation not converged'.format(file))
                        uu = fill
                        br = False

                else:
                    print('u mean')
                    uu = fill

                """
                Average v
                """
                if un.loc[index][1] > 0.05 and len(u.index) > threshold:
                    print('v interpolation')
                    print(v.size)
                    if comp == 'v':
                        br = True
                    # Interpolation
                    vv = poly2(x=y, y=z, z=v, x_new=yym, y_new=zzm)
                    vvv = rbf2d(x=y, y=z, z=v, x_new=yym, y_new=zzm, epsilon=epsilon, smooth=smooth, function=function)
                    vvvv = rbf2d(x=y, y=z, z=v, x_new=yym, y_new=zzm, epsilon=epsilon2 if not isinstance(epsilon2, type(None)) else epsilon, smooth=smooth2, function=function)
                    if vv is False or vvv is False or vvvv is False:
                        print('{}: u interpolation not converged'.format(file))
                        vv = fill
                        br = False

                else:
                    print('v mean')
                    vv = fill

                """
                Average w
                """
                if un.loc[index][2] > 0.05 and len(u.index) > threshold:
                    print('w interpolation')
                    print(w.size)
                    if comp == 'w':
                        br = True
                    # Interpolation
                    ww = poly2(x=y, y=z, z=w, x_new=yym, y_new=zzm)
                    www = rbf2d(x=y, y=z, z=w, x_new=yym, y_new=zzm, epsilon=epsilon, smooth=smooth, function=function)
                    wwww = rbf2d(x=y, y=z, z=w, x_new=yym, y_new=zzm, epsilon=epsilon2 if not isinstance(epsilon2, type(None)) else epsilon, smooth=smooth2, function=function)
                    if ww is False or www is False or wwww is False:
                        print('{}: u interpolation not converged'.format(file))
                        ww = fill
                        br = False
                else:
                    print('w mean')
                    ww = fill

            else:
                print('Empty file')
                uu = fill
                vv = fill
                ww = fill

            if br is True:
                break

        if br is True:
            break

    if comp == 'u':
        return [u, uu, uuu, uuuu], y, yy, z, zz, loc_x, loc_y, loc_z, file
    if comp == 'v':
        return [v, vv, vvv, vvvv], y, yy, z, zz, loc_x, loc_y, loc_z, file
    if comp == 'w':
        return [w, ww, www, wwww], y, yy, z, zz, loc_x, loc_y, loc_z, file

"""
Interpolation
"""

fill = 10
epsilon = 1
epsilon2 = 500
smooth = 0.0
smooth2 = 0.02
comp = 'v'
plane = 'x=-10'
threshold = 150
function = 'multiquadric'

arrcomp, y, yy, z, zz, loc_x, loc_y, loc_z, file = interpolate_all(fill=fill, epsilon=epsilon, epsilon2=epsilon2,
                                                                   smooth=smooth, smooth2=smooth2,
                                                                   comp=comp, plane=plane, threshold=threshold,
                                                                   function=function
                                                                   )

function = function.capitalize().replace("_", " ")

"""
Plot
"""

save = False
if save is True:
    dpi = 150
    if epsilon2 == 500:
        filename = r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\images\2DInterpolation_epsilon500_{}.png'.format(comp)
    else:
        filename = r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\images\2DInterpolation_{}_{}.png'.format(function, comp)
else:
    dpi = None
    filename = None

"""
Scatter
"""
widths = [1, 1, 1, 1]
heights = [1]
fig = mplPlotter(light=True).setup2d(figsize=(22, 7))
legendloc=(0.6, -0.5)
y_ticks = 6
x_ticks = 6
degree = 2
tsize=16
axsize = 15
pad = 15
tity = 1.05
shrink = 0.32

cb_vmin = arrcomp[0].min()
cb_vmax = arrcomp[0].max()
cmap = 'RdBu'

mplPlotter(light=True, fig=fig, shape_and_position=141).plot2d(scatter=True, x=y, y=z, c=arrcomp[0],
                                                               aspect=1,
                                                               pointsize=60,
                                                               marker='o',
                                                               grid=False, resize_axes=True,
                                                               color_bar=True,
                                                               cb_vmin=cb_vmin,
                                                               cb_vmax=cb_vmax,
                                                               cb_top_title=True,
                                                               cb_top_title_x=-0.8,
                                                               cb_axis_labelpad=10,
                                                               plot_title='${}$ data in ${}$ analysis plane \n bin: {}'.format(comp, plane, file[9:-5]),
                                                               title_size=tsize,
                                                               title_y=tity,
                                                               xaxis_labelpad=pad-10,
                                                               yaxis_labelpad=pad+10,
                                                               xaxis_label_size=axsize,
                                                               yaxis_label_size=axsize,
                                                               tick_ndecimals=1,
                                                               xaxis_bold=True,
                                                               yaxis_bold=True,
                                                               x_label='y $[mm]$', y_label='z $[mm]$',
                                                               cb_title='{} $[m/s]$'.format(comp),
                                                               cb_title_size=axsize,
                                                               cb_title_weight='bold',
                                                               cb_top_title_y=1.025,
                                                               x_bounds=[loc_y, loc_y+10],
                                                               y_bounds=[loc_z, loc_z+10],
                                                               x_tick_number=x_ticks,
                                                               y_tick_number=y_ticks,
                                                               linewidth=0,
                                                               more_subplots_left=True,
                                                               shrink=shrink,
                                                               cmap=cmap
                                                               )

"""
Polynomial Interpolation
"""
mplPlotter(light=True, fig=fig, shape_and_position=142).heatmap(x=yy, y=zz, z=arrcomp[1], tick_ndecimals=1,
                                                                xaxis_labelpad=0, yaxis_labelpad=0,
                                                                plot_title='Polynomial regression \n $2^{nd}$ degree',
                                                                x_label=None, y_label=None,
                                                                title_size=tsize,
                                                                title_y=tity,
                                                                color_bar=True,
                                                                cb_vmin=cb_vmin,
                                                                cb_vmax=cb_vmax,
                                                                cb_axis_labelpad=10,
                                                                x_bounds=[loc_y, loc_y + 10],
                                                                y_bounds=[loc_z, loc_z + 10],
                                                                more_subplots_left=True,
                                                                x_tick_number=x_ticks,
                                                                y_tick_number=y_ticks,
                                                                shrink=shrink,
                                                                cmap=cmap
                                                                )

"""
RBF Interpolation
"""
mplPlotter(light=True, fig=fig, shape_and_position=143).heatmap(x=yy, y=zz, z=arrcomp[2], tick_ndecimals=1,
                                                                xaxis_labelpad=0, yaxis_labelpad=0,
                                                                plot_title='{} radial basis interpolation \n $\epsilon={}$ | Scipy $smooth={}$'.format(function, epsilon, smooth),
                                                                x_label=None, y_label=None,
                                                                color_bar=True,
                                                                cb_vmin=cb_vmin,
                                                                cb_vmax=cb_vmax,
                                                                title_size=tsize,
                                                                title_y=tity,
                                                                cb_axis_labelpad=10,
                                                                x_bounds=[loc_y, loc_y + 10],
                                                                y_bounds=[loc_z, loc_z + 10],
                                                                more_subplots_left=True,
                                                                x_tick_number=x_ticks,
                                                                y_tick_number=y_ticks,
                                                                shrink=shrink,
                                                                cmap=cmap
                                                                )
# yyw, zzw = np.meshgrid(yy, zz)
# mplPlotter(light=True, fig=fig, shape_and_position=143).surface3d(x=yyw, y=zzw, z=arrcomp[2])
# save = False

"""
RBF Interpolation
"""
mplPlotter(light=True, fig=fig, shape_and_position=144).heatmap(x=yy, y=zz, z=arrcomp[3], tick_ndecimals=1,
                                                                xaxis_labelpad=0, yaxis_labelpad=0,
                                                                plot_title='{} radial basis interpolation \n $\epsilon={}$ | Scipy $smooth={}$'.format(function, epsilon2 if not isinstance(epsilon2, type(None)) else epsilon, smooth2),
                                                                x_label=None, y_label=None,
                                                                color_bar=True,
                                                                cb_vmin=cb_vmin,
                                                                cb_vmax=cb_vmax,
                                                                title_size=tsize,
                                                                title_y=tity,
                                                                cb_axis_labelpad=10,
                                                                x_bounds=[loc_y, loc_y + 10],
                                                                y_bounds=[loc_z, loc_z + 10],
                                                                more_subplots_left=True,
                                                                x_tick_number=x_ticks,
                                                                y_tick_number=y_ticks,
                                                                shrink=shrink,
                                                                cmap=cmap,
                                                                )

# yyw, zzw = np.meshgrid(yy, zz)
# mplPlotter(light=True, fig=fig, shape_and_position=144).surface3d(x=yyw, y=zzw, z=arrcomp[3])
# save = False

plt.subplots_adjust(left=0.075, right=0.975, wspace=0.2, top=1.2, bottom=-0.2)

if save is True:
    plt.savefig(dpi=dpi, fname=filename)

plt.show()
