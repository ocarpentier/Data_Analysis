import pandas as pd
import numpy as np

import scipy.interpolate as int
from scipy.signal import lfilter
from scipy.signal import savgol_filter
import scipy.optimize as optimize

from mpl_plotter_mpl_plotting_methods import MatPlotLibPublicationPlotter as mplPlotter

# Plot setup
fig = mplPlotter(light=True).setup2d(figsize=(7, 15))
legendloc = (0.6, -0.65)
pointsize = 3

# File read
df = pd.read_csv(
        r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\stagnation_line\stagnation_line_z=0.csv', index_col=0)

df.columns = ['x','y','z','u','v','w']

xticks = 6
yticks = 6

# Experimental data
x = df['x']
u = df['u']
ax1 = mplPlotter(light=True, fig=fig, shape_and_position=411, usetex=False).plot2d(x=x, y=u, grid=True,
                                                                                   resize_axes=True, scatter=True,
                                                                                   color='red', label='Experimental',
                                                                                   pointsize=pointsize,
                                                                                   more_subplots_left=True,
                                                                                   )

"""
Denoising
"""
def rolling(x, u, n=2):
    u = u.to_numpy()
    uret = np.cumsum(u)
    uret[n:] = uret[n:] - uret[:-n]

    x = x.to_numpy()
    xret = np.cumsum(x)
    xret[n:] = xret[n:] - xret[:-n]
    return uret[n - 1:]/n, xret[n - 1:]/n

def Savitzky_Golay(u):
    u_smooth = savgol_filter(u, 25, 2)
    return u_smooth

def IIR(u):
    n = 15                  # The larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    u_smooth = lfilter(b, a, u)
    return u_smooth

u_smooth, x_smooth = rolling(x=x, u=u)

mplPlotter(fig=fig, ax=ax1, shape_and_position=412, usetex=False).plot2d(x=x_smooth, y=u_smooth, grid=True,
                                                                         resize_axes=True, scatter=True, color='blue',
                                                                         plot_title='Rolling average denoising',
                                                                         title_y=1.05,
                                                                         legend=True, legendloc=legendloc,
                                                                         label='Denoised', pointsize=pointsize, title_size=15,
                                                                         xaxis_label_size=20, yaxis_labelpad=10,
                                                                         y_label='$u$', yaxis_label_size=20,
                                                                         x_ticklabel_size=10, y_ticklabel_size=10,
                                                                         y_tick_number=yticks, x_tick_number=xticks,
                                                                         gridlines='dotted',
                                                                         more_subplots_left=True,
                                                                         aspect=3,
                                                                         x_bounds=[-200, -70],
                                                                         y_bounds=[0, 12],
                                                                         x_ticklabels=[-200, -70],
                                                                         y_ticklabels=[0, 12],
                                                                         tick_ndecimals=2,
                                                                         prune=None)

"""
Interpolation
"""
def poly_fit(x, u, x_smooth, u_smooth, degree):
    std = u.std()
    std_smooth = np.std(u_smooth)
    # f_exp_coeff = np.polyfit(x=x, y=u, deg=degree, w=u/std)
    # f_smooth_coeff = np.polyfit(x=x_smooth, y=u_smooth, deg=degree, w=u_smooth/std_smooth)
    p_exp = np.poly1d(np.polyfit(x=x, y=u, deg=degree,
                                 # w=u/std
                                 ))
    p_smooth = np.poly1d(np.polyfit(x=x_smooth, y=u_smooth, deg=degree,
                                    # w=u_smooth/std_smooth
                                    ))
    xx = np.linspace(-200, -70, 1000)
    u_exp_interpolated = p_exp(xx)
    u_smooth_interpolated = p_smooth(xx)
    return u_exp_interpolated, u_smooth_interpolated, xx

def rbf(x, u, x_smooth, u_smooth, variant, smooth):
    u_exp_int = int.Rbf(x, u, function=variant, smooth=smooth)
    u_smooth_int = int.Rbf(x_smooth, u_smooth, function=variant, smooth=smooth)
    xx = np.linspace(-200, -70, 1000)
    u_exp_interpolated = u_exp_int(xx)
    u_smooth_interpolated = u_smooth_int(xx)
    return u_exp_interpolated, u_smooth_interpolated, xx

def cubicspline(x, u, x_smooth, u_smooth, variant, smooth):
    if variant == 'linear':
        k = 1
    if variant == 'quadratic':
        k = 2
    if variant == 'cubic':
        k = 3
    df_exp = pd.DataFrame([x, u]).T
    df_exp.columns = ['x', 'u']
    df_exp.sort_values(['x'], inplace=True)

    df_smooth = pd.DataFrame([x_smooth, u_smooth]).T
    df_smooth.columns = ['x_smooth', 'u_smooth']
    df_smooth.sort_values(['x_smooth'], inplace=True)

    tck_exp = int.splrep(x=df_exp['x'], y=df_exp['u'], s=smooth, k=k)
    tck_smooth = int.splrep(x=df_smooth['x_smooth'], y=df_smooth['u_smooth'], s=smooth, k=k)
    xx = np.linspace(-200, -70, 1000)
    u_exp_new = int.splev(xx, tck_exp, der=0)
    u_smooth_new = int.splev(xx, tck_smooth, der=0)
    return u_exp_new, u_smooth_new, xx




"""
Poly fit plot
"""
degree = 2
u_exp_fit, u_smooth_fit, xx = poly_fit(x=x, u=u, x_smooth=x_smooth, u_smooth=u_smooth, degree=2)

ax2 = mplPlotter(light=True, fig=fig, shape_and_position=412, usetex=False).plot2d(x=x, y=u,
                                                                                   grid=True, resize_axes=False,
                                                                                   scatter=True, color='red',
                                                                                   xaxis_label_size=20,
                                                                                   yaxis_labelpad=10,
                                                                                   y_label='$u$',
                                                                                   yaxis_label_size=20,
                                                                                   x_ticklabel_size=10,
                                                                                   y_ticklabel_size=10,
                                                                                   y_tick_number=yticks,
                                                                                   x_tick_number=xticks,
                                                                                   more_subplots_left=True,
                                                                                   pointsize=pointsize)
mplPlotter(light=True, fig=fig, ax=ax2, shape_and_position=412, usetex=False).plot2d(x=xx, y=u_exp_fit,
                                                                                     grid=True, resize_axes=False,
                                                                                     color='darkred', linewidth=1,
                                                                                     more_subplots_left=True)

mplPlotter(light=True, fig=fig, ax=ax2, shape_and_position=412, usetex=False).plot2d(x=x_smooth, y=u_smooth,
                                                                                     grid=True,
                                                                                     resize_axes=False,
                                                                                     scatter=True,
                                                                                     color='blue',
                                                                                     more_subplots_left=True,
                                                                                     pointsize=pointsize)
mplPlotter(light=True, fig=fig, ax=ax2, shape_and_position=412, usetex=False).plot2d(x=xx, y=u_smooth_fit,
                                                                                     grid=True, resize_axes=True,
                                                                                     more_subplots_left=True,
                                                                                     color='cornflowerblue',
                                                                                     linewidth=2, gridlines='dotted',
                                                                                     title_y=1.05,
                                                                                     plot_title='Polynomial regression of {}nd degree'.format(degree),
                                                                                     title_size=15, xaxis_label_size=20,
                                                                                     yaxis_labelpad=10, y_label='$u$',
                                                                                     yaxis_label_size=20,
                                                                                     x_ticklabel_size=10,
                                                                                     y_ticklabel_size=10,
                                                                                     y_tick_number=yticks, x_tick_number=xticks,
                                                                                     x_bounds=[-200, -70],
                                                                                     y_bounds=[0, 12],
                                                                                     aspect=3, prune=None)

"""
Spline interpolation
"""
interpolation = 'thin_plate'
smooth = 0
smooth_title = '\n Scipy $smooth={}$'.format(smooth) if smooth > 0 else ''
title_y = 1.05 if smooth_title == '' else 1.0
title = interpolation.capitalize().replace("_", " ")
u_exp_fit, u_smooth_fit, xx = rbf(x=x, u=u, x_smooth=x_smooth, u_smooth=u_smooth, variant=interpolation, smooth=smooth)

ax3 = mplPlotter(light=True, fig=fig, shape_and_position=413, usetex=False).plot2d(x=x, y=u,
                                                                                   grid=True,
                                                                                   resize_axes=False, scatter=True,
                                                                                   color='red',  xaxis_label_size=20,
                                                                                   yaxis_labelpad=10, y_label='$u$',
                                                                                   yaxis_label_size=20,
                                                                                   x_ticklabel_size=10,
                                                                                   y_ticklabel_size=10,
                                                                                   y_tick_number=yticks, x_tick_number=xticks,
                                                                                   more_subplots_left=True,
                                                                                   pointsize=pointsize)
mplPlotter(light=True, fig=fig, ax=ax3, shape_and_position=413, usetex=False).plot2d(x=xx, y=u_exp_fit, grid=True,
                                                                                     resize_axes=False,
                                                                                     more_subplots_left=True,
                                                                                     color='darkred',
                                                                                     linewidth=1)

mplPlotter(light=True, fig=fig, ax=ax3, shape_and_position=413, usetex=False).plot2d(x=x_smooth, y=u_smooth,
                                                                                     grid=True, resize_axes=False,
                                                                                     scatter=True, color='blue',
                                                                                     more_subplots_left=True,
                                                                                     pointsize=pointsize)
mplPlotter(light=True, fig=fig, ax=ax3, shape_and_position=413, usetex=False).plot2d(x=xx, y=u_smooth_fit,
                                                                                     grid=True, resize_axes=True,
                                                                                     more_subplots_left=True,
                                                                                     color='cornflowerblue',
                                                                                     linewidth=2, gridlines='dotted',
                                                                                     plot_title='{} radial basis interpolation'.format(title) + smooth_title,
                                                                                     title_y=title_y,
                                                                                     title_size=15, xaxis_label_size=20,
                                                                                     yaxis_labelpad=10, y_label='$u$',
                                                                                     yaxis_label_size=20,
                                                                                     x_ticklabel_size=10,
                                                                                     y_ticklabel_size=10,
                                                                                     y_tick_number=yticks,
                                                                                     x_tick_number=xticks,
                                                                                     x_bounds=[-200, -70],
                                                                                     y_bounds=[0, 12],
                                                                                     aspect=3, prune=None)

"""
Spline interpolation
"""
interpolation = 'linear'
smooth = 0
smooth_title = '\n Scipy $smooth={}$'.format(smooth) if smooth > 0 else ''
title_y = 1.05 if smooth_title == '' else 1.0
title = interpolation.capitalize().replace("_", " ")
u_exp_fit, u_smooth_fit, xx = cubicspline(x=x, u=u, x_smooth=x_smooth, u_smooth=u_smooth, variant=interpolation, smooth=smooth)

ax4 = mplPlotter(light=True, fig=fig, shape_and_position=414, usetex=False).plot2d(x=x, y=u,
                                                                                   grid=True, resize_axes=False,
                                                                                   scatter=True, color='red',
                                                                                   xaxis_label_size=20,
                                                                                   yaxis_labelpad=10,
                                                                                   y_label='$u$',
                                                                                   yaxis_label_size=20,
                                                                                   x_ticklabel_size=10,
                                                                                   y_ticklabel_size=10, y_tick_number=yticks,
                                                                                   x_tick_number=xticks,
                                                                                   more_subplots_left=True,
                                                                                   pointsize=pointsize)
mplPlotter(light=True, fig=fig, ax=ax4, shape_and_position=414, usetex=False).plot2d(x=xx, y=u_exp_fit,
                                                                                     grid=True, resize_axes=False,
                                                                                     more_subplots_left=True,
                                                                                     color='darkred', linewidth=1)

mplPlotter(light=True, fig=fig, ax=ax4, shape_and_position=414, usetex=False).plot2d(x=x_smooth, y=u_smooth,
                                                                                     grid=True, resize_axes=False,
                                                                                     scatter=True, color='blue',
                                                                                     more_subplots_left=True,
                                                                                     pointsize=pointsize)
mplPlotter(light=True, fig=fig, ax=ax4, shape_and_position=414, usetex=False).plot2d(x=xx, y=u_smooth_fit,
                                                                                     grid=True, resize_axes=True,
                                                                                     more_subplots_left=False,
                                                                                     color='cornflowerblue',
                                                                                     linewidth=2, gridlines='dotted',
                                                                                     plot_title='{} spline interpolation'.format(title) + smooth_title,
                                                                                     title_y=title_y,
                                                                                     title_size=15, xaxis_label_size=20,
                                                                                     yaxis_labelpad=10, y_label='$u$',
                                                                                     yaxis_label_size=20,
                                                                                     x_ticklabel_size=10,
                                                                                     y_ticklabel_size=10,
                                                                                     y_tick_number=yticks,
                                                                                     x_tick_number=xticks,
                                                                                     dpi=150, filename=r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\images\Interpolation.png',
                                                                                     x_bounds=[-200, -70],
                                                                                     y_bounds=[0, 12],
                                                                                     aspect=3, prune=None
                                                                                     )




