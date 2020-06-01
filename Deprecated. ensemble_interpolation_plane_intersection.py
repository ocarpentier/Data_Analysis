import pandas as pd
import numpy as np

import plotly.graph_objs as go

from mpl_plotter_ply_plotting_methods import PlotLyPublicationPlotter as plyPlotter
from mpl_plotter_colormaps import ColorMaps

"""
Setup
"""

cb_vmax = 6
cb_vmin = -6
nticks = 10
ticksize = 20
pointsize = 7.5
symbol = 'square'
comp = 'v'

"""
x=-10
"""

plane = 'x=-10'
version = '1.0'
u_mosaic = np.loadtxt(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_ensemble_averaged\{}\{}_{}.txt'.format(plane, comp, version))
if plane == 'x=10' or plane == 'x=-10':
    array_shape = (30, 30)
    k_top = 15
else:
    array_shape = (30, 40)
    k_top = 20
x1 = np.linspace(-k_top, k_top, 2*k_top)
x2 = np.linspace(-15, 15, 30)
x1, x2 = np.meshgrid(x1, x2)
x_m10 = np.ones((30, 2*k_top))*-10
fig = plyPlotter().scatter3D(x=x_m10, y=x1, z=x2, c=u_mosaic,
                             opacity=1,
                             more_subplots_left=True,
                             cb_vmax=cb_vmax,
                             cb_vmin=cb_vmin,
                             cb_tickfontsize=ticksize,
                             cb_nticks=nticks,
                             cb_title='[m/s]',
                             cb_y=0.5,
                             w=1000,
                             h=1000,
                             pointsize=pointsize,
                             symbol=symbol
                             )


"""
y=0
"""

plane = 'y=0'
version = '1.0'
u_mosaic = np.loadtxt(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_ensemble_averaged\{}\{}_{}.txt'.format(plane, comp, version))
if plane == 'x=10' or plane == 'x=-10':
    array_shape = (30, 30)
    k_top = 15
else:
    array_shape = (30, 40)
    k_top = 20

x1 = np.linspace(-k_top, k_top, 2*k_top)
x2 = np.linspace(-15, 15, 30)
x1, x2 = np.meshgrid(x1, x2)
z0 = np.ones((30, 2*k_top))*0
plyPlotter().scatter3D(x=x1, y=x2, z=z0, c=u_mosaic, fig=fig,
                       opacity=1,
                       more_subplots_left=True,
                       cb_vmax=cb_vmax,
                       cb_vmin=cb_vmin,
                       cb_tickfontsize=ticksize,
                       cb_nticks=nticks,
                       pointsize=pointsize,
                       symbol=symbol
                       )

"""
z=0
"""

plane = 'z=0'
version = '1.0'
u_mosaic = np.loadtxt(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_ensemble_averaged\{}\{}_{}.txt'.format(plane, comp, version))
if plane == 'x=10' or plane == 'x=-10':
    array_shape = (30, 30)
    k_top = 15
else:
    array_shape = (30, 40)
    k_top = 20
x1 = np.linspace(-k_top, k_top, 2*k_top)
x2 = np.linspace(-15, 15, 30)
x1, x2 = np.meshgrid(x1, x2)
y0 = np.ones((30, 2*k_top))*0
plyPlotter().scatter3D(x=x1, y=y0, z=x2, c=np.flip(u_mosaic, axis=0), fig=fig,
                       opacity=1,
                       more_subplots_left=True,
                       cb_vmax=cb_vmax,
                       cb_vmin=cb_vmin,
                       cb_tickfontsize=ticksize,
                       cb_nticks=nticks,
                       plot_title='Ensemble averaged <i><b>{}<b><i>'.format(comp),
                       pointsize=pointsize,
                       symbol=symbol)

theta = np.linspace(0, 2*np.pi, 100)
phi = np.linspace(0, np.pi, 100)
x = 0 + np.outer(np.cos(theta), np.sin(phi))*7.5
y = 0 + np.outer(np.sin(theta), np.sin(phi))*7.5
z = 0 + np.outer(np.ones(100), np.cos(phi))*7.5  # note this is 2d now

color = []
for i in range(len(x)):
    color.append('black')

fig.add_trace(go.Surface(x=x, y=y, z=z, showscale=False, surfacecolor=color))

fig.show()
