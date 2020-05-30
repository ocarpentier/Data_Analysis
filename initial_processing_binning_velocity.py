import pandas as pd
import numpy as np
import os

path, dirs, files = next(os.walk(r"E:\AE BSc\AE Year 2\Aerodynamics Project\Binning\Row_bins"))
file_count = len(files)

hnumber = 0
inumber = 0

for h in range(hnumber, file_count):

    print(h)

    binh = pd.read_csv(r'E:\AE BSc\AE Year 2\Aerodynamics Project\Binning\Row_bins\{}.csv'.format(h))

    binh['x bin'] = binh['x'].divide(10.).apply(np.floor)
    binh['y bin'] = binh['y'].divide(10.).apply(np.floor)
    binh['z bin'] = binh['z'].divide(10.).apply(np.floor)

    save = 0
    if h == hnumber:
        save = inumber

    for i in range(save, len(binh)):
        name = '{}_{}_{}'.format(int(binh.loc[i]['x bin']), int(binh.loc[i]['y bin']), int(binh.loc[i]['z bin']))

        threedbin = pd.read_csv(r'E:\AE BSc\AE Year 2\Aerodynamics Project\Binning\3D_bins\{}.csv'.format(name))

        threedbin = threedbin.append(binh.iloc[i])

        threedbin.to_csv(r'E:\AE BSc\AE Year 2\Aerodynamics Project\Binning\3D_bins\{}.csv'.format(name))

        del threedbin
