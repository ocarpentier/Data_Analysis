import csv
import time
import numpy as np
import pandas as pd
from os import listdir

# Pandas

# Banned words to clean up the final numpy array
bannedword = ['Unnamed: 0',
              'Name:', 'x ',
              'Name:', ' y',
              'Name:', ' z',
              'Name: u',
              'Name: v',
              'Name: w',
              'dtype: float64']

# Measures
progress = 0
max_time = 0.17250823974609375 * 22656
time_passed = 0
n = 0

# Mean averaging
means = np.empty((22656, 6))

# Loop through files
for file in listdir(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_binned'):

    start_time = time.time()

    with open(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_binned\{}'.format(
            file)) as csvDataFile:
        data = list(csv.reader(csvDataFile))

    amountofcoordinates = int(len(data) / 10)
    allpoints = np.empty((amountofcoordinates, 6))

    for coordinate in range(amountofcoordinates):
        xc = 2 + 10 * coordinate
        yc = xc + 1
        zc = xc + 2
        uc = xc + 3
        vc = xc + 4
        wc = xc + 5
        xx = data[xc][1]
        yy = data[yc][1]
        zz = data[zc][1]
        uu = data[uc][1]
        vv = data[vc][1]
        ww = data[wc][1]
        allpoints[coordinate] = [xx, yy, zz, uu, vv, ww]

    df = pd.DataFrame(allpoints)
    df.to_csv(
        r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_binned_formatted\{}'.format(
            file))

    means[n] = df.to_numpy(dtype='float').mean(axis=0)

    time_passed = time_passed + time.time() - start_time
    progress = round(progress + 1 / 22656, 2)
    seconds = round(time.time() - start_time, 4)
    hours = time.strftime('%H:%M:%S', time.gmtime(max_time - time_passed))

    print('{} seconds    {} left    {} %'.format(seconds, hours, progress))

    n += 1

df = pd.DataFrame(means)
df.to_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_averaged_mean\vam.csv')
