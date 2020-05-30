import pandas as pd
import os.path

from shutil import copyfile
from itertools import chain
from os import listdir


def z_0():
    z = 0

    for x in chain(range(-20, 20)):
        for y in range(-15, 15):
            if os.path.isfile(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_binned_formatted\{}_{}_{}.csv'.format(x, y, z)):
                copyfile(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_binned_formatted\{}_{}_{}.csv'.format(x, y, z),
                         r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\z=0\[0, 1)\{}_{}_{}.csv'.format(x, y, z))

    for file in listdir(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\z=0\[0, 1)'):

        df = pd.read_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\z=0\[0, 1)\{}'.format(file))

        df = df[df['2'] < 1].sort_values(['2'])

        df.to_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\z=0\[0, 0.1)\{}'.format(file))

    for file in listdir(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\z=0\[0, 0.1)'):

        df = pd.read_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\z=0\[0, 0.1)\{}'.format(file))

        df['2'] = 0

        df.to_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\z=0\z=0_projected\{}'.format(file))


def y_0():
    y = 0

    for x in chain(range(-20, 20)):
        for z in chain(range(-15, 15)):
            if os.path.isfile(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_binned_formatted\{}_{}_{}.csv'.format(x, y, z)):
                copyfile(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_binned_formatted\{}_{}_{}.csv'.format(x, y, z),
                         r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\y=0\[0, 1)\{}_{}_{}.csv'.format(x, y, z))

    for file in listdir(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\y=0\[0, 1)'):
        df = pd.read_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\y=0\[0, 1)\{}'.format(file))

        df = df[df['1'] < 1].sort_values(['1'])

        df.to_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\y=0\[0, 0.1)\{}'.format(file))

    for file in listdir(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\y=0\[0, 0.1)'):

        df = pd.read_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\y=0\[0, 0.1)\{}'.format(file))

        df['1'] = 0

        df.to_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\y=0\y=0_projected\{}'.format(file))


def x_m10():
    x = -10

    for y in range(-15, 15):
        for z in chain(range(-15, 15)):
            if os.path.isfile(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_binned_formatted\{}_{}_{}.csv'.format(x, y, z)):
                copyfile(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_binned_formatted\{}_{}_{}.csv'.format(x, y, z),
                         r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\x=-10\[-10, -9)\{}_{}_{}.csv'.format(x, y, z))

    for file in listdir(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\x=-10\[-10, -9)'):
        df = pd.read_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\x=-10\[-10, -9)\{}'.format(file))

        df = df[df['0'] < -99].sort_values(['0'])

        df.to_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\x=-10\[-10, -9.9)\{}'.format(file))

    for file in listdir(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\x=-10\[-10, -9.9)'):

        df = pd.read_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\x=-10\[-10, -9.9)\{}'.format(file))

        df['0'] = -100

        df.to_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\x=-10\x=-10_projected\{}'.format(file))


def x_10():
    x = 10

    for y in range(-15, 15):
        for z in chain(range(-15, 15)):
            if os.path.isfile(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_binned_formatted\{}_{}_{}.csv'.format(x, y, z)):
                copyfile(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_binned_formatted\{}_{}_{}.csv'.format(x, y, z),
                         r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\x=10\[10, 11)\{}_{}_{}.csv'.format(x, y, z))

    for file in listdir(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\x=10\[10, 11)'):
        df = pd.read_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\x=10\[10, 11)\{}'.format(file))

        df = df[df['0'] < 101].sort_values(['0'])

        df.to_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\x=10\[10, 10.1)\{}'.format(file))

    for file in listdir(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\x=10\[10, 10.1)'):

        df = pd.read_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\x=10\[10, 10.1)\{}'.format(file))

        df['0'] = 100

        df.to_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\x=10\x=10_projected\{}'.format(file))

z_0()
y_0()
x_10()
x_m10()