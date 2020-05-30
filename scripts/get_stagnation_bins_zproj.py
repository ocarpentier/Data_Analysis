import os
from shutil import copyfile
from itertools import chain


def z_proj():
    zpath, dirs, zproj = next(os.walk(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\z=0\z=0_projected'))
    ypath, dirs, yproj = next(os.walk(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\y=0\y=0_projected'))

    for zbin in zproj:
        for ybin in yproj:
            print(zbin == ybin)
            if zbin == ybin:
                copyfile(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\z=0\z=0_projected\{}'.format(zbin),
                         r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\stagnation_line\{}'.format(zbin))

z_proj()
