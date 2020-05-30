import pandas as pd
import os
import numpy as np


def z_0_mean_std():

    path, dirs, files = next(os.walk(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\z=0\z=0_projected'))

    print(len(files))

    means = np.empty((int(len(files)), 9))

    std = np.empty((int(len(files)), 9))

    n = 0

    for file in files:

        df = pd.read_csv(path+'\\'+file)

        means[n] = df.to_numpy(dtype='float').mean(axis=0)

        std[n] = df.std(axis=0)

        """
        Outlier removal to be done by Tinka, Siya
        """

        df = df[df['2'] < 1].sort_values(['2'])

        # df.to_csv(r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\z=0\[0, 0.1)\{}'.format(file))

        n += 1
