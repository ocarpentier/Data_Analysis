import pandas as pd

def z_proj():

    df = pd.read_excel(
        r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\velocity_planes_of_interest\z=0\z=0_wo_outliers\zzero.xlsx')

    df = df[df[0] < -7].sort_values([1])

    df = df[df[1] > -0.5]
    df = df[df[1] < 0.5].sort_values([2])

    # Set y=0
    df[1] = 0
    # Set z=0
    df[2] = 0

    df.to_csv(
        r'C:\Users\xXY4n\AE BSc\AE Year 2\Aerodynamics project\Data Analysis\data\stagnation_line\stagnation_line_z=0.csv')


z_proj()
