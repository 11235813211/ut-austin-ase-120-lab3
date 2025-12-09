import pandas as pd
import numpy as np
import os
import re
from pprint import pprint
import matplotlib.pyplot as plt

filename = "Data/111425 114545BlackRPM500AOA05.csv"
f2 = "Data/111425 123426BlackRPM500AOA05Pitching.csv"

df = pd.read_csv(filename, encoding="latin1")
df2 = pd.read_csv(f2, encoding="latin1")

def GetFiles(folder):
    """
    Returns a list of tuples containing:
    (filename, color, rpm, aoa, is_pitching)
    """
    
    pattern = re.compile(
        r".*?(Black|Blue|Red|Green|White)"     # color
        r"RPM(\d{3})"                          # rpm
        r"AOA(\d{2})"                          # aoa
        r"(Pitching)?"                         # optional 'Pitching'
        r"\.csv$"
    )

    results = []
    
    for fname in os.listdir(folder):
        match = pattern.match(fname)
        if match:
            color = match.group(1)
            rpm = int(match.group(2))
            aoa = int(match.group(3))
            is_pitching = match.group(4) is not None
            results.append((fname, color, rpm, aoa, is_pitching))
    
    return results


def GetStat(df, stat_string):
    if len(df[stat_string]) < 501:
        raise ValueError(f"DataFrame has {len(df)} rows; at least 501 rows are required for '{stat_string}'.")
    stat = df[stat_string][1:501].astype(float)

    stat_avg = stat.mean()
    stat_std = stat.std()

    return [stat_avg, stat_std]

# vel = GetStat(df, vel_str)
# normal = GetStat(df, normal_str)
# axial = GetStat(df, axial_str)
# moment = GetStat(df, moment_str)
# print("Velocity: ", vel)
# print("Normal: ", normal)
# print("Axial: ", axial)
# print("Moment: ", moment)

# print("\n#2:")

# vel = GetStat(df2, vel_str)
# normal = GetStat(df2, normal_str)
# axial = GetStat(df2, axial_str)
# moment = GetStat(df2, moment_str)
# print("Velocity: ", vel)
# print("Normal: ", normal)
# print("Axial: ", axial)
# print("Moment: ", moment)

def GetFileData(folder, file: list) -> list:
    df = pd.read_csv(os.path.join(folder, file[0]), encoding="latin1")

    vel_str = "Velocity"
    axial_str = "PGB Axial"
    normal_str = "PGB Normal"
    moment_str = "PGB Moment"

    vel = GetStat(df, vel_str)
    normal = GetStat(df, normal_str)
    axial = GetStat(df, axial_str)
    moment = GetStat(df, moment_str)

    return [vel, axial, normal, moment]

def AverageSharedData(index, d1: list, d2: list):
    """
    Returns a copy of d1 with the element at `index` replaced
    by the average of d1[index] and d2[index] (assuming same sample size).
    
    d1, d2: lists of [mean, stdev]
    """
    # Make a copy of d1
    result = d1.copy()
    
    # Extract mean and stdev at the given index
    mean1, std1 = d1[index]
    mean2, std2 = d2[index]
    
    # Average of the means
    avg_mean = (mean1 + mean2) / 2
    
    # Combine standard deviations assuming same sample size n
    # Variance of the combined data: mean of the two variances + squared difference of means / 2
    avg_variance = (std1**2 + std2**2) / 2 + ((mean1 - mean2)**2) / 4
    avg_std = np.sqrt(avg_variance)
    
    # Replace the element at index
    result[index] = [avg_mean, avg_std]
    
    return result



def ExtractFileData(folder, files):
    data = {}
    '''
    data format:
        key: (color, RPM, AOA)
        data: ([vel_avg, vel_std],
            [axial_avg, axial_std],
            [normal_avg, normal_std],
            [pitch_moment_avg, pitch_moment_std]
            [side_avg, side_std],
            [yaw_moment_avg, yaw_moment_std])
    '''
    for file in files:
        key = tuple(file[1:4])
        stats = GetFileData(folder, file)
        if file[4]:
            # pitching is true
            stats.insert(4, [0,0])
            stats.insert(5, [0,0])
            if key not in data:
                data[key] = stats
            else:
                print(f"Pitching: Updating at {key}")
                data[key][0] = AverageSharedData(0, data[key], stats)[0]
                data[key][1] = AverageSharedData(1, data[key], stats)[1]

                data[key][2] = stats[2]
                data[key][3] = stats[3]
        else:
            stats.insert(2, [0,0])
            stats.insert(3, [0,0])
            if key not in data:
                data[key] = stats
            else:
                print(f"Yawing: Updating at {key}")
                data[key][0] = AverageSharedData(0, data[key], stats)[0]
                data[key][1] = AverageSharedData(1, data[key], stats)[1]
                data[key][4] = stats[4]
                data[key][5] = stats[5]
    return data


def SubVariables(a: list, b: list):
    c = [0, 0]
    c[0] = a[0] - b[0]
    c[1] = ((a[1])**2 + (b[1])**2) ** (1/2)
    return c

def MultVariables(a: list, b: list):
    c = [0, 0]
    c[0] = a[0] * b[0]
    c[1] = abs(c[0]) * ((a[0]/a[1])**2 + (b[0]/b[1])**2) ** (1/2)
    return c

def DivVariables(a: list, b: list):
    c = [0, 0]
    c[0] = a[0] / b[0]
    c[1] = abs(c[0]) * ((a[0]/a[1])**2 + (b[0]/b[1])**2) ** (1/2)
    return c

def CalcPercentRootChord(v, color):
    '''
    Docstring for CalcPercentRootChord
    
    :param v: [val, err]
    :param length: length of midpoint to end of PGB in chords
    :param delta: delta
    '''
    root_chords = {
        "Black": 8.4,
        "Blue": 8.46,
        "White": 6.5
    }
    root_chord = root_chords[color]

    offsets = {
        "Black": 2.5195,
        "Blue": 3,
        "White": 0.75
    }
    offset = offsets[color]

    val = v[0]
    err = v[1]
    if color== "Black":
        print(color, val, err)

    m_to_in = 39.37

    val = val*m_to_in - 1.145 + offset
    val = 0.5 - val/root_chord
    val = val * 100
    err = err * m_to_in / root_chord * 100
    return [val, err]

def GetMomentArms(stats, color):
    '''
    ([vel_avg, vel_std],
            [axial_avg, axial_std],
            [normal_avg, normal_std],
            [pitch_moment_avg, pitch_moment_std]
            [side_avg, side_std],
            [yaw_moment_avg, yaw_moment_std])
    '''
    axial = stats[1]
    normal = stats[2]
    pitch = stats[3]
    side = stats[4]
    yaw = stats[5]

    axial_arm = DivVariables(yaw, side)
    # if abs(abs(side[0]) - 0.7592556) < 0.0001:
    #     print("YAW, SIDE, AXIAL", yaw, side, axial_arm)

    normal_pitching_moment = MultVariables(normal, axial_arm)
    axial_pitching_moment = SubVariables(pitch, normal_pitching_moment)
    normal_arm = DivVariables(axial_pitching_moment, axial)

    axial_arm = CalcPercentRootChord(axial_arm, color)

    return [axial_arm, normal_arm]



def CalculateMoments(data):

    '''
    data format:
        key: (color, RPM, AOA)
        data: ([vel_avg, vel_std],
            [axial_avg, axial_std],
            [normal_avg, normal_std],
            [pitch_moment_avg, pitch_moment_std]
            [side_avg, side_std],
            [yaw_moment_avg, yaw_moment_std])
    '''
    for key in data:
        color = key[0]
        stats = data[key]
        stats += GetMomentArms(stats, color)
        data[key] = stats
    
    return data
    
def plot_3x7_panel(data):
    colors = ['Black', 'Blue', 'White']
    x_indices = [200, 300, 400, 500]  # velocities
    y_labels = ["Axial Force (N)", "Normal Force (N)", "Pitching Moment (Nm)",
                "Side Force (N)", "Yawing Moment (Nm)",
                "CP Chord Position (%)", "CP Lateral Position (m)"]

    fig, axes = plt.subplots(nrows=3, ncols=7, figsize=(21, 8), sharex=False)

    # Precompute y-axis limits for the first 5 columns (shared per column)
    col_y_limits = []
    for j in range(5):
        col_min, col_max = float('inf'), float('-inf')
        for color in colors:
            for x in x_indices:
                values = data[(color, x, 5)]
                y_mean, y_std = values[j + 1]  # skip index 0 (x-axis)
                col_min = min(col_min, y_mean - y_std)
                col_max = max(col_max, y_mean + y_std)
        col_max += 0.1 * (col_max - col_min)
        col_min -= 0.099 * (col_max - col_min)
        col_y_limits.append((col_min, col_max))

    # Plotting
    for i, color in enumerate(colors):
        for j in range(7):
            color = colors[i]
            ax = axes[i, j]
            x_means, x_stds, y_means, y_stds = [], [], [], []


            for x in x_indices:
                values = data[(color, x, 5)]
                x_mean, x_std = values[0]  # always the velocity for x-axis
                y_mean, y_std = values[j + 1]  # y-values correctly aligned
                x_means.append(x_mean)
                x_stds.append(x_std)
                y_means.append(y_mean)
                y_stds.append(y_std)
            


            ax.errorbar(x_means, y_means, xerr=x_stds, yerr=y_stds,
                        fmt='o', capsize=5, label=color)

            ax.set_title(f'{color} - {y_labels[j]}')
            ax.set_xlabel("Velocity (m/s)")
            ax.set_ylabel(y_labels[j])
            ax.grid(True)

            # Shared y-limits for columns 0-4, independent for columns 5-6
            if j < 5:
                ymin, ymax = col_y_limits[j]
                ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    plt.show()


files = GetFiles("Data/")
# print(files)

data = ExtractFileData("Data", files)
# pprint(data)

data = CalculateMoments(data)
pprint(data)

s = data[("Blue", 400, 5)]

print("Side: ", s[4])
print("Yaw: ", s[5])
print("Axial Arm: ", s[6])

plot_3x7_panel(data)


# print(df["PTA 1"])