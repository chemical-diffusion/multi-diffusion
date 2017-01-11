import numpy as np


def load_profiles(file_list, t=1):
    # last component will be substracted
    profiles = []
    x_points = []
    permutation = [0, 3, 1, 2]
    for file_name in file_list:
        data = np.genfromtxt(file_name, delimiter='\t', skip_header=1).T
        order = np.argsort(data[1])
        profiles.append(data[2:][permutation][:, order])
        x_points.append(data[1, order])
    return x_points, profiles


def load_profiles_1360(file_list, t=1):
    # last component will be substracted
    profiles = []
    x_points = []
    permutation = [0, 3, 1, 2]
    for file_name in file_list:
        data = np.genfromtxt(file_name, delimiter='\t', skip_header=1).T
        order = np.argsort(data[0])
        profiles.append(data[1:][permutation][:, order])
        x_points.append(data[0, order])
    return x_points, profiles


def load_profiles_tg(file_list, t=1):
    # last component will be substracted
    profiles = []
    x_points = []
    permutation = [0, 3, 2, 1]
    for file_name in file_list:
        data = np.loadtxt(file_name).T
        order = np.argsort(data[0])
        profiles.append(data[1:][permutation][:, order])
        x_points.append(data[0, order])
    return x_points, profiles


def load_profiles_ternary(file_list, t=1):
    # last component will be substracted
    profiles = []
    x_points = []
    permutation = [2, 1, 0]
    for file_name in file_list:
        data = np.genfromtxt(file_name, skip_header=1).T
        order = np.argsort(data[0])
        profiles.append(data[-3:][permutation][:, order])
        x_points.append(data[0, order])
    return x_points, profiles


def load_profiles_nas(file_list):
    # last component will be substracted
    profiles = []
    x_points = []
    for file_name in file_list:
        data = np.loadtxt(file_name).T
        profiles.append(data)
        x_points.append(10 * np.arange(len(data[0]), dtype=np.float))
    return x_points, profiles


def load_profiles_bns(file_list):
    # last component will be substracted
    profiles = []
    x_points = []
    for file_name in file_list:
        data = np.loadtxt(file_name).T
        profiles.append(data[1:])
        x_points.append(data[0])
    return x_points, profiles
