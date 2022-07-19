import numpy as np
from constants import *
from sklearn.model_selection import train_test_split

def split_train_dev_test(df, test_size=0.2, dev_size=0.25):
    # use split to get train, dev and test
    X_train, X_test = train_test_split(df, test_size=test_size, random_state=0)
    X_train, X_dev = train_test_split(X_train, test_size=dev_size, random_state=0)
    return X_train, X_dev, X_test


def distance_coord(coords):
    x_1, y_1, x_2, y_2 = coords
    return np.sqrt(((y_1-y_2)**2)+(x_1 - x_2)**2)


def distance_from_center(coord_1):
    """
    Calculate the distance from the center of Tel Aviv
    """
    return np.sqrt(np.sum((coord_1 - TEL_AVIV_CENTER)**2))


def get_season(ts):
    if isinstance(ts, dt.datetime):
        ts = ts.date()
    ts = ts.replace(year=Y)
    return next(season for season, (start, end) in SEASONS
                if start <= ts <= end)


def convert_posix(ts):
    return dt.datetime.utcfromtimestamp(int(ts) / POSIX_FACTOR)


def time_in_range(window, ts):
    """Return true if x is in the range [start, end]"""
    start, end = window

    start = start.time()
    end = end.time()
    ts = ts.time()

    if start <= end:
        return start <= ts <= end
    else:
        return start <= ts or ts <= end
