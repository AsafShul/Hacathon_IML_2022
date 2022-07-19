# imports:
import sys
import numpy as np

from task_1 import genarate_labels_task1
from task_2 import generate_labels_task2
from preprocess import load_data, preprocess_data, preprocess_test


if __name__ == '__main__':
    # configure run:
    np.random.seed(0)
    # args = configure_run_params()

    test_path = sys.argv[1]
    dates = sys.argv[2:]

    # load data:
    raw_train = load_data('data/waze_data.csv')
    raw_test = load_data(test_path)
    total_raw_df = raw_train.append(raw_test)

    # process data:
    q1_train, q2_train = preprocess_data(raw_train)
    test_df = preprocess_test(raw_test, raw_train)

    # task 1:
    genarate_labels_task1(q1_train, test_df)

    # task 2:
    generate_labels_task2(q2_train, dates)


