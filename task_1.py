import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, f1_score, euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from constants import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import GradientBoostingClassifier


def task1_split_xy(data):
    data = data[data.shape[0] % 5:]
    group_data = data.reshape((int(data.shape[0] / 5), 5, data.shape[1]))
    y = group_data[:, 4, :]
    X = group_data[:, 0:4, :].reshape((int(data.shape[0] / 5), data.shape[1] * 4))
    return X, y

def baseline_task1_process(df):
    df.sort_values(by=['update_date'])
    X = df.loc[:, ['linqmap_type', 'linqmap_subtype', 'x', 'y']]
    X = pd.get_dummies(X, columns=['linqmap_type', 'linqmap_subtype'])
    X = X.to_numpy()
    return X

def task1_split_xy_test(df):
    data = df.to_numpy()
    group_data = data.reshape((int(data.shape[0] / 4), 4, data.shape[1]))
    X = group_data.reshape((int(data.shape[0] / 4), data.shape[1] * 4))
    return X


def load_filtered_data(train_path, dev_path, test_path):
    """
    Load data from csv
    """
    train_data = pd.read_csv(train_path, index_col=0)
    dev_data = pd.read_csv(dev_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)
    train_data = pd.concat([train_data, dev_data], axis=0)
    return train_data, test_data


def generate_baseline_task1(df):
    train_size = []
    test_loss = []
    train_loss = []
    f1_type = []
    f1_subtype = []
    distance = []

    data = baseline_task1_process(df)

    for i in range(1, 20):
        train, test = train_test_split(data, test_size=i / 20, random_state=0)
        train_X, train_y = task1_split_xy(train)
        m = MultiOutputRegressor(LinearRegression()).fit(train_X, train_y)
        train_pred = m.predict(train_X)

        test_X, test_y = task1_split_xy(test)
        pred = m.predict(test_X[-100:])

        train_size.append(train_X.shape[0])
        test_loss.append(mean_squared_error(test_y[-100:], pred))
        train_loss.append(mean_squared_error(train_y, train_pred))
        f1_type.append(f1_score(np.argmax(test_y[-100:, 2:6], axis=1),
                                np.argmax(pred[:, 2:6], axis=1),
                                average='macro'))
        f1_subtype.append(f1_score(np.argmax(test_y[-100:, 6:], axis=1),
                                   np.argmax(pred[:, 6:], axis=1),
                                   average='macro'))
        distance.append(
            np.mean(euclidean_distances(test_y[-100:, :2], pred[:, :2])))

        print(f"test loss for iteration {i}: ",
              mean_squared_error(test_y[-100:], pred))
        print(f"type prediction f1_score for iteration {i}: ",
              f1_score(np.argmax(test_y[-100:, 2:6], axis=1),
                       np.argmax(pred[:, 2:6], axis=1), average='macro'))
        print(f"subtype prediction f1_score for iteration {i}: ",
              f1_score(np.argmax(test_y[-100:, 6:], axis=1),
                       np.argmax(pred[:, 6:], axis=1), average='macro'))
        print(f"euclidean distance mean for iteration {i}: ",
              np.mean(euclidean_distances(test_y[-100:, :2], pred[:, :2])))

    go.Figure([go.Scatter(x=train_size, y=test_loss, mode='markers+lines',
                          marker=dict(color="blue"),
                          name=f'test error'),
               go.Scatter(x=train_size, y=train_loss, mode='markers+lines',
                          marker=dict(color="red"),
                          name=f'train error')]
              ).show()
    go.Figure([go.Scatter(x=train_size, y=f1_type, mode='markers+lines',
                          marker=dict(color="blue"),
                          name=f'f1 score for type'),
               go.Scatter(x=train_size, y=f1_subtype, mode='markers+lines',
                          marker=dict(color="red"),
                          name=f'f1 score for subtype')]
              ).show()
    go.Figure([go.Scatter(x=train_size, y=distance, mode='markers+lines',
                          marker=dict(color="blue"),
                          name=f'mean distance')]
              ).show()



def load_final_data(train_path, dev_path, test_path, final_path):
    """
    Load data from csv
    """
    train_data = pd.read_csv(train_path, index_col=0)
    dev_data = pd.read_csv(dev_path, index_col=0)
    test_data = pd.read_csv(test_path, index_col=0)
    train_data = pd.concat([train_data, dev_data, test_data], axis=0)
    test_data = pd.read_csv(final_path, index_col=0)
    return train_data, test_data


def Q_1_regression_predict(train_path, dev_path, test_path):
    # load the data
    train_data, test_data = load_filtered_data(train_path, dev_path, test_path)
    # fit model over train data
    model = MultiOutputRegressor(SVR(degree=50, kernel='poly', C=3))
    y_start_index = train_data.columns.get_loc("x")
    x_train = train_data.iloc[:, :y_start_index]
    y_train = train_data.iloc[:, y_start_index:y_start_index+2]
    m = model.fit(x_train, train_data[['x', 'y']])
    # predict the test data
    y_start_index = test_data.columns.get_loc("x")
    x_test = test_data.iloc[:, :y_start_index]
    y_test = test_data.iloc[:, y_start_index:y_start_index+2]
    y_pred = m.predict(x_test)
    # calculate the error
    print(f"mean_squared_error over test: {mean_squared_error(y_test, y_pred)}")
    # print the euclidean distance
    print(f"euclidean distance over test: {np.mean(euclidean_distances(y_test, y_pred))}")
    # print error over train data
    print(f"mean_squared_error over train: {mean_squared_error(y_train, m.predict(x_train))}")
    # print the euclidean distance
    print(f"euclidean distance over train: {np.mean(euclidean_distances(y_train, m.predict(x_train)))}")


def final_regression_predict(train_data, test_data):
    # fit model over train data
    model = MultiOutputRegressor(SVR(degree=50, kernel='poly', C=3))
    y_start_index = train_data.columns.get_loc("x")
    x_train = train_data.iloc[:, :y_start_index]
    y_train = train_data.iloc[:, y_start_index:y_start_index+2]
    m = model.fit(x_train, y_train)
    # predict the test data
    m_prediction = m.predict(test_data)
    # convert the prediction to dataframe
    m_prediction = pd.DataFrame(m_prediction, columns=['x', 'y'])
    return pd.concat([test_data, m_prediction], axis=1), m_prediction


def split_labels(df):
    X = df.to_numpy()
    y = X[:, 207:211]
    X = X[:, :205]
    return X, y


def split_labels2(df):
    X = df.to_numpy()
    y = X[:, 211:]
    X = X[:, :205]
    return X, y


def sub_split(X, m):
    X = np.c_[X, np.argmax(m.predict_proba(X), axis=1)]
    return X


def genarate_labels_task1(q1_train, final_data):
    # regression:

    _, final_pred = final_regression_predict(q1_train, final_data)
    # classification:
    X, y = split_labels(q1_train)
    y = y.astype('int')

    m = OneVsRestClassifier(
        GradientBoostingClassifier(n_estimators=90, learning_rate=0.5)).fit(X, y)

    type_pred = [TYPES[np.argmax(p)] for p in m.predict_proba(final_data.to_numpy())]

    X, y = split_labels2(q1_train)
    X = sub_split(X, m)
    y = y.astype('int')
    m2 = OneVsRestClassifier(
        GradientBoostingClassifier(n_estimators=90, learning_rate=0.5)).fit(X,
                                                                            y)

    test_X = sub_split(final_data.to_numpy(), m)
    sub_type_pred = [SUB_TYPES[np.argmax(p)] for p in m2.predict_proba(test_X)]

    res = pd.concat([pd.DataFrame({'linqmap_type': type_pred,
                                   'linqmap_subtype': sub_type_pred}), final_pred], axis=1)
    res.to_csv('predictions.csv')
