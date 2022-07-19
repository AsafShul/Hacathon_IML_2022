import pandas as pd
import numpy as np
import plotly.graph_objects as go
import mlflow

from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from plotly.subplots import make_subplots


def split_labels(df):
    X = df.to_numpy()
    y = X[:, 208:212]
    X = X[:, :206]
    return X, y


def split_labels2(df):
    X = df.to_numpy()
    y = X[:, 212:]
    X = X[:, :206]
    return X, y

def sub_split(X, m):
    new_X, new_y = split_labels2(X)
    X = np.c_[new_X, np.argmax(m.predict_proba(new_X), axis=1)]
    return X






def run_mlflow_classification(train_X, train_y, test_X, test_y):
    mlflow.set_experiment("classification_hyper_params_visualization")

    for alpha in range(5, 100, 10):
        for beta in range(5, 100, 10):
            with mlflow.start_run(run_name=f'run RandomForestAdaBoost {alpha}') as run:
                print('RandomForestAdaBoost alpha: ', alpha)
                mlflow.log_param("algo", "RandomForestAdaBoost")
                mlflow.log_param("lambda", alpha)
                mlflow.log_param("c", beta)

                clf = RandomForestClassifier(n_estimators=beta)
                m = OneVsRestClassifier(AdaBoostClassifier(clf, n_estimators=alpha)).fit(train_X, train_y)

                train_pred = m.predict(train_X)
                pred = m.predict(test_X)

                mlflow.log_metric('dev_mse',
                                  mean_squared_error(test_y[:, :2], pred[:, :2]))
                mlflow.log_metric('train_mse', mean_squared_error(train_y[:, :2],
                                                                  train_pred[:,
                                                                  :2]))
                mlflow.log_metric('dev_f1_macro', f1_score(np.argmax(pred, axis=1), np.argmax(test_y, axis=1), average='macro'))
                mlflow.log_metric('train_f1_macro', f1_score(np.argmax(train_pred, axis=1), np.argmax(train_y, axis=1), average='macro'))

    for alpha in range(5, 105, 5):
        with mlflow.start_run(run_name=f'run SVC {alpha}') as run:
            print('SVC alpha: ', alpha)
            mlflow.log_param("algo", "SVC")
            mlflow.log_param("lambda", alpha)

            m = OneVsRestClassifier(SVC(kernel="poly", degree=alpha)).fit(train_X, train_y)

            train_pred = m.predict(train_X)
            pred = m.predict(test_X)

            mlflow.log_metric('dev_mse',
                              mean_squared_error(test_y[:, :2], pred[:, :2]))
            mlflow.log_metric('train_mse', mean_squared_error(train_y[:, :2],
                                                              train_pred[:,
                                                              :2]))
            mlflow.log_metric('dev_f1_macro', f1_score(np.argmax(pred, axis=1),
                                                       np.argmax(test_y,
                                                                 axis=1),
                                                       average='macro'))
            mlflow.log_metric('train_f1_macro',
                              f1_score(np.argmax(train_pred, axis=1),
                                       np.argmax(train_y, axis=1),
                                       average='macro'))

    for alpha in range(10, 320, 20):
        with mlflow.start_run(run_name=f'run RandomForestClassifier {alpha}') as run:
            print('RandomForestClassifier alpha: ', alpha)
            mlflow.log_param("algo", "RandomForestClassifier")
            mlflow.log_param("lambda", alpha)

            m = MultiOutputClassifier(RandomForestClassifier(n_estimators=alpha)).fit(train_X, train_y)

            train_pred = m.predict(train_X)
            pred = m.predict(test_X)

            mlflow.log_metric('dev_mse',
                              mean_squared_error(test_y[:, :2], pred[:, :2]))
            mlflow.log_metric('train_mse', mean_squared_error(train_y[:, :2],
                                                              train_pred[:,
                                                              :2]))
            mlflow.log_metric('dev_f1_macro', f1_score(np.argmax(pred, axis=1),
                                                       np.argmax(test_y,
                                                                 axis=1),
                                                       average='macro'))
            mlflow.log_metric('train_f1_macro',
                              f1_score(np.argmax(train_pred, axis=1),
                                       np.argmax(train_y, axis=1),
                                       average='macro'))

    for alpha in range(1, 10000, 1000):
        alpha /= 10000
        for c in range(1, 10, 2):
            c /= 10
            with mlflow.start_run(run_name=f'run SGDClassifier {alpha}') as run:
                print('SGDClassifier alpha: ', alpha)
                mlflow.log_param("algo", "SGDClassifier")
                mlflow.log_param("lambda", alpha)
                mlflow.log_param("c", c)

                m = OneVsRestClassifier( SGDClassifier(loss="modified_huber", alpha=alpha, l1_ratio=c, tol=100)).fit(train_X, train_y)

                train_pred = m.predict(train_X)
                pred = m.predict(test_X)

                mlflow.log_metric('dev_mse', mean_squared_error(test_y[:, :2],
                                                                pred[:, :2]))
                mlflow.log_metric('train_mse',
                                  mean_squared_error(train_y[:, :2],
                                                     train_pred[:, :2]))
                mlflow.log_metric('dev_f1_macro', f1_score(np.argmax(pred, axis=1), np.argmax(test_y, axis=1), average='macro'))
                mlflow.log_metric('train_f1_macro', f1_score(np.argmax(train_pred, axis=1), np.argmax(train_y, axis=1), average='macro'))

    for alpha in range(10, 210, 10):
        for c in range(1, 10, 2):
            c /= 10
            with mlflow.start_run(run_name=f'run GradientBoostingClassifier {alpha}') as run:
                print('GradientBoostingClassifier alpha: ', alpha)
                mlflow.log_param("algo", "GradientBoostingClassifier")
                mlflow.log_param("lambda", alpha)
                mlflow.log_param("c", c)

                m = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=alpha, learning_rate=c)).fit(train_X, train_y)

                train_pred = m.predict(train_X)
                pred = m.predict(test_X)

                mlflow.log_metric('dev_mse',
                                  mean_squared_error(test_y[:, :2], pred[:, :2]))
                mlflow.log_metric('train_mse', mean_squared_error(train_y[:, :2],
                                                                  train_pred[:,
                                                                  :2]))
                mlflow.log_metric('dev_f1_macro', f1_score(np.argmax(pred, axis=1), np.argmax(test_y, axis=1), average='macro'))
                mlflow.log_metric('train_f1_macro', f1_score(np.argmax(train_pred, axis=1), np.argmax(train_y, axis=1), average='macro'))


if __name__ == "__main__":
    run_mlflow = False

    train_df = pd.read_csv("data/waze_data_train_tlv.csv")
    dev_df = pd.read_csv("data/waze_data_dev_tlv.csv")

    classifier_names = ['AdaBoostRandomForrest', 'SVC', 'RandomForest',
                        'SGDClassifier', 'GradientBoostingClassifier']

    train_X, train_y = split_labels(train_df)
    train_y = train_y.astype('int')
    test_X, test_y = split_labels(dev_df)

    if run_mlflow:
        run_mlflow_classification(train_X, train_y, test_X, test_y)

    train_size = []

    classifier_num = len(classifier_names)
    iterations = 50

    res = dict(zip(range(classifier_num), [dict(test_loss=np.zeros(iterations-1),
                                    train_loss=np.zeros(iterations-1),
                                    test_f1_macro=np.zeros(iterations-1),
                                    train_f1_macro=np.zeros(iterations-1)) for _ in range(classifier_num)]))

    for i in range(1, iterations):
        print(f'iteration {i}.')
        train_size.append(train_X.shape[0])
        test_X, test_y = split_labels(dev_df)

        train, _ = train_test_split(train_df, train_size=i / (iterations), random_state=0)
        train_X, train_y = split_labels(train)
        train_y = train_y.astype('int')

        # fit:
        clf = RandomForestClassifier(n_estimators=100)
        AdaRf = OneVsRestClassifier(AdaBoostClassifier(clf, n_estimators=40)).fit(train_X, train_y)
        svc = OneVsRestClassifier(SVC(kernel="poly", degree=100)).fit(train_X, train_y)
        rf = MultiOutputClassifier(RandomForestClassifier(n_estimators=300)).fit(train_X, train_y)
        sgd = OneVsRestClassifier(SGDClassifier(loss="modified_huber",alpha=0.0001,l1_ratio=0.5,tol=100)).fit(train_X, train_y)
        gbc = OneVsRestClassifier(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)).fit(train_X, train_y)

        # predict:
        classifiers = [AdaRf, svc, rf, sgd, gbc]
        train_predictions = [classifier.predict(train_X) for classifier in classifiers]
        test_predictions = [classifier.predict(test_X) for classifier in classifiers]

        # log:
        idx = i-1
        for j in range(classifier_num):
            res[j]['test_loss'][idx] = mean_squared_error(test_y[:, :2], test_predictions[j][:, :2])
            res[j]['train_loss'][idx] = mean_squared_error(train_y[:, :2], train_predictions[j][:, :2])
            res[j]['test_f1_macro'][idx] = np.mean(euclidean_distances(test_predictions[j][:, :2], test_predictions[j][:, :2]))
            res[j]['train_f1_macro'][idx] = np.mean(euclidean_distances(train_predictions[j][:, :2], train_predictions[j][:, :2]))


    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Error as a function of train size",
                                        "f1 macro as a function of train size"],
                        horizontal_spacing=0.05, vertical_spacing=.03)
    for j in range(classifier_num):
        fig.add_trace(go.Scatter(x=train_size[1:-1], y=res[j]['test_loss'][1:-1], name=f'{classifier_names[j]} test loss'), row=1, col=1)
        fig.add_trace(go.Scatter(x=train_size[1:-1], y=res[j]['train_loss'][1:-1], name=f'{classifier_names[j]} train loss'), row=1, col=1)
        fig.add_trace(go.Scatter(x=train_size[1:-1], y=res[j]['test_f1_macro'][1:-1], name=f'{classifier_names[j]} test f1 macro'), row=1, col=2)
        fig.add_trace(go.Scatter(x=train_size[1:-1], y=res[j]['train_f1_macro'][1:-1], name=f'{classifier_names[j]} train f1 macro'), row=1, col=2)


    fig.update_layout(title="Classification classifiers comparison",
                      xaxis_title="Train size",
                      yaxis_title="Error",
                      showlegend=True)

    fig.write_html("ClassificationClassifiersComparison.html")
