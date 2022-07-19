import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV, \
    MultiTaskLassoCV, ElasticNet, SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import SVR, NuSVR
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow


def split_labels(df):
    X = df.to_numpy()
    y = X[:, 206:208]
    X = X[:, :206]
    return X, y


def run_mlflow_regression(train_X, train_y, test_X, test_y):
    mlflow.set_experiment("regression_hyper_params_visualization")

    with mlflow.start_run(run_name=f'run LinearRegression') as run:
        print('LinearRegression')
        mlflow.log_param("algo", "LinearRegression")
        mlflow.log_param("lambda", 0)
        m = MultiOutputRegressor(LinearRegression()).fit(train_X, train_y)

        train_pred = m.predict(train_X)
        pred = m.predict(test_X)

        mlflow.log_metric('dev_mse',
                          mean_squared_error(test_y[:, :2], pred[:, :2]))
        mlflow.log_metric('train_mse', mean_squared_error(train_y[:, :2],
                                                          train_pred[:, :2]))
        mlflow.log_metric('dev_euclidean_distances', np.mean(
            euclidean_distances(test_y[:, :2], pred[:, :2])))
        mlflow.log_metric('train_euclidean_distances', np.mean(
            euclidean_distances(train_y[:, :2], train_pred[:, :2])))

    for alpha in range(1, 1500, 50):
        with mlflow.start_run(run_name=f'run RidgeRegression {alpha}') as run:
            print('RidgeRegression alpha: ', alpha)
            mlflow.log_param("algo", "Ridge")
            mlflow.log_param("lambda", alpha)
            m = MultiOutputRegressor(Ridge(alpha=alpha)).fit(train_X, train_y)

            train_pred = m.predict(train_X)
            pred = m.predict(test_X)

            mlflow.log_metric('dev_mse',
                              mean_squared_error(test_y[:, :2], pred[:, :2]))
            mlflow.log_metric('train_mse', mean_squared_error(train_y[:, :2],
                                                              train_pred[:,
                                                              :2]))
            mlflow.log_metric('dev_euclidean_distances', np.mean(
                euclidean_distances(test_y[:, :2], pred[:, :2])))
            mlflow.log_metric('train_euclidean_distances', np.mean(
                euclidean_distances(train_y[:, :2], train_pred[:, :2])))

    for alpha in range(1, 1500, 50):
        with mlflow.start_run(run_name=f'run LassoRegression {alpha}') as run:
            print('Lasso alpha: ', alpha)
            mlflow.log_param("algo", "Lasso")
            mlflow.log_param("lambda", alpha)

            m = MultiOutputRegressor(Lasso(alpha=alpha)).fit(train_X, train_y)

            train_pred = m.predict(train_X)
            pred = m.predict(test_X)

            mlflow.log_metric('dev_mse',
                              mean_squared_error(test_y[:, :2], pred[:, :2]))
            mlflow.log_metric('train_mse', mean_squared_error(train_y[:, :2],
                                                              train_pred[:,
                                                              :2]))
            mlflow.log_metric('dev_euclidean_distances', np.mean(
                euclidean_distances(test_y[:, :2], pred[:, :2])))
            mlflow.log_metric('train_euclidean_distances', np.mean(
                euclidean_distances(train_y[:, :2], train_pred[:, :2])))

    for alpha in range(1, 1500, 50):
        with mlflow.start_run(run_name=f'run LassoCV {alpha}') as run:
            print('LassoCV alpha: ', alpha)
            mlflow.log_param("algo", "LassoCV")
            mlflow.log_param("lambda", alpha)

            m = MultiOutputRegressor(
                LassoCV(eps=1e-4, n_alphas=alpha, tol=1e-4)).fit(train_X,
                                                                 train_y)

            train_pred = m.predict(train_X)
            pred = m.predict(test_X)

            mlflow.log_metric('dev_mse',
                              mean_squared_error(test_y[:, :2], pred[:, :2]))
            mlflow.log_metric('train_mse', mean_squared_error(train_y[:, :2],
                                                              train_pred[:,
                                                              :2]))
            mlflow.log_metric('dev_euclidean_distances', np.mean(
                euclidean_distances(test_y[:, :2], pred[:, :2])))
            mlflow.log_metric('train_euclidean_distances', np.mean(
                euclidean_distances(train_y[:, :2], train_pred[:, :2])))

    for alpha in range(50, 300, 50):
        for c in range(1, 10, 2):
            with mlflow.start_run(run_name=f'run SVR {alpha}') as run:
                print('SVR alpha: ', alpha)
                mlflow.log_param("algo", "SVR")
                mlflow.log_param("lambda", alpha)
                mlflow.log_param("c", c)

                m = MultiOutputRegressor(
                    SVR(tol=100, kernel="poly", degree=alpha, C=c)).fit(
                    train_X, train_y)

                train_pred = m.predict(train_X)
                pred = m.predict(test_X)

                mlflow.log_metric('dev_mse', mean_squared_error(test_y[:, :2],
                                                                pred[:, :2]))
                mlflow.log_metric('train_mse',
                                  mean_squared_error(train_y[:, :2],
                                                     train_pred[:, :2]))
                mlflow.log_metric('dev_euclidean_distances', np.mean(
                    euclidean_distances(test_y[:, :2], pred[:, :2])))
                mlflow.log_metric('train_euclidean_distances', np.mean(
                    euclidean_distances(train_y[:, :2], train_pred[:, :2])))

    for alpha in range(1, 2001, 100):
        with mlflow.start_run(run_name=f'run NuSVR {alpha}') as run:
            print('NuSVR alpha: ', alpha)
            mlflow.log_param("algo", "NuSVR")
            mlflow.log_param("lambda", alpha)

            m = MultiOutputRegressor(NuSVR(nu=0.99, C=alpha)).fit(train_X,
                                                                  train_y)

            train_pred = m.predict(train_X)
            pred = m.predict(test_X)

            mlflow.log_metric('dev_mse',
                              mean_squared_error(test_y[:, :2], pred[:, :2]))
            mlflow.log_metric('train_mse', mean_squared_error(train_y[:, :2],
                                                              train_pred[:,
                                                              :2]))
            mlflow.log_metric('dev_euclidean_distances', np.mean(
                euclidean_distances(test_y[:, :2], pred[:, :2])))
            mlflow.log_metric('train_euclidean_distances', np.mean(
                euclidean_distances(train_y[:, :2], train_pred[:, :2])))

    for alpha in range(1, 101, 10):
        with mlflow.start_run(run_name=f'run MultiTaskLassoCV {alpha}') as run:
            print('MultiTaskLassoCV alpha: ', alpha)
            mlflow.log_param("algo", "MultiTaskLassoCV")
            mlflow.log_param("lambda", alpha)

            m = MultiTaskLassoCV(n_alphas=alpha).fit(train_X, train_y)

            train_pred = m.predict(train_X)
            pred = m.predict(test_X)

            mlflow.log_metric('dev_mse',
                              mean_squared_error(test_y[:, :2], pred[:, :2]))
            mlflow.log_metric('train_mse', mean_squared_error(train_y[:, :2],
                                                              train_pred[:,
                                                              :2]))
            mlflow.log_metric('dev_euclidean_distances', np.mean(
                euclidean_distances(test_y[:, :2], pred[:, :2])))
            mlflow.log_metric('train_euclidean_distances', np.mean(
                euclidean_distances(train_y[:, :2], train_pred[:, :2])))

    for alpha in range(0, 2000, 100):
        with mlflow.start_run(run_name=f'run ElasticNet {alpha}') as run:
            print('ElasticNet alpha: ', alpha)
            mlflow.log_param("algo", "ElasticNet")
            mlflow.log_param("lambda", alpha)

            m = MultiOutputRegressor(ElasticNet(alpha=alpha, tol=10)).fit(
                train_X, train_y)

            train_pred = m.predict(train_X)
            pred = m.predict(test_X)

            mlflow.log_metric('dev_mse',
                              mean_squared_error(test_y[:, :2], pred[:, :2]))
            mlflow.log_metric('train_mse', mean_squared_error(train_y[:, :2],
                                                              train_pred[:,
                                                              :2]))
            mlflow.log_metric('dev_euclidean_distances', np.mean(
                euclidean_distances(test_y[:, :2], pred[:, :2])))
            mlflow.log_metric('train_euclidean_distances', np.mean(
                euclidean_distances(train_y[:, :2], train_pred[:, :2])))

    for alpha in range(10, 300, 10):
        with mlflow.start_run(
                run_name=f'run RandomForestRegressor {alpha}') as run:
            print('RandomForestRegressor alpha: ', alpha)
            mlflow.log_param("algo", "RandomForestRegressor")
            mlflow.log_param("lambda", alpha)

            m = RandomForestRegressor(n_estimators=alpha).fit(train_X, train_y)

            train_pred = m.predict(train_X)
            pred = m.predict(test_X)

            mlflow.log_metric('dev_mse',
                              mean_squared_error(test_y[:, :2], pred[:, :2]))
            mlflow.log_metric('train_mse', mean_squared_error(train_y[:, :2],
                                                              train_pred[:,
                                                              :2]))
            mlflow.log_metric('dev_euclidean_distances', np.mean(
                euclidean_distances(test_y[:, :2], pred[:, :2])))
            mlflow.log_metric('train_euclidean_distances', np.mean(
                euclidean_distances(train_y[:, :2], train_pred[:, :2])))

    for alpha in range(20, 120, 20):
        with mlflow.start_run(
                run_name=f'run AdaBoostRandomForestRegressor {alpha}') as run:
            print('AdaBoostRandomForestRegressor alpha: ', alpha)
            mlflow.log_param("algo", "AdaBoostRandomForestRegressor")
            mlflow.log_param("lambda", alpha)

            m = MultiOutputRegressor(
                AdaBoostRegressor(RandomForestRegressor(n_estimators=alpha),
                                  n_estimators=5)).fit(train_X, train_y)

            train_pred = m.predict(train_X)
            pred = m.predict(test_X)

            mlflow.log_metric('dev_mse',
                              mean_squared_error(test_y[:, :2], pred[:, :2]))
            mlflow.log_metric('train_mse', mean_squared_error(train_y[:, :2],
                                                              train_pred[:,
                                                              :2]))
            mlflow.log_metric('dev_euclidean_distances', np.mean(
                euclidean_distances(test_y[:, :2], pred[:, :2])))
            mlflow.log_metric('train_euclidean_distances', np.mean(
                euclidean_distances(train_y[:, :2], train_pred[:, :2])))

    for alpha in range(20, 200, 20):
        with mlflow.start_run(run_name=f'run SGDRegressor {alpha}') as run:
            print('SGDRegressor alpha: ', alpha)
            mlflow.log_param("algo", "SGDRegressor")
            mlflow.log_param("lambda", alpha)

            m = MultiOutputRegressor(
                SGDRegressor(penalty="l2", alpha=0.01, l1_ratio=0.6,
                             power_t=alpha, early_stopping=True,
                             validation_fraction=0.01, eta0=0.0001)).fit(
                train_X, train_y)

            train_pred = m.predict(train_X)
            pred = m.predict(test_X)

            mlflow.log_metric('dev_mse',
                              mean_squared_error(test_y[:, :2], pred[:, :2]))
            mlflow.log_metric('train_mse', mean_squared_error(train_y[:, :2],
                                                              train_pred[:,
                                                              :2]))
            mlflow.log_metric('dev_euclidean_distances', np.mean(
                euclidean_distances(test_y[:, :2], pred[:, :2])))
            mlflow.log_metric('train_euclidean_distances', np.mean(
                euclidean_distances(train_y[:, :2], train_pred[:, :2])))

if __name__ == "__main__":
    run_mlflow = False

    train_df = pd.read_csv("data/waze_data_train_tlv.csv")
    dev_df = pd.read_csv("data/waze_data_dev_tlv.csv")

    classifier_names = ['Ridge', 'Lasso', 'LassoCV', 'SVR',
                        'NuSVR', 'MultiTaskLassoCV', 'ElasticNet',
                        'RandomForestRegressor', 'AdaBoostRandomForest']

    train_X, train_y = split_labels(train_df)
    test_X, test_y = split_labels(dev_df)

    if run_mlflow:
        run_mlflow_regression(train_X, train_y, test_X, test_y)

    train_size = []

    classifier_num = len(classifier_names)
    iterations = 50

    res = dict(zip(range(classifier_num), [dict(test_loss=np.zeros(iterations-1),
                                    train_loss=np.zeros(iterations-1),
                                    test_distance=np.zeros(iterations-1),
                                    train_distance=np.zeros(iterations-1)) for _ in range(classifier_num)]))

    for i in range(1, iterations):
        print(f'iteration {i}.')
        train_size.append(train_X.shape[0])
        test_X, test_y = split_labels(dev_df)

        train, _ = train_test_split(train_df, train_size=i / (iterations), random_state=0)
        train_X, train_y = split_labels(train)

        # fit:
        # lr = MultiOutputRegressor(LinearRegression()).fit(train_X, train_y)
        ridge = MultiOutputRegressor(Ridge(alpha=200)).fit(train_X, train_y)
        lasso = MultiOutputRegressor(Lasso(alpha=100)).fit(train_X, train_y)
        lassoCv = MultiOutputRegressor(LassoCV(eps=1e-4,n_alphas=100, tol=1e-4)).fit(train_X, train_y)
        svr = MultiOutputRegressor(SVR(tol=100,kernel="poly",degree=100, C=5)).fit(train_X,train_y)
        nuSvr = MultiOutputRegressor(NuSVR(nu=0.99,C=1000.0)).fit(train_X, train_y)
        MTlasso = MultiTaskLassoCV(n_alphas=20,).fit(train_X, train_y)
        en = MultiOutputRegressor(ElasticNet(alpha=100,tol=10)).fit(train_X, train_y)
        rf = RandomForestRegressor(n_estimators=160).fit(train_X, train_y)
        adaRf = MultiOutputRegressor(AdaBoostRegressor(RandomForestRegressor(n_estimators=100), n_estimators=5)).fit(train_X, train_y)
        # sgd = MultiOutputRegressor(SGDRegressor(penalty="l2",alpha=0.01,l1_ratio=0.6,power_t=100,early_stopping=True,validation_fraction=0.01,eta0=0.0001)).fit(train_X, train_y)

        # predict:
        classifiers = [ridge, lasso, lassoCv, svr, nuSvr, MTlasso, en, rf, adaRf]
        train_predictions = [classifier.predict(train_X) for classifier in classifiers]
        test_predictions = [classifier.predict(test_X) for classifier in classifiers]

        # log:
        idx = i-1
        for j in range(classifier_num):
            res[j]['test_loss'][idx] = mean_squared_error(test_y[:, :2], test_predictions[j][:, :2])
            res[j]['train_loss'][idx] = mean_squared_error(train_y[:, :2], train_predictions[j][:, :2])
            res[j]['test_distance'][idx] = np.mean(euclidean_distances(test_predictions[j][:, :2], test_predictions[j][:, :2]))
            res[j]['train_distance'][idx] = np.mean(euclidean_distances(train_predictions[j][:, :2], train_predictions[j][:, :2]))


    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Error as a function of train size",
                                        "mean distance as a function of train size"],
                        horizontal_spacing=0.05, vertical_spacing=.03)
    for j in range(classifier_num):
        fig.add_trace(go.Scatter(x=train_size[1:-1], y=res[j]['test_loss'][1:-1], name=f'{classifier_names[j]} test loss'), row=1, col=1)
        fig.add_trace(go.Scatter(x=train_size[1:-1], y=res[j]['train_loss'][1:-1], name=f'{classifier_names[j]} train loss'), row=1, col=1)
        fig.add_trace(go.Scatter(x=train_size[1:-1], y=res[j]['test_distance'][1:-1], name=f'{classifier_names[j]} test distance'), row=1, col=2)
        fig.add_trace(go.Scatter(x=train_size[1:-1], y=res[j]['train_distance'][1:-1], name=f'{classifier_names[j]} train distance'), row=1, col=2)


    fig.update_layout(title="Regression classifiers comparison",
                      xaxis_title="Train size",
                      yaxis_title="Error",
                      showlegend=True)

    fig.write_html("RegressionClassifiersComparison.html")
