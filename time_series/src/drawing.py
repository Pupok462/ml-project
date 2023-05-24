import plotly.graph_objects as go
import statsmodels.api as sm
import matplotlib.pyplot as plt


def show_loss_after_train(train_loss, val_loss, metric_score):
    """
    :param train_loss: train loss from model
    :param val_loss: val loss from model
    :param metric_score: metric score from model
    :return: Graph
    """
    plt.figure(figsize=(15, 9))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.subplot(1, 2, 2)
    plt.plot(metric_score)
    plt.xlabel("epochs")
    plt.ylabel("score")

    plt.show()


def show_forecast_test_plot(time, train, forecasts, test):
    """
    :param time: column Date of pd.DataFrame from datasets with time
    :param train: column target_value[:train_size] of pd.DataFrame from dateset with value
    :param forecasts: values of predicton ur model
    :param test: column target_value[train_size:] of pd.DataFrame from dateset with value
    :return: Graph
    """
    fig = go.Figure([go.Scatter(x=time[:len(train)], y=train),
                     go.Scatter(x=time[len(train):], y=forecasts, name='forecast'),
                     go.Scatter(x=time[len(train):], y=test, name='test')])
    fig.show()


def show_time_series_only(time, target):
    """
    :param time: column Date of pd.DataFrame from datasets with time
    :param target: column target_value of pd.DataFrame from dateset with value
    :return: Graph
    """
    fig = go.Figure(go.Scatter(x=time[:len(target)], y=target))

    fig.show()


def show_acf_pacf(ts, lags):
    """
    ACF поможет нам определить q, т. к. по ее коррелограмме можно определить количество автокорреляционных коэффициентов сильно отличных от 0 в модели MA \n
    PACF поможет нам определить p, т. к. по ее коррелограмме можно определить последний номер коэффициента сильно отличный от 0 в модели AR

    """
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(ts.values.squeeze(), lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(ts, lags=lags, ax=ax2, method='ywm')

    fig.show()
