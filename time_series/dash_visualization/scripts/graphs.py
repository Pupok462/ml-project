import matplotlib.pyplot as plt
import plotly.graph_objects as go
import statsmodels.api as sm


def show_ts(ts):
    time = ts.Date
    target = ts.target_value

    fig = go.Figure(go.Scatter(x=time, y=target, ))
    fig.update_layout(
        title={
            'text': "Исходный график временного ряда",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Дата",
        yaxis_title="Цена валюты",
    )
    return fig


def show_forecast_test_plot(ts, train, forecasts, test, model_name):

    time = ts.Date

    fig = go.Figure([go.Scatter(x=time[:len(train)], y=train, name='train'),
                     go.Scatter(x=time[len(train):], y=forecasts, name='forecast'),
                     go.Scatter(x=time[len(train):], y=test, name='test')])
    fig.update_layout(
        title={
            'text': f"Предсказание модели {model_name}",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Дата",
        yaxis_title="Цена валюты",
    )
    return fig


def draw_acf_pacf(ts, lags):
    fig = plt.figure(figsize=(24,10))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(ts.values.squeeze(), lags=lags, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(ts, lags=lags, ax=ax2, method='ywm')
