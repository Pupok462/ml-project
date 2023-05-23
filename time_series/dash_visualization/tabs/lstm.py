from dash import dcc
import dash_bootstrap_components as dbc
from dash import html
from scripts.graphs import show_ts, show_forecast_test_plot
from model import data, lstm_test, lstm_pred, lstm_train


button_style = {
    'appearance': 'button',
    'backface-visibility': 'hidden',
    'background-color': '#405cf5',
    'border-radius': '6px',
    'border-width': '0',
    'box-shadow': 'rgba(50, 50, 93, .1) 0 0 0 1px inset,rgba(50, 50, 93, .1) 0 2px 5px 0,rgba(0, 0, 0, .07) 0 1px 1px 0',
    'box-sizing': 'border-box',
    'color': '#fff',
    'cursor': 'pointer',
    'font-family': '-apple-system,system-ui,"Segoe UI",Roboto,"Helvetica Neue",Ubuntu,sans-serif',
    'font-size': '100%',
    'height': '44px',
    'line-height': '1.15',
    'margin': '12px 0 0',
    'outline': 'none',
    'overflow': 'hidden',
    'padding': '0 25px',
    'position': 'relative',
    'text-align': 'center',
    'text-transform': 'none',
    'transform': 'translateZ(0)',
    'transition': 'all .2s,box-shadow .08s ease-in',
    'user-select': 'none',
    '-webkit-user-select': 'none',
    'touch-action': 'manipulation',
    'width': '100%',
}


tab_content = [
    dbc.Row(html.Button('Обновить данные', id='submit-val', n_clicks=0, style=button_style), style={'width': '10%', 'margin-left': '45%', 'margin-top': '10px'}),
    dbc.Row(
        [
            dbc.Col(
                [
                    dcc.Graph(figure=show_ts(data), id='base_ts')
                ], width={'size': 6}),
            dbc.Col(
                [
                    dcc.Graph(figure=show_forecast_test_plot(data, lstm_train, lstm_pred, lstm_test, 'LSTM'), id='arima_pred')
                ], width={'size': 6})
        ]),
]

