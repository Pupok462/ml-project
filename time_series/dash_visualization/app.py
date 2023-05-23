import dash_bootstrap_components as dbc
from dash import Dash, dcc, html
import pandas as pd
from features import scale_data_selector, scale_data_picker
from tabs.arima import tab_content as arima
from tabs.lstm import tab_content as lstm
from tabs.data import tab_content as data_tab
from tabs.page_default import tab_content as page_default


app = Dash(__name__,  external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([
    dbc.Tabs([
        dbc.Tab(arima, label='ARIMA'),
        dbc.Tab(lstm, label='LSTM'),
        dbc.Tab(page_default, label='XGBOOST'),
        dbc.Tab(page_default, label='Prophet'),
        dbc.Tab(page_default, label='Gluon'),
        dbc.Tab(page_default, label='LightGBM'),
        dbc.Tab(data_tab, label='Data'),
    ], style={'padding-left': '37%'})

], style={'margin-left': '40px', 'margin-right': '40px', 'margin-top': '20px'})


if __name__ == '__main__':
    app.run_server(debug=True)
