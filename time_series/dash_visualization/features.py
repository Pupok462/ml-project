from dash import dcc
from model import data
import pandas as pd


# currency_selector = dcc.Dropdown(
#     id='card_selector',
#     options='d',
#     value=['ETH', 'BTC', 'DOG', 'SMTH'],
#     multi=True
# )

scale_data_selector = dcc.RangeSlider(
    id="scale_slider",
    min=min(data.Date),
    max=max(data.Date),
    marks={25: '2021-12-20', 30: '2022-02-20', 35: '2022-04-20', 40: '2022-06-20', 45: '2022-08-20', 50: '2022-10-20', 55: '2022-12-20'},
    step=1,
    value=[min(data.Date), max(data.Date)]
)

scale_data_picker = dcc.DatePickerRange(
        id='scale_data_picker',
        clearable=True,
        start_date=min(data.Date),
        end_date=max(data.Date),
        min_date_allowed=min(data.Date),
        max_date_allowed=max(data.Date),
)
