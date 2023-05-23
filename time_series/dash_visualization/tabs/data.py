from dash import dash_table, html
from model import data


tab_content = html.Div(
    dash_table.DataTable(
        data.to_dict('records'), [{"name": i, "id": i} for i in data.columns]),
    style={'padding-left': '15%', 'padding-right': '15%'}
)
