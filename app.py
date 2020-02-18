import dash
import dash_core_components as dcc
import dash_html_components as html

style = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=style)

app.layout = html.Div(children=[
    html.H1(children="Hello Khoa!"),

    html.Div(children='''
    Bad framework'''),
])

if __name__=='__main__':
    app.run_server(debug=True)