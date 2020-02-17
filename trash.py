import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import base64
from io import BytesIO
import datetime
import cv2
import numpy as np
from PIL import Image
import random
from keras.models import load_model, model_from_json

label = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', "truck"]

app = dash.Dash(__name__)

new_img = False
click = False
cur_content = None
res_default = 'Let click!'
cur_acc = res_default

def pre_process(img):
    return cv2.resize(img, (32,32))

def compute(img):
    global label
    image = pre_process(img)
    image = image.reshape((1,32,32,3))
    with open('model.json') as j:
        model = j.read()

    model = model_from_json(model)
    model.load_weights('w.h5')

    pre = model.predict(image)

    x = pre.argmax()
    lb = label[x]
    acc = int(pre[0][x]*100)

    return lb+': '+str(acc)+'%'

app.layout = html.Div([
    html.H1('Classify Web'),
    html.P("We have 10 classes: 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'"),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drop here',
            html.Button('Select File'),
        ]),
        multiple=True
    ),
    html.Div(id='output-data'),
    html.Button(id='button', n_clicks=0, children='Get Result'),
    html.Div(id='res')
])


def process(content, mes):
    global res_default
    if mes is not None:
        return html.Div([
        # html.H3(filename),
        # html.H4(datetime.datetime.fromtimestamp(date)),

        html.Img(src=content),
        html.P(mes)
        # html.Button(id='button', n_clicks=0, children='Get Result')
    ])
    i = content.split(',')[1]
    t = BytesIO(base64.b64decode(i))
    img = Image.open(t)
    T = np.array(img)
    img = cv2.cvtColor(T, cv2.COLOR_BGR2RGB)
    pre = compute(img)
    return html.Div([
        # html.H3(filename),
        # html.H4(datetime.datetime.fromtimestamp(date)),

        html.Img(src=content),
        html.P(pre)
        # html.Button(id='button', n_clicks=0, children='Get Result')
    ])


@app.callback(
    Output(component_id='output-data', component_property='children'),
    [Input(component_id='upload-data', component_property='contents'),
     Input('button', 'n_clicks')]
)
def update(content, n_click):
    global click, cur_content, res_default
    if content is not None:
        if cur_content!=content:
            # new_img = True
            cur_content = content
            click = True
            return [process(C, res_default) for C in cur_content]
        else:
            return [process(C, None) for C in cur_content]

@app.callback(
    Output('res', 'children'),
    [Input(component_id='upload-data', component_property='contents'),
     Input('button', 'n_clicks')]
)
def get_result(content, n_click):
    global click, cur_acc, cur_content
    if content is not None:
        if cur_content!=content:
            return ""
        else:
            if click == True:
                # acc = compute()
                # cur_acc = acc
                click = False
            return "Thanks for use my app!"

    


if __name__ == '__main__':
    app.run_server(debug=True)
