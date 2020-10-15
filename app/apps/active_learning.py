from app import app
from catalogo.util.extract_text import ExtractText
from catalogo.algo.summarize import Summarize
import os
import pathlib
import statistics
import sys
from collections import OrderedDict
from wordcloud import WordCloud, STOPWORDS

import pathlib
import dash
import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd

sys.path.append("..")
#app = dash.Dash(__name__)
#server = app.server
#

df_lda = pd.read_csv("outputs/summary_df.csv.gz", compression='gzip')
columns_to_display = ['topics', 'topic_txt',
                      'title', 'author', 'src', 'summary']
df_lda = df_lda[columns_to_display]

ls_topic = df_lda['topics'].unique().tolist()
list_topic_map = [{'label': t, 'value': t} for t in ls_topic]


df_topic_coherence = pd.read_csv("outputs/df_topic_coherence.csv")

control_layout = html.Div(
    className="container",
    children=[
        html.Div(
            className="row",
            style={},
            children=[
                html.Div(
                    className="five columns control-active-learning-settings",
                    children=[
                        #html.P(["Enter url or local html file path:"]),
                        html.Div(
                            [
                                html.Br(),
                                html.P(
                                    "Pick a topic to examine: "),
                                dcc.Dropdown(
                                    id='active-learning-dropdown',
                                    options=list_topic_map,
                                    value=ls_topic[0]),

                                html.Br(),
                                html.P(
                                    "Pick a text to update: "),
                                dcc.Dropdown(
                                    id='opt-text-dropdown', style={
                                        'width': '100%', 'height': '100%', 'font-size': '9px'}),

                                html.Br(),
                                html.P(
                                    "Enter Keywords for selected text in descending order, seperated by comma: "),
                                dcc.Textarea(
                                    id='al-input-Keywords',
                                    style={
                                        'width': '100%', 'height': '100%', 'font-size': '14px'},
                                    value='keyword1, keyword2, keyword3'),
                                #html.Div(id='intermediate-df', style={'display': 'none'})
                            ],),
                    ],
                ),
                html.Div(
                    className="seven columns ",
                    children=[
                        html.Div(
                            [
                                html.Br(),
                                html.P(
                                    "Coherence Scores by Topic ID: "),
                                html.Div(
                                    [dcc.Graph(id="single-topic-bar"), ],
                                    className="app__barchart"),
                            ],),
                    ],
                ),
            ],),
    ],)

table_layout = html.Div(
    className="",
    children=[

        html.Div(
            className="container",
            children=[
                html.Div(
                    className="row",
                    style={},
                    children=[

                        html.Div(
                            className="fifteen columns ",
                            children=[
                                dash_table.DataTable(
                                    id='ida_table',
                                    style_cell={
                                        'whiteSpace': 'normal',
                                        # 'height': 'auto',  # wrapped text in column
                                        'textAlign': 'left',
                                        'overflowY': 'auto',
                                        'overflowX': 'auto',
                                        'fontSize': 12,
                                        'font-family': 'sans-serif',
                                        'minWidth': '40px',
                                        'maxHeight': '100px',
                                        'padding': '8px',
                                    },
                                    style_data={
                                        'maxHeight': '100px',
                                        # 'width': '150px', 'minWidth': '150px', 'maxWidth': '200px',
                                        # 'textOverflow': 'ellipsis',
                                    },
                                    style_table={
                                        'maxHeight': '100%',
                                        'width': '100%',
                                        'maxWidth': '200%',
                                        'margin': '8px',
                                    },
                                    # style header
                                    style_header={
                                        'fontWeight': 'bold',
                                        'fontSize': 13,
                                        'color': 'white',
                                        'backgroundColor': '#357EC7',
                                        'textAlign': 'center',
                                    },
                                    style_data_conditional=[
                                        {
                                            'if': {'column_id': 'topic_txt'},
                                            'width': '160px',
                                        },
                                        {
                                            'if': {'column_id': 'topics'},
                                            'width': '80px',
                                        },
                                        {
                                            'if': {'column_id': 'title'},
                                            'width': '260px',
                                        },
                                        {
                                            'if': {'column_id': 'summary'},
                                            'width': '260px',
                                        },
                                        {
                                            'if': {'column_id': 'filename'},
                                            'width': '260px',
                                        },
                                        {
                                            'if': {'column_id': 'author'},
                                            'width': '80px',
                                        },
                                        {
                                            'if': {'column_id': 'date'},
                                            'width': '80px',
                                        },

                                    ],
                                    columns=(
                                        [{'id': p, 'name': p}
                                            for p in columns_to_display]
                                    ),
                                    # data=df_lda.to_dict(orient='records'),
                                    fixed_rows={'headers': True, 'data': 0},
                                    editable=False,
                                    filter_action='native',
                                    sort_action='native',
                                    sort_mode='multi',
                                    # page_action='native',
                                    # page_size=10,
                                    # virtualization=True, # this returns error
                                    page_action='none',
                                    # row_selectable="multi",
                                    row_deletable=True,
                                ),
                            ],
                        ),
                    ],
                ),

            ],
        ),
    ],
)

layout = html.Div([control_layout, table_layout])


def gen_bar_chart(topic_number):

    import ast
    df = df_topic_coherence[df_topic_coherence['topic_number'] == topic_number]

    sel = ast.literal_eval(df['keywords'].values[0])

    x_val = [t[1] for t in sel]
    y_val = [t[0] for t in sel]
    coh = df['coherence'].values[0]
    fig = go.Figure([go.Bar(x=x_val, y=y_val)])
    fig.update_layout(title_text='Topic Number: ' +
                      str(topic_number) + ', coherence: ' + str(round(coh, 2)))
    return fig




@app.callback([
    Output("ida_table", "data"),
    Output("single-topic-bar", "figure"),
    Output('opt-text-dropdown', 'options'),
],
    [Input("active-learning-dropdown", "value")]
)
def filter_by_topic(topic_num):
    """
    """
    df_lda = pd.read_csv("outputs/summary_df.csv.gz", compression='gzip')
    df_lda = df_lda[df_lda['topics'] == topic_num]
    data = df_lda.to_dict(orient='records')

    topic_number = int(topic_num.split(":")[1])
    fig = gen_bar_chart(topic_number)

    opt = [{'label': i, 'value': i} for i in df_lda['title'].values.tolist()]

    return data, fig, opt
