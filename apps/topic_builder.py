from app import app
import json
import umap
from sklearn.manifold import TSNE
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_cytoscape as cyto
import plotly.graph_objects as go
import dash
import plotly.express as px
import numpy as np
import pandas as pd
import logging

# ===== START LOGGER =====
logger = logging.getLogger(__name__)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
sh = logging.StreamHandler()
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
sh.setFormatter(formatter)
root_logger.addHandler(sh)


# network_df = pd.read_csv('outputs/network_df.csv', index_col=0)  # ~8300 nodes
network_df = pd.read_csv("outputs/network_df.csv")

# Prep data
network_df["target_title"] = network_df["target_title"].fillna("")
network_df["topic_id"] = network_df["topic_id"].astype(str)
topic_ids = [str(i) for i in range(len(network_df["topic_id"].unique()))]
# lda_val_arr = network_df[topic_ids].values

with open("outputs/lda_topics.json", "r") as f:
    lda_topics = json.load(f)
topics_txt = [lda_topics[str(i)] for i in range(len(lda_topics))]
topics_txt = [[j.split("*")[1].replace('"', "")
               for j in i] for i in topics_txt]
topics_txt = ["; ".join(i) for i in topics_txt]

src_ser = network_df.groupby("src")["0"].count().sort_values(ascending=False)
src_ser = src_ser.append(pd.Series({'All': src_ser.sum()}))

# with open("outputs/startup_elms.json", "r") as f:
#    startup_elms = json.load(f)


# startup_elm_list = startup_elms["elm_list"]

lda_df = pd.read_csv("outputs/lda_df.csv")

col_swatch = px.colors.qualitative.Dark24
def_stylesheet = [
    {
        "selector": "." + str(i),
        "style": {"background-color": col_swatch[i], "line-color": col_swatch[i]},
    }
    for i in range(len(network_df["topic_id"].unique()))
]
def_stylesheet += [
    {
        "selector": "node",
        "style": {"width": "data(node_size)", "height": "data(node_size)"},
    },
    {"selector": "edge", "style": {"width": 1, "curve-style": "bezier"}},
]


def gen_scatter_plot(lda_df, tsne_perp=40, algo='tsne'):

    arr_cols = list(range(12))
    arr_cols = map(str, arr_cols)
    lda_arr = np.array(lda_df[arr_cols])
    # for tsne_perp in [20, 35, 50, 100, 200]:  # Test out different perplexity values
    if algo == 'tsne':
        for tsne_perp in [tsne_perp]:  # Test out different perplexity values
            node_locs = TSNE(
                n_components=2,
                perplexity=tsne_perp,
                n_iter=350,
                n_iter_without_progress=100,
                learning_rate=500,
                random_state=42,
            ).fit_transform(lda_arr)
        plot_title = "t-SNE, perplexity: " + str(tsne_perp)

    elif algo == "umap":
        reducer = umap.UMAP(n_components=2)
        node_locs = reducer.fit_transform(lda_arr)
        plot_title = "UMAP"

    lda_df = pd.concat([lda_df, pd.DataFrame(
        node_locs, columns=["a", "b"])], axis=1)

    topic_ids = "Topic: " + lda_df["topic_id"].astype(str).values

    # prep color mapping to be the same as html topic list
    set_topics = np.unique(topic_ids)
    set_topics = sorted(set_topics, key=lambda x: int(x.split(' ')[-1]))

    t_c_map = list(zip(set_topics, col_swatch))
    colorsIdx = {k: v for k, v in t_c_map}
    cols = pd.Series(topic_ids).map(colorsIdx)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lda_df['a'],
        y=lda_df['b'],
        marker_color=cols,
        mode="markers",
        text=lda_df['title'])
    )
    fig.update_traces(marker=dict(size=8,
                                  line=dict(width=1,
                                            color='white')),
                      selector=dict(mode='markers'),)
    fig.update_layout(showlegend=False, title=plot_title)
    return fig


topics_html = list()
for topic_html in [
    html.Span([str(i) + ": " + topics_txt[i]], style={"color": col_swatch[i]})
    for i in range(len(topics_txt))
]:
    topics_html.append(topic_html)
    topics_html.append(html.Br())

body_layout = dbc.Container(
    [
        dbc.Row(
            [dbc.Col(
                [dcc.Markdown(
                    f"""
                -----
                # Data:
                -----
                For this demonstration, {len(network_df)} confluence and medium.com documentations dataset* were categorised into
                {len(network_df.topic_id.unique())} topics using
                [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) analysis.

                Each topic is shown in different color on the citation map, as shown on the right.
                """)],
                sm=20,
                md=4,
            ),
                dbc.Col(
                    [dcc.Markdown(
                        """
                -----
                # Topics:
                -----
                """
                    ),
                        html.Div(
                            topics_html,
                            style={
                                "fontSize": 13,
                                "height": "160px",
                                "overflow": "auto",
                            },
                            className="app__lda_topic_list",
                    ), ],
                    sm=20,
                    md=8,
            ), ],
        ),
        html.Br(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dcc.Graph(
                                    id="scatter_dim_reduced",
                                    # layout={"name": "preset"},
                                    style={"width": "100%", "height": "600px"},
                                    # elements=startup_elm_list,
                                    # stylesheet=def_stylesheet,
                                    # minZoom=0.06,
                                )
                            ]
                        ),
                        # dbc.Row(
                        #    [
                        #        dbc.Alert(
                        #            id="node-data",
                        #            children="Click on a node to see its details here",
                        #            color="secondary",
                        #        )
                        #    ]
                        # ),
                    ],
                    sm=12,
                    md=8,
                ),
                dbc.Col(
                    [
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        html.Br(),
                        dbc.Badge(
                            "Data Sources:", color="info", className="mr-1"
                        ),
                        dbc.FormGroup(
                            [
                                dcc.Dropdown(
                                    id="src_dropdown",
                                    options=[
                                        {
                                            "label": i
                                            + " ("
                                            + str(v)
                                            + " documents(s))",
                                            "value": i,
                                        }
                                        for i, v in src_ser.items()
                                    ],
                                    value='All',
                                    multi=False,
                                    style={"width": "100%"},
                                ),
                            ]
                        ),
                        html.Br(),
                        dbc.Badge(
                            "Dimensionality reduction algorithm",
                            color="info",
                            className="mr-1",
                        ),
                        dbc.FormGroup(
                            [
                                dcc.RadioItems(
                                    id="dim_red_algo",
                                    options=[
                                        {"label": "UMAP", "value": "umap"},
                                        {"label": "t-SNE", "value": "tsne"},
                                    ],
                                    value="tsne",
                                    labelStyle={
                                        "display": "inline-block",
                                        "color": "DarkSlateGray",
                                        "fontSize": 12,
                                        "margin-right": "10px",
                                    },
                                )
                            ]
                        ),
                        html.Br(),
                        dbc.Badge(
                            "t-SNE parameters (not applicable to UMAP):",
                            color="info",
                            className="mr-1",
                        ),
                        dbc.Container(
                            "Current perplexity: 40 (min: 10, max:100)",
                            id="tsne_para",
                            style={"color": "DarkSlateGray", "fontSize": 12},
                        ),
                        dbc.FormGroup(
                            [
                                dcc.Slider(
                                    id="tsne_perp",
                                    min=10,
                                    max=100,
                                    step=1,
                                    marks={10: "10", 100: "100", },
                                    value=40,
                                ),
                                # html.Div(id='slider-output')
                            ]
                        ),
                    ],
                    sm=12,
                    md=4,
                ),
            ]
        ),
    ],
    style={"marginTop": 20},
)

model_eval_layout = dbc.Container([html.Br(),
                                   dbc.Row([
                                       dbc.Col([html.Br(),
                                                # html.H4('Some detailed explaination of the methods we covered:'),
                                                # html.Label(['Some detailed explaination of the methods we covered', html.A('link', href='documentation.html')]),
                                                html.A('LDA visualized',
                                                       href='/assets/lda_vis.html', target="_blank"),
                                                html.Br(),
                                                html.Iframe(src=app.get_asset_url('lda_vis.html'), className="iframe-lda-vis",
                                                #style=dict(position="relative", left="0", top="0", width="100%", height="100%")
                                                ),
                                                html.Br(),
                                                html.Iframe(src=app.get_asset_url('lda_cv_coherence.html'), className="iframe-lda-vis", style={'border': 'none'}),
                                            ],)
                                   ],
    className="link-to-doc")], style={"marginTop": 20},)


@app.callback(
    dash.dependencies.Output("tsne_para", "children"),
    [dash.dependencies.Input("tsne_perp", "value")],
)
def update_output(value):
    return f"Current t-SNE perplexity: {value} (min: 10, max:100)"


@app.callback(
    Output("scatter_dim_reduced", "figure"),
    [
        Input("src_dropdown", "value"),
        Input("dim_red_algo", "value"),
        Input("tsne_perp", "value"),
    ],
)
def filter_nodes(data_src, dim_red_algo, tsne_perp):

    if data_src == 'All':
        filter_df = lda_df
    else:
        filter_df = lda_df[lda_df['src'] == data_src]

    return gen_scatter_plot(filter_df, tsne_perp, dim_red_algo)


@app.callback(
    Output("node-data",
           "children"), [Input("scatter_dim_reduced", "selectedNodeData")]
)
def display_nodedata(datalist):
    contents = "Click on a node to see its details here"
    if datalist is not None:
        if len(datalist) > 0:
            data = datalist[-1]
            contents = []
            contents.append(html.H5("Title: " + data["title"].title()))
            contents.append(
                html.P(
                    # "Data source: "
                    # + data["doc_src"].title()
                    " Published: "
                    + data["pub_date"]
                )
            )
            contents.append(
                html.P(
                    "Author(s): "
                    + str(data["authors"])
                )
            )

    return contents


layout = html.Div([body_layout, model_eval_layout])
