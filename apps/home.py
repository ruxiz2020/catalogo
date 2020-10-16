import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_core_components as dcc

# needed only if running this as a single page app
#external_stylesheets = [dbc.themes.LUX]

#app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# change to app.layout if running as single page app instead
layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Welcome to Topic Builder  ",
                            className="text-center main_title"), className="mb-5 mt-5")
        ]),
        dbc.Row([
            dbc.Col(html.H4(
                children='A research topic library builder '), className="text-center sub_title")
        ]),
        dbc.Row([
            dbc.Col([html.Br(),
                     html.H5('There are 3 parts of this topic builder tool:'),
                     html.Br(),
                     html.P(
                         '1. A Topic Modeling Library which uses LDA to automatically cluster documents into K topics.'),
                    html.P(
                        '2. An Active Learning mechanism that automatically generates minimal sample for user labeling, which then feeds back to the topic clustering result.'),
                    html.P(
                        '3. A single text explore helper that extracts keywords and summary for easy user labeling.'),
                     ])
        ]),

        html.Br(),
        html.Br(),

        dbc.Row([
            dbc.Col(dbc.Card(children=[html.H5(children='LDA: Topic Model Automatic Builder',
                                               className=" text-center"),
                                       dbc.Button("Click",
                                                  href="/library",
                                                  color="primary",
                                                  className="mt-3 col-s-9 home_button"),

                                       ],
                             body=True, color="dark", outline=True, className="home_click_box"), className="mb-3 "),

            dbc.Col(dbc.Card(children=[html.H5(children='Active-learning: User Labeling',
                                               className="text-center"),
                                       dbc.Button("Click",
                                                  href="/active_learning",
                                                  color="primary",
                                                  className="mt-3 col-s-9 home_button_label"),
                                       ],
                             body=True, color="dark", outline=True, className="home_click_box"), className="mb-3"),

            dbc.Col(dbc.Card(children=[html.H5(children='Single Article Explore: Summary',
                                               className="text-center"),
                                       dbc.Button("Click",
                                                  href="/single_text",
                                                  color="primary",
                                                  className="mt-3 col-s-9 home_button_text"),
                                       ],
                             body=True, color="dark", outline=True, className="home_click_box"),className="mb-3"),

        ], className="mb-5"),

        html.Br(),
        dbc.Row([
            dbc.Col([html.Br(),
                     #html.H4('Some detailed explaination of the methods we covered:'),
                     #html.Label(['Some detailed explaination of the methods we covered', html.A('link', href='documentation.html')]),
                     html.A('Some detailed explaination of the methods we covered',
                        href='/assets/documentation.html', target="_blank"),
                     html.Br(),
                     ])
        ], className="link-to-doc"),
    ])

])


# needed only if running this as a single page app
# if __name__ == '__main__':
#     app.run_server(host='127.0.0.1', debug=True)
