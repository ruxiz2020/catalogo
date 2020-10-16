import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc


# must add this line in order for the app to be deployed successfully on Heroku
from app import server
from app import app
# import all pages in the app
from apps import home, single_article, topic_builder, active_learning

# building the navigation bar
# https://github.com/facultyai/dash-bootstrap-components/blob/master/examples/advanced-component-usage/Navbars.py
dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("Home", href="/home", className="drop-zone-dropdown"),
        dbc.DropdownMenuItem("LDA Topic Builder", href="/library", className="drop-zone-dropdown"),
        dbc.DropdownMenuItem("Active Learning Labeling", href="/active_learning", className="drop-zone-dropdown"),
        dbc.DropdownMenuItem("Single Text Explore", href="/single_text", className="drop-zone-dropdown"),
        dbc.DropdownMenuItem("Documentation of Methods", href="/assets/documentation.html", className="drop-zone-dropdown", target="_blank"),
    ],
    nav = True,
    in_navbar = True,
    label = "Explore",
    className="nav-dropdown",
)

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src="/assets/library.png", height="55px")),
                        dbc.Col(dbc.NavbarBrand("A Topic Library Builder", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="/home",
            ),
            dbc.NavbarToggler(id="navbar-toggler2"),
            dbc.Collapse(
                dbc.Nav(
                    # right align dropdown menu with ml-auto className
                    [dropdown], className="ml-auto", navbar=True
                ),
                id="navbar-collapse2",
                navbar=True,
            ),
        ]
    ),
    color="dark",
    dark=True,
    className="nav_bar",
)

def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

for i in [2]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)

# embedding the navigation bar
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/home':
        return home.layout
    elif pathname == '/single_text':
        return single_article.layout
    elif pathname == '/library':
        return topic_builder.layout
    elif pathname == '/active_learning':
        return active_learning.layout
    else:
        return home.layout

if __name__ == '__main__':
    app.run_server(debug=True, port=9000)
