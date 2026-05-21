from dash import Dash, dcc, html

from pages.q1 import layout as q1_layout
from pages.q2 import layout as q2_layout
from pages.q3 import layout as q3_layout
from pages.q4 import layout as q4_layout
from pages.q5 import layout as q5_layout

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("GAStech Visual Analytics Dashboard"),

    dcc.Tabs([
        dcc.Tab(label="Question 1", children=q1_layout()),
        dcc.Tab(label="Question 2", children=q2_layout()),
        dcc.Tab(label="Question 3", children=q3_layout()),
        dcc.Tab(label="Question 4", children=q4_layout()),
        dcc.Tab(label="Question 5", children=q5_layout()),
    ])
])

if __name__ == "__main__":
    app.run(debug=True)