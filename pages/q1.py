from dash import html, dcc, callback, Input, Output

from data_loader import locations
from figures import (
    make_cc_vs_loyalty_locations,
    make_anomaly_chart,
    make_anomaly_table,
    make_purchase_activity_by_hour,
    make_day_hour_heatmap,
    make_weekday_activity,
    make_location_hour_heatmap,
    make_location_weekday_heatmap,
    make_location_timeline
)


def layout():
    return html.Div([
        html.H2("Question 1 — Credit and Loyalty Card Analysis"),

        dcc.Graph(figure=make_cc_vs_loyalty_locations()),
        dcc.Graph(figure=make_anomaly_chart()),

        html.H3("Top Credit/Loyalty Anomalies"),
        make_anomaly_table(),

        dcc.Graph(figure=make_purchase_activity_by_hour()),
        dcc.Graph(figure=make_day_hour_heatmap()),
        dcc.Graph(figure=make_weekday_activity()),
        dcc.Graph(figure=make_location_hour_heatmap()),
        dcc.Graph(figure=make_location_weekday_heatmap()),

        html.H3("Inspect One Location"),
        html.Label("Select location"),
        dcc.Dropdown(
            id="q1-location-dropdown",
            options=[{"label": loc, "value": loc} for loc in locations],
            value=locations[0],
            clearable=False
        ),

        dcc.Graph(id="q1-location-timeline")
    ])


@callback(
    Output("q1-location-timeline", "figure"),
    Input("q1-location-dropdown", "value")
)
def update_location_timeline(location):
    return make_location_timeline(location)