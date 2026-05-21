import pandas as pd
from dash import html, dcc, callback, Input, Output, dash_table

from data_loader import gps, cc, vehicle_options
from figures import (
    make_match_status_chart,
    make_unmatched_location_chart,
    make_vehicle_map
)


def layout():
    return html.Div([
        html.H2("Question 2 — Vehicle, Credit, and Loyalty Analysis"),

        dcc.Graph(figure=make_match_status_chart()),
        dcc.Graph(figure=make_unmatched_location_chart()),

        html.H3("Vehicle Map"),

        html.Label("Select date"),
        dcc.DatePickerSingle(
            id="q2-date-picker",
            min_date_allowed=gps["timestamp"].min().date(),
            max_date_allowed=gps["timestamp"].max().date(),
            initial_visible_month=gps["timestamp"].min().date(),
            date=gps["timestamp"].min().date()
        ),

        html.Br(),
        html.Br(),

        html.Label("Select time range"),
        dcc.RangeSlider(
            id="q2-time-slider",
            min=0,
            max=23,
            step=1,
            value=[8, 18],
            marks={i: f"{i}:00" for i in range(0, 24, 2)}
        ),

        html.Br(),

        html.Label("Select vehicle"),
        dcc.Dropdown(
            id="q2-vehicle-dropdown",
            options=[
                {"label": row["display_name"], "value": row["id"]}
                for _, row in vehicle_options.iterrows()
            ],
            value=vehicle_options.iloc[0]["id"],
            clearable=False
        ),

        html.Br(),

        html.Label("Map layers"),
        dcc.Checklist(
            id="q2-map-layers",
            options=[
                {"label": "Vehicle Path", "value": "path"},
                {"label": "Businesses", "value": "businesses"},
                {"label": "Purchase Hotspots", "value": "hotspots"},
                {"label": "Detected Stops", "value": "stops"}
            ],
            value=["path", "hotspots"],
            inline=True
        ),

        dcc.Graph(id="q2-vehicle-map"),

        html.H3("Credit Card Transactions in Selected Time Window"),
        dash_table.DataTable(
            id="q2-selected-transactions-table",
            columns=[
                {"name": "Timestamp", "id": "timestamp"},
                {"name": "Location", "id": "location"},
                {"name": "Card", "id": "last4ccnum"},
                {"name": "Price", "id": "price"}
            ],
            page_size=10,
            style_table={"overflowX": "auto"}
        )
    ])


@callback(
    Output("q2-vehicle-map", "figure"),
    Input("q2-vehicle-dropdown", "value"),
    Input("q2-date-picker", "date"),
    Input("q2-time-slider", "value"),
    Input("q2-map-layers", "value")
)
def update_vehicle_map(vehicle_id, selected_date, time_range, layers):
    df = gps[gps["id"] == vehicle_id].copy()

    selected_date_obj = pd.to_datetime(selected_date).date()

    df = df[df["timestamp"].dt.date == selected_date_obj]
    df = df[
        (df["timestamp"].dt.hour >= time_range[0]) &
        (df["timestamp"].dt.hour <= time_range[1])
    ]

    return make_vehicle_map(df, layers, selected_date_obj, time_range)


@callback(
    Output("q2-selected-transactions-table", "data"),
    Input("q2-date-picker", "date"),
    Input("q2-time-slider", "value")
)
def update_selected_transactions_table(selected_date, time_range):
    selected_date = pd.to_datetime(selected_date).date()

    filtered = cc[
        (cc["timestamp"].dt.date == selected_date) &
        (cc["timestamp"].dt.hour >= time_range[0]) &
        (cc["timestamp"].dt.hour <= time_range[1])
    ].copy()

    filtered["timestamp"] = filtered["timestamp"].astype(str)

    return filtered[
        ["timestamp", "location", "last4ccnum", "price"]
    ].sort_values("timestamp").to_dict("records")