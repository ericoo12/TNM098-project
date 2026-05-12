import base64
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

# -------------------------
# Settings
# -------------------------
IMAGE_WIDTH = 2740
IMAGE_HEIGHT = 1535
image_path = "assets/map.jpg"

vehicle_col = "id"

MAP_LON_MIN, MAP_LON_MAX = 24.8250, 24.9100
MAP_LAT_MIN, MAP_LAT_MAX = 36.0450, 36.0950

# -------------------------
# Load image
# -------------------------
with open(image_path, "rb") as image_file:
    encoded = base64.b64encode(image_file.read()).decode()

# -------------------------
# Load GPS data
# -------------------------
gps = pd.read_csv("data/gps.csv")
gps.columns = gps.columns.str.strip().str.lower()

gps["timestamp"] = pd.to_datetime(gps["timestamp"])

gps["x_pix"] = (
    (gps["long"] - MAP_LON_MIN) /
    (MAP_LON_MAX - MAP_LON_MIN)
) * IMAGE_WIDTH

gps["y_pix"] = (
    (gps["lat"] - MAP_LAT_MIN) /
    (MAP_LAT_MAX - MAP_LAT_MIN)
) * IMAGE_HEIGHT

gps = gps.sort_values([vehicle_col, "timestamp"])
vehicles = sorted(gps[vehicle_col].unique())

# -------------------------
# Load transaction data
# -------------------------
cc = pd.read_csv("data/cc_data.csv", encoding="cp1252")
loyalty = pd.read_csv("data/loyalty_data.csv", encoding="cp1252")

location_coords = pd.read_csv("data/location_coords.csv")

cc.columns = cc.columns.str.strip().str.lower()
loyalty.columns = loyalty.columns.str.strip().str.lower()

cc["timestamp"] = pd.to_datetime(cc["timestamp"])
loyalty["timestamp"] = pd.to_datetime(loyalty["timestamp"])

cc["hour"] = cc["timestamp"].dt.hour
cc["day"] = cc["timestamp"].dt.date

loyalty["hour"] = loyalty["timestamp"].dt.hour
loyalty["day"] = loyalty["timestamp"].dt.date


locations = sorted(cc["location"].unique())

# -------------------------
# Figures
# -------------------------
def make_map(df):
    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=df["x_pix"],
        y=df["y_pix"],
        mode="lines+markers",
        marker=dict(size=3),
        line=dict(width=1),
        customdata=df[["timestamp", "lat", "long"]],
        hovertemplate=
            "Time: %{customdata[0]}<br>" +
            "Lat: %{customdata[1]:.6f}<br>" +
            "Long: %{customdata[2]:.6f}" +
            "<extra></extra>"
    ))

    fig.update_layout(
        images=[
            dict(
                source="data:image/jpeg;base64," + encoded,
                xref="x",
                yref="y",
                x=0,
                y=IMAGE_HEIGHT,
                sizex=IMAGE_WIDTH,
                sizey=IMAGE_HEIGHT,
                sizing="stretch",
                opacity=1,
                layer="below"
            )
        ],
        xaxis=dict(range=[0, IMAGE_WIDTH], visible=False),
        yaxis=dict(range=[0, IMAGE_HEIGHT], visible=False, scaleanchor="x"),
        height=800,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    fig.add_trace(go.Scatter(
        x=location_coords["x_pix"],
        y=location_coords["y_pix"],
        mode="markers+text",
        text=location_coords["location"],
        textposition="top center",
        marker=dict(size=10, symbol="x"),
        name="Businesses",
        hovertemplate=
        "Location: %{text}<extra></extra>"
    ))

    return fig


def make_popular_locations():
    counts = cc["location"].value_counts().reset_index()
    counts.columns = ["location", "count"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=counts["location"],
        y=counts["count"]
    ))

    fig.update_layout(
        title="Most Popular Credit Card Locations",
        xaxis_title="Location",
        yaxis_title="Number of Purchases",
        height=500
    )

    return fig


def make_purchase_activity_by_hour():
    hourly = cc.groupby("hour").size().reset_index(name="count")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hourly["hour"],
        y=hourly["count"],
        mode="lines+markers"
    ))

    fig.update_layout(
        title="Credit Card Purchases by Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Number of Purchases",
        height=400
    )

    return fig


def make_loyalty_locations():
    counts = loyalty["location"].value_counts().reset_index()
    counts.columns = ["location", "count"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=counts["location"],
        y=counts["count"]
    ))

    fig.update_layout(
        title="Most Popular Loyalty Card Locations",
        xaxis_title="Location",
        yaxis_title="Number of Purchases",
        height=500
    )

    return fig

def make_location_timeline(location):
    df = cc[cc["location"] == location].copy()
    df["date_hour"] = df["timestamp"].dt.floor("h")

    counts = df.groupby("date_hour").size().reset_index(name="count")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=counts["date_hour"],
        y=counts["count"],
        mode="lines+markers"
    ))

    fig.update_layout(
        title=f"Purchases over Time: {location}",
        xaxis_title="Time",
        yaxis_title="Purchases",
        height=400
    )

    return fig
# -------------------------
# Layout
# -------------------------
app.layout = html.Div([
    html.H1("GAStech Visual Analytics Dashboard"),
html.Label("Select Date"),
    dcc.DatePickerSingle(
        id='date-picker',
        min_date_allowed=gps['timestamp'].min().date(),
        max_date_allowed=gps['timestamp'].max().date(),
        initial_visible_month=gps['timestamp'].min().date(),
        date=gps['timestamp'].min().date()  # Default to the first day
    ),
html.Label("Select Time Range (Hours)"),
    dcc.RangeSlider(
        id='time-slider',
        min=0, max=23, step=1,
        value=[8, 18], # Default to 8am - 6pm
        marks={i: f'{i}:00' for i in range(0, 25, 2)}
    ),
    dcc.Tabs([
        dcc.Tab(label="Vehicle Map", children=[
            html.Br(),

            html.Label("Select vehicle"),
            dcc.Dropdown(
                id="vehicle-dropdown",
                options=[
                    {"label": str(v), "value": v}
                    for v in vehicles
                ],
                value=vehicles[0],
                clearable=False
            ),

            dcc.Graph(id="vehicle-map")
        ]),

        dcc.Tab(label="Transactions", children=[
            html.Br(),

            dcc.Graph(
                id="popular-credit-locations",
                figure=make_popular_locations()
            ),

            dcc.Graph(
                id="purchase-activity-hour",
                figure=make_purchase_activity_by_hour()
            ),

            dcc.Graph(
                id="popular-loyalty-locations",
                figure=make_loyalty_locations()
            ),

            html.Label("Select location"),
            dcc.Dropdown(
                id="location-dropdown",
                options=[
                  {"label": loc, "value": loc}
                  for loc in locations
                 ],
                value=locations[0],
                clearable=False
            ),


dcc.Graph(id="location-timeline"),
        ])
    ])
])

# -------------------------
# Callback
# -------------------------
@app.callback(
    Output("vehicle-map", "figure"),
    [Input("vehicle-dropdown", "value"),
     Input("date-picker", "date"),
     Input("time-slider", "value")]
)
def update_vehicle_map(vehicle_id, selected_date, time_range):
    df = gps[gps[vehicle_col] == vehicle_id].copy()

    # Filter Date
    df = df[df['timestamp'].dt.date == pd.to_datetime(selected_date).date()]

    # Filter Time Range
    df = df[(df['timestamp'].dt.hour >= time_range[0]) &
            (df['timestamp'].dt.hour <= time_range[1])]

    return make_map(df)

def update_location_timeline(location):
    return make_location_timeline(location)


# Run app
if __name__ == "__main__":
    app.run(debug=False)