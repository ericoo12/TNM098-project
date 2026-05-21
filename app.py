import base64
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table

app = Dash(__name__)

# -------------------------
# Settings
# -------------------------
IMAGE_WIDTH = 2740
IMAGE_HEIGHT = 1535
IMAGE_PATH = "assets/map.jpg"

TIME_WINDOW_MINUTES = 20
DISTANCE_THRESHOLD_PIXELS = 150

STOP_DISTANCE_THRESHOLD_PIXELS = 85
MIN_STOP_MINUTES = 10

MAP_LON_MIN, MAP_LON_MAX = 24.8250, 24.9100
MAP_LAT_MIN, MAP_LAT_MAX = 36.0450, 36.0950

WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

# -------------------------
# Load map image
# -------------------------
with open(IMAGE_PATH, "rb") as image_file:
    encoded_map = base64.b64encode(image_file.read()).decode()

# -------------------------
# Load data
# -------------------------
gps = pd.read_csv("data/gps.csv")
cc = pd.read_csv("data/cc_data.csv", encoding="cp1252")
loyalty = pd.read_csv("data/loyalty_data.csv", encoding="cp1252")
location_coords = pd.read_csv("data/location_coords.csv")

for df in [gps, cc, loyalty, location_coords]:
    df.columns = df.columns.str.strip().str.lower()

gps["timestamp"] = pd.to_datetime(gps["timestamp"])
cc["timestamp"] = pd.to_datetime(cc["timestamp"])
loyalty["timestamp"] = pd.to_datetime(loyalty["timestamp"])

for df in [cc, loyalty]:
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.date
    df["weekday"] = df["timestamp"].dt.day_name()

gps["x_pix"] = ((gps["long"] - MAP_LON_MIN) / (MAP_LON_MAX - MAP_LON_MIN)) * IMAGE_WIDTH
gps["y_pix"] = ((gps["lat"] - MAP_LAT_MIN) / (MAP_LAT_MAX - MAP_LAT_MIN)) * IMAGE_HEIGHT

gps = gps.sort_values(["id", "timestamp"])

vehicles = sorted(gps["id"].unique())
locations = sorted(cc["location"].unique())
car_assignments = pd.read_csv("data/car-assignments.csv")

car_assignments.columns = car_assignments.columns.str.strip().str.lower()
car_assignments["name"] = (
    car_assignments["firstname"].astype(str) + " " +
    car_assignments["lastname"].astype(str)
)
car_assignments = car_assignments.rename(columns={"carid": "id"})
car_assignments["id"] = car_assignments["id"].astype("Int64")

gps = gps.merge(
    car_assignments[["id", "name", "currentemploymenttype", "currentemploymenttitle"]],
    on="id",
    how="left"
)

gps["display_name"] = gps.apply(
    lambda row: f"{int(row['id'])} — {row['name']}"
    if pd.notna(row["name"])
    else f"{int(row['id'])} — Unknown driver",
    axis=1
)

vehicle_options = (
    gps[["id", "display_name"]]
    .drop_duplicates()
    .sort_values("id")
)

# -------------------------
# Helper functions
# -------------------------
def load_location_coords():
    df = pd.read_csv("data/location_coords.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df


def get_valid_location_coords():
    coords = load_location_coords()
    return coords[
        (coords["x_pix"] > 10) &
        (coords["y_pix"] > 10)
    ].copy()


def get_location_counts():
    cc_counts = cc["location"].value_counts().reset_index()
    cc_counts.columns = ["location", "credit_count"]

    loyalty_counts = loyalty["location"].value_counts().reset_index()
    loyalty_counts.columns = ["location", "loyalty_count"]

    merged = pd.merge(cc_counts, loyalty_counts, on="location", how="outer").fillna(0)
    merged["difference"] = merged["credit_count"] - merged["loyalty_count"]
    merged["abs_difference"] = merged["difference"].abs()

    return merged.sort_values("credit_count", ascending=False)


def get_purchase_location_summary():
    counts = cc["location"].value_counts().reset_index()
    counts.columns = ["location", "purchase_count"]

    valid_locations = get_valid_location_coords()

    summary = pd.merge(
        valid_locations,
        counts,
        on="location",
        how="left"
    ).fillna({"purchase_count": 0})

    return summary


def detect_vehicle_stops(vehicle_df):
    vehicle_df = vehicle_df.sort_values("timestamp").copy()

    if len(vehicle_df) < 2:
        return pd.DataFrame(columns=[
            "vehicle_id", "start_time", "end_time",
            "duration_minutes", "x_pix", "y_pix"
        ])

    stops = []
    start_idx = 0

    while start_idx < len(vehicle_df) - 1:
        start_row = vehicle_df.iloc[start_idx]
        cluster = [start_row]

        for idx in range(start_idx + 1, len(vehicle_df)):
            row = vehicle_df.iloc[idx]

            distance = (
                (row["x_pix"] - start_row["x_pix"]) ** 2 +
                (row["y_pix"] - start_row["y_pix"]) ** 2
            ) ** 0.5

            if distance <= STOP_DISTANCE_THRESHOLD_PIXELS:
                cluster.append(row)
            else:
                break

        start_time = cluster[0]["timestamp"]
        end_time = cluster[-1]["timestamp"]
        duration_minutes = (end_time - start_time).total_seconds() / 60

        if duration_minutes >= MIN_STOP_MINUTES:
            stops.append({
                "vehicle_id": start_row["id"],
                "start_time": start_time,
                "end_time": end_time,
                "duration_minutes": round(duration_minutes, 1),
                "x_pix": sum(r["x_pix"] for r in cluster) / len(cluster),
                "y_pix": sum(r["y_pix"] for r in cluster) / len(cluster)
            })

        start_idx += max(1, len(cluster))

    return pd.DataFrame(stops)


def get_transactions():
    cc_tx = cc.copy()
    cc_tx["source"] = "credit"
    cc_tx["card_id"] = cc_tx["last4ccnum"].astype(str)

    loyalty_tx = loyalty.copy()
    loyalty_tx["source"] = "loyalty"
    loyalty_tx["card_id"] = loyalty_tx["loyaltynum"].astype(str)

    common_cols = ["timestamp", "location", "source", "card_id"]

    return pd.concat(
        [cc_tx[common_cols], loyalty_tx[common_cols]],
        ignore_index=True
    ).sort_values("timestamp")


def match_transactions_to_vehicles():
    transactions = get_transactions()
    valid_locations = get_valid_location_coords()

    results = []

    for _, tx in transactions.iterrows():
        tx_time = tx["timestamp"]
        tx_location = tx["location"]

        business = valid_locations[
            valid_locations["location"] == tx_location
        ]

        if business.empty:
            results.append({
                "timestamp": tx_time,
                "location": tx_location,
                "source": tx["source"],
                "card_id": tx["card_id"],
                "matched_vehicle": None,
                "min_distance": None,
                "match_status": "Unknown location"
            })
            continue

        business_x = business.iloc[0]["x_pix"]
        business_y = business.iloc[0]["y_pix"]

        time_min = tx_time - pd.Timedelta(minutes=TIME_WINDOW_MINUTES)
        time_max = tx_time + pd.Timedelta(minutes=TIME_WINDOW_MINUTES)

        nearby_gps = gps[
            (gps["timestamp"] >= time_min) &
            (gps["timestamp"] <= time_max)
        ].copy()

        if nearby_gps.empty:
            results.append({
                "timestamp": tx_time,
                "location": tx_location,
                "source": tx["source"],
                "card_id": tx["card_id"],
                "matched_vehicle": None,
                "min_distance": None,
                "match_status": "No GPS nearby"
            })
            continue

        nearby_gps["distance"] = (
            (nearby_gps["x_pix"] - business_x) ** 2 +
            (nearby_gps["y_pix"] - business_y) ** 2
        ) ** 0.5

        best_match = nearby_gps.sort_values("distance").iloc[0]

        if best_match["distance"] <= 80:
            match_status = "Strong match"
        elif best_match["distance"] <= DISTANCE_THRESHOLD_PIXELS:
            match_status = "Possible match"
        else:
            match_status = "No nearby vehicle"

        results.append({
            "timestamp": tx_time,
            "location": tx_location,
            "source": tx["source"],
            "card_id": tx["card_id"],
            "matched_vehicle": best_match["id"],
            "min_distance": round(best_match["distance"], 2),
            "match_status": match_status
        })

    return pd.DataFrame(results)

def get_driver_name(vehicle_id):
    match = car_assignments[car_assignments["id"] == vehicle_id]

    if match.empty:
        return "Unknown driver"

    return match.iloc[0]["name"]


def get_vehicle_card_matches():
    transactions = get_transactions()
    valid_locations = get_valid_location_coords()

    results = []

    for _, tx in transactions.iterrows():
        tx_time = tx["timestamp"]
        tx_location = tx["location"]

        business = valid_locations[valid_locations["location"] == tx_location]

        if business.empty:
            continue

        business_x = business.iloc[0]["x_pix"]
        business_y = business.iloc[0]["y_pix"]

        time_min = tx_time - pd.Timedelta(minutes=TIME_WINDOW_MINUTES)
        time_max = tx_time + pd.Timedelta(minutes=TIME_WINDOW_MINUTES)

        nearby_gps = gps[
            (gps["timestamp"] >= time_min) &
            (gps["timestamp"] <= time_max)
        ].copy()

        if nearby_gps.empty:
            continue

        nearby_gps["distance"] = (
            (nearby_gps["x_pix"] - business_x) ** 2 +
            (nearby_gps["y_pix"] - business_y) ** 2
        ) ** 0.5

        nearby_gps = nearby_gps[
            nearby_gps["distance"] <= DISTANCE_THRESHOLD_PIXELS
        ]

        if nearby_gps.empty:
            continue

        best_per_vehicle = (
            nearby_gps
            .sort_values("distance")
            .groupby("id")
            .first()
            .reset_index()
        )

        for _, vehicle in best_per_vehicle.iterrows():
            results.append({
                "timestamp": tx_time,
                "day": tx_time.date(),
                "location": tx_location,
                "source": tx["source"],
                "card_id": tx["card_id"],
                "vehicle_id": vehicle["id"],
                "driver_name": vehicle["name"],
                "distance": vehicle["distance"]
            })

    return pd.DataFrame(results)


vehicle_card_matches = get_vehicle_card_matches()


def infer_card_owners():
    if vehicle_card_matches.empty:
        return pd.DataFrame()

    grouped = (
        vehicle_card_matches
        .groupby(["source", "card_id", "vehicle_id", "driver_name"])
        .agg(
            match_count=("timestamp", "count"),
            unique_days=("day", "nunique"),
            unique_locations=("location", "nunique"),
            avg_distance=("distance", "mean"),
            min_distance=("distance", "min")
        )
        .reset_index()
    )

    grouped["confidence_score"] = (
        grouped["match_count"] * 2
        + grouped["unique_days"] * 3
        + grouped["unique_locations"] * 2
        - grouped["avg_distance"] / 20
    )

    grouped["confidence_score"] = grouped["confidence_score"].round(2)
    grouped["avg_distance"] = grouped["avg_distance"].round(2)
    grouped["min_distance"] = grouped["min_distance"].round(2)

    grouped = grouped.sort_values(
        ["source", "card_id", "confidence_score"],
        ascending=[True, True, False]
    )

    grouped["rank_for_card"] = (
        grouped
        .groupby(["source", "card_id"])
        .cumcount() + 1
    )

    return grouped


card_owner_scores = infer_card_owners()


def get_best_card_owner_table():
    best = card_owner_scores[card_owner_scores["rank_for_card"] == 1].copy()

    best = best.sort_values("confidence_score", ascending=False)

    return best[[
        "source",
        "card_id",
        "vehicle_id",
        "driver_name",
        "match_count",
        "unique_days",
        "unique_locations",
        "avg_distance",
        "confidence_score"
    ]]

def get_home_candidate_points():
    df = gps.copy()
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    candidates = []

    for (vehicle_id, date), day_df in df.groupby(["id", "date"]):
        day_df = day_df.sort_values("timestamp")

        morning = day_df[
            (day_df["hour"] >= 5) &
            (day_df["hour"] <= 10)
        ]

        evening = day_df[
            (day_df["hour"] >= 17) &
            (day_df["hour"] <= 23)
        ]

        if not morning.empty:
            first = morning.iloc[0]
            candidates.append({
                "id": vehicle_id,
                "name": first["name"] if pd.notna(first["name"]) else "Unknown / company truck",
                "date": date,
                "type": "morning_start",
                "x_pix": first["x_pix"],
                "y_pix": first["y_pix"]
            })

        if not evening.empty:
            last = evening.iloc[-1]
            candidates.append({
                "id": vehicle_id,
                "name": last["name"] if pd.notna(last["name"]) else "Unknown / company truck",
                "date": date,
                "type": "evening_end",
                "x_pix": last["x_pix"],
                "y_pix": last["y_pix"]
            })

    return pd.DataFrame(candidates)


home_candidate_points = get_home_candidate_points()


def get_home_locations():
    home = (
        home_candidate_points
        .groupby(["id", "name"], dropna=False)
        .agg(
            home_x=("x_pix", "median"),
            home_y=("y_pix", "median"),
            evidence_points=("date", "count"),
            days_observed=("date", "nunique")
        )
        .reset_index()
    )

    return home


home_locations = get_home_locations()


def get_sleepover_anomalies():
    candidates = home_candidate_points.merge(
        home_locations[["id", "home_x", "home_y"]],
        on="id",
        how="left"
    )

    candidates["distance_from_home"] = (
        (candidates["x_pix"] - candidates["home_x"]) ** 2 +
        (candidates["y_pix"] - candidates["home_y"]) ** 2
    ) ** 0.5

    anomalies = candidates[candidates["distance_from_home"] > 250].copy()
    anomalies["distance_from_home"] = anomalies["distance_from_home"].round(1)

    return anomalies.sort_values("distance_from_home", ascending=False)


sleepover_anomalies = get_sleepover_anomalies()

sleepover_anomalies = get_sleepover_anomalies()

def make_home_location_map():
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=home_locations["home_x"],
        y=home_locations["home_y"],
        mode="markers+text",
        text=home_locations["name"],
        textposition="top center",
        marker=dict(
            size=12,
            symbol="circle",
            line=dict(width=2)
        ),
        name="Estimated home locations",
        customdata=home_locations[["id", "days_observed", "evidence_points"]],
        hovertemplate=(
            "Driver: %{text}<br>"
            "Vehicle: %{customdata[0]}<br>"
            "Days observed: %{customdata[1]}<br>"
            "Evidence points: %{customdata[2]}"
            "<extra></extra>"
        )
    ))

    fig.add_trace(go.Scatter(
        x=sleepover_anomalies["x_pix"],
        y=sleepover_anomalies["y_pix"],
        mode="markers",
        marker=dict(
            size=14,
            symbol="x",
            line=dict(width=3)
        ),
        name="Possible sleepover / away overnight",
        customdata=sleepover_anomalies[[
            "name",
            "id",
            "date",
            "distance_from_home"
        ]],
        hovertemplate=(
            "Driver: %{customdata[0]}<br>"
            "Vehicle: %{customdata[1]}<br>"
            "Date: %{customdata[2]}<br>"
            "Distance from usual overnight location: %{customdata[3]} px"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title="Estimated Home Locations and Overnight Anomalies",
        images=[dict(
            source="data:image/jpeg;base64," + encoded_map,
            xref="x",
            yref="y",
            x=0,
            y=IMAGE_HEIGHT,
            sizex=IMAGE_WIDTH,
            sizey=IMAGE_HEIGHT,
            sizing="stretch",
            opacity=1,
            layer="below"
        )],
        xaxis=dict(
            range=[0, IMAGE_WIDTH],
            visible=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[0, IMAGE_HEIGHT],
            visible=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="x"
        ),
        height=850,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig


def make_sleepover_table():
    table = sleepover_anomalies.copy()
    table["date"] = table["date"].astype(str)

    return dash_table.DataTable(
        columns=[
            {"name": "Vehicle", "id": "id"},
            {"name": "Driver", "id": "name"},
            {"name": "Date", "id": "date"},
            {"name": "Distance from Home", "id": "distance_from_home"},
            {"name": "Type", "id": "type"}
        ],
        data=table[[
            "id",
            "name",
            "date",
            "distance_from_home",
            "type"
        ]].to_dict("records"),
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"}
    )

# -------------------------
# Figures
# -------------------------
def make_vehicle_map(df, layers):
    fig = go.Figure()

    purchase_summary = get_purchase_location_summary()
    business_locations = get_valid_location_coords()
    stops = detect_vehicle_stops(df)

    if "path" in layers:
        fig.add_trace(go.Scattergl(
            x=df["x_pix"],
            y=df["y_pix"],
            mode="lines+markers",
            marker=dict(size=3),
            line=dict(width=1),
            name="Vehicle path",
            customdata=df[["timestamp", "lat", "long"]],
            hovertemplate=(
                "Time: %{customdata[0]}<br>"
                "Lat: %{customdata[1]:.6f}<br>"
                "Long: %{customdata[2]:.6f}"
                "<extra></extra>"
            )
        ))

    if "businesses" in layers:
        fig.add_trace(go.Scatter(
            x=business_locations["x_pix"],
            y=business_locations["y_pix"],
            mode="markers",
            marker=dict(
                size=9,
                symbol="x",
                line=dict(width=2)
            ),
            name="Businesses",
            text=business_locations["location"],
            hovertemplate="Location: %{text}<extra></extra>"
        ))

    if "hotspots" in layers:
        fig.add_trace(go.Scatter(
            x=purchase_summary["x_pix"],
            y=purchase_summary["y_pix"],
            mode="markers",
            text=purchase_summary["location"],
            marker=dict(
                size=6 + purchase_summary["purchase_count"] / 30,
                color=purchase_summary["purchase_count"],
                colorscale="YlOrRd",
                showscale=True,
                colorbar=dict(title="Purchases"),
                opacity=0.75,
                line=dict(width=1, color="black")
            ),
            name="Purchase hotspots",
            customdata=purchase_summary[["purchase_count"]],
            hovertemplate=(
                "Location: %{text}<br>"
                "Purchases: %{customdata[0]}"
                "<extra></extra>"
            )
        ))

    if "stops" in layers and not stops.empty:
        fig.add_trace(go.Scatter(
            x=stops["x_pix"],
            y=stops["y_pix"],
            mode="markers",
            marker=dict(
                size=10 + stops["duration_minutes"] / 3,
                symbol="circle-open",
                color="cyan",
                line=dict(width=3),
                opacity=0.9
            ),
            name="Detected stops",
            customdata=stops[[
                "vehicle_id",
                "start_time",
                "end_time",
                "duration_minutes"
            ]],
            hovertemplate=(
                "Vehicle: %{customdata[0]}<br>"
                "Start: %{customdata[1]}<br>"
                "End: %{customdata[2]}<br>"
                "Duration: %{customdata[3]} min"
                "<extra></extra>"
            )
        ))

    fig.update_layout(
        #title="Vehicle Movement Map",
        images=[dict(
            source="data:image/jpeg;base64," + encoded_map,
            xref="x",
            yref="y",
            x=0,
            y=IMAGE_HEIGHT,
            sizex=IMAGE_WIDTH,
            sizey=IMAGE_HEIGHT,
            sizing="stretch",
            opacity=1,
            layer="below"
        )],
        xaxis=dict(
            range=[0, IMAGE_WIDTH],
            visible=True,
            showgrid=True,
            zeroline=False,
            title="x_pix"
        ),
        yaxis=dict(
            range=[0, IMAGE_HEIGHT],
            visible=True,
            showgrid=True,
            zeroline=False,
            scaleanchor="x",
            title="y_pix"
        ),
        height=800,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    return fig


def make_cc_vs_loyalty_locations():
    counts = get_location_counts()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=counts["location"],
        y=counts["credit_count"],
        name="Credit Card"
    ))

    fig.add_trace(go.Bar(
        x=counts["location"],
        y=counts["loyalty_count"],
        name="Loyalty Card"
    ))

    fig.update_layout(
        title="Credit vs Loyalty Purchases by Location",
        xaxis_title="Location",
        yaxis_title="Number of Purchases",
        barmode="group",
        height=500
    )

    return fig


def make_anomaly_chart():
    anomalies = get_location_counts().sort_values("abs_difference", ascending=False)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=anomalies["location"],
        y=anomalies["difference"],
        name="Credit - Loyalty Difference"
    ))

    fig.update_layout(
        title="Credit and Loyalty Card Discrepancies by Location",
        xaxis_title="Location",
        yaxis_title="Credit Count - Loyalty Count",
        height=500
    )

    return fig


def make_anomaly_table():
    anomalies = get_location_counts().sort_values("abs_difference", ascending=False)

    return html.Table([
        html.Thead(html.Tr([
            html.Th("Location"),
            html.Th("Credit"),
            html.Th("Loyalty"),
            html.Th("Difference")
        ])),
        html.Tbody([
            html.Tr([
                html.Td(row["location"]),
                html.Td(int(row["credit_count"])),
                html.Td(int(row["loyalty_count"])),
                html.Td(int(row["difference"]))
            ])
            for _, row in anomalies.head(10).iterrows()
        ])
    ])


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


def make_day_hour_heatmap():
    heat = cc.groupby(["weekday", "hour"]).size().reset_index(name="count")

    pivot = heat.pivot(
        index="weekday",
        columns="hour",
        values="count"
    ).fillna(0)

    pivot = pivot.reindex(WEEKDAY_ORDER)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="YlOrRd"
    ))

    fig.update_layout(
        title="Purchases by Weekday and Hour",
        xaxis_title="Hour",
        yaxis_title="Weekday",
        height=450
    )

    return fig


def make_weekday_activity():
    counts = (
        cc["weekday"]
        .value_counts()
        .reindex(WEEKDAY_ORDER)
        .reset_index()
    )

    counts.columns = ["weekday", "count"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=counts["weekday"],
        y=counts["count"]
    ))

    fig.update_layout(
        title="Credit Card Purchases by Weekday",
        xaxis_title="Weekday",
        yaxis_title="Number of Purchases",
        height=400
    )

    return fig


def make_location_hour_heatmap():
    heat = cc.groupby(["location", "hour"]).size().reset_index(name="count")

    pivot = heat.pivot(
        index="location",
        columns="hour",
        values="count"
    ).fillna(0)

    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False).drop(columns="total")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="YlOrRd"
    ))

    fig.update_layout(
        title="Purchases by Location and Hour",
        xaxis_title="Hour of Day",
        yaxis_title="Location",
        height=800
    )

    return fig


def make_location_weekday_heatmap():
    top_locations = cc["location"].value_counts().head(15).index
    filtered = cc[cc["location"].isin(top_locations)]

    heat = filtered.groupby(["location", "weekday"]).size().reset_index(name="count")

    pivot = heat.pivot(
        index="location",
        columns="weekday",
        values="count"
    ).fillna(0)

    pivot = pivot.reindex(columns=WEEKDAY_ORDER)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=False).drop(columns="total")

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="YlOrRd"
    ))

    fig.update_layout(
        title="Purchases by Location and Weekday — Top 15 Locations",
        xaxis_title="Weekday",
        yaxis_title="Location",
        height=650
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
        title=f"Purchases Over Time: {location}",
        xaxis_title="Time",
        yaxis_title="Purchases",
        height=400
    )

    return fig

def make_card_owner_heatmap(source_type="credit"):
    df = card_owner_scores[card_owner_scores["source"] == source_type].copy()

    if df.empty:
        return go.Figure()

    pivot = df.pivot_table(
        index="driver_name",
        columns="card_id",
        values="confidence_score",
        aggfunc="max",
        fill_value=0
    )

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="YlOrRd"
    ))

    fig.update_layout(
        title=f"{source_type.title()} Card Ownership Evidence",
        xaxis_title="Card ID",
        yaxis_title="Likely Driver",
        height=900
    )

    return fig


def make_best_card_owner_table():
    table = get_best_card_owner_table()

    return dash_table.DataTable(
        columns=[
            {"name": "Source", "id": "source"},
            {"name": "Card ID", "id": "card_id"},
            {"name": "Vehicle", "id": "vehicle_id"},
            {"name": "Driver", "id": "driver_name"},
            {"name": "Matches", "id": "match_count"},
            {"name": "Days", "id": "unique_days"},
            {"name": "Locations", "id": "unique_locations"},
            {"name": "Avg Distance", "id": "avg_distance"},
            {"name": "Confidence", "id": "confidence_score"}
        ],
        data=table.to_dict("records"),
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"}
    )

# -------------------------
# Precompute matching
# -------------------------
matched_transactions = match_transactions_to_vehicles()


def make_match_status_chart():
    counts = matched_transactions["match_status"].value_counts().reset_index()
    counts.columns = ["match_status", "count"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=counts["match_status"],
        y=counts["count"]
    ))

    fig.update_layout(
        title="Transaction-to-Vehicle Match Status",
        xaxis_title="Match Status",
        yaxis_title="Number of Transactions",
        height=400
    )

    return fig


def make_unmatched_location_chart():
    unmatched = matched_transactions[
        matched_transactions["match_status"] != "Strong match"
    ]

    counts = unmatched["location"].value_counts().head(15).reset_index()
    counts.columns = ["location", "count"]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=counts["location"],
        y=counts["count"]
    ))

    fig.update_layout(
        title="Locations with Most Weak/Unmatched Transactions",
        xaxis_title="Location",
        yaxis_title="Transactions",
        height=500
    )

    return fig


# -------------------------
# Page layouts
# -------------------------
def question1_layout():
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
            id="location-dropdown",
            options=[{"label": loc, "value": loc} for loc in locations],
            value=locations[0],
            clearable=False
        ),

        dcc.Graph(id="location-timeline")
    ])


def question2_layout():
    return html.Div([
        html.H2("Question 2 — Vehicle, Credit, and Loyalty Analysis"),

        dcc.Graph(figure=make_match_status_chart()),
        dcc.Graph(figure=make_unmatched_location_chart()),

        html.H3("Vehicle Map"),

        html.Label("Select date"),
        dcc.DatePickerSingle(
            id="date-picker",
            min_date_allowed=gps["timestamp"].min().date(),
            max_date_allowed=gps["timestamp"].max().date(),
            initial_visible_month=gps["timestamp"].min().date(),
            date=gps["timestamp"].min().date()
        ),

        html.Br(),
        html.Br(),

        html.Label("Select time range"),
        dcc.RangeSlider(
            id="time-slider",
            min=0,
            max=23,
            step=1,
            value=[8, 18],
            marks={i: f"{i}:00" for i in range(0, 24, 2)}
        ),

        html.Br(),

        html.Label("Select vehicle"),
        dcc.Dropdown(
            id="vehicle-dropdown",
            options=[{"label": str(v), "value": v} for v in vehicles],
            value=vehicles[0],
            clearable=False
        ),

        html.Br(),

        html.Label("Map layers"),
        dcc.Checklist(
            id="map-layers",
            options=[
                {"label": "Vehicle Path", "value": "path"},
                {"label": "Businesses", "value": "businesses"},
                {"label": "Purchase Hotspots", "value": "hotspots"},
                {"label": "Detected Stops", "value": "stops"}
            ],
            value=["path", "hotspots"],
            inline=True
        ),

        dcc.Graph(id="vehicle-map"),

        html.H3("Credit Card Transactions in Selected Time Window"),
        dash_table.DataTable(
            id="selected-transactions-table",
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

def question3_layout():
    return html.Div([
        html.H2("Question 3 — Card Ownership Inference"),


        html.H3("Best Inferred Owner for Each Card"),
        make_best_card_owner_table(),

        html.H3("Credit Card Ownership Heatmap"),
        dcc.Graph(figure=make_card_owner_heatmap("credit")),

        html.H3("Loyalty Card Ownership Heatmap"),
        dcc.Graph(figure=make_card_owner_heatmap("loyalty")),

        html.H3("Estimated Home Locations and Possible Sleepovers"),
        html.P(
            "Home locations are estimated from where vehicles remain overnight. "
            "Large deviations from usual overnight locations may indicate sleepovers, "
            "shared residences, unusual travel, or noisy GPS behavior."
        ),
        dcc.Graph(figure=make_home_location_map()),

        html.H3("Overnight Anomalies"),
        make_sleepover_table()
    ])

def placeholder_layout(question_number, title):
    return html.Div([
        html.H2(f"Question {question_number} — {title}")
    ])


# -------------------------
# Layout
# -------------------------
app.layout = html.Div([
    html.H1("GAStech Visual Analytics Dashboard"),

    dcc.Tabs([
        dcc.Tab(label="Question 1", children=question1_layout()),
        dcc.Tab(label="Question 2", children=question2_layout()),
        dcc.Tab(label="Question 3", children=question3_layout()),
        dcc.Tab(label="Question 4", children=placeholder_layout(4, "Informal Relationships")),
        dcc.Tab(label="Question 5", children=placeholder_layout(5, "Suspicious Activity"))
    ])
])


# -------------------------
# Callbacks
# -------------------------
@app.callback(
    Output("vehicle-map", "figure"),
    Input("vehicle-dropdown", "value"),
    Input("date-picker", "date"),
    Input("time-slider", "value"),
    Input("map-layers", "value")
)
def update_vehicle_map(vehicle_id, selected_date, time_range, layers):
    df = gps[gps["id"] == vehicle_id].copy()

    selected_date = pd.to_datetime(selected_date).date()

    df = df[df["timestamp"].dt.date == selected_date]
    df = df[
        (df["timestamp"].dt.hour >= time_range[0]) &
        (df["timestamp"].dt.hour <= time_range[1])
    ]

    return make_vehicle_map(df, layers)


@app.callback(
    Output("location-timeline", "figure"),
    Input("location-dropdown", "value")
)
def update_location_timeline(location):
    return make_location_timeline(location)


@app.callback(
    Output("selected-transactions-table", "data"),
    Input("date-picker", "date"),
    Input("time-slider", "value")
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


# -------------------------
# Run app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)