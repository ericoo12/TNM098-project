import plotly.graph_objects as go
from dash import html, dash_table

from data_loader import (
    cc, encoded_map,
    IMAGE_WIDTH, IMAGE_HEIGHT, WEEKDAY_ORDER
)

from helpers import (
    get_location_counts,
    get_purchase_location_summary,
    get_valid_location_coords,
    detect_vehicle_stops,
    matched_transactions,
    card_owner_scores,
    get_best_card_owner_table,
    home_locations,
    sleepover_anomalies
)

# --------------------------------------------------
# Shared map figure
# --------------------------------------------------

def make_vehicle_map(df, layers, selected_date=None, time_range=None):
    """
    Creates the interactive vehicle map used in Question 2.

    Layers can include:
    - vehicle path,
    - inferred business locations,
    - purchase hotspots,
    - detected vehicle stops.

    If selected_date and time_range are provided, purchase hotspots are filtered
    to the same selected time window as the vehicle path.
    """
    fig = go.Figure()

    # Default: use all credit card purchases.
    filtered_cc = cc

    # Optional: filter purchases to selected date and hour range.
    if selected_date is not None and time_range is not None:
        filtered_cc = cc[
            (cc["timestamp"].dt.date == selected_date) &
            (cc["timestamp"].dt.hour >= time_range[0]) &
            (cc["timestamp"].dt.hour <= time_range[1])
        ].copy()

    # Purchase summary joins transaction counts with inferred location coordinates.
    purchase_summary = get_purchase_location_summary(filtered_cc)

    # Hide locations with zero purchases in the selected time window.
    purchase_summary = purchase_summary[
        purchase_summary["purchase_count"] > 0
    ]

    # Business location markers.
    business_locations = get_valid_location_coords()

    # Stop markers for the selected vehicle/time window.
    stops = detect_vehicle_stops(df)

    # Vehicle movement path.
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

    # Static business locations from location_coords.csv.
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

    # Dynamic purchase hotspots filtered by selected time window.
    if "hotspots" in layers:
        fig.add_trace(go.Scatter(
            x=purchase_summary["x_pix"],
            y=purchase_summary["y_pix"],
            mode="markers",
            text=purchase_summary["location"],
            marker=dict(
                size=8 + purchase_summary["purchase_count"] * 2,
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
                "Purchases in selected window: %{customdata[0]}"
                "<extra></extra>"
            )
        ))

    # Detected stops for the selected vehicle/time window.
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

    # Add JPG map image as background.
    fig.update_layout(
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


# --------------------------------------------------
# Question 1 figures
# --------------------------------------------------

def make_cc_vs_loyalty_locations():
    """
    Bar chart comparing purchase counts from credit cards and loyalty cards.
    Useful for finding mismatches between the two transaction sources.
    """
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
    """
    Shows credit-minus-loyalty discrepancies per location.

    Positive values mean more credit transactions than loyalty transactions.
    Negative values mean more loyalty transactions than credit transactions.
    """
    anomalies = get_location_counts().sort_values(
        "abs_difference",
        ascending=False
    )

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
    """
    Table of the top 10 largest credit/loyalty count mismatches.
    """
    anomalies = get_location_counts().sort_values(
        "abs_difference",
        ascending=False
    )

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
    """
    Line chart of all credit card purchases grouped by hour of day.
    Used to identify morning, lunch, evening, or late-night activity.
    """
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
    """
    Heatmap showing which weekday/hour combinations contain the most purchases.
    Useful for spotting time-based anomalies.
    """
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
    """
    Bar chart of total purchases per weekday.
    """
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
    """
    Heatmap of purchase activity by location and hour.
    Shows when each business is popular.
    """
    heat = cc.groupby(["location", "hour"]).size().reset_index(name="count")

    pivot = heat.pivot(
        index="location",
        columns="hour",
        values="count"
    ).fillna(0)

    # Sort locations by total transaction volume.
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
    """
    Heatmap of top 15 purchase locations across weekdays.
    Useful for identifying weekday/weekend differences.
    """
    top_locations = cc["location"].value_counts().head(15).index
    filtered = cc[cc["location"].isin(top_locations)]

    heat = (
        filtered
        .groupby(["location", "weekday"])
        .size()
        .reset_index(name="count")
    )

    pivot = heat.pivot(
        index="location",
        columns="weekday",
        values="count"
    ).fillna(0)

    pivot = pivot.reindex(columns=WEEKDAY_ORDER)

    # Sort by total activity.
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
    """
    Time series of purchases for one selected location.
    Used to inspect specific businesses in more detail.
    """
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


# --------------------------------------------------
# Question 2 figures
# --------------------------------------------------

def make_match_status_chart():
    """
    Shows how many transactions have strong, possible, weak, or missing vehicle evidence.
    """
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
    """
    Shows which locations have the most transactions without strong vehicle support.
    """
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


# --------------------------------------------------
# Question 3 figures
# --------------------------------------------------

def make_card_owner_heatmap(source_type="credit"):
    """
    Heatmap showing evidence linking cards to likely vehicle drivers.

    Darker/higher values mean stronger inferred ownership evidence.
    """
    df = card_owner_scores[
        card_owner_scores["source"] == source_type
    ].copy()

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
    """
    Table showing the top inferred owner for each card.
    """
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


def make_home_location_map():
    """
    Map of estimated home locations and unusual overnight locations.

    Home is estimated from repeated overnight GPS positions.
    """
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
        customdata=home_locations[["id", "nights_observed"]],
        hovertemplate=(
            "Driver: %{text}<br>"
            "Vehicle: %{customdata[0]}<br>"
            "Nights observed: %{customdata[1]}"
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
    """
    Table of overnight locations far from the vehicle's usual overnight position.
    """
    table = sleepover_anomalies.copy()
    table["date"] = table["date"].astype(str)

    return dash_table.DataTable(
        columns=[
            {"name": "Vehicle", "id": "id"},
            {"name": "Driver", "id": "name"},
            {"name": "Date", "id": "date"},
            {"name": "Distance from Home", "id": "distance_from_home"},
            {"name": "GPS Points", "id": "point_count"}
        ],
        data=table[[
            "id",
            "name",
            "date",
            "distance_from_home",
            "point_count"
        ]].to_dict("records"),
        page_size=15,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"}
    )