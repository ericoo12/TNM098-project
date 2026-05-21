import pandas as pd

from data_loader import (
    gps, cc, loyalty, car_assignments,
    TIME_WINDOW_MINUTES, DISTANCE_THRESHOLD_PIXELS,
    STOP_DISTANCE_THRESHOLD_PIXELS, MIN_STOP_MINUTES
)

# --------------------------------------------------
# Location coordinate helpers
# --------------------------------------------------

def load_location_coords():
    """
    Reloads location_coords.csv from disk.

    This is useful because business coordinates are manually refined during analysis.
    By loading from disk, the dashboard can pick up coordinate edits without changing
    the code.
    """
    df = pd.read_csv("data/location_coords.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df


def get_valid_location_coords():
    """
    Returns only usable business coordinates.

    Some unknown locations are stored as placeholders like x=1, y=1.
    Those should not be used for plotting, distance matching, or hotspot analysis.
    """
    coords = load_location_coords()

    return coords[
        (coords["x_pix"] > 10) &
        (coords["y_pix"] > 10)
    ].copy()


# --------------------------------------------------
# Question 1 helpers: transaction summaries
# --------------------------------------------------

def get_location_counts():
    """
    Compares credit card and loyalty card purchase counts per location.

    Used for:
    - most popular locations,
    - credit vs loyalty discrepancy chart,
    - anomaly table.
    """
    cc_counts = cc["location"].value_counts().reset_index()
    cc_counts.columns = ["location", "credit_count"]

    loyalty_counts = loyalty["location"].value_counts().reset_index()
    loyalty_counts.columns = ["location", "loyalty_count"]

    merged = pd.merge(
        cc_counts,
        loyalty_counts,
        on="location",
        how="outer"
    ).fillna(0)

    merged["difference"] = merged["credit_count"] - merged["loyalty_count"]
    merged["abs_difference"] = merged["difference"].abs()

    return merged.sort_values("credit_count", ascending=False)


def get_purchase_location_summary(filtered_cc=None):
    """
    Counts purchases per mapped location.

    If filtered_cc is provided, only those transactions are counted.
    This is used on the vehicle map so purchase hotspots can update when
    the user changes the selected day or time range.
    """
    if filtered_cc is None:
        filtered_cc = cc

    counts = filtered_cc["location"].value_counts().reset_index()
    counts.columns = ["location", "purchase_count"]

    valid_locations = get_valid_location_coords()

    summary = pd.merge(
        valid_locations,
        counts,
        on="location",
        how="left"
    ).fillna({"purchase_count": 0})

    return summary


# --------------------------------------------------
# Vehicle stop detection
# --------------------------------------------------

def detect_vehicle_stops(vehicle_df):
    """
    Detects simple vehicle stops.

    A stop is defined as a sequence of GPS points where the vehicle remains
    within STOP_DISTANCE_THRESHOLD_PIXELS for at least MIN_STOP_MINUTES.

    Output columns:
    - vehicle_id
    - start_time
    - end_time
    - duration_minutes
    - x_pix
    - y_pix

    These stop points are useful for identifying:
    - likely business visits,
    - home/work locations,
    - suspicious meetings,
    - vehicle presence near transaction locations.
    """
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

        # Build a cluster of consecutive points that stay close to the start point.
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

        # Keep only stops longer than the minimum duration.
        if duration_minutes >= MIN_STOP_MINUTES:
            stops.append({
                "vehicle_id": start_row["id"],
                "start_time": start_time,
                "end_time": end_time,
                "duration_minutes": round(duration_minutes, 1),
                "x_pix": sum(r["x_pix"] for r in cluster) / len(cluster),
                "y_pix": sum(r["y_pix"] for r in cluster) / len(cluster)
            })

        # Move past this cluster.
        start_idx += max(1, len(cluster))

    return pd.DataFrame(stops)


# --------------------------------------------------
# Unified transaction table
# --------------------------------------------------

def get_transactions():
    """
    Combines credit card and loyalty card transactions into one standard format.

    Output columns:
    - timestamp
    - location
    - source: 'credit' or 'loyalty'
    - card_id
    """
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


# --------------------------------------------------
# Question 2: transaction-to-vehicle matching
# --------------------------------------------------

def match_transactions_to_vehicles():
    """
    Matches each transaction to the nearest vehicle GPS point in time and space.

    For each transaction:
    1. Find the business coordinate from location_coords.csv.
    2. Find vehicle GPS points within TIME_WINDOW_MINUTES.
    3. Compute distance from GPS point to business.
    4. Classify match quality.

    Match status:
    - Strong match: very close vehicle evidence
    - Possible match: plausible but less precise
    - No nearby vehicle: GPS exists, but no vehicle is close enough
    - No GPS nearby: no GPS data in the time window
    - Unknown location: business has no usable coordinates
    """
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


# --------------------------------------------------
# Question 3: card ownership inference
# --------------------------------------------------

def get_vehicle_card_matches():
    """
    Produces repeated evidence linking vehicles/drivers to credit or loyalty cards.

    Unlike match_transactions_to_vehicles(), this keeps every vehicle close enough
    to a transaction, not only the single closest one.

    This is useful for estimating card ownership because repeated close matches
    across days and locations suggest a relationship between:
    - a vehicle,
    - a driver,
    - a card.
    """
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


# Precomputed table used by Question 3.
vehicle_card_matches = get_vehicle_card_matches()


def infer_card_owners():
    """
    Scores likely ownership between cards and vehicle drivers.

    The score rewards:
    - many matches,
    - matches across multiple days,
    - matches across multiple locations,
    - shorter average distance.

    This does not prove ownership; it provides ranked evidence.
    """
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
    """
    Returns only the top-ranked inferred driver for each card.
    """
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


# --------------------------------------------------
# Home / overnight behavior inference
# --------------------------------------------------

def get_overnight_locations():
    """
    Estimates nightly vehicle locations from GPS points between 22:00 and 05:00.

    These are not guaranteed to be homes, but repeated overnight locations
    are useful evidence for home-location inference.
    """
    night_gps = gps[
        (gps["timestamp"].dt.hour >= 22) |
        (gps["timestamp"].dt.hour <= 5)
    ].copy()

    overnight = (
        night_gps
        .groupby(["id", "name", night_gps["timestamp"].dt.date])
        .agg(
            x_pix=("x_pix", "mean"),
            y_pix=("y_pix", "mean"),
            point_count=("timestamp", "count")
        )
        .reset_index()
        .rename(columns={"timestamp": "date"})
    )

    return overnight


overnight_locations = get_overnight_locations()


def get_home_locations():
    """
    Estimates each vehicle's usual overnight location.

    The median x/y coordinate is used to reduce the effect of noisy GPS points
    or occasional unusual nights.
    """
    return (
        overnight_locations
        .groupby(["id", "name"])
        .agg(
            home_x=("x_pix", "median"),
            home_y=("y_pix", "median"),
            nights_observed=("date", "nunique")
        )
        .reset_index()
    )


home_locations = get_home_locations()


def get_sleepover_anomalies():
    """
    Finds nights where a vehicle is far from its usual overnight location.

    These may indicate:
    - sleepovers,
    - travel,
    - shared residences,
    - GPS noise,
    - suspicious overnight activity.
    """
    overnight = overnight_locations.merge(
        home_locations[["id", "home_x", "home_y"]],
        on="id",
        how="left"
    )

    overnight["distance_from_home"] = (
        (overnight["x_pix"] - overnight["home_x"]) ** 2 +
        (overnight["y_pix"] - overnight["home_y"]) ** 2
    ) ** 0.5

    anomalies = overnight[overnight["distance_from_home"] > 250].copy()
    anomalies["distance_from_home"] = anomalies["distance_from_home"].round(1)

    return anomalies.sort_values("distance_from_home", ascending=False)


sleepover_anomalies = get_sleepover_anomalies()

# Precomputed table used by Question 2.
matched_transactions = match_transactions_to_vehicles()