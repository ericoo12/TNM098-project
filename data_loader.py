import base64
import pandas as pd

# --------------------------------------------------
# Global constants
# --------------------------------------------------
# These values define how the JPG map image relates to the GPS coordinate system.
# The GPS data uses longitude/latitude, while the dashboard plots on image pixels.
IMAGE_WIDTH = 2740
IMAGE_HEIGHT = 1535
IMAGE_PATH = "assets/map.jpg"

# Transaction-to-vehicle matching settings.
# A transaction is compared with GPS points within this time window.
TIME_WINDOW_MINUTES = 20

# Maximum pixel distance for considering a vehicle "near" a transaction location.
DISTANCE_THRESHOLD_PIXELS = 150

# Stop detection settings.
# A stop is detected when a vehicle remains within this pixel radius
# for at least MIN_STOP_MINUTES.
STOP_DISTANCE_THRESHOLD_PIXELS = 85
MIN_STOP_MINUTES = 10

# Longitude/latitude bounds of the map image.
# Used to transform GPS coordinates into x/y pixel positions.
MAP_LON_MIN, MAP_LON_MAX = 24.8250, 24.9100
MAP_LAT_MIN, MAP_LAT_MAX = 36.0450, 36.0950

# Consistent weekday order for charts and heatmaps.
WEEKDAY_ORDER = [
    "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday"
]

# --------------------------------------------------
# Load and encode map image
# --------------------------------------------------
# Plotly needs the map image as a base64 string so it can be used
# as a background layer in figures.
with open(IMAGE_PATH, "rb") as image_file:
    encoded_map = base64.b64encode(image_file.read()).decode()

# --------------------------------------------------
# Load raw data files
# --------------------------------------------------
# gps: vehicle movement traces
# cc: credit card transactions
# loyalty: loyalty card transactions
# location_coords: manually/inferred business coordinates on the map
# car_assignments: maps vehicle IDs to employees
gps = pd.read_csv("data/gps.csv")
cc = pd.read_csv("data/cc_data.csv", encoding="cp1252")
loyalty = pd.read_csv("data/loyalty_data.csv", encoding="cp1252")
location_coords = pd.read_csv("data/location_coords.csv")
car_assignments = pd.read_csv("data/car-assignments.csv")

# --------------------------------------------------
# Standardize column names
# --------------------------------------------------
# Lowercase and strip whitespace to avoid bugs from inconsistent CSV headers.
for df in [gps, cc, loyalty, location_coords, car_assignments]:
    df.columns = df.columns.str.strip().str.lower()

# --------------------------------------------------
# Parse timestamps
# --------------------------------------------------
# Convert timestamp strings to pandas datetime objects.
gps["timestamp"] = pd.to_datetime(gps["timestamp"])
cc["timestamp"] = pd.to_datetime(cc["timestamp"])
loyalty["timestamp"] = pd.to_datetime(loyalty["timestamp"])

# --------------------------------------------------
# Add time features to transaction datasets
# --------------------------------------------------
# These columns are used for weekday/hour heatmaps and filtering.
for df in [cc, loyalty]:
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.date
    df["weekday"] = df["timestamp"].dt.day_name()

# --------------------------------------------------
# Convert GPS longitude/latitude to image pixel coordinates
# --------------------------------------------------
# This allows GPS points to be plotted directly on top of the JPG map.
gps["x_pix"] = (
    (gps["long"] - MAP_LON_MIN) /
    (MAP_LON_MAX - MAP_LON_MIN)
) * IMAGE_WIDTH

gps["y_pix"] = (
    (gps["lat"] - MAP_LAT_MIN) /
    (MAP_LAT_MAX - MAP_LAT_MIN)
) * IMAGE_HEIGHT

# --------------------------------------------------
# Prepare car assignment data
# --------------------------------------------------
# Combine first and last name into a single driver name.
car_assignments["name"] = (
    car_assignments["firstname"].astype(str) + " " +
    car_assignments["lastname"].astype(str)
)

# Rename carid to id so it matches the GPS vehicle column.
car_assignments = car_assignments.rename(columns={"carid": "id"})
car_assignments["id"] = car_assignments["id"].astype("Int64")

# --------------------------------------------------
# Attach driver information to GPS records
# --------------------------------------------------
# After this merge, each GPS point knows which employee/vehicle it belongs to.
gps = gps.merge(
    car_assignments[[
        "id",
        "name",
        "currentemploymenttype",
        "currentemploymenttitle"
    ]],
    on="id",
    how="left"
)

# Human-readable vehicle label for dropdowns.
gps["display_name"] = gps.apply(
    lambda row: f"{int(row['id'])} — {row['name']}"
    if pd.notna(row["name"])
    else f"{int(row['id'])} — Unknown driver",
    axis=1
)

# Sort GPS points chronologically per vehicle.
gps = gps.sort_values(["id", "timestamp"])

# --------------------------------------------------
# Shared dropdown / lookup values
# --------------------------------------------------
vehicles = sorted(gps["id"].unique())
locations = sorted(cc["location"].unique())

# Vehicle dropdown options with employee names.
vehicle_options = (
    gps[["id", "display_name"]]
    .drop_duplicates()
    .sort_values("id")
)