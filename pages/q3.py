from dash import html, dcc

from figures import (
    make_best_card_owner_table,
    make_card_owner_heatmap,
    make_home_location_map,
    make_sleepover_table
)


def layout():
    return html.Div([
        html.H2("Question 3 — Card Ownership Inference"),

        html.P(
            "This page estimates which employee owns each credit card and loyalty card "
            "by matching card transactions to nearby vehicle GPS traces. Higher scores "
            "mean stronger repeated evidence across time, locations, and days."
        ),

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