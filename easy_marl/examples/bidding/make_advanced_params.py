"""
File to test workflow when given aeres params

NOTE: we also need to build an upated environment to incoporate:
 - Ramp limits and cost
 - LF capacity limits

N Plants
T Hours

Neccessary Data

Scalars ndim = 1:
 - max_bid_delta

Plant Vectors ndim = (1,N):
 - Capacity
 - SRMC
 - Ramp Limit (this is not implemented in the env yet)
 - Ramp Cost  (this is not implemented in the env yet)
 - LF series pointer

Hourly Vectors = (1,T):
 - LF series
 - Demand Series

"""

# import pandas as pd
import numpy as np


def _read_plant_data():
    # pd.read_csv("dummy.csv")
    pass


def _read_demand_profile():
    pass


def example_plant_data():
    plant_data = {}
    # solar and wind first
    plant_data["capacities"] = [40, 40, 20, 20, 20]
    plant_data["srmc"] = [0, 0, 10, 20, 30]
    plant_data["ramp_limit"] = [
        1,
        1,
        0.25,
        0.5,
        1,
    ]  # % of the capacity per timestep
    plant_data["ramp_cost"] = [0, 0, 1, 1, 1]
    plant_data["lf_series"] = ["wof", "sol", "nan", "nan", "nan"]

    for key in plant_data:
        plant_data[key] = np.asarray(plant_data[key])

    return plant_data


def example_demand_profile():
    from scipy.interpolate import CubicSpline

    # Define key points for interpolation (Hour, Value)
    # Weekday: Low at midnight (0.5), peak at 8am (0.85), low at midday (0.7), highest peak at 7pm (1.0)
    wd_x = [0, 8, 13, 19, 24]
    wd_y = [0.5, 0.85, 0.7, 1.0, 0.5]

    # Create smooth spline for weekday with periodic boundary conditions
    wd_spline = CubicSpline(wd_x, wd_y, bc_type="periodic")

    # Weekend: Similar but generally lower demand
    we_x = [0, 10, 14, 20, 24]
    we_y = [0.45, 0.75, 0.6, 0.85, 0.45]

    # Create smooth spline for weekend with periodic boundary conditions
    we_spline = CubicSpline(we_x, we_y, bc_type="periodic")

    # Generate hourly profiles
    hours = np.arange(24)
    weekday_profile = wd_spline(hours)
    weekend_profile = we_spline(hours)

    # Construct weekly profile (5 weekdays, 2 weekends)
    demand_profile = np.concatenate([weekday_profile] * 5 + [weekend_profile] * 2)

    return demand_profile


def read_plant_data():
    try:
        _read_plant_data()
        raise ImportError("Feature not created yet")
    except Exception:
        plant_data = example_plant_data()

    return plant_data


def read_demand_profile():
    try:
        _read_demand_profile()
        raise ImportError("Feature not created yet")
    except Exception:
        demand_profile = example_demand_profile()

    demand_profile = np.asarray(demand_profile)
    demand_profile = demand_profile / demand_profile.max()

    return demand_profile


def get_params(N=5, T=7 * 24) -> dict:
    plant_data = read_plant_data()
    demand_profile = read_demand_profile()

    # temporary fix
    demand_profile = demand_profile * sum(plant_data["capacities"]) * 0.75

    params = {
        "N_generators": N,
        "T": T,
        "demand_profile": demand_profile.tolist(),
        "capacities": plant_data["capacities"].tolist(),
        "costs": plant_data["srmc"].tolist(),
        "max_bid_delta": 50.0,
        "lambda_bid_penalty": 0.01,
    }
    return params
