# For reading from partitioned cloud storage paths.
import pyarrow.dataset as pds
import pyarrow.parquet as pq
import pandas as pd
import time, dateutil
from dateutil.relativedelta import relativedelta

# Define some global variables.
s3_partitions = pds.dataset(
    "s3://anonymous@air-example-data/ursa-labs-taxi-data/by_year/2018/",
    partitioning=["month"],
)
models_to_train = [f"s3://anonymous@{file}" for file in s3_partitions.files]
SAMPLE_UNIQUE_ID = 141
FORECAST_LENGTH = 28

def train_model(file_path: str):
    # Read a pyarrow parquet S3 file.
    data = pq.read_table(file_path,
                         filters=[ ("pickup_location_id", "=", SAMPLE_UNIQUE_ID) ],
                         columns=[ "pickup_at", "pickup_location_id", "trip_distance" ],
                        ).to_pandas()
    data["ds"] = data["pickup_at"].dt.to_period("D").dt.to_timestamp()
    data.rename(columns={"trip_distance": "y"}, inplace=True)
    data.drop("pickup_at", inplace=True, axis=1)
    unique_id = data["pickup_location_id"][0]

    # Split data into train, test.
    train_end = data.ds.max() - relativedelta(days=FORECAST_LENGTH - 1)
    train_df = data.loc[(data.ds <= train_end), :].copy()
    test_df = data.iloc[-(FORECAST_LENGTH):, :].copy()
    
    # Define Prophet model with 75% confidence interval.
    import prophet
    from prophet import Prophet
    model = Prophet(interval_width=0.75, seasonality_mode="multiplicative")      

    # Train and fit Prophet model.
    model = model.fit(train_df[["ds", "y"]])

    return train_df, test_df, model, unique_id

############
# SERIAL PYTHON
############
start = time.time()
for file in models_to_train:
    print("Training model serially", file)
    train_df, test_df, model, unique_id = train_model(file)
    print(train_df.shape, test_df.shape, type(model), type(unique_id))

time_regular_python = time.time() - start
print(f"Total number of models: {len(models_to_train)}")
print(f"TOTAL TIME TAKEN: {time_regular_python/60:.2f} minutes")

# Total number of models: 12
# TOTAL TIME TAKEN: 2.34 minutes
# Ran on 1-node AWS cluster of m5.4xlarges


############
# RAY MULTIPROCESSING
############
import ray
from ray.util.multiprocessing import Pool
import tqdm

# Restart ray
if ray.is_initialized():
    ray.shutdown()

start = time.time()
# Create a pool, where each worker is assigned 1 CPU by Ray.
pool = Pool(ray_remote_args={"num_cpus": 1})

# Use the pool to run `train_model` on the data, in batches of 1.
iterator = pool.imap_unordered(train_model, models_to_train, chunksize=1)

# Track the progress using tqdm and retrieve the results into a list.
results = list(tqdm.tqdm(iterator, total=len(models_to_train)))

time_ray_multiprocessing = time.time() - start
print(f"Total number of models: {len(results)}")
print(f"TOTAL TIME TAKEN: {time_ray_multiprocessing/60:.2f} minutes")

assert len(results) == len(models_to_train)
print(type(results[0][0]), type(results[0][1]), type(results[0][2]))

# Total number of models: 12
# TOTAL TIME TAKEN: 0.57 minutes

# Calculate the speed-up between serial Python and Ray Multiprocessing Pool
import numpy as np
speedup = time_regular_python / time_ray_multiprocessing
print(f"Speedup from running Ray Multiprocessing vs serial Python: {np.round(speedup, 1)}x"
      f", or {(np.round(speedup, 0)-1) * 100}%")

# Speedup from running Ray Multiprocessing vs serial Python: 3.7x, or 300.0%


############
# RAY TUNE WITH AIR
############
import numpy as np
from typing import Tuple
from ray import air, tune
from ray.air import session, ScalingConfig
from ray.air.checkpoint import Checkpoint
RAY_IGNORE_UNHANDLED_ERRORS=1

def evaluate_model_prophet(
    model: "prophet.forecaster.Prophet",
) -> Tuple[float, pd.DataFrame]:

    # Inference model using FORECAST_LENGTH.
    future_dates = model.make_future_dataframe(
        periods=FORECAST_LENGTH, freq="D"
    )
    future = model.predict(future_dates)

    # Calculate mean absolute error.
    temp = future.copy()
    temp["forecast_error"] = np.abs(temp["yhat"] - temp["trend"])
    error = np.mean(temp["forecast_error"])

    return error, future


# This function is exactly the same as train_model(), except:
# - Change the input parameter to config type dict.
# - Change the model based on 'algorithm' input parameter
# - Add evaluate model error and Ray AIR checkpointing
def trainable_func(config: dict):
    # Read a pyarrow parquet S3 file.
    data = pq.read_table(config["file_path"],
                         filters=[ ("pickup_location_id", "=", SAMPLE_UNIQUE_ID) ],
                         columns=[ "pickup_at", "pickup_location_id", "trip_distance" ],
                        ).to_pandas()
    data["ds"] = data["pickup_at"].dt.to_period("D").dt.to_timestamp()
    data.rename(columns={"trip_distance": "y"}, inplace=True)
    data.drop("pickup_at", inplace=True, axis=1)
    unique_id = data["pickup_location_id"][0]

    # Split data into train, test.
    train_end = data.ds.max() - relativedelta(days=FORECAST_LENGTH - 1)
    train_df = data.loc[(data.ds <= train_end), :].copy()
    test_df = data.iloc[-(FORECAST_LENGTH):, :].copy()
    
    # Define Prophet model with 75% confidence interval.
    import prophet
    from prophet import Prophet
    if config["algorithm"] == "prophet_additive":
        model = Prophet(interval_width=0.75, seasonality_mode="additive")
    elif config["algorithm"] == "prophet_multiplicative":
        model = Prophet(interval_width=0.75, seasonality_mode="multiplicative")

    # Train and fit the Prophet model.
    model = model.fit(train_df[["ds", "y"]])

    # Inference model and evaluate error.
    error, future = evaluate_model_prophet(model)

    # Define a model checkpoint using AIR API.
    # https://docs.ray.io/en/latest/tune/tutorials/tune-checkpoints.html
    checkpoint = ray.air.checkpoint.Checkpoint.from_dict(
        {
            "model": model,
            "forecast_df": future,
            "location_id": unique_id,
        }
    )
    # Save checkpoint and report back metrics, using ray.air.session.report()
    metrics = dict(error=error)
    session.report(metrics, checkpoint=checkpoint)

# Tune is designed for up to thousands of trials.
param_space = {
    "file_path": tune.grid_search(
        [ f"s3://anonymous@{file}" for file in s3_partitions.files ]
    ),
    "algorithm": tune.grid_search(["prophet_additive", "prophet_multiplicative"]),
}

start = time.time()
tuner = tune.Tuner(trainable_func, param_space=param_space)
results = tuner.fit()

time_ray_tune = time.time() - start
print(f"Total number of models: {len(results)}")
print(f"TOTAL TIME TAKEN: {time_ray_tune/60:.2f} minutes")

# Total number of models: 24
# TOTAL TIME TAKEN: 1.42 minutes

# Calculate the speed-up between serial Python and Ray Tune
speedup = time_regular_python / time_ray_tune * 2
print(f"Speedup from running Ray Tune with AIR vs serial Python: {np.round(speedup, 1)}x"
      f", or {(np.round(speedup, 0)-1) * 100}%")

# Speedup from running Ray Tune with AIR vs serial Python: 242.3x, or 24100.0%
