# For reading from partitioned cloud storage paths.
from smart_open import smart_open
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
MODELS_TO_TRAIN = [f"s3://anonymous@{file}" for file in s3_partitions.files]
SAMPLE_UNIQUE_ID = 141
FORECAST_LENGTH = 28

def train_model(file_path: str):
    data = pq.read_table(file_path,
                         filters=[ ("pickup_location_id", "=", SAMPLE_UNIQUE_ID) ],
                         columns=[ "pickup_at", "pickup_location_id", "trip_distance" ],
                        ).to_pandas()
    data["ds"] = data["pickup_at"].dt.to_period("D").dt.to_timestamp()
    data.rename(columns={"trip_distance": "y"}, inplace=True)
    data.drop("pickup_at", inplace=True, axis=1)

    # Split data into train, test.
    train_end = data.ds.max() - relativedelta(days=FORECAST_LENGTH - 1)
    train_df = data.loc[(data.ds <= train_end), :].copy()
    test_df = data.iloc[-(FORECAST_LENGTH):, :].copy()
    
    # Define Prophet model with 75% confidence interval.
    model = Prophet(interval_width=0.75, seasonality_mode="multiplicative")      

    # Train and fit Prophet model.
    model = model.fit(train_df[["ds", "y"]])

    return train_df, test_df, model

# This will take much too long serially.
start = time.time()
for file in MODELS_TO_TRAIN:
    print("Training model serially", file)
    train_df, test_df, model = train_model(file)
    print(train_df.shape, test_df.shape, type(model))

time_regular_python = time.time() - start
print(f"Total number of models: {len(MODELS_TO_TRAIN)}")
print(f"TOTAL TIME TAKEN: {time_regular_python/60:.2f} minutes")

# Total number of models: 12
# TOTAL TIME TAKEN: 2.34 minutes
# Ran on 1-node AWS cluster of m5.4xlarges
