"""Utilities for SSML Dataset, Trainers, Tune demo
"""
from typing import Tuple
import ray
from ray.data import Dataset
# import pyarrow as pa  #is something else overwriting pa?
import pyarrow.dataset as pads
import pandas as pd


def pushdown_read_data(files_list: list) -> Dataset:
    """Read data using projection (col selection) and row filter pushdown at parquet read-level. 

    Args:
        files_list (list): list of files

    Returns:
        Dataset: Ray Dataset data
    """
    filter_expr = (
        (pads.field("passenger_count") > 0)
        & (pads.field("trip_distance") > 0)
        & (pads.field("fare_amount") > 0)
    )

    the_dataset = ray.data.read_parquet(
        files_list,
        columns=[
            'pickup_at', 'dropoff_at',
            'passenger_count', 'trip_distance', 'fare_amount', 
            # might need these later for plotting, but drop for now
            # 'pickup_longitude', 'pickup_latitude',
            # 'dropoff_longitude', 'dropoff_latitude'
                ], 
        filter=filter_expr,
    )

    # Force full execution of both of the file reads.
    the_dataset = the_dataset.fully_executed()
    return the_dataset

def transform_batch(the_df: pd.DataFrame) -> pd.DataFrame:
    """UDF function to transform a pandas dataframe

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: updated dataframe
    """
    df = the_df.copy()
    df["trip_duration"] = (df["dropoff_at"] - df["pickup_at"]).dt.seconds
    df = df[df["trip_duration"] >= 60]
    return df

def prepare_data(files_list: list, target: str) -> Tuple[Dataset, Dataset, Dataset]:
    """Function to load data, clean it, and split into train, validation, and test datasets.

    Args:
        files_list (list): list of files
        target (str): column name of the target_column or y-variable you want to predict

    Returns:
        Tuple[Dataset, Dataset, Dataset]: train, valid, test
    """
    # pushdown read data
    the_dataset = pushdown_read_data(files_list)
    
    # calculate trip_duration using pandas UDF `transform_batch`
    the_dataset = the_dataset.map_batches(
            transform_batch, 
            batch_format="pandas")   
        
    # drop 2 columns we do not need anymore
    the_dataset = the_dataset.drop_columns(["dropoff_at", "pickup_at"])
        
    # perform a global shuffle
    the_dataset = the_dataset.random_shuffle()
    
    # split data into train/valid
    train_dataset, valid_dataset = \
        the_dataset.train_test_split(test_size=0.2)
    
    # create test data same as valid
    test_dataset = valid_dataset.drop_columns([target])
    
    # return train, valid, test data
    return train_dataset, valid_dataset, test_dataset


