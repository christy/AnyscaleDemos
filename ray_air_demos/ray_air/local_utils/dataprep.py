"""Utilities for SSML Dataset batch training demo
"""
from typing import Tuple
import ray
from ray.data import Dataset
# import pyarrow as pa  #is something else overwriting pa?
import pyarrow.dataset as pds
import pandas as pd
import numpy as np

# Define some global variables.
location_ids = np.array([  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
        14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
        27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
        40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
        53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
        66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,
        79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
        92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 105, 106,
       107, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
       121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
       134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
       147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
       160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
       173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185,
       186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198,
       200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
       213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225,
       226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238,
       239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
       252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263], dtype='int32')
from typing import Tuple
import ray
from ray.data import Dataset
# import pyarrow as pa  #is something else overwriting pa?
import pyarrow.dataset as pads
import pandas as pd


def pushdown_read_data(files_list: list,
                       sample_ids: list) -> Dataset:
    """Read data using projection (col selection) and row filter pushdown at parquet read-level. 

    Args:
        files_list (list): list of files
        sample_ids (list): list of sampling ids

    Returns:
        Dataset: Ray Dataset data
    """
    filter_expr = (
        (pads.field("passenger_count") > 0)
        & (pads.field("trip_distance") > 0)
        & (pads.field("fare_amount") > 0)
        & (pads.field("pickup_location_id").isin(sample_ids))
    )

    the_dataset = ray.data.read_parquet(
        files_list,
        columns=[
            'pickup_at', 'dropoff_at', 'pickup_location_id',
            'passenger_count', 'trip_distance', 'fare_amount'],
        filter=filter_expr,
    )

    # Force full execution of both of the file reads.
    the_dataset = the_dataset.fully_executed()
    return the_dataset

def transform_batch(the_df: pd.DataFrame) -> pd.DataFrame:
    """UDF function to transform a pandas dataframe.
        Calculates trip duration in seconds.
        Drops rows with shorter than 1 minute trip duration.
        Drops pickup/dropoff timestamps since not needed anymore
        Fillna missing pickup_location_id

    Args:
        df (pd.DataFrame): input dataframe

    Returns:
        pd.DataFrame: updated dataframe
    """
    df = the_df.copy()    
    df["trip_duration"] = (df["dropoff_at"] - df["pickup_at"]).dt.seconds    
    df = df[df["trip_duration"] >= 60]    
    df.drop(["dropoff_at", "pickup_at"], axis=1, inplace=True)
    df['pickup_location_id'] = df['pickup_location_id'].fillna(-1)
    return df

def prepare_data(files_list: list, 
                 target: str, 
                 sample_ids: list) -> Tuple[Dataset, Dataset, Dataset]:
    """Function to load data, clean it, and split into train, validation, and test datasets.

    Args:
        files_list (list): list of files
        target (str): column name of the target_column or y-variable you want to predict
        sample_ids (list): list of sampling ids

    Returns:
        Tuple[Dataset, Dataset, Dataset]: train, valid, test
    """
    # pushdown read data
    the_dataset = pushdown_read_data(files_list, sample_ids)
    
    # calculate trip_duration using pandas UDF `transform_batch`
    the_dataset = the_dataset.map_batches(
            transform_batch, 
            batch_format="pandas")   
        
    # perform a global shuffle
    the_dataset = the_dataset.random_shuffle()
    
    # split data into train/valid
    train_dataset, valid_dataset = \
        the_dataset.train_test_split(test_size=0.2)
    
    # create test data same as valid
    test_dataset = valid_dataset.drop_columns([target])
    
    # return train, valid, test data
    return train_dataset, valid_dataset, test_dataset


