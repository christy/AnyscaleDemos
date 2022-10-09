from typing import Tuple
import random
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
        sample_ids (list): list of ids for sampling

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



if __name__ == "__main__":

    # Test the prepare_data function
    target = "trip_duration"
    data_files = \
    ['s3://air-example-data/ursa-labs-taxi-data/by_year/2019/05/data.parquet/359c21b3e28f40328e68cf66f7ba40e2_000000.parquet',
    's3://air-example-data/ursa-labs-taxi-data/by_year/2019/06/data.parquet/ab5b9d2b8cc94be19346e260b543ec35_000000.parquet']
    sample_locations = list(range(1, 21))
    
    #True: sample the data instead of using all of it
    SMOKE_TEST = True 

    if SMOKE_TEST:
        data_files = data_files[0]
        sample_locations = random.sample(sample_locations, 3)
        
    train_ds, valid_ds, test_ds = prepare_data(data_files, target, sample_locations)

    print(f"Number rows train, test: ", end="")
    print(f"{train_ds.count()}, {test_ds.count()}")