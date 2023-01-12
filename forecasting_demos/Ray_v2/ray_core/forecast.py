##########
## Standalone Python file for Prophet.
##########

# Import required libraries.
import time, os, logging
from typing import Tuple
import dateutil
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
# Import forecasting libraries.
import prophet
from prophet import Prophet
print(f"prophet: {prophet.__version__}")

# Import and start ray.
import ray
if ray.is_initialized():
    ray.shutdown()
ray.init()
print(ray.cluster_resources())

##########
## STEP 1.  Read data and ray.put() data into Ray shared cluster memory.
##########
# Read data into a Pandas dataframe.
filename = "../../../forecasting_demos/Ray_v1/data/clean_taxi_monthly.parquet"
g_month = pd.read_parquet(filename)

# Prophet requires timstamp is 'ds' and target_value name is 'y'
# Prophet requires at least 2 data points per timestamp
# StatsForecast requires location name is 'unique_id'

g_month.reset_index(inplace=True)
g_month.rename(columns={"pickup_monthly": "ds"}, inplace=True)
g_month.rename(columns={"pulocationid": "unique_id"}, inplace=True)
g_month.rename(columns={"trip_quantity": "y"}, inplace=True)
g_month.drop(['pickup_lat', 'pickup_lon'], axis=1, inplace=True)

# Put data in shared ray object store.
input_data_ref = ray.put(g_month)

# Define some global variables.
TARGET = "trip_quantity"
FORECAST_LENGTH = 2
UNIQUE_ID="unique_id"
ID_LIST = list(g_month[UNIQUE_ID].unique())
    
# define file handler, not appending, to avoid growing logs
file_handler = logging.FileHandler('training.log', mode='w')
formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
file_handler.setFormatter(formatter)

# prophet logger - also need Class below to stop the noisy PyStan messages
prophet_logger = logging.getLogger('fbprophet')
prophet_logger.setLevel(logging.CRITICAL)
prophet_logger.addHandler(file_handler)

# This class is to suppress the Pystan noisy messages coming from Prophet
# Thanks to https://github.com/facebook/prophet/issues/223#issuecomment-326455744
class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

# Define a train_model function.
def train_model_prophet(input_df:pd.DataFrame, input_value:str) -> Tuple[pd.DataFrame, pd.DataFrame, bytes]:
    
    # Subset pandas df
    df = input_df.loc[(input_df[UNIQUE_ID] == input_value), :].copy()

    # Split data into train, test.
    train_end = df.ds.max() - relativedelta(months=FORECAST_LENGTH - 1)
    train_df = df.loc[(df.ds <= train_end), :].copy()
    test_df = df.iloc[-(FORECAST_LENGTH):, :].copy()
    
    # Define Prophet model with 75% confidence interval.
    model = Prophet(interval_width=0.75, seasonality_mode="multiplicative")      

    # Train and fit Prophet model.
    with suppress_stdout_stderr():  
        model = model.fit(train_df[["ds", "y"]])
    
    return train_df, test_df, model

# Define inference_model function.
def inference_model_prophet(
    model: "prophet.forecaster.Prophet",
    train: pd.DataFrame,
    valid: pd.DataFrame,
    input_value:str,
) -> Tuple[float, pd.DataFrame]:

    # Inference Prophet model using FORECAST_LENGTH.
    future_dates = model.make_future_dataframe(
        periods=FORECAST_LENGTH, freq="M"
    )
    future = model.predict(future_dates)
    
    # Merge in the actual y-values for convenience.
    future = pd.merge(future, train[['ds', 'y']], on=['ds'], how='left')
    future = pd.merge(future, valid[['ds', 'y']], on=['ds'], how='left')
    future['y'] = future.y_x.combine_first(future.y_y)
    future.drop(['y_x', 'y_y'], inplace=True, axis=1)
    future['unique_id'] = input_value

    # Calculate mean absolute forecast error.
    temp = future.copy()
    temp["forecast_error"] = np.abs(temp["yhat"] - temp["y"])
    error = np.mean(temp["forecast_error"])

    return error, future


###########
# Regular Python program flow to train and inference Prophet models
###########

# Train every model.
print("Start training with regular Python...")
start = time.time()
train, valid, model = map(
    list, zip(*(
        [ train_model_prophet(g_month, input_value=v,)
          for v in ID_LIST ]
    )),)

# Inference every model.
error, forecast = map(
    list, zip(*(
        [ inference_model_prophet(model[p],train[p],
                                  valid[p], input_value=v,)
          for p,v in enumerate(ID_LIST) ]
    )),)

# Print some training stats
time_regular_python = time.time() - start
print(f"Total number of models: {len(model)}")
print(f"TOTAL TIME TAKEN: {time_regular_python/60:.2f} minutes")

# Verify you have 1 model and 1 forecast per unique ID.
assert len(model) == len(ID_LIST)
assert len(forecast) == len(ID_LIST)

# View first two forecasts
for p, v in enumerate(ID_LIST[0:2]):
    print(forecast[p].tail(2))
    
    
###########
# Ray distributed program flow to train and inference ARIMA models
###########

# Convert your regular python functions to ray remote functions
train_model_prophet_remote = ray.remote(train_model_prophet).options(num_returns=3)
inference_model_prophet_remote = ray.remote(inference_model_prophet).options(num_returns=2)

# Train every model.
start = time.time()
train_obj_refs, valid_obj_refs, model_obj_refs = map(
    list, zip(*(
        [ train_model_prophet_remote.remote(input_data_ref, input_value=v,)
          for v in ID_LIST ]
    )),)

# Inference every model.
error_obj_refs, forecast_obj_refs = map(
    list, zip(*(
        [ inference_model_prophet_remote.remote(
                    model_obj_refs[p], train_obj_refs[p],
                    valid_obj_refs[p],input_value=v)
           for p,v in enumerate(ID_LIST) ]
    )),)

# ray.get() means block until all objectIDs requested are available
forecast = ray.get(forecast_obj_refs)
error = ray.get(error_obj_refs)
model = ray.get(model_obj_refs)

# Print some training stats
time_ray_local = time.time() - start
print(f"Total number of models: {len(model)}")
print(f"TOTAL TIME TAKEN: {time_ray_local/60:.2f} minutes")

# Calculate speedup:
speedup = time_regular_python / time_ray_local
print(f"Speedup from running Ray parallel code on your laptop: {np.round(speedup, 1)}x"
      f", or {(np.round(speedup, 0)-1) * 100}%")

# Verify you have 1 model and 1 forecast per unique ID.
assert len(model) == len(ID_LIST)
assert len(forecast) == len(ID_LIST)

# View first two forecasts
for p, v in enumerate(ID_LIST[0:2]):
    print(forecast[p].tail(2))