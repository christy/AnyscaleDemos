
# forecasting_demo

This library uses [Ray](https://docs.ray.io/en/latest/) for quick and easy distributed forecasting - training and inference - converting existing code so it can run in parallel on multiple compute nodes.  The compute can be cores on your laptop or clusters in the cloud.  

These forecasting demos show how to leverage the benefits of well-known open-source [ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average), [Prophet](https://facebook.github.io/prophet/), [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/), and [TensorFlow](https://www.tensorflow.org/) algorithms and Deep Learning frameworks. The PyTorch demo uses the [Ray plug-in for PyTorch Lightning.](https://github.com/ray-project/ray_lightning?ref=pythonrepo.com)



### Data

------

This library uses the public NYC Taxi rides dataset.  The forecast goal is to predict the number of NY City yellow taxi rides that will be requested for each location in NYC in the future.

- Raw data original source: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

- Raw data hosted publicly by AWS:  https://registry.opendata.aws/nyc-tlc-trip-records-pds/

- 8 months of cleaned data in this repo under folder data/

  

### Notebooks

------

Each notebook can be run in 2 ways.  Change the variables in the notebook and then run either the local or cloud parts.  The "SMALL" and "MEDIUM" versions of the notebooks show example timings.

Suggested - copy the notebooks - run the copy.  That way you'll keep the original outputs for reference.

| Description                                                  | Instructions                          | Notebook                                                     |
| ------------------------------------------------------------ | ------------------------------------- | ------------------------------------------------------------ |
| **ARIMA** strategic forecasting number of rides at each location in the next 2 months at monthly granularity. **Original data 260 items, 2 data points forecast horizon.** | Change the variables in the notebook. | [Link to notebook](https://github.com/anyscale/demos/blob/master/forecasting_demo/nyctaxi_arima_simple_SMALL_data.ipynb) |
| **ARIMA** strategic forecasting number of rides at each location in the next 2 months at monthly granularity. **Fake data 2860 items, 2 data points forecast horizon.** | Change the variables in the notebook. | [Link to notebook](https://github.com/anyscale/demos/blob/master/forecasting_demo/nyctaxi_arima_simple_MEDIUM_data.ipynb) |
| **Prophet** strategic forecasting number of rides at each location in the next 2 months at monthly granularity. **Original data 260 items, 2 data points forecast horizon.** | Change the variables in the notebook. | [Link to notebook](https://github.com/anyscale/demos/blob/master/forecasting_demo/nyctaxi_prophet_simple_SMALL_data.ipynb) |
| **Prophet** strategic forecasting number of rides at each location in the next 2 months at monthly granularity. **Fake data 2860 items, 2 data points forecast horizon.** | Change the variables in the notebook. | [Link to notebook](https://github.com/anyscale/demos/blob/master/forecasting_demo/nyctaxi_prophet_simple_MEDIUM_data.ipynb) |
| **Pytorch Lightning** Deep Learning notebook operational forecasting number of rides at each location in the next week at hourly granularity.  **Original data 260 items, 24 * 7 = 168 data points forecast horizon.** | Change the variables in the notebook. | [Link to notebook](https://github.com/anyscale/demos/blob/master/forecasting_demo/pytorch_forecasting_ray_local.ipynb) |
| **Tensorflow2** Distributed Deep Learning notebook operational forecasting number of rides at each location in the next week at hourly granularity.  **Original data 260 items, 24 * 7 = 168 data points forecast horizon.** |                                       | coming soon                                                  |



### Installation

------

To install Ray, install libraries in this order:

1. `conda install -y grpcio`
2. `pip install ray`

To install Anyscale:  `pip install -U anyscale`

To install Prophet from scratch, install libraries in this order:

1. `conda install -y llvmlite`
1. `pip install kats`
3. If you get error about numpy version, `conda uninstall -y numpy`
3. `conda install -y numba=0.52.0`
3. `conda install -y scipy`
3. `conda install -c conda-forge statsmodels`
4. `pip install kats`

To install ARIMA from scratch, install libraries in this order:

1. `conda install -y scipy`

2. `conda install -y scikit-learn`

3. `conda install -y statsmodels`

4. `conda install -y cython`

5. `pip install pmdarima`

   

### Instructions

------

**Multi-node Training using your own laptop's multiple cores:** <br>

1. RUN_RAY_LOCAL = True 
2. RUN_RAY_ON_A_CLOUD = False 
3. Run everything in the notebook up to, but not including the cloud part.

Uses a [Ray local server](https://docs.ray.io/en/latest/walkthrough.html) to automatically handle CUDA whether or not you have GPU.



**Multi-node Training using a cluster of compute nodes in any cluster:** <br>

1. RUN_RAY_LOCAL = False 
2. RUN_RAY_ON_A_CLOUD = True 
3. Run everything in the notebook up to, but not including the Ray local server part, then run the cloud part.

Uses [Anyscale](https://docs.anyscale.com/) to automatically start a cloud cluster, auto-scale, handle CUDA whether or not GPU are available, and automatically shuts down the cloud cluster.


