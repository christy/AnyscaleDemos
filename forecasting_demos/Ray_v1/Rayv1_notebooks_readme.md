### Instructions for running Ray v1 notebooks

------

**Multi-node Training using your own laptop's multiple cores:** <br>

1. Change the variables in the notebook.
   1. RUN_RAY_LOCAL = True 
   2. RUN_RAY_ON_A_CLOUD = False 

Run everything in the notebook up to, but not including the cloud part.

Uses a [Ray local server](https://docs.ray.io/en/latest/walkthrough.html) to automatically handle CUDA whether or not you have GPU.



**Multi-node Training using a cluster of compute nodes in any cluster:** <br>

1. Change the variables in the notebook.
   1. RUN_RAY_LOCAL = False 
   2. RUN_RAY_ON_A_CLOUD = True 


Run everything in the notebook up to, but not including the Ray local server part, then run the cloud part.

Uses [Anyscale](https://docs.anyscale.com/) to automatically start a cloud cluster, auto-scale, handle CUDA whether or not GPU are available, and automatically shuts down the cloud cluster.