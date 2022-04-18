# ODSC Conference April, 2022

## Quick setup instructions:

#### 1. Install Conda if needed
https://www.anaconda.com/products/individual <br>
$ conda env list  # list conda envs <br>
<br>

#### 2. Conda install RLlib environment for tutorial
```
$ conda create -yn rllib_tutorial python=3.9
$ conda activate rllib_tutorial
$ pip install jupyterlab "ray[rllib,serve,tune]" sklearn
conda install -y tensorflow  # either version works!
pip install torch gputil  # any latest version works!
pip install "ray[default]"  # updates ray dashboard

# Mac - see extra install notes below
# Win10 - see extra install notes below

$ git clone https://github.com/christy/AnyscaleDemos
$ cd rllib_demos/ODSC_conference
$ jupyter-lab
```

##### Mac only
$ pip install grpcio  
Note: In case you are getting a "requires TensorFlow version >= 2.8" error at some point in the notebook, try the following: <br>
$ pip uninstall -y tensorflow <br>
$ python -m pip install tensorflow-macos --no-cache-dir

##### Win10 only
$ pip install pywin32  # <- Win10 only
