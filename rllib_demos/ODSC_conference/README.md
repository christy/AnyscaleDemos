# ODSC Conference April, 2022

## Tutorial setup instructions:

You can run the tutorial notebook either: <br>

Option #1.  Run the .ipynb as a Colab. This is easiest but the tutorial will run more slowly.<br>

Option #2. Run the .ipynb in your own local development environment that you need to create yourself. You'll also need to download from github this .ipynb.<br>
<br>

#### Option #1. Run tutorial from Colab

Click https://colab.research.google.com/github/christy/AnyscaleDemos/blob/main/rllib_demos/ODSC_conference/tutorial_notebook.ipynb
<br>
<br>


#### Option #2. Conda install RLlib environment for tutorial (more setup steps but tutorial runs quicker)

#### 1. Install Conda if needed
https://www.anaconda.com/products/individual <br>
$ conda env list  # list conda envs <br>

#### 2. Create conda env and install the libraries below
```
$ conda create -yn rllib_tutorial python=3.9
$ conda activate rllib_tutorial
$ pip install jupyterlab "ray[rllib,serve,tune]" sklearn
$ conda install -y tensorflow  # either version works!
$ pip install recsim torch gputil  # any latest version works!
$ pip install "ray[default]"  # updates ray dashboard

# Win10 only  - required extra step
$ pip install pywin32  # <- Win10 only

# Mac - see possible extra install notes below

# Now run the tutorial notebook locally
$ git clone https://github.com/christy/AnyscaleDemos
$ cd rllib_demos/ODSC_conference
$ jupyter-lab
```

##### Mac only - potential extra steps
$ conda install grpcio  <br>

In case you are getting a "requires TensorFlow version >= 2.8" error at some point in the notebook, try the following: <br>
$ pip uninstall -y tensorflow <br>
$ python -m pip install tensorflow-macos --no-cache-dir
