## Installing the required environment

Create a new conda environment and follow installation instructions in [pySlowFast](https://github.com/facebookresearch/SlowFast/blob/main/INSTALL.md), install a pytorch version matching your cuda version. Note the code was tested on pytorch version 1.8.
For our code to run you'll need to install pySlowFast from the github, not through pypi.

There is a known issue with building pySlowFast when using pytorch >= 2.0, to avoid it either use an older version of torch or use [this repo](https://github.com/shir3bar/slowfast).

Once you're done installing dependencies, clone this repository and you're good to go.
```commandline
git clone https://github.com/shir3bar/larval_fish_behavior_analysis
cd larval_fish_behavior_analysis
```

### Using the env_ubuntu.yml
If you're running Ubuntu, you can use the environment_ubuntu.yml to install all dependencies and then install pySlowFast.
Note you may need to change the pytorch distribution to match your cuda.

```commandline
git clone https://github.com/shir3bar/larval_fish_behavior_analysis
cd larval_fish_behavior_analysis

conda env create -f environment_ubuntu.yml
conda activate larvaeAction

git clone https://github.com/facebookresearch/SlowFast/

export PYTHONPATH=/path/to/SlowFast/slowfast:$PYTHONPATH

cd SlowFast
python setup.py build develop
```

### Download the models
