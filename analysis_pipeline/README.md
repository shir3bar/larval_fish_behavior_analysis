# Analysis pipeline
This is a code implementation of the deep learning pipeline for the detection of the sparsely-occurring feeding strike behavior of larval fish in continuous video data as described in [Assessing the determinants of larval fish strike rates using computer vision](https://doi.org/10.1016/j.ecoinf.2023.102195).

We apply two models: a fish detector ([Faster-R-CNN](https://github.com/facebookresearch/detectron2)) and an action classifier ([SlowFast network](https://github.com/facebookresearch/SlowFast/)) to detect and classify the behavior of larval fish in aquaculture rearing pools.

### Getting started
This code was tested on Ubuntu 20.04 LTS and Windows 10, it requires:
* Python 3.8
* A CUDA-capable GPU (optional, but would greatly improve runtimes)

It is also highly recommended to use conda or another virtual environment manager.

### Setup with Conda
**Soon** We'll upload the conda environment configuration for Windows. For now, this environment was tested on Ubuntu:

```commandline
git clone https://github.com/shir3bar/larval_fish_behavior_analysis
cd larval_fish_behavior_analysis

conda env create -f environment_ubuntu.yml
conda activate larvaeAction

cd ./analysis_pipeline
```

### Models
Download our trained models for this pipeline [here](https://drive.google.com/open?id=1yxH-69Qd1w0-bfyjRpa32NHBhXbcJmXT) and place them in the `./models` folder for the code to work. **OR** place your own custom-trained models in this folder.
We wrote the code to work with the Detectron2 and Pytorchvideo libraries, if you're using something different you'll have to modify the `load_detector` and `load_action_classifer` functions found in `./loaders.py`.
<!--, for object detector we recommend using detectron2, YoloV5, megadetector (for terrestrials) or megafishdetector (for fish).
-->
### Video format
Our code currently supports `.avi` or NorPix's proprietary `.seq` format (for which we provide a parser). Since we base our video handling on OpenCV, extension to other formats such as `.mp4` should be fairly straightforward.

### Configuration file
A config file for the action detection model is placed in the `./models` folder, this is used when loading the model.
If you're using custom models, you might need to change the configs or the model loading function in `./loaders.py`.

### Run pipeline
Run pipeline on all untrimmed videos in a folder:
```commandline
python fish_for_fish.py /path/to/save/results /path/to/video/data
```
If you'd like a to run on a single video, use:
```commandline
python fish_for_fish.py /path/to/save/results /path/to/video/data -video_name video_file_name
```
These option will save all clips resulting from the dataset as `.avi` files, using the 
`-no_clips` flag will only save `.csv`s with model predictions (see sample video section for an example of output).

To use the pipeline with pre-existing detections, use the `-classify_only` flag and check out our `detections.csv` for an example of the expected format.

For other options such as clip sizes, durations, and duration of samples from pre-existing videos, please see `./parser.py` or:

```commandline
python fish_for_fish.py --help
```

### ***SOON:*** Sample video
You can test our system using this video. 




More code and documentation coming soon.
