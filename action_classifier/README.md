## Action classifier training
Code to train and evaluate action classifiers for larval fish behavior analysis 
using the [pySlowFast](https://github.com/facebookresearch/SlowFast/) 
and [Pytorchvideo](https://pytorchvideo.org/) frameworks. <br>
More nitty gritty details of ablation studies can be found [here](https://www.biorxiv.org/content/10.1101/2022.11.14.516417v1.abstract).

If you want to try the code without installation hassles try this [Colab notebook](./Train_action_classifier.ipynb).

### Getting started
This code was test on Ubuntu 20.04 LTS and requires:
* Python 3.8
* A CUDA-capable GPU

It is also highly recommended to use conda or another virtual environment manager.
We will also provide a Colab notebook for those who would like to try out the code/models without the setup hassle.

### Setup 
See [INSTALL.md](./INSTALL.md) for instructions.

### Data structure
```
dataset_folder
|
├── train
│   ├── strike
│   └── swim
└── val
│   ├── strike
│   └── swim
└── test
    ├── strike
    └── swim
```
Our training dataset, structured as above, can be downloaded 
[here](https://drive.google.com/open?id=1HRHSlyNn7QrczMqEmqH2UHRXKwSsOhrS).
Our extended test set, used for further evaluation of classifiers under more challenging conditions, 
can be downloaded [here](https://drive.google.com/open?id=196CT-FqsH9EpEuLbffzytRFfFpJuaAKc).

### Custom dataset
Our code uses pytorchvideo's `labeleddatapath` function and can thus be used with other binary classification
problems with minimal modifications.
The dataset and dataloader can be used for any folder with a train/val/test subfolders and 
within them folders with class names. 
The scoring and the decision on which class is the positive class should be modified.
If you have more than two action classes, 
look into the [pySlowFast](https://github.com/facebookresearch/SlowFast/) code.

### Configuration file
All model, dataset and training options should be specified in cfg.yaml files.
Including: type of model, path to data, path to save output, etc. See samples in `./configs`.

### Train a model
To fine-tune a SlowFast network pretrained on Kinetics, use:
```commandline
python run_net.py --cfg ./configs/SLOWFAST_8x8_pretrained.yaml --pretrained
```
To fine-tune a network pretrained on SSv2, use:
```commandline
python run_net.py --cfg ./configs/SLOWFAST_8x8_pretrained.yaml --pretrained --ssv2
```

### Evaluate a model

To evaluate a model on it's training dataset (all splits) and plot ROC/PRC curves:
```commandline
python eval_net.py /path/to/experiment_dir/checkpoints ./configs/SLOWFAST_8x8_pretrained.yaml 
--epoch 49 --plot
```
This will also save a `.csv` with all predictions. 

To evaluate a model on an alternative test set (e.g., the extended test set):
```commandline
python eval_net /path/to/experiment_dir/checkpoints ./configs/SLOWFAST_8x8_pretrained.yaml 
--epoch 49 --plot --alt_testset_dir /path/to/your/test_set
```

###Acknowledgements:
This code is based on:
* [pySlowFast](https://github.com/facebookresearch/SlowFast/)
* [Pytorchvideo](https://pytorchvideo.org/) 

