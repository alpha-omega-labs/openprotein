# Installation of OpenProtein from scratch for 3070, 3080, 3090 GPU

## Lambda AI stack install
https://lambdalabs.com/blog/install-tensorflow-and-pytorch-on-rtx-30-series

- 1. Install clean Ubuntu 20.04 LTS (Focal Fossa)
- 2. Install Lambda (~4GB to download, ~13GB installed)
<pre>
LAMBDA_REPO=$(mktemp) && \
wget -O${LAMBDA_REPO} https://lambdalabs.com/static/misc/lambda-stack-repo.deb && \
sudo dpkg -i ${LAMBDA_REPO} && rm -f ${LAMBDA_REPO} && \
sudo apt-get update && sudo apt-get install -y lambda-stack-cuda
</pre>
- 3. reboot

## OpenProtein installation

- 1. <code> git clone https://github.com/alpha-omega-labs/openprotein.git</code>
- 2. 
<pre>
cd openprotein
git clone https://github.com/JeppeHallgren/pytorch-crf.git
cd pytorch-crf
pip install -U Flask
pip install -U flask-cors
pip install PeptideBuilder
pip install requests
pip install -r requirements.txt
pip install -e .
cd ..
sudo apt-get install npm
python3 __main__.py --use-gpu --experiment-id deepprotein --minibatch-size 32 --learning-rate 0.001 --min-updates 16000
</pre>

# OpenProtein

This is a fork from openprotein, A PyTorch framework for tertiary protein structure prediction.


# Additions to the OpenProtein
- CASP7 dataset from [ProteinNet](https://github.com/aqlaboratory/proteinnet) is added.
- A deep residual CNN architecture is added

# Run:
you can use the following commands:
`__main__.py --use-gpu --evaluate-on-test --experiment-id deepprotein --minibatch-size 6 --learning-rate 0.001 --min-updates 14000`

##

![Alt text](examplemodelrun.png?raw=true "OpenProtein")


## Getting started
To run this project, simply git clone the repository, install dependencies using `pipenv install` and then type `pipenv run python __main__.py` in the terminal to run the sample experiment:
```
$ pipenv run python __main__.py
------------------------
--- OpenProtein v0.1 ---
------------------------
Live plot deactivated, see output folder for plot.
Starting pre-processing of raw data...
Preprocessed file for testing.txt already exists.
force_pre_processing_overwrite flag set to True, overwriting old file...
Processing raw data file testing.txt
Wrote output to 81 proteins to data/preprocessed/testing.txt.hdf5
Completed pre-processing.
2018-09-27 19:27:34: Train loss: -781787.696391812
2018-09-27 19:27:35: Loss time: 1.8300042152404785 Grad time: 0.5147676467895508
...
```

## Developing a Predictive Model
See `models.py` for examples of how to create your own model. 

## Using a Predictive Model
See `prediction.py` for examples of how to use pre-trained models. 

## Memory Usage
OpenProtein includes a preprocessing tool (`preprocessing.py`) which will transform the standard ProteinNet format into a hdf5 file and save it in `data/preprocessed/`. This is done in a memory-efficient way (line-by-line). 

The OpenProtein PyTorch data loader is memory optimized too - when reading the hdf5 file it will only load the samples needed for each minibatch into memory.

## License
Please see the LICENSE file in the root directory.
