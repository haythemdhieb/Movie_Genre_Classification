# Movie_Genre_Classification
###In this project we aim to classify the genre of movie using a free public data set(https://github.com/ishmeetkohli/imdbGenreClassification/tree/master/data).
##Environement Setup
================================================================================
1- We use:
- [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
  to setup our environment,
- and python 3.7

Setup our environment:
```bash
conda --version
#clone the repo from
git clone https://github.com/haythemdhieb/Movie_Genre_Classification
cd Movie_Genre_Classification/
# Create a conda env
conda env create -f environment.yml
# Activate
conda activate MGC
# Update the environment
conda env update MGC --file environment.yml
```
## 1. Project structure (Script Mode)
================================================================================
This is the folder structure of the project:

```
README.md                            # a simple readme file
src/                                 # source code
   config.yaml                      # config file
   Data_loader.py                   # data loader module
   model.py                         # the architecture model
   movie_classifier.py              # the inference script
   train.py                         # training file
   utils.py                         # utility functions
input/
    trainingSet.csv
tests/
    test_processing.py

```

## 2. Running the workflow (Script Mode)
--------------------------------------------------------------------------------
To run an experiment on the optimization workflow:

1- ``` python src/train.py
```
2- To run an inference type this command ``` python python movie_classifier.py  --title="heat" --description="Hunters and their prey--Neil and his professional criminal crew hunt to score big money targets (banks, vaults, armored cars) and are, in turn, hunted by Lt. Vincent Hanna and his team of cops in the Robbery/Homicide police division. A botched job puts Hanna onto their trail while they regroup and try to put together one last big retirement score. Neil and Vincent are similar in many ways, including their troubled personal lives. At a crucial moment in his life, Neil disobeys the dictum taught to him long ago by his criminal mentor--'Never have anything in your life that you can't walk out on in thirty seconds flat, if you spot the heat coming around the corner'--as he falls in love. Thus the stage is set for the suspenseful ending...."
```
## 3. Testing
 ``` python tests/test_processing.py
```
