# ECE 6502: Tensors for Data Science Final Project
This repository contains all the necessary code
and data needed to reproduce the analysis I performed for the final project.
In the sections below,
I will detail how once can generate my results.

# Installing the Correct Dependencies
All of the denoted by the `pyproject.toml` file.
Please take the following steps to initialize the repository.

1. `git clone https://github.com/zblanks/tensor_completion.git` to your desired location
2. Ensure you have Python and Poetry installed on your machine
    * See https://python-poetry.org/docs/ for instructions on how to get Poetry installed
3. Navigate to the directory containing the `tensor_completion` repository
4. `poetry install`
    * This will build the dependencies 
    and install the pvals package in the virtual environment
5. `poetry shell`
    * This spawns a .venv shell allowing you to run these experiments

# Re-Mapping Arrivals and Departures using K-Means Clustering
To get the arrivals and departures data to belong to a finite alphabet,
we used k-means clustering with five clusters.
We selected five using the "elbow method,"
but of course you are able to choose your own value as you desire.
To train and then save these re-mapped values,
use the following commands:

`python kmeans.py --nclusters $nclusters --count_type {arrivals, departures}`

where $nclusters denotes the desired number of clusters,
and count_type corresponds to either clustering the arrivals or departures data.
By default the code saves both the clustering error and the labels,
but one can also adjust these options at the command line.


# Training the CPD, Neural Network, and Random Forest Models
To compare the statistical performance of the three models,
you need to train them multiple times using a bootstrapped dataset.
For the CPD model,
you cans specify the random seed to bootstrap the training data through the command:

`python cpd.py --seed $seed`

where $seed denotes the value you would like to input (e.g., 17).
The `nn.py` script has the same format.
By default the random forest model runs and scores itself $20$ times,
but one can easily update the code to modify this behavior.

# Performing the T-Test to Compare Model Results
After training the CPD, Neural Network, and Random Forest models multiple times,
ideally somewhere in the range of $15$ to $20$ (we did $20$, but you can do more),
you are now ready to perform the one-sided t-test to compare the out-of-sample performance.
To do this,
for the CPD and NN models, run

`python model_results.py --model_type {cpd, nn}`

where you either specify using a CPD or neural network model.
This command will go through all of the stored models in the `/models` directory,
and score their out-of-sample performance.
Recall that by default the random forest model does this automaticlaly,
and does not save the model to disk.
After saving these results to disk you are now ready to perform the t-test.
Do this by entering the command:

`python t_test.py`

This script goes through all of the available models,
and performs a one-sided t-test for comparing the CPD approach to the neural network
and the random forest,
and prints the results.

And that's it!
I summarized all of the results in the final report,
but feel free to play around with the various settings,
e.g., rank, sequence length, etc.,
to see how that affects the results.
Everything I've included should be self-contained and readily reproducible.
