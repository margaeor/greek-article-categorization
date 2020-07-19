
# Greek Article Categorization
In this project the task was to take Greek newspaper articles from enet 
dataset and categorize them using Natural Language Processing
into the following categories:
- Art
- Economy
- Creece
- Politics
- Sports
- World

The process followed in order to perform this task
is described in the image below:

![Preprocessing Steps](./presentation/images/process2.png?raw=true "Preprocessing Steps")

# Terms - Bigrams
By performing the preprocessing steps on our dataset
and by performing dimensionality reduction,
we were able to detect the features (terms and bigrams)
which are more important in determining the category
an article belongs to.
Those terms (and bigrams) are shown in the images below:

![Terms and Bigrams](./presentation/images/cloud.png?raw=true "Terms")

# Results
The classification accuracy for different models using PCA
as dimensionality reduction is shown below:

![Results](./presentation/images/results2.png?raw=true "Results")

# Running
In order to run this project you need tensorflow 1.8, numpy and matplotlib.
After installing those packages there are 2 scripts you can run:
1. `run_single_model.py` which performs dimensionality reduction using LDA or PCA, performs data transformation and then performs classification using one of the following models:
   	- NB
	- SVM
	- RandomForest
	- GMM
	- KNN
	- ANN
	- CNN

2. `model_param_plot.py` which has the following options:
   1. **Visualize bigrams**: Creates a word-cloud visualization of the most important bigrams.
   2. **Visualize terms**: Creates a word-cloud visualization of the most important terms.
   3. **Run model grid param search**: For every model (GMM, MEAN, RandomForest, SVM, KNN) it evaluates and displays the model accuracy for different sets of parameters.
   4. **Run all models kfold**: Runs every model using kfold cross validation and displays the accuracy in a common figure.
     
