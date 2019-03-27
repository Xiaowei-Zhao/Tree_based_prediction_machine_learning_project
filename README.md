# Tree_based_prediction_machine_learning_project

The overall goal is to predict whether a payment by a company to a medical doctor or facility
was made as part of a research project or not.
Data is downloaded from 2017 provided here:
https://www.cms.gov/OpenPayments/Explore-the-Data/Dataset-Downloads.html

Step 1 Identify Features
I assemble a dataset consisting of features and target (for example in a dataframe or in two
arrays X and y). I try to find features that are relevant for the prediction task or should be excluded because they leak the target information.
I also show visualizations or statistics to support the selection.

Step 2 Preprocessing and Baseline Model
I create a simple minimum viable model by doing an initial selection of features, doing
appropriate preprocessing and cross-validating a linear model. I exclude features or do simplified preprocessing for this task. 

Step 3 Feature Engineering
I create derived features and perform more in-depth preprocessing and data cleaning. 
I encode categorical variables using One-hot-encoding and Target-encoding.

Step 4 Model Selection
I tried different classification model such as trees, random forests, gradient boosting, SVM to improve
 my result. I change my preprocessing and feature engineering
to be suitable for the model.  I also tune parameters as appropriate.

Step 5 Feature Selections
I identify features that are important for my best model: Gradient Boosting. I find features that are most influential,
or could be removed without decrease in performance.

Step 6 Find an explainable model
I create an "explainable" model that is nearly as good as my best model
An explainable model should be small enough to be easily inspected - say a linear model with
few enough coefficients that you can reasonable look at all of them, or a tree with a small
number of leafs etc.