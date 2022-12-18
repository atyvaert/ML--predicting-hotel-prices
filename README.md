Hotel room price prediction

Team 10 : Simon De Lange, Artur Tyvaert, Viktor Vandenbulcke, Stijn Van Ruymbeke
---------------------------------------------------------------------------------------------------
INTRODUCTION
---------------------------------------------------------------------------------------------------

This project aims at creating predictive models to be albe to predict the average daily rates of hotel rooms. 
This allows hotel managers to (dynamically) set the hotel room prices at an optimal level and thus maximize
there profits. In this README we will give an overview of the files present in this folder and what purpose 
these files serve. We will also explain how the users has to use these files. 

---------------------------------------------------------------------------------------------------
OVERVIEW OF FOLDERS AND FILES 
---------------------------------------------------------------------------------------------------

SRC (r files):
- Data_exploration
- Data_cleaning 
- Feature engineering
- Modelling 
- Neural_networks 
- LayerX_model

MODELS (r workspace)

DATA (csv files)
- Bronze_data 
- Silver_data
- Gold_data 
- Sample Submission

Tuning

---------------------------------------------------------------------------------------------------
USING THE FILES 
---------------------------------------------------------------------------------------------------


To be able to understand the reasoning behind some of the decisions that were made, it's important that the user 
goes through the R files in the same order as the creators of the files. So in this section we will state in what 
order to go through the files. Firstly go through the Data_exploration file, secondly the Data_cleaning file and lastly 
the Feature_engineering file. After these three files are run the order of the other files doesn't matter anymore, 
although it might be better to first go through the Modelling file and afterwards the Nearal networks file. Because the 
Modelling file contains simpler models and the neural networks are quite complex. 


---------------------------------------------------------------------------------------------------
PURPOSE OF THE FILES
---------------------------------------------------------------------------------------------------
DATA_EXPLORATION 
---------------------------------------------------------------------------------------------------


In this R file we take a first glance at the data that was given to us. We do this by plotting barplots, 
boxplots and histograms, depending on whether the variables is categrorical or numerical. We also checked
the presence of missing values. The insights that we derive from this are necessary for the next steps of 
our data preprocessing. 

---------------------------------------------------------------------------------------------------
DATA_CLEANING 
---------------------------------------------------------------------------------------------------


In this R file we missing values, outliers and regular expression were treated. To treat missing 
values in an effictient way, we created a function that was able to detect missing values. When the missing 
values were detected, these were imputed by the mean for numerical variables and the model for categorical 
variables. For some cases missing values corresponded to a value of 0, which was derived from the data exploration, 
in these cases the missing values were imputed by 0. Next to imputing these missing values we also created flag 
functions that created flag varaibles to indicate whether a certain variable was missing, as the absence of some 
variables could be insightfull. Furthermore, functions were created to detect and handle outliers based on their 
z-score. For some exceptions outlier boundaries were set arbitrary based on the data exploration. Lastly, some data 
variables were transformed into useful parts using regular expressions. 

---------------------------------------------------------------------------------------------------
FEATURE_ENGINEERING
---------------------------------------------------------------------------------------------------




---------------------------------------------------------------------------------------------------
MODELLING
---------------------------------------------------------------------------------------------------


In this section we fit multiple models on our data. To do this in an efficient way, we splitted the training 
set into a train set and validation set, with 80% and 20% of the observations respectively. The new train 
set was used to train the model and the validation set was used to make predictions and to compute the RMSE of 
each model. Based on the RMSE the best performing models(s) of each category were selected and trained on 
the complete train set (train + validation). Afterwards the test set was used to make predictions. 

Now we wil give an overview of the used models, the models are subdivided into linear and non-linear 
models 

1 LINEAR MODELS 


1.1. STEPWISE SELECTION
Stepwise selection is a procedure that constructs models, only with the best features.

1.1.1 FORWARD STEPWISE SELECTION
This method is a selection method that adds features once at a time starting with the most significant feature. 
By using an evaluation metric we can identify how much parameters the optimal model has. In our case this is X 
parameters. We fit a model using these parameters. 

1.1.2 BACKWARD STEPWISE SELECTION 
Backward stepwise selection is similar to forward stepwise selection, but with the opposite approach. It starts
with a model with all features and removes one feature at a time, starting with the least significant. By using 
an evaluation metric we can identify how much parameters the optimal model has. In our case this is X parameters.
We fit a model using these parameters. 

1.1.3 SEQUENTIAL REPLACEMENT SELECTION
Sequential replacement selection is a combination of both forward stepwise selection and backward stepwise 
selection. At each steps features can either be removed or added based on their impact on the evaluation metric. 
We identified that in our case the optimal number of parameters was X and we can fit a model using thes parameters. 

1.2 SHRINKAGE METHODS
Shrinkage methods constrain or shrinksthe coefficient estimates. To control the shrinking, a hyperparameter (λ) 
and a penalty term are added to the minimization problem. 

1.2.1 RIDGE REGRESSION
In ridge regression the penalty term (l2 norm) is defined as the spuare of the Euclidean norm of the coefficients. 
We found the omptimal λ by performing 10-fold cross validation and used this λ to train or ridge model. 

1.2.2 LASSO REGRESSION 
The difference between lasso and ridge is that for lasso the penalty term is defined as the sum of the absolute
values of the model coefficients. Here we used the same method as for ridge to find the optimal model. 


2 NON-LINEAR MODELS 


2.1 NON-LINEAR TRANSFORMATIONS
In this section we perform non-linear transformations on numerical variables. 

2.1.1 POLYNOMIAL REGRESSION 
Polynomial regression models the relation between the independent and dependent variables as nth degree polynomial. 
We constructed models that contained all our varaibles and augmented them with polynomial terms of our numerical 
variables, untill the 4th degree. We compared these models in an anova table and found that the model till the 3th 
degree had the best performance. 

2.1.2 SPLINES 
Splines are piecewise polynomials with smoothness conditions. Data is typically split into sections, defined by knots. 
Because in our case we cannot define the knots we make use of the degrees of freedom (df) parmameters. 
We do the same as with polynomials but augment or models with splines with df = 4, 5, 6. We compare the models 
in an anova table and find the model with 5 df performs best. 

2.1.3 GENERALIZED ADDITIVE MODELS (GAMs)
GAMs extend standard linear models by allowing non linear functions, while maintaining additivity. We 
constructed two GAMs that contained all variables and a combination of polynomials and splines of our 
numerical variables. We found by comparing these models in an anova table that they had similar performance. 

2.2 TREE BASED MODELS

2.2.1 REGRESSION TREES 

2.2.2 BAGGING 

2.2.3 RANDOM FORESTS 
In Random Forests, a large number of trees is grown where a split only considers a random sample of m predictors.

2.3 SUPPORT VECTOR MACHINES 

---------------------------------------------------------------------------------------------------
NEURAL_NETWORKS
---------------------------------------------------------------------------------------------------


---------------------------------------------------------------------------------------------------
LAYERX_MODEL
---------------------------------------------------------------------------------------------------


---------------------------------------------------------------------------------------------------
MODELS
---------------------------------------------------------------------------------------------------


Because some models had to train for a long time, we stored these models as a workspace object after they 
were trained. In this way the user is able to load these models into the workspace if retraining would be 
necessary. The folder called models contains all these workspace objects of the models. 

---------------------------------------------------------------------------------------------------
BRONZE DATA 
---------------------------------------------------------------------------------------------------


This folder contains our test an train dataset. These datasets are the sets containing the raw data 
that were given to us for the assignment. No preprocessing steps have been applied to this data. 
This data will be used in the data_exploration and data_cleaning file.

--------------------------------------------------------------------------------------------------
SILVER DATA 
---------------------------------------------------------------------------------------------------


This folder contains the train, test and validation set of our data after it was cleaned in the 
data_cleaning file. This data will be used in the feature_engineering file. 

-------------------------------------------------------------------------------------------------
GOLD DATA 
---------------------------------------------------------------------------------------------------


This folder contains the train, test and validation set of our data after feature engineerig was 
performed in the feature_engineering file. This data will be used in the modelling and Neural network file.

--------------------------------------------------------------------------------------------------
SAMPLE SUBMISSIONS 
---------------------------------------------------------------------------------------------------


In the X folder there are also multiple CSV files that have the following naming convention:
sample_submission_nameModel. These sample submissions contain the predictions on our test set of the 
models with the best performance on the validation set for each category. These are the files that 
were handed in, in the Kaggle competition (only the best performing models). 






