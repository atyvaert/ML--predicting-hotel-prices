Hotel room price prediction

Team 10 : Simon De Lange, Artur Tyvaert, Viktor Vandenbulcke, Stijn Van Ruymbeke

---------------------------------------------------------------------------------------------------
INTRODUCTION
---------------------------------------------------------------------------------------------------

This project aims at creating predictive models to be able to predict the average daily rates of hotel rooms. 
This allows hotel managers to (dynamically) set the hotel room prices at an optimal level and thus maximize
there profits. In this README the authors will give an overview of the files present in this folder and what purpose 
these files serve. It will also explained how the users have to use these files. 

---------------------------------------------------------------------------------------------------
OVERVIEW OF FOLDERS AND FILES 
---------------------------------------------------------------------------------------------------

TEAM10_TECHNICAL_REPORT (pdf)

SRC (r files):
- Data_exploration
- Data_cleaning 
- Feature engineering
- Feature engineering2
- Feature_engineering_PCA
- Modelling 
- Neural_networks 
- LayerX_model

TUNINGX (folders)

MODELS (r workspace)

DATA (csv files)
- Bronze_data 
- Silver_data
- Gold_data 
- Sample_Submission

---------------------------------------------------------------------------------------------------
USING THE FILES 
---------------------------------------------------------------------------------------------------

To be able to understand the reasoning behind some of the decisions that were made, it's important that the user 
goes through the R files in the same order as the creators of the files. So in this section it will be stated in what 
order to go through the files. Firstly go through the Data_exploration file, secondly the Data_cleaning file and lastly 
the Feature_engineering file. After these three files are run the order of the other files doesn't matter anymore, 
although it might be better to first go through the Modelling file and afterwards the Neural networks file. Because the 
Modelling file contains simpler models and the neural networks are quite complex. 

The file "Feature_engineering_PCA" contains extra code where the creators used PCA to reduce the number of numerical variables.
They didn't end up using this as this only reduced the number of features by 2 and didn't increase predictive performance on the validation set. 

---------------------------------------------------------------------------------------------------
PURPOSE OF THE FILES
---------------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------------
TEAM10_TECHNICAL_REPORT
---------------------------------------------------------------------------------------------------

This is a technical report of the project. In this report the findings of the project are discussed in further detail, while using the CRISP-DM framework.

---------------------------------------------------------------------------------------------------
DATA_EXPLORATION 
---------------------------------------------------------------------------------------------------

In this R file as first glance is taken at the data that was given to us. This is done by plotting barplots, 
boxplots and histograms, depending on whether the variables is categorical or numerical. The presence of missing 
values was checked as well. The insights that were derived from this are necessary for the next steps of 
our data preprocessing. 

---------------------------------------------------------------------------------------------------
DATA_CLEANING 
---------------------------------------------------------------------------------------------------

In this R file missing values, outliers and regular expression were treated. When the missing 
values were detected, these were imputed by the mean (numerical variables) and the mode (categorical 
variables) computed on the training set. For some cases missing values corresponded to a value of 0, which was derived from the data exploration, 
in these cases the missing values were imputed by 0. Next to imputing these missing values, flag variables were created to indicate whether a certain variable had missing values as missing information can provide insights. Furthermore, outliers of the training data were handled based on their 
z-score. For some exceptions outlier boundaries were set arbitrary based on the data exploration. Lastly, the dates were parsed for all the datasets and some timing variables were extracted.

---------------------------------------------------------------------------------------------------
FEATURE_ENGINEERING
---------------------------------------------------------------------------------------------------

In this file feature engineering is performed. Feature engineering is the step in which features are created 
from the data, these features are created to improve the performance of the models. Our data contained a lot of 
categorical variables, for these variables we had to create dummy variables. To not have to many variables,
a maximum cardinality of 10 was imposed for most of these variables. 
3 additional variables were created as well:
- room_type_conflict: equals 1 if the assigned room is not equal to the reserved room type.
- time_between_arrival_checkout and time_between_arrival_cancel: difference in days between arrival and
checkout (positive) or cancellation (negative). 
- nr_weekdays and nr_weekenddays: decomposes nr_nights in weekdays and weeekend days. 

Next to that, variables that have a high correlation were deleted. Lastly, log transformations were performed
on some variables to make them less skewed and reduce the range of their data. 

---------------------------------------------------------------------------------------------------
FEATURE_ENGINEERING2
---------------------------------------------------------------------------------------------------
Same as Feature_engineering, but the variable day_of_month_arrival is added. The authors created a separate file
because all information in the report is on the first version of FE. We only decided to add this variable because of the 
better model performance.

---------------------------------------------------------------------------------------------------
FEATURE_ENGINEERING_PCA
---------------------------------------------------------------------------------------------------

Principal Component Analysis was tried to reduce the number of numerical features,
but this only reduced the number of features by 2 and didn't increase predictive performance on the validation set.
All submissions were based on the original "Feature_engineering" file.

---------------------------------------------------------------------------------------------------
MODELLING
---------------------------------------------------------------------------------------------------

In this section multiple models are fitted on our data. To do this in an efficient way, the training set is split  
into a train set and validation set, with 80% and 20% of the observations respectively. The new train 
set was used to train the model and the validation set was used to make predictions and to compute the RMSE of 
each model. Based on the RMSE the best performing models(s) of each category were selected and trained on 
the complete train set (train + validation). Afterwards the test set was used to make predictions. 

Now an overview of the used models will be given, the models are subdivided into linear and non-linear 
models 

1 LINEAR MODELS 

1.1 LINEAR REGRESSION 
A linear model was used to regress average_daily_rate on all other variables of the gold data. This model 
resulted in RMSE of 30.71916 which served as a benchmark for this project.

1.2 STEPWISE SELECTION
Stepwise selection is a procedure that constructs models, only with the best features.

1.2.1 FORWARD STEPWISE SELECTION
This method is a selection method that adds features once at a time starting with the most significant feature. 
By using an evaluation metric it can identified how much parameters the optimal model has. In our case this is 95
parameters. A model is fitted using these parameters. This resulted in a RMSE of 30.71786. 

1.2.2 BACKWARD STEPWISE SELECTION 
Backward stepwise selection is similar to forward stepwise selection, but with the opposite approach. It starts
with a model with all features and removes one feature at a time, starting with the least significant. By using 
an evaluation metric it can be identified how much parameters the optimal model has. In our case this is 95 parameters.
A model can be fitted using these parameters. This model resulted in a RMSE of 30.71776. 

1.2.3 SEQUENTIAL REPLACEMENT SELECTION
Sequential replacement selection is a combination of both forward stepwise selection and backward stepwise 
selection. At each steps features can either be removed or added based on their impact on the evaluation metric. 
It was identified that in our case the optimal number of parameters was 99 and a model was fitted using these parameters. 
This model resulted in a RMSE of 30.70939. 

1.3 SHRINKAGE METHODS
Shrinkage methods constrain or shrink the coefficient estimates. To control the shrinking, a hyperparameter (λ) 
and a penalty term are added to the minimization problem. 

1.3.1 RIDGE REGRESSION
In ridge regression the penalty term (l2 norm) is defined as the square of the Euclidean norm of the coefficients. 
The optimal λ was found by performing 10-fold cross validation and used this λ to train or ridge model. This resulted 
in a RMSE of 32.33805. 

1.3.2 LASSO REGRESSION 
The difference between lasso and ridge is that for lasso the penalty term is defined as the sum of the absolute
values of the model coefficients. Here the same method was as for ridge to find the optimal model. This resulted
in a RMSE of 30.71731. 

2 NON-LINEAR MODELS 

2.1 NON-LINEAR TRANSFORMATIONS
In this section non-linear transformations were performed on numerical variables. 

2.1.1 POLYNOMIAL REGRESSION 
Polynomial regression models the relation between the independent and dependent variables as nth degree polynomial. 
Models were constructed that contained all our variables and were augmented with polynomial terms of our numerical 
variables, until the 4th degree. These models were compared in an anova table and found that the model till the 3th 
degree had the best performance. This model resulted in a RMSE of 30.72.

2.1.2 SPLINES 
Splines are piecewise polynomials with smoothness conditions. Data is typically split into sections, defined by knots. 
Because in our case the knots cannot be defined, so degrees of freedom (df) is taken as parameter. 
The same is done as with polynomials but the models are augmented with splines with df = 4, 5, 6. The models are compared
in an anova table and the model with 5 df performs best. This model resulted in a RMSE of 30.72

2.1.3 GENERALIZED ADDITIVE MODELS (GAMs)
GAMs extend standard linear models by allowing non-linear functions, while maintaining additivity. Two GAMs were constructed 
that contained all variables and a combination of polynomials and splines of our 
numerical variables. By comparing these models in an anova table it was found that they had similar performance. 
With RMSE equal to 30.61176.

2.2 TREE BASED MODELS

2.2.1 REGRESSION TREES 
A simple single decision tree was fitted. As expected the performance of this model on the validation set
was rather low, with a RMSE of 36.44005

2.2.2 BAGGING 
Bagging is an ensemble technique in which multiple trees samples and the resulting predictions are averaged 
to make the final prediction. A RMSE of 21.83 was obtained which is a big improvement over the previous one. 

2.2.3 XGBOOST
XGBoost is a highly scalable implementation of gradient boosted decision trees. 
To maximize the performance of our model hyperparameter tuning was performed, which resulted in the following parameters:
- nrounds: 775
- max_depth: 10
- eta: 0.1095579
- gamma: 8.648076
- colsample_bytree: 0.47439.6
- min_child_weight: 16 
- subsample: 0.77999961

The model with the optimal parameters resulted in a RMSE of 17.44413 on the validation set. 

2.2.4 RANDOM FOREST
Random forest is an extension of bagging, here a split only considers a random sample of (mtry) of the p predictors. 
As a consequence, the trees are decorrelated. Because the authors thought random forest would have high predictive power, 
different RF models were created 
Model 1 was created with mtry = p/3 (default) predictors and ntree = 150. This resulted in a RMSE of 18.59
Model 2 was run with 5-fold cross validation. A grid was used to tun mtry, mtry = c(28, 31, 34, 37, 40). 
This resulted in a RMSE of 18.61, which is higher than model 1, what's quite surprising. 

2.3 SUPPORT VECTOR MACHINES 
Support vector machines are mainly used for classification purposes, but work relatively well in regression
tasks. Therefore the authors decided to create a basic SVM. Which resulted in a quit high RMSE of 36.42701

---------------------------------------------------------------------------------------------------
NEURAL_NETWORKS
---------------------------------------------------------------------------------------------------
To find the optimal neural network architecture, different architectures were build. The authors build models with one up until four hidden layers. Next, different parameters were tuned for these models with a grid search, including the number of hidden units per layer. The models were constructed with the following guide:

Architecture:
Our approach to tune the parameters started by creating a wide model with few layers, evolving to less wide but deeper models that have more model complexity. So for the one layer model, a grid search for values from 128 to 640 neurons was performed. For the models with multiple layers, there was also a grid search, but the number of neurons were gradually decreased for each layer in the model. 

Architecture regularization:
A dropout layer was added after each regular layer to reduce overfitting. An optimal dropout rate was found by performing a grid search with drop-out values of 0.3 and 0.4. A maxnorm constraint was used as well, this technique reduces overfitting by limiting the maximum Euclidean norm of weights of the network. The evaluated values were 0.5, 1, 2 and 3. Lastly an early stopping criterion was used as well. If the RMSE did not improve for 20 iterations, the training process was interrupted and the current model was used as the final model. 

Learning convergence:
The learning rate of our model was tuned by performing a grid search using values 0.01 and 0.001. The learning rate determines how quickly or slowly the model learns. Further, the batch size was set to 128. This parameter determines the number of observations used for each gradient step. Lastly, the number of epochs used to train each model had to be determined. This was done by visually checking the convergence rate of the models while it was running. For the one- and two-layer models, 200 epochs were used. For the three and four layered models, 150 epochs were used. When the optimal model was found for each model, the optimal model was retrained using 200 epochs as the one and two layered models still had incremental improvements after 200 epochs.

After training all the models, the authors found that the one layer model with 512 neurons, dropout value = 0.4, maxnorm constraint = 1 and learning rate of 0.001, had the best performance with a RMSE of 22.26. For the other models the performance decreased substantially, with 2 layer = 51.49 , 3 layer =  35.88 and 4 layer = 32.19. This decrease in performance could be due to the fact that too much complexity was added to our models. Therefore, it is logical that the one-layer models yields the best performance. 

---------------------------------------------------------------------------------------------------
LAYERX_MODEL AND TUNINGX
---------------------------------------------------------------------------------------------------
In order to train the different neural networks, multiple R-files were used. First,
the main file ‘Neural_networks’ defines the grid search parameters for each model. Furthermore, there are different R-scripts in which we define each model: layer1_model,layer2_model. . . When performing a tuning run, the main file uses these R-scripts to train a particular model with the corresponding grid search values. While tuning the models, the results are stored in the tuning directory of each layer: _tuning, _tuning2, _tuning3 and _tuning4

---------------------------------------------------------------------------------------------------
MODELS
---------------------------------------------------------------------------------------------------

Because some models had to train for a long time, these models were as a workspace object after they 
were trained. In this way the user is able to load these models into their environment if retraining would be 
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
This folder contains the train, test and validation set of our data after feature engineering was 
performed in the feature_engineering file. This data will be used in the modelling and Neural network file.

--------------------------------------------------------------------------------------------------
SAMPLE_SUBMISSIONS 
---------------------------------------------------------------------------------------------------
In the sample_submissions folder there are also multiple CSV files that have the following naming convention:
sample_submission_nameModel. These sample submissions contain the predictions on our test set of the 
models with the best performance on the validation set for each category. These are the files that 
were handed in, in the Kaggle competition (only the best performing models).








