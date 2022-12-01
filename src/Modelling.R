##############################################################
##############################################################
# Modelling
##############################################################
##############################################################

# libraries
library(glmnet)
library(ISLR)
library(leaps)
library(tree)
library(gbm)
library(rpart) #for fitting decision trees
library(randomForest)
library(doParallel)
library(caret)
library(xgboost)
library(e1071)


# import data
rm(list = ls())
train <- read.csv('./data/gold_data/train.csv')
val <- read.csv('./data/gold_data/val.csv')
test_X <- read.csv('./data/gold_data/test.csv')

# separate dependent and independent variables for training and validation set
train_X <- subset(train, select = -c(average_daily_rate))
train_y <- train$average_daily_rate

val_X <- subset(val, select = -c(average_daily_rate))
val_y <- val$average_daily_rate


# create a dataset to train the final model on with the train and validation set combined
train_and_val <- rbind(train, val)
train_and_val_X <- subset(train_and_val, select = -c(average_daily_rate))
train_and_val_y <- train_and_val$average_daily_rate

# inspect
str(train)
str(val)
str(test_X)



##############################################################
##############################################################
# STRUCTURE OF OUR MODELS
##############################################################
##############################################################
# For this assignment, we choose to have a clear structure in order to find the best model
# First, we classify our models per category. Within each category, we predict the average daily rate
# of our dataset using multiple models. 

# Then, we assess the performance of our model by making predictions on the validation set. This way,
# we get the RMSE of each model.

# Then, at the end of the category, the best performing model is then retrained on the training and validation
# set in order to predict the test set. If the RMSE is very close for multiple models, it is possible than we 
# retrain more than 1 model.

# NOTE: The baseline models are an exception on this rule and there each model is trained
# on all data (train + validation set) after finding the best model.



##############################################################
##############################################################
# BASELINE MODELS
##############################################################
##############################################################


##############################################################
# 1.1 Linear regression
##############################################################
# train the model on the training data
lm.fit <- lm(average_daily_rate ~ ., data = train)
summary(lm.fit)
# make predictions on the validation set and calculate RMSE
linR_pred <- predict(lm.fit, val_X)
sqrt(mean((val_y - linR_pred)^2))

# train the model on all the available training data
lm.fit <- lm(average_daily_rate ~ ., data = train_and_val)

# make predictions on the test set and save in submission folder
linR_pred_test <- predict(lm.fit, test_X)
linR_preds_df <- data.frame(id = as.integer(test_X$id),
                               average_daily_rate= linR_pred_test)

#str(linR_preds_df)

# save submission file
write.csv(linR_preds_df, file = "./data/sample_submission_linR.csv", row.names = F)

##############################################################
#This function returns how many features we should use based on RMSE on the validation set
# this is used when performing subset selection
min_validation_error <- function(model) {
  val.mat <- model.matrix(average_daily_rate ~ ., data = val)
  
  val.errors <- rep(NA, 95)
  for (i in 1:95) {
    coefi <- coef(model, id = i)
    pred <- val.mat[, names(coefi)] %*% coefi
    val.errors[i] <- sqrt(mean((val_y - pred)^2))
  }
  return(which.min(val.errors))
}


##############################################################
# 1.2 Forward Stepwise selection
##############################################################

str(train)

#### DIT IS VAN VOOR DE DATA UPDATE DUS NU 102 EXPL VAR!!!

# perform forward stepwise selection and look at the results
regfit.full_for <- regsubsets(train$average_daily_rate ~ ., data = train, nvmax = 95, really.big = T, method = "forward")
regF.summary <- summary(regfit.full_for)
#regF.summary
#regF.summary$rsq
#regF.summary$adjr2
#regF.summary$rss

# plot the results
par(mfrow = c(2, 1))
plot(regF.summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(regF.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted Rsq", type = "l")

# look at the optimal number of parameters by applying the model on the validation set
# and looking for the minimal RMSE
optimal_nr_predictors_forward =  min_validation_error(regfit.full_for) #59

# train the model on the training data and calculate the RMSE of the validation set
# coef(regfit.full_for, optimal_nr_predictors_forward)
lm.cols.forward <- names(coef(regfit.full_for, optimal_nr_predictors_forward))[-1]
modeltrainmatrixforward <- cbind(train_X[lm.cols.forward], train_y)
best_model_forward = lm(train_y ~ ., data = modeltrainmatrixforward)

# make predictions on the validation set and calculate RMSE
forward_pred <- predict(best_model_forward, val_X)
sqrt(mean((forward_pred - val_y)^2))

# train the model with the optimal parameters to all available training data (train + val set)
model_train_matrix_forward <- cbind(train_and_val_X[lm.cols.forward], train_and_val_y)

best_model_forward = lm(train_and_val_y ~ ., data = model_train_matrix_forward)

# make predictions on the test set and save in submission folder
forward_pred_test <- predict(best_model_forward, test_X)
forward_preds_df <- data.frame(id = as.integer(test_X$id),
                               average_daily_rate= forward_pred_test)

# str(forward_preds_df)

# save submission file
write.csv(forward_preds_df, file = "./data/sample_submission_forwardsel.csv", row.names = F)


##############################################################
# 1.3 Backward Stepwise selection 
##############################################################

### same comment

# perform backwards stepwise selection and look at the results
regfit.full_back <- regsubsets(average_daily_rate ~ ., data = train, nvmax = 96, really.big = T, method = "backward")
regB.summary <- summary(regfit.full_back)
#regB.summary
#regB.summary$rsq
#regB.summary$adjr2
#regB.summary$rss

# plot the results
par(mfrow = c(2, 1))
plot(regB.summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(regB.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted Rsq", type = "l")

# look at the optimal number of parameters by applying the model on the validation set
# and looking for the minimal RMSE
optimal_nr_predictors_backward =  min_validation_error(regfit.full_back) # 54

# train the model on the training data and calculate the RMSE of the validation set
# coef(regfit.full_back, optimal_nr_predictors_backward)
lm.cols.backward <- names(coef(regfit.full_back, optimal_nr_predictors_backward))[-1]
modeltrainmatrixbackward <- cbind(train_X[lm.cols.backward], train_y)
best_model_backward =  lm(train_y ~., data = modeltrainmatrixbackward)

# make predictions on the validation set and calculate RMSE
backward_pred <- predict(best_model_backward, val_X)
sqrt(mean((backward_pred - val_y)^2))

# train the model with the optimal parameters to all available training data (train + val set)
model_train_matrix_backward <- cbind(train_and_val_X[lm.cols.backward], train_and_val_y)
best_model_backward =  lm(train_and_val_y ~., data = model_train_matrix_backward)

# predictions on the test set and save in submission folder
backward_pred_test <- predict(best_model_backward, test_X)
backward_preds_df <- data.frame(id = as.integer(test_X$id),
                                average_daily_rate= backward_pred_test)

# str(backward_preds_df)

# save submission file
write.csv(backward_preds_df, file = "./data/sample_submission_backwardsel.csv", row.names = F)

##############################################################
# 1.4 Sequential replacement Stepwise selection 
##############################################################

regfit.full_seq <- regsubsets(average_daily_rate ~ ., data = train, nvmax = 99, really.big = T, method = "seqrep")
regS.summary <- summary(regfit.full_seq)
#regS.summary
#regS.summary$rsq
#regS.summary$adjr2
#regS.summary$rss

# plot the results
par(mfrow = c(2, 1))
plot(regS.summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(regS.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted Rsq", type = "l")

# look at the optimal number of parameters by applying the model on the validation set
# and looking for the minimal RMSE
optimal_nr_predictors_seqrep =  min_validation_error(regfit.full_seq) #54

# train the model on the training data and calculate the RMSE of the validation set
# coef(regfit.full_seq, optimal_nr_predictors_seqrep)
lm.cols.seqrep <- names(coef(regfit.full_seq, optimal_nr_predictors_seqrep))[-1]
modeltrainmatrixseqrep <- cbind(train_X[lm.cols.seqrep], train_y)
best_model_seqrep =  lm(train_y ~., data = modeltrainmatrixseqrep)

# make predictions on the validation set and calculate RMSE
seqrep_pred <- predict(best_model_seqrep, val_X)
sqrt(mean((seqrep_pred - val_y)^2))


# train the model with the optimal parameters to all available training data (train + val set)
model_train_matrix_seqrep <- cbind(train_and_val_X[lm.cols.seqrep], train_and_val_y)
best_model_seqrep =  lm(train_and_val_y ~., data = model_train_matrix_seqrep)



# predictions on the test set and save in submission folder
seqrep_pred_test <- predict(best_model_seqrep, test_X)
seqrep_preds_df <- data.frame(id = as.integer(test_X$id),
                              average_daily_rate= seqrep_pred_test)

# save submission file
write.csv(seqrep_preds_df, file = "./data/sample_submission_seqrepsel.csv", row.names = F)


##############################################################
# 1.5 Ridge Regression
##############################################################

# look for the best lambda value to perform the ridge regression with 10- fold cross validation
# use the training data to do tune the parameters with cross validation
# First, transform the variables
x_train <- model.matrix(average_daily_rate ~., train)[,-1]
all_train <- model.matrix(average_daily_rate ~., train_and_val)[,-1]

set.seed(1)
cv.out <- cv.glmnet(x_train, train_y, alpha = 0, standardize = F)
bestlam <- cv.out$lambda.min
bestlam
plot(cv.out) # Draw plot of training MSE as a function of lambda
# look at smallest RMSE 
sqrt(min(cv.out$cvm))

# train the model on all training data 
ridge.mod <- glmnet(all_train, train_and_val_y, alpha = 0, lambda = bestlam, standardize = F)

# look at the coefficients of the model
# predict(ridge.mod, s = bestlam, type = 'coefficients')

# make predictions for the test set with the optimal lambda
rigde_pred_test = predict(ridge.mod, s = bestlam, newx = as.matrix(test_X[, -1]))

# predictions
ridge_preds_df <- data.frame(id = as.integer(test_X$id),
                              average_daily_rate= rigde_pred_test)
colnames(ridge_preds_df)[2] <- 'average_daily_rate'
str(ridge_preds_df)
# save submission file
write.csv(ridge_preds_df, file = "./data/sample_submission_ridge.csv", row.names = F)


##############################################################
# 1.6 Lasso Regression
##############################################################
# look for the best lambda value to perform the ridge regression with 10- fold cross validation
# use the training data to do tune the parameters with cross validation
# First, transform the variables
x_train <- model.matrix(average_daily_rate ~., train)[,-1]
all_train <- model.matrix(average_daily_rate ~., train_and_val)[,-1]

set.seed(1)
cv.out <- cv.glmnet(x_train, train_y, alpha = 1, standardize = F)
bestlam <- cv.out$lambda.min
bestlam
plot(cv.out) # Draw plot of training MSE as a function of lambda
# look at smallest RMSE (not sure if this is right)
sqrt(min(cv.out$cvm))

# train the model on the training data 
lasso.mod <- glmnet(all_train, train_and_val_y, alpha = 1, lambda = bestlam, standardize = F)

# look at the coefficients of the model
# predict(lasso.mod, s = bestlam, type = 'coefficients')

# make predictions for the test set with the optimal lambda
lasso_pred_test = predict(lasso.mod, s = bestlam, newx = as.matrix(test_X[, -1]))

# predictions
lasso_preds_df <- data.frame(id = as.integer(test_X$id),
                             average_daily_rate = lasso_pred_test)

colnames(lasso_preds_df)[2] <- 'average_daily_rate'
str(lasso_preds_df)
# save submission file
write.csv(lasso_preds_df, file = "./data/sample_submission_lasso.csv", row.names = F)


#####@
# AAN TE PASSEN:
# 1) FOR EACH MODEL: DO HYPERPARAMETER TUNING ON TRAIN SET WITH CROSS VALIDATION
# 2) RETRAIN ON TRAIN SET WITH OPTIMAL PARAMETERS AND PREDICT ON VALIDATION SET
# 3) RETRAIN BEST-PERFORMING MODEL ON TRAIN + VAL SET TO PREDICT ON TEST SET



##############################################################
##############################################################
# MOVING BEYOND LINEARITY
##############################################################
##############################################################

##############################################################
# EERST NOG POLYNOMIALS, SPLINES...: TELLEN DUS OOK AANPASSEN

# @Viktor
# 1) FOR EACH MODEL: DO HYPERPARAMETER TUNING ON TRAIN SET WITH CROSS VALIDATION
# 2) RETRAIN ON TRAIN SET WITH OPTIMAL PARAMETERS AND PREDICT ON VALIDATION SET
# 3) RETRAIN BEST-PERFORMING MODEL ON TRAIN + VAL SET TO PREDICT ON TEST SET
##############################################################



##############################################################
##############################################################
# TREE-BASED METHODES
##############################################################
##############################################################

##############################################################
##############################################################
# 1. FIT A REGRESSION TREE
##############################################################
##############################################################

##############################################################
# 1.1 Fit a standard regression tree
##############################################################
# 1) We train the model on the training data, we see that the MSE is 1791 (RMSE = 42.3)
# Besides, we see that have 13 terminal nodes
tree.rate <- tree(average_daily_rate ~ ., train)
summary(tree.rate)

# visualize the tree
plot(tree.rate)
#text(tree.rate, pretty = 0)

# 2) We make predictions on the validation set, which results in an RMSE of 36.4
tree_pred_val <- predict(tree.rate, newdata = val_X)
sqrt(mean((tree_pred_val - val_y)^2))

##############################################################
# 1.2 Fit a regression tree  with cross validation
##############################################################
# fit a regression tree using cross validation
cv.rate <- cv.tree(tree.rate)

# plot the tree
plot(cv.rate$size, cv.rate$dev, type = 'b')
# We see that the most comlex tree is chosen by cross validation
# We could prune the tree to make it less complex, but we do not think this is usefull here
# As the CV chooses 13 terminal nodes, this regression has the same results as 3.1


##############################################################
# 2. BAGGING
##############################################################

# score = 20.4

# As these models become computationally very intensive, we set up parallel processing to speed
# up the process. Change number of clusters according to CPU
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)

# Besides, we also start saving our models so we do not have to run these models again

# you can close the parellel processing with the following code:
#stopCluster(cluster)

# 1) train the bagging model on the training data to do hyperparameter tuning
# WARNING: this takes some time to run
set.seed(1)
bagging.rate <- randomForest(average_daily_rate ~ ., data = train, mtry = 95, importance = TRUE)

# save the model
save(bagging.rate, file = "models/bagging_model_training.Rdata")

# 2) We make predictions on the validation set, which results in an RMSE 
bagging_pred_val <- predict(bagging.rate, newdata = val_X)
sqrt(mean((bagging_pred_val - val_y)^2))



##############################################################
# 3. Random Forest
##############################################################

###############################################
# 3.1 random Forest with standard parameters
###############################################
#score = 19.5
# By default, randomForest() uses p/3 variables when building a random forest of regression trees


# 1) train the bagging model on the training data to do hyperparameter tuning
set.seed(1)
rf.rate <- randomForest(average_daily_rate ~ ., data = train, mtry = 33,  ntree = 110, importance = TRUE)

# save the model
save(rf.rate, file = "models/rf_model_train.Rdata")

# 2) We make predictions on the validation set, which results in an RMSE 
rf_pred_val <- predict(rf.rate, newdata = val_X)
sqrt(mean((rf_pred_val - val_y)^2))

###############################################
# 3.2 random Forest with CV 
###############################################
#score = 19.5

#We tune over 3 values of interaction depth
# TO DO: KIJKEN NAAR BESTE VALUE EN ERROND EXTRA GRID OF RANDOM SEARCH @ Simon
rfGrid <-  expand.grid(mtry = c(20, 34, 40))

# 1) train the random forest model on the training data to do hyperparameter tuning
set.seed(1)
trainControl <- trainControl(method = 'cv', number = 5, verboseIter = TRUE, allowParallel = TRUE)
cv.rf.rate <- train(average_daily_rate ~ .,
                    data = train,
                    method = 'rf',
                    trControl = trainControl,
                    metric = 'RMSE',
                    tuneGrid = rfGrid
)

# save the model
save(cv.rf.rate, file = "models/cv_rf_model_train.Rdata")

# 2) We make predictions on the validation set, which results in an RMSE 
cv_rf_pred_val <- predict(cv.rf.rate, newdata = val_X)
sqrt(mean((cv_rf_pred_val - val_y)^2))


##############################################################
# 4. BOOSTING
##############################################################

##############################################################
# 4.1 Boosting standard model
##############################################################

# 1) train the boosting model on the training data to do hyperparameter tuning
boosting.rate <- gbm(average_daily_rate ~ ., data = train, distribution = "gaussian", n.trees = 5000, 
                     interaction.depth = 4, shrinkage = 0.2, verbose = F)

# save the model
save(boosting.rate, file = "models/boosting_model_train.Rdata")

# 2) We make predictions on the validation set, which results in an RMSE 
boosting_pred_val <- predict(boosting.rate, newdata = val_X)
sqrt(mean((boosting_pred_val - val_y)^2))




##############################################################
# 4.2 Boosting with cross validation
##############################################################

# TO DO: BETTER USE OF RANDOM SEARCH INSTEAD OF ALL THESE GRID SEARCHES
# IF GRID SEARCH: ALSO LOOK WHAT WAS BEST VALUE AND NEXT GRID SEARCH MUST
# BE CLOSER TO THIS VALUE TO FIND VALUE AROUND OPTIMAL, WEET DAT JE DIT GEDAAN 
# HEBT MAAR BEST OOK VERMELDEN IN DE CODE

##########
# A) First sequence of tuning parameters
##########

# First, we tune over 3 values of interaction depth when building the model
gbmGrid <-  expand.grid(interaction.depth = c(1, 4, 6), 
                        n.trees = c(1000, 1500, 2000), 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)

# 1) train the boosting model on the training data to do hyperparameter tuning
set.seed(1)
trainControl <- trainControl(method = 'cv', number = 3, verboseIter = TRUE, allowParallel = TRUE)
cv.boosting1.rate <- train(average_daily_rate ~ .,
                           data = train_and_val,
                           method = 'gbm',
                           trControl = trainControl,
                           metric = 'RMSE',
                           tuneGrid = gbmGrid
)

# save model so we do not have to run it time and time again 
save(cv.boosting1.rate, file = "models/cv_boosting1_model_train.Rdata")

# 2) We make predictions on the validation set, which results in an RMSE 
cv_boosting1_pred_val <- predict(cv.boosting1.rate, newdata = val_X)
sqrt(mean((cv_boosting1_pred_val - val_y)^2))


##########
# B) Second sequence of tuning parameters
##########

# Next, we tune over 3 values of interaction depth
gbmGrid <-  expand.grid(interaction.depth = c(7, 9, 11), 
                        n.trees = c(1000, 1500, 2000), 
                        shrinkage = 0.05,
                        n.minobsinnode = 20)

# 1) train the boosting model on the training data to do hyperparameter tuning
set.seed(1)
trainControl <- trainControl(method = 'cv', number = 4, verboseIter = TRUE, allowParallel = TRUE)
cv.boosting2.rate <- train(average_daily_rate ~ .,
                           data = train,
                           method = 'gbm',
                           trControl = trainControl,
                           metric = 'RMSE',
                           tuneGrid = gbmGrid
)

# save model 
save(cv.boosting2.rate, file = "models/cv_boosting2_model_train.Rdata")

# 2) We make predictions on the validation set, which results in an RMSE 
cv_boosting2_pred_val <- predict(cv.boosting2.rate, newdata = val_X)
sqrt(mean((cv_boosting2_pred_val - val_y)^2))


##########
# C) Third sequence of tuning parameters
##########

# Next, we tune over 3 values of interaction depth
gbmGrid <-  expand.grid(interaction.depth = c(11, 13, 15), 
                        n.trees = 3000, 
                        shrinkage = c(0.01, 0.001),
                        n.minobsinnode = 20)

# 1) train the boosting model on the training data to do hyperparameter tuning
set.seed(1)
trainControl <- trainControl(method = 'cv', number = 4, verboseIter = TRUE, allowParallel = TRUE)
cv.boosting3.rate <- train(average_daily_rate ~ .,
                   data = train,
                   method = 'gbm',
                   trControl = trainControl,
                   metric = 'RMSE',
                   tuneGrid = gbmGrid
)

# save model
save(cv.boosting3.rate, file = "models/cv_boosting3_model_train.Rdata")

# 2) We make predictions on the validation set, which results in an RMSE 
cv_boosting3_pred_val <- predict(cv.boosting3.rate, newdata = val_X)
sqrt(mean((cv_boosting3_pred_val - val_y)^2))


##############################################################
# 4.2 XGBoost
##############################################################

#hyperparameters:
#https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/  
# lambda (regul) default, tree depth (default 6),
# pruning param: gamma (compare with gain) 5?, 
# nthreas = detectcores-11
# eta 0.01-0.2

#We set up for parallel processing, change number of clusters according to CPU cores
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)

##########
# A) First sequence of grid parameters
##########
XGBgrid1 <-  expand.grid(nrounds = c(500, 1000, 1500), 
                        max_depth = 6, 
                        eta = 0.05,
                        gamma = 0,
                        colsample_bytree = 1,
                        min_child_weight = 1,
                        subsample = 1)
#result: 1500, 6


##########
# B) Second sequence of grid parameters
##########
XGBgrid2 <-  expand.grid(nrounds = c(1500, 2000), 
                        max_depth = c(6, 8), 
                        eta = c(0.05, 0.01),
                        gamma = 0,
                        colsample_bytree = 1,
                        min_child_weight = 1,
                        subsample = 1)

set.seed(1)
trainControl <- trainControl(method = 'cv', number = 3, verboseIter = TRUE, allowParallel = TRUE)


xgb_tune <- train(x = train_and_val_X,
                  y = train_and_val_y,
                  method = 'xgbTree',
                  trControl = trainControl,
                  metric = 'RMSE',
                  tuneGrid = XGBgrid2,
                  verbose = TRUE
)

##########
# C) Adaptive_cv + random search
##########
trainControl <- trainControl(method = 'adaptive_cv',
                             number = 10,
                             repeats = 10,
                             adaptive = list(min = 5, alpha = 0.05, method = "gls", complete = TRUE),
                             verboseIter = TRUE,
                             search = 'random',
                             allowParallel = TRUE)

# 1) train the XG boosting model on the training data to do hyperparameter tuning
set.seed(1)
xgb.tune.rate <- train(x = train_X,
                  y = train_y,
                  method = 'xgbTree',
                  trControl = trainControl,
                  metric = 'RMSE',
                  tuneLength = 25,
                  verbose = TRUE
)

#close parallel
stopCluster(cluster)

# save model
save(xgb_tune, file = "models/xgb_model4_train.Rdata")

# 2) We make predictions on the validation set, which results in an RMSE 
XGB_pred_val <- predict(xgb.tune.rate, newdata = val_X)
sqrt(mean((XGB_pred_val - val_y)^2))


##############################################################
# 4.3 Adaboost
##############################################################

##############################################################
# 5. Retrain the best performing model(s) of the decision trees 
# and make predictions on the test set
##############################################################
# Train the model X on all the data (train + val) 

# make predictions, bv:
# make prediction on the test set and save
#cv_boosting1_pred_test <- predict(cv.boosting1.rate.all, newdata = test_X, n.trees = 1000)

# VERGEET MODEL NIET TE SAVEN ZODAT JE NIET OPNIEUW MOET RUNNEN (NIET ENKEL PREDICTIONS)

# save 
#cv_boosting1_df <- data.frame(id = as.integer(test_X$id),
#                              average_daily_rate= cv_boosting1_pred_test)


#colnames(cv_boosting1_df)[2] <- 'average_daily_rate'
#str(cv_boosting1_df)
# save submission file
#write.csv(cv_boosting1_df, file = "./data/sample_submission_boosting.csv", row.names = F)



##############################################################
# SUPPORT VECTOR MACHINES
##############################################################

#36 very bas

##############################################################
# 1. Perform standard regression with SVM
##############################################################

# 1) train the SVM model on the training data to do hyperparameter tuning
set.seed(1)
svm.rate = svm(average_daily_rate ~ ., data = train, scale = FALSE)

# save model
save(svm.rate, file = "models/svm_model_train.Rdata")

# 2) We make predictions on the validation set, which results in an RMSE 
svm_pred_val <- predict(svm.rate, newdata = val_X)
sqrt(mean((svm_pred_val - val_y)^2))

# 3) As this is the only model we make for SVM, we immediately train the model on all the data
set.seed(1)
svm.rate.all = svm(average_daily_rate ~ ., data = train_and_val, scale = FALSE)

# Make predictions on the test set
svm_pred_test <- predict(svm.rate.all, newdata = val_X)

SVM_reg_pred_df <- data.frame(id = as.integer(test_X$id),
                          average_daily_rate= svm_pred_test)


colnames(SVM_reg_pred_df)[2] <- 'average_daily_rate'
str(SVM_reg_pred_df)
SVM_reg_pred_df
# save submission file
write.csv(SVM_reg_pred_df, file = "./data/sample_submission_SupportVectorRegression.csv", row.names = F)
