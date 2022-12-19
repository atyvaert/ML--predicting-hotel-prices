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
library(splines)
library(mgcv)
library(lightgbm)

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
# set in order to predict the test set. If the RMSE is very close for multiple models, it is possible that we 
# retrain more than 1 model.


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
# RMSE = 30.71916


##############################################################
# This function returns how many features we should use based on RMSE on the validation set
# this is used when performing subset selection
min_validation_error <- function(model) {
  val.mat <- model.matrix(average_daily_rate ~ ., data = val)
  
  val.errors <- rep(NA, ncol(train_X))
  for (i in 1:ncol(train_X)) {
    coefi <- coef(model, id = i)
    pred <- val.mat[, names(coefi)] %*% coefi
    val.errors[i] <- sqrt(mean((val_y - pred)^2))
  }
  return(which.min(val.errors))
}


##############################################################
# 1.2 Forward Stepwise selection
##############################################################

# perform forward stepwise selection and look at the results
regfit.full_for <- regsubsets(train$average_daily_rate ~ ., data = train, nvmax = ncol(train_X), really.big = T, method = "forward")
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
optimal_nr_predictors_forward =  min_validation_error(regfit.full_for) #95

# train the model on the training data and calculate the RMSE of the validation set
lm.cols.forward <- names(coef(regfit.full_for, optimal_nr_predictors_forward))[-1]
modeltrainmatrixforward <- cbind(train_X[lm.cols.forward], train_y)
best_model_forward = lm(train_y ~ ., data = modeltrainmatrixforward)

# make predictions on the validation set and calculate RMSE
forward_pred <- predict(best_model_forward, val_X)
sqrt(mean((forward_pred - val_y)^2))
#RMSE = 30.71786


##############################################################
# 1.3 Backward Stepwise selection 
##############################################################

# perform backwards stepwise selection and look at the results
regfit.full_back <- regsubsets(average_daily_rate ~ ., data = train, nvmax = ncol(train_X), really.big = T, method = "backward")
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
optimal_nr_predictors_backward =  min_validation_error(regfit.full_back) # 95

# train the model on the training data and calculate the RMSE of the validation set
# coef(regfit.full_back, optimal_nr_predictors_backward)
lm.cols.backward <- names(coef(regfit.full_back, optimal_nr_predictors_backward))[-1]
modeltrainmatrixbackward <- cbind(train_X[lm.cols.backward], train_y)
best_model_backward =  lm(train_y ~., data = modeltrainmatrixbackward)

# make predictions on the validation set and calculate RMSE
backward_pred <- predict(best_model_backward, val_X)
sqrt(mean((backward_pred - val_y)^2))
# RMSE = 30.71776


##############################################################
# 1.4 Sequential replacement Stepwise selection 
##############################################################

regfit.full_seq <- regsubsets(average_daily_rate ~ ., data = train, nvmax = ncol(train_X), really.big = T, method = "seqrep")
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
optimal_nr_predictors_seqrep =  min_validation_error(regfit.full_seq) #99

# train the model on the training data and calculate the RMSE of the validation set
# coef(regfit.full_seq, optimal_nr_predictors_seqrep)
lm.cols.seqrep <- names(coef(regfit.full_seq, optimal_nr_predictors_seqrep))[-1]
modeltrainmatrixseqrep <- cbind(train_X[lm.cols.seqrep], train_y)
best_model_seqrep =  lm(train_y ~., data = modeltrainmatrixseqrep)

# make predictions on the validation set and calculate RMSE
seqrep_pred <- predict(best_model_seqrep, val_X)
sqrt(mean((seqrep_pred - val_y)^2))
# RMSE = 30.70939


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

# train the model with best parameters and make predictions on validation data 
ridge.mod <- glmnet(train_X, train_y, alpha = 0, lambda = bestlam, standardize = F)
ridge_pred_val = predict(ridge.mod, s = bestlam, newx = as.matrix(val_X))
sqrt(mean((val_y - ridge_pred_val)^2))
# RMSE = 32.33805


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

# train the model with best parameters and make predictions on validation data 
lasso.mod <- glmnet(train_X, train_y, alpha = 0, lambda = bestlam, standardize = F)
lasso_pred_val = predict(lasso.mod, s = bestlam, newx = as.matrix(val_X))
sqrt(mean((val_y - lasso_pred_val)^2))
# RMSE = 30.71731


##############################################################
# 1.7 Retrain the best performing model of the linear models 
# and make predictions on the test set
##############################################################

# Best model: subset selection with sequential replacement
# RMSE = 30.70939

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
##############################################################
# 2 MOVING BEYOND LINEARITY
##############################################################
##############################################################
# We can only perform polynomial functions, splines and GAMs on numerical features
# First we will look at the numerical features
par(mfrow = c(1, 1))
plot(train$nr_adults, train$average_daily_rate, col = "gray")
plot(train$nr_weekdays, train$average_daily_rate, col = "gray")
plot(train$nr_weekenddays, train$average_daily_rate, col = "gray")
plot(train$special_requests, train$average_daily_rate, col = "gray")
plot(train$days_in_waiting_list, train$average_daily_rate, col = "gray")
plot(train$previous_bookings_not_canceled, train$average_daily_rate, col = "gray")
plot(train$previous_cancellations, train$average_daily_rate, col = "gray")
plot(train$car_parking_spaces, train$average_daily_rate, col = "gray")
plot(train$special_requests, train$average_daily_rate, col = "gray")
plot(train$time_between_arrival_cancel, train$average_daily_rate, col = "gray")
plot(train$lead_time, train$average_daily_rate, col = "gray")
# We can see from the plots, that after normalization only 'time between arrival and cancel' and 'lead time' are still numerical variables
# So we will only perform polynomial functions, splines and generalized additive models on these variables

##############################################################
# 2.1 Polynomial Regression
##############################################################
# We perform polynomial functions untill the 4th degree
poly1.rate <- lm(average_daily_rate ~., data = train)
poly2.rate <- lm(average_daily_rate ~. +  poly(lead_time, 2) +  poly(time_between_arrival_cancel, 2), data = train)
poly3.rate <- lm(average_daily_rate ~. +  poly(lead_time, 3) +  poly(time_between_arrival_cancel, 3), data = train)
poly4.rate <- lm(average_daily_rate ~. +  poly(lead_time, 4) +  poly(time_between_arrival_cancel, 4), data = train)
anova(poly1.rate, poly2.rate, poly3.rate, poly4.rate)
# from the anova table we can see that the 2nd degree model doesnt really improve the linear model, but the 3th improves the second
# so out of the 3 polynomial functions, the 3th degree function will be the best 
poly_rate <- poly3.rate
# We make predictions on the validation set, RMSE = 30.72
poly_pred_val <- predict(poly_rate, newdata = val_X)
sqrt(mean((poly_pred_val - val_y)^2))

##############################################################
# 2.2 Splines
##############################################################
# First we fit splines for our numerical variables, with different degrees of freedom 
spline1.rate <- lm(average_daily_rate ~., data = train)
spline4.rate <- lm(average_daily_rate ~. +  bs(lead_time, df = 4) +  bs(time_between_arrival_cancel, df = 4), data = train)
spline5.rate <- lm(average_daily_rate ~. +  bs(lead_time, df = 5) +  bs(time_between_arrival_cancel, df = 5), data = train)
spline6.rate <- lm(average_daily_rate ~. +  bs(lead_time, df = 6) +  bs(time_between_arrival_cancel, df = 6), data = train)
anova(spline1.rate, spline4.rate, spline5.rate, spline6.rate)
# From the anova table we can see that the spline with 5 df imporves the models with 4 df, but the model with 6 df doesn't impove the one with 5df
# so out of our splines the one with 5df will be the best
spline.rate <- spline5.rate
# We make predictions on the validation set
spline_pred_val <- predict(spline.rate, newdata = val_X)
sqrt(mean((spline_pred_val - val_y)^2))
# RMSE = 30.637

##############################################################
# 2.3 Generative additive models
##############################################################
# In the Gam function we cannot use the point to select all our independent variables 
# therefore we create a function that return a string that sums up all the independent variables 
gam_with_All_variables <- function(col_names){
  names = noquote(col_names)
  string = "average_daily_rate ~ "
  for(item in names){
    string = paste(string, item, " + ")
  }
  return(string)
}
# apply this function to train_x 
all_variables <- gam_with_All_variables(names(train_X))
all_variables
# we create 2 gam models, these are combinations of splines ant polynonial functions of the numerical variables
# the parameters that are used, are the optimal parameters derived from step 1.1 and 1.2
gam2 <- as.formula(paste(all_variables, " bs(lead_time, df = 5) + poly(time_between_arrival_cancel, 3)"))
gam3 <- as.formula(paste(all_variables, " poly(time_between_arrival_cancel, 3) + bs(lead_time, df = 5)"))
# we fit a linear regression and the two gam models
gam1.rate <- lm(average_daily_rate ~., data = train)
gam2.rate <- gam(gam2, data = train)
gam3.rate <- gam(gam3 , data = train)
anova(gam1.rate, gam2.rate, gam3.rate)
# From the anova table we can see that the second model is significantly better than the linear regression model
gam.rate <- gam2.rate
# we make predictions on the validation set
gam_pred_val <- predict(gam.rate, newdata = val_X)
sqrt(mean((gam_pred_val - val_y)^2))
# RMSE = 30.61176
##############################################################

##############################################################
# 2.4 Retrain the best performing model of non-linear models 
# and make predictions on the test set
##############################################################
# Train the model X on all the data (train + val) 
# The best performing model is spline 5
spline5.all <- lm(average_daily_rate ~. +  bs(lead_time, df = 5) +  bs(time_between_arrival_cancel, df = 5), data = train_and_val)
# make predictions, bv:
# make prediction on the test set and save
spline_pred_test <- predict(spline5.all, newdata = test_X)
# save 
spline_pred_df <- data.frame(id = as.integer(test_X$id),
                             average_daily_rate= spline_pred_test)
write.csv(spline_pred_df, file = "./data/sample_submission_Spline.csv", row.names = F)

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
# RMSE = 36.44005

##############################################################
# 1.2 Fit a regression tree  with cross validation
##############################################################
# fit a regression tree using cross validation
cv.rate <- cv.tree(tree.rate)

# plot the tree
plot(cv.rate$size, cv.rate$dev, type = 'b')
# We see that the most complex tree is chosen by cross validation
# We could prune the tree to make it less complex, but we do not think this is usefull here
# As the CV chooses 13 terminal nodes, this regression has the same results as 3.1


##############################################################
# 2. BAGGING
##############################################################

# 1) train the bagging model on the training data to do hyperparameter tuning
set.seed(1)
bagging.rate <- randomForest(average_daily_rate ~ ., data = train, mtry = ncol(train_X), ntrees = 150, importance = TRUE)

# save the model
save(bagging.rate, file = "models/bagging_model_training.Rdata")

# 2) We make predictions on the validation set, which results in an RMSE 
bagging_pred_val <- predict(bagging.rate, newdata = val_X)
sqrt(mean((bagging_pred_val - val_y)^2))
# score = 20.4
# RMSE = 


##############################################################
# 3. Random Forest
##############################################################

###############################################
# 3.1 Random Forest with standard parameters
###############################################
# Using p/3 variables in each tree is a good starting point when building a random forest of regression trees

# 1) train the bagging model on the training data to do hyperparameter tuning
set.seed(1)
rf.rate <- randomForest(average_daily_rate ~ ., data = train, mtry = 33,  ntree = 150, importance = TRUE)

# save the model
save(rf.rate, file = "models/rf_model_train.Rdata")

# 2) We make predictions on the validation set, which results in an RMSE 
rf_pred_val <- predict(rf.rate, newdata = val_X)
sqrt(mean((rf_pred_val - val_y)^2))
# 18.38
# RMSE = 


###############################################
# 3.2 random Forest with CV 
###############################################

#We tune over 3 values of interaction depth around p/3 (=34)
rfGrid <-  expand.grid(mtry = c(28, 31, 34, 37, 40))

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


###############################################
# 3.3 random Forest with adaptiveCV 
###############################################

#We set up for parallel processing, change number of clusters according to CPU cores
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)

# 1) train the random forest model on the training data to do hyperparameter tuning
set.seed(1)
trainControl <- trainControl(method = 'adaptive_cv',
                             number = 5,
                             repeats = 2,
                             adaptive = list(min = 5, alpha = 0.05, method = "gls", complete = TRUE),
                             verboseIter = TRUE,
                             search = 'random',
                             allowParallel = TRUE)

cv.rf.rate <- train(average_daily_rate ~ .,
                    data = train,
                    method = 'rf',
                    trControl = trainControl,
                    metric = 'RMSE',
                    tuneLength = 5,
                    verbose = TRUE
)

stopCluster(cluster)

# save the model
save(cv.rf.rate, file = "models/adaptive_cv_rf_model_train.Rdata")

# 2) We make predictions on the validation set, which results in an RMSE 
cv_rf_pred_val <- predict(cv.rf.rate, newdata = val_X)
sqrt(mean((cv_rf_pred_val - val_y)^2))
# RMSE = 


##############################################################
# 4. BOOSTING
##############################################################

##############################################################
# 4.1 Boosting standard model
##############################################################

#We set up for parallel processing, change number of clusters according to CPU cores
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)

# 1) train the boosting model on the training data to do hyperparameter tuning
boosting.rate <- gbm(average_daily_rate ~ ., data = train, distribution = "gaussian", n.trees = 5000, 
                     interaction.depth = 4, shrinkage = 0.2, verbose = F)

stopCluster(cluster)

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
# A) Randomized search
##########

#We set up for parallel processing, change number of clusters according to CPU cores
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)

# 1) train the boosting model on the training data to do hyperparameter tuning
trainControl <- trainControl(method = 'adaptive_cv',
                             number = 5,
                             repeats = 3,
                             adaptive = list(min = 5, alpha = 0.05, method = "gls", complete = TRUE),
                             verboseIter = TRUE,
                             search = 'random',
                             allowParallel = TRUE)

set.seed(1)
cv.boosting1.rate <- train(average_daily_rate ~ .,
                    data = train,
                    method = 'gbm',
                    trControl = trainControl,
                    metric = 'RMSE',
                    tuneLength = 5,
                    verbose = TRUE
)
stopCluster(cluster)


# save model so we do not have to run it time and time again 
save(cv.boosting1.rate, file = "models/adaptivecv_boosting_model_train.Rdata")

# 2) We make predictions on the validation set, which results in an RMSE 
cv_boosting1_pred_val <- predict(cv.boosting1.rate, newdata = val_X)
sqrt(mean((cv_boosting1_pred_val - val_y)^2))
RMSE = 

##########
# B) Grid search
##########

# Number of tested combinations is limited because of limited resources

# We set up for parallel processing, change number of clusters according to CPU cores
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)

# After running some other grids, we increased the values for interaction depth. This was our final grid: 

gbmGrid <-  expand.grid(interaction.depth = c(11, 13, 15), 
                        n.trees = 3000, 
                        shrinkage = c(0.01, 0.001),
                        n.minobsinnode = 20)

# 1) train the boosting model on the training data to do hyperparameter tuning
set.seed(1)
trainControl <- trainControl(method = 'cv', number = 3, verboseIter = TRUE, allowParallel = TRUE)
cv.boosting3.rate <- train(average_daily_rate ~ .,
                           data = train,
                           method = 'gbm',
                           trControl = trainControl,
                           metric = 'RMSE',
                           tuneGrid = gbmGrid
)

#close parallel
stopCluster(cluster)

# save model
save(cv.boosting3.rate, file = "models/cv_boosting_model_train.Rdata")

# 2) We make predictions on the validation set 
cv_boosting3_pred_val <- predict(cv.boosting3.rate, newdata = val_X)
sqrt(mean((cv_boosting3_pred_val - val_y)^2))


##############################################################
# 4.2 XGBoost
##############################################################

# to tune hyperparameters:
#https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

##########
# C) Adaptive_cv + random search
##########

# We set up for parallel processing, change number of clusters according to CPU cores
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)

trainControl <- trainControl(method = 'adaptive_cv',
                             number = 10,
                             repeats = 5,
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
                       tuneLength = 100,
                       verbose = TRUE
)

#close parallel
stopCluster(cluster)

# save model
save(xgb.tune.rate, file = "models/xgb_model_train.Rdata")

# 2) We make predictions on the validation set, which results in an RMSE 
XGB_pred_val <- predict(xgb.tune.rate, newdata = val_X)
sqrt(mean((XGB_pred_val - val_y)^2))
# RMSE = 17.44413
# 17.63677


##############################################################
# 5. Retrain the best performing model(s) of the decision trees 
# and make predictions on the test set
##############################################################

# Train the model X on all the data (train + val) 
load(file = "models/xgb_model_train_try.Rdata")
xgb.tune.rate$bestTune
xgb.rate.all <- xgboost(xgb.DMatrix(label = train_and_val_y, data = as.matrix(train_and_val_X)),
                        nrounds = xgb.tune.rate$bestTune$nrounds,
                        max_depth = xgb.tune.rate$bestTune$max_depth,
                        eta = xgb.tune.rate$bestTune$eta,
                        gamma = xgb.tune.rate$bestTune$gamma,
                        colsample_bytree = xgb.tune.rate$bestTune$colsample_bytree,
                        min_child_weight = xgb.tune.rate$bestTune$min_child_weight,
                        subsample = xgb.tune.rate$bestTune$subsample
)
#save model
save(xgb.rate.all, file="models/xgb_train_and_val.RData")

# make predictions, bv:
# make prediction on the test set and save
xgb_pred_test <- predict(xgb.rate.all, newdata = xgb.DMatrix(as.matrix(test_X[, -1])))


xgb_df <- data.frame(id = as.integer(test_X$id),
                     average_daily_rate= xgb_pred_test)
colnames(xgb_df)[2] <- 'average_daily_rate'
str(xgb_df)
# save submission file
write.csv(xgb_df, file = "./data/sample_submission_xgb.csv", row.names = F)



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
# RMSE = 36.42701

# 3) As this is the only model we make for SVM, we immediately train the model on all the data
set.seed(1)
svm.rate.all = svm(average_daily_rate ~ ., data = train_and_val, scale = FALSE)

# Make predictions on the test set
svm_pred_test <- predict(svm.rate.all, newdata = test_X)

SVM_reg_pred_df <- data.frame(id = as.integer(test_X$id),
                              average_daily_rate= svm_pred_test)


colnames(SVM_reg_pred_df)[2] <- 'average_daily_rate'
str(SVM_reg_pred_df)
SVM_reg_pred_df
# save submission file
write.csv(SVM_reg_pred_df, file = "./data/sample_submission_SupportVectorRegression.csv", row.names = F)
