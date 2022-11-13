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
#str(train)
#str(val)
#str(test_X)

##############################################################
# 0. Linear regression
##############################################################
# train the model on the training data
lm.fit <- lm(average_daily_rate ~ ., data = train)

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
  
  val.errors <- rep(NA, 96)
  for (i in 1:96) {
    coefi <- coef(model, id = i)
    pred <- val.mat[, names(coefi)] %*% coefi
    val.errors[i] <- sqrt(mean((val_y - pred)^2))
  }
  return(which.min(val.errors))
}


##############################################################
# 1. Forward Stepwise selection
##############################################################



#### DIT IS VAN VOOR DE DATA UPDATE DUS NU 102 EXPL VAR!!!

# perform forward stepwise selection and look at the results
regfit.full_for <- regsubsets(train$average_daily_rate ~ ., data = train, nvmax = 96, really.big = T, method = "forward")
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
optimal_nr_predictors_forward =  min_validation_error(regfit.full_for) #54

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
# 2. Backward Stepwise selection 
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
# 3. Sequential replacement Stepwise selection 
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
# 4. Ridge Regression
##############################################################

# look for the best lambda value to perform the ridge regression with 10- fold cross validation
# use all the available data as we perform cross validation
# First, transform the variables
x_train <- model.matrix(average_daily_rate ~., train_and_val)[,-1]

set.seed(1)
cv.out <- cv.glmnet(x_train, train_and_val_y, alpha = 0, standardize = F)
bestlam <- cv.out$lambda.min
bestlam
plot(cv.out) # Draw plot of training MSE as a function of lambda
# look at smallest RMSE 
sqrt(min(cv.out$cvm))

# train the model on all training data 
ridge.mod <- glmnet(x_train, train_and_val_y, alpha = 0, lambda = bestlam, standardize = F)

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
# 5. Lasso Regression
##############################################################
# look for the best lambda value to perform the ridge regression with 10- fold cross validation
# use all the available data as we perform cross validation
# First, transform the variables
x_train <- model.matrix(average_daily_rate ~., train_and_val)[,-1]

set.seed(1)
cv.out <- cv.glmnet(x_train, train_and_val_y, alpha = 1, standardize = F)
bestlam <- cv.out$lambda.min
bestlam
plot(cv.out) # Draw plot of training MSE as a function of lambda
# look at smallest RMSE (not sure if this is right)
sqrt(min(cv.out$cvm))

# train the model on the training data 
lasso.mod <- glmnet(x_train, train_and_val_y, alpha = 1, lambda = bestlam, standardize = F)

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




##############################################################
# 6. Regression Tree
##############################################################

# basic tree, no cv

# We fit the model
tree.rate <- tree(average_daily_rate ~ ., train_and_val, control=rpart.control(cp=.0001))
summary(tree.rate)
plot(tree.rate)
#We make predictions
tree_pred_test <- predict(tree.rate, newdata = test_X)
tree_preds_df <- data.frame(id = as.integer(test_X$id),
                             average_daily_rate= tree_pred_test)
colnames(tree_preds_df)[2] <- 'average_daily_rate'
str(tree_preds_df)
# save submission file
write.csv(tree_preds_df, file = "./data/sample_submission_tree.csv", row.names = F)


##############################################################
# 7. Regression Tree with CV
##############################################################





##############################################################
# 8. Bagging
##############################################################
# score = 20.4
set.seed(1)
baggingModel <- randomForest(average_daily_rate ~ ., data = train_and_val, mtry = 102,ntree = 110, importance = TRUE)
bagging_pred <- predict(baggingModel, newdata = test_X)

bagging_pred

bagging_preds_df = data.frame(id = as.integer(test_X$id),
                              average_daily_rate= bagging_pred)


colnames(bagging_preds_df)[2] <- 'average_daily_rate'
str(bagging_preds_df)
# save submission file
write.csv(bagging_preds_df, file = "./data/sample_submission_bagging.csv", row.names = F)



##############################################################
# 9. random Forest
##############################################################

###############################################
#9.1 random Forest with suboptimal parameters 
###############################################

#score = 21
# By default, randomForest() uses p/3 variables when building a random forest of regression trees
# By default, randomForest() uses sqrt(p) variables when building a random forest of classification trees



#build rf model
set.seed(1)
rf.model <- randomForest(average_daily_rate ~ ., data = train_and_val, mtry = 10,  ntree = 110, importance = TRUE)

#get predictions
rf.pred <- predict(rf.model, newdata = test_X)

rf.pred



rf_preds_df <- data.frame(id = as.integer(test_X$id),
                          average_daily_rate= rf.pred)


colnames(rf_preds_df)[2] <- 'average_daily_rate'
str(rf_preds_df)
# save submission file
write.csv(rf_preds_df, file = "./data/sample_submission_randomForest.csv", row.names = F)





###############################################
#9.2 random Forest with optimal parameters 
###############################################
# kan 34 eens proberen, is eig 102 pred var nu

#score = 19.5
# By default, randomForest() uses p/3 variables when building a random forest of regression trees


#build rf model
set.seed(1)
rf.model2 <- randomForest(average_daily_rate ~ ., data = train_and_val, mtry = 33,  ntree = 110, importance = TRUE)

#get predictions
rf.pred2 <- predict(rf.model2, newdata = test_X)

rf.pred2



rf_preds_df2 <- data.frame(id = as.integer(test_X$id),
                          average_daily_rate= rf.pred2)


colnames(rf_preds_df2)[2] <- 'average_daily_rate'
str(rf_preds_df2)
# save submission file
write.csv(rf_preds_df2, file = "./data/sample_submission_randomForest2.csv", row.names = F)





##############################################################
# 10. Boosting
##############################################################

boosting_model <- gbm(average_daily_rate ~ ., data = train_and_val, distribution = "gaussian", n.trees = 5000, interaction.depth = 4, shrinkage = 0.2, verbose = F)
boosting.pred <- predict(boosting_model, newdata = test_X, n.trees = 5000)
boosting_pred

boosting_df <- data.frame(id = as.integer(test_X$id),
                           average_daily_rate= boosting_pred)




colnames(boosting_df)[2] <- 'average_daily_rate'
str(boosting_df)
# save submission file
write.csv(boosting_df, file = "./data/sample_submission_boosting.csv", row.names = F)












