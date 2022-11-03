##############################################################
##############################################################
# Modelling
##############################################################
##############################################################

# libraries
library(glmnet)


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

# inspect
#str(train)
#str(val)
#str(test_X)


##############################################################
# 1. Forward Stepwise selection
##############################################################

regfit.full_for <- regsubsets(train$average_daily_rate ~ ., data = train, nvmax = 99, really.big = T, method = "forward")
regF.summary <- summary(regfit.full_for)
regF.summary
regF.summary$rsq
regF.summary$adjr2
regF.summary$rss

par(mfrow = c(2, 1))
plot(regF.summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(regF.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted Rsq", type = "l")

max_r_squared_forward = max(regF.summary$adjr2) # 0.5593856
optimal_nr_predictors_forward =  which.max(regF.summary$adjr2) #82
optimal_nr_predictors_forward

coef(regfit.full_for, optimal_nr_predictors_forward)
lm.cols.forward <- names(coef(regfit.full_for, optimal_nr_predictors_forward))[-1]

modeltrainmatrixforward <- cbind(train_X[lm.cols.forward], train_y)

best_model_forward = lm(train_y ~ ., data = modeltrainmatrixforward)
forward_pred <- predict(best_model_forward, val_X)

# calculate the RMSE
sqrt(mean((forward_pred - val_y)^2))

# predictions
forward_pred_test <- predict(best_model_forward, test_X)
forward_preds_df <- data.frame(id = test_X$id,
                               average_daily_rate= forward_pred_test)
forward_preds_df$id <- as.integer(forward_preds_df$id)
# save submission file
write.csv(forward_preds_df, file = "./data/sample_submission_forwardsel.csv", row.names = F)


##############################################################
# 2. Backward Stepwise selection 
##############################################################

regfit.full_back <- regsubsets(average_daily_rate ~ ., data = train, nvmax = 99, really.big = T, method = "backward")
regB.summary <- summary(regfit.full_back)
regB.summary
regB.summary$rsq
regB.summary$adjr2
regB.summary$rss

par(mfrow = c(2, 1))
plot(regB.summary$rss, xlab = "Number of Variables", ylab = "RSS", type = "l")
plot(regB.summary$adjr2, xlab = "Number of Variables", ylab = "Adjusted Rsq", type = "l")

max_r_squared_backward = max(regB.summary$adjr2)
optimal_nr_predictors_backward =  which.max(regB.summary$adjr2) 

coef(regfit.full_back, optimal_nr_predictors_backward)
lm.cols.backward <- names(coef(regfit.full_back, optimal_nr_predictors_backward))[-1]

modeltrainmatrixbackward <- cbind(train_X[lm.cols.backward], train_y)


best_model_backward =  lm(train_y ~., data = modeltrainmatrixbackward)
backward_pred <- predict(best_model_backward, val_X)
# calculate the RMSE
sqrt(mean((backward_pred - val_y)^2))

# predictions
backward_pred_test <- predict(best_model_backward, test_X)
backward_preds_df <- data.frame(id = test_X$id,
                                average_daily_rate= backward_pred_test)
backward_preds_df$id <- as.integer(backward_preds_df$id)
# save submission file
write.csv(backward_preds_df, file = "./data/sample_submission_backwardsel.csv", row.names = F)


##############################################################
# 3. Ridge Regression
##############################################################
# transform the variables
x_train <- model.matrix(average_daily_rate ~., train)[,-1]
x_val <- model.matrix(average_daily_rate ~., train)[,-1]

# look for the best lambda value to perform the ridge regression with 10- fold cross validation
set.seed(1)
cv.out <- cv.glmnet(x_train, train_y, alpha = 0, standardize = F)
bestlam <- cv.out$lambda.min
bestlam
plot(cv.out) # Draw plot of training MSE as a function of lambda

# train the model on the training data 
ridge.mod <- glmnet(x_train, train_y, alpha = 0, lambda = bestlam, standardize = F)

# look at the coefficients of the model
# predict(ridge.mod, s = bestlam, type = 'coefficients')

# make predictions for the validation set with the optimal lambda
ridge_pred = predict(ridge.mod, s = bestlam, newx = x_val)

# calculate the RMSE
sqrt(mean((ridge_pred - val_y)^2))

# make predictions for the test set with the optimal lambda
rigde_pred_test = predict(ridge.mod, s = bestlam, newx = as.matrix(test_X[, -1]))

# predictions
ridge_preds_df <- data.frame(id = as.integer(test_X$id),
                              average_daily_rate= rigde_pred_test)
colnames(ridge_preds_df)[2] <- 'average_daily_rate'
ridge_preds_df
# save submission file
write.csv(ridge_preds_df, file = "./data/sample_submission_ridge.csv", row.names = F)


##############################################################
# 4. Lasso Regression
##############################################################
# transform the variables
x_train <- model.matrix(average_daily_rate ~., train)[,-1]
x_val <- model.matrix(average_daily_rate ~., train)[,-1]

# look for the best lambda value to perform the ridge regression with 10- fold cross validation
set.seed(1)
cv.out <- cv.glmnet(x_train, train_y, alpha = 1, standardize = F)
bestlam <- cv.out$lambda.min
bestlam
plot(cv.out) # Draw plot of training MSE as a function of lambda

# train the model on the training data 
lasso.mod <- glmnet(x_train, train_y, alpha = 1, lambda = bestlam, standardize = F)

# look at the coefficients of the model
# predict(lasso.mod, s = bestlam, type = 'coefficients')

# make predictions for the validation set with the optimal lambda
lasso_pred = predict(lasso.mod, s = bestlam, newx = x_val)

# calculate the RMSE
sqrt(mean((lasso_pred - val_y)^2))

# make predictions for the test set with the optimal lambda
lasso_pred_test = predict(lasso.mod, s = bestlam, newx = as.matrix(test_X[, -1]))

# predictions
lasso_preds_df <- data.frame(id = as.integer(test_X$id),
                             average_daily_rate = lasso_pred_test)

colnames(lasso_preds_df)[2] <- 'average_daily_rate'
lasso_preds_df
# save submission file
write.csv(lasso_preds_df, file = "./data/sample_submission_lasso.csv", row.names = F)






















































