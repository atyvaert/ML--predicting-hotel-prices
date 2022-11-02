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
# 2. Ridge Regression
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
rigde_pred_test = predict(ridge.mod, s = bestlam, newx = as.matrix(test_X))

# predictions
ridge_preds_df <- data.frame(id = test_X$id,
                              average_daily_rate= rigde_pred_test)
# save submission file
write.csv(ridge_preds_df, file = "./data/sample_submission_ridge.csv", row.names = F)


##############################################################
# 2. Lasso Regression
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
lasso_pred_test = predict(lasso.mod, s = bestlam, newx = as.matrix(test_X))

# predictions
lasso_preds_df <- data.frame(id = test_X$id,
                             average_daily_rate= lasso_pred_test)
# save submission file
write.csv(lasso_preds_df, file = "./data/sample_submission_lasso.csv", row.names = F)























































