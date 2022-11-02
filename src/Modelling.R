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
str(train)
str(val)
str(test_X)


##############################################################
# 2. Ridge Regression
##############################################################
# transform the variables



























































