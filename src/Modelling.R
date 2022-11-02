##############################################################
##############################################################
# Modelling
##############################################################
##############################################################

# libraries



# import data
rm(list = ls())
train <- read.csv('./data/gold_data/train.csv')
test_X <- read.csv('./data/gold_data/test.csv')


# # separate dependent and independent variables
train_X <- subset(train, select = -c(average_daily_rate))
train_y <- train$average_daily_rate

# inspect
str(train)
str(test_X)































































