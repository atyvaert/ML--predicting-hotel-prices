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
library(keras)

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
nrow(train)

#####@
# AAN TE PASSEN:
# 1) FOR EACH MODEL: DO HYPERPARAMETER TUNING ON TRAIN SET WITH CROSS VALIDATION
# 2) RETRAIN ON TRAIN SET WITH OPTIMAL PARAMETERS AND PREDICT ON VALIDATION SET
# 3) RETRAIN BEST-PERFORMING MODEL ON TRAIN + VAL SET TO PREDICT ON TEST SET


##############################################################
##############################################################
# Neural networks
##############################################################
##############################################################

# As these models become computationally very intensive, we set up parallel processing to speed
# up the process. Change number of clusters according to CPU
cluster <- makeCluster(detectCores()-1)
registerDoParallel(cluster)

# Besides, we also start saving our models so we do not have to run these models again

# you can close the parellel processing with the following code:
#stopCluster(cluster)


# Transform the data in the right format
x_train <- model.matrix(average_daily_rate ~ ., data = train)[, -1]
y_train <- train$average_daily_rate

x_val <- model.matrix(average_daily_rate ~ ., data = val)[, -1]
y_val <- val$average_daily_rate



# build the optimal neural network architecture
# The approach to finding the optimal neural network model will have some tweakable constants.
# We start with a number of layers equal to one, but we will increase this number later.
# Further, we have a minimum of 64 and a maximum of 256 neurons per layer. The step size between the number
# of neurons will be 64, giving a wide range of possibilities.
# https://towardsdatascience.com/how-to-find-optimal-neural-network-architecture-with-tensorflow-the-easy-way-50575a03d060

#####################################################################
#####################################################################
# IDEEEN EN TO DO:
# best model is momenteel 1 layer met 512 neurons: kijk naar meer neurons
# best run moet nog gefixt worden
# kijk naar andere mogelijkheden om overfitting tegen te gaan
# 1) Overfit:  speel met nr of hidden layers en nr of hidden units
# 2) Look at learning convergence
# => stel epochs in per model (soms veel soms weinig nodig)
# learning rate wordt hier ook bekeken bij ons via grid
# 3) regularization tuning parameters
# drop out  rate al gebruikt hier
# Stopping criteria for early stopping al gebruikt hier
# kijk ook nog eens naar de rest:
# => maximum value for maxnorm
# => lasso and ridge penalty
# maxnorm works well with dropout staat er in de slides
#####################################################################
#####################################################################


#####################################################################
#####################################################################
# STRUCTURE: First look at models how long it takes before reaching max
# look at validation curve to check learning convergence
# adjust the number of epochs in the file of the model
# play with the architecture (start wide and go narrow)
#####################################################################
#####################################################################



# Use this guide
# https://www.roelpeters.be/using-keras-in-r-hypertuning-a-model/


# In essence, hypertuning is done through flags. Anywhere in your neural network, 
# you can replace a parameter with a flag
FLAGS <- flags(
  flag_numeric('dropout1', 0.3),
  flag_integer('neurons1', 128),
  flag_integer('neurons2', 128),
  flag_integer('neurons3', 128),
  flag_numeric('l2', 0.001),
  flag_numeric('lr', 0.001),
  flag_numeric('maxnorm', 3)
)




##############
# wide model with 1 layer
#############

# In the following lines of code I define the possible values of all the parameters I want to hypertune. 
# This will produce a lot of possible combinations of parameters. 
# That’s why in the tuning_run() function, I specify I only want to try 10% of the possible combinations (sampled).
if(!require('tfruns')) install.packages('tfruns')
library(tfruns)
par <- list(
  dropout1 = c(0.3,0.4,0.5),
  neurons1 = c(64,128,256, 512, 700, 900),
  lr = c(0.0001,0.001,0.01),
  maxnorm1 = c(1,2, 3, 4)
)
runs1 <- tuning_run('./src/layer1_model.R', runs_dir = '_tuning', flags = par, sample = 0.4)


# Finally, I simply list all the runs, by referring to its running directory, where all the information 
# from the run is stored and I ask for it to be ordered according to the mean squared error.
ls_runs1(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning')

# Finally, I select the best model parameters and I train the model with it.
# The best model has a dropout value, 512 neurons and a learning rate of 0.001
best_run1 <- ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning')[1,]

# Run the best model again and save the model
layer1_model <- keras_model_sequential()
layer1_model %>%
  layer_dense(units = best_run1$flag_neurons1, units = FLAGS$neurons1, activation = "relu",
              input_shape = ncol(x_train),
              constraint_maxnorm(max_value = best_run1$maxnorm1, axis = 0)) %>%
  layer_dropout(rate = best_run1$flag_dropout1) %>%
  layer_dense(units = 1, activation = "linear")
layer1_model

layer1_model %>% compile(loss = "mse",
                   optimizer = optimizer_rmsprop(learning_rate = best_run1$lr),
                   metrics = list("mean_squared_error"))
history <- layer1_model %>%
  fit(x_train, y_train, epochs = 500, batch_size = 128,
      validation_split = 0.2,
      verbose = 1)

# predict on validation set
nn_run1_pred_val <- layer1 %>% predict(x_val)
score <- sqrt(mean((nn_run1_pred_val - val_y)^2))
score

# save
save(run1, file = './nn_models/layer1.Rdata')

##############
# less wide model but add a layer
#############
par <- list(
  dropout1 = c(0.3,0.4,0.5),
  neurons1 = c(32,64,128,256),
  neurons2 = c(16, 32,64,128),
  lr = c(0.0001,0.001,0.01),
  maxnorm1 = c(1,2, 3, 4)
)

# perform runs
runs2 <- tuning_run('./src/layer2_model.R', sample = 0.5, runs_dir = '_tuning', flags = par)


# Finally, I simply list all the runs, by referring to its running directory, where all the information 
# from the run is stored and I ask for it to be ordered according to the mean squared error.
ls_runs2(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning')

# Finally, I select the best model parameters and I train the model with it.
# The best model has a dropout value, 512 neurons and a learning rate of 0.001
best_run2 <- ls_runs2(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning')[1,]

# hier nog probleem: run the best model again lukt nog niet
# Run the best model again and save the model
layer2_model <- keras_model_sequential()
layer2_model %>%
  layer_dense(units = best_run2$neurons1, activation = "relu",
              input_shape = ncol(x_train),
              constraint_maxnorm(max_value = best_run2$maxnorm1, axis = 0)) %>%
  layer_dropout(rate = best_run2$dropout1) %>%
  layer_dense(units = best_run2$neurons2, activation = "relu",
              constraint_maxnorm(max_value = best_run2$maxnorm1, axis = 0)) %>%
  layer_dropout(rate = best_run2$dropout1) %>%
  layer_dense(units = 1, activation = "linear")
layer2_model

layer2_model %>% compile(loss = "mse",
                         optimizer = optimizer_rmsprop(learning_rate = best_run2$lr),
                         metrics = list("mean_squared_error"))
history <- layer2_model %>%
  fit(x_train, y_train, epochs = 500, batch_size = 128,
      validation_split = 0.2,
      verbose = 1)



# predict on validation set
#nn_run2_pred_val <- run2 %>% predict(x_val)
#score <- sqrt(mean((nn_run2_pred_val - val_y)^2))
#score
nn_run2_pred_val <- sqrt(best_run2$metric_val_mean_squared_error)
nn_run2_pred_val

# save
save(run2, file = './nn_models/layer2.Rdata')


#######################@
# to test the best model
#######################

# plot the best models
system.time(
  history <- modelnn %>%
    fit(x_train, y_train, epochs = 30, batch_size = 128,
        validation_split = 0.2))

history
plot(history, smooth = FALSE)


x_test <- as.matrix(test_X[, -1])





