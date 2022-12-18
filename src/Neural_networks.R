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

x_train_and_val <-model.matrix(average_daily_rate ~ ., data = train_and_val)[, -1]
y_train_and_val <- train_and_val_y

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



#####################################################################
#####################################################################
# 1. ONE LAYER MODEL
# build a wide model that overfits the data and regularize with dropout and maxnorm
#####################################################################
#####################################################################

# In the following lines of code I define the possible values of all the parameters I want to hypertune. 
# This will produce a lot of possible combinations of parameters. 
# That’s why in the tuning_run() function, I specify I only want to try 10% of the possible combinations (sampled).
if(!require('tfruns')) install.packages('tfruns')
library(tfruns)
par <- list(
  dropout1 = c(0.3,0.4),
  neurons1 = c(128,256, 384, 512, 640),
  lr = c(0.0001,0.001),
  maxnorm1 = c(0.5, 1,2, 3)
)
runs1 <- tuning_run('./src/layer1_model.R', runs_dir = '_tuning', flags = par, sample = 0.4)


# Finally, I simply list all the runs, by referring to its running directory, where all the information 
# from the run is stored and I ask for it to be ordered according to the mean squared error.
ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning')

# Finally, I select the best model parameters and I train the model with it.
# The best model has a dropout value of 0.4, 512 neurons and a learning rate of 0.001 and a maxnorm of 1
best_run1 <- ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning')[1,]

# Run the best model again and save the model
layer1_model <- keras_model_sequential()
layer1_model %>%
  layer_dense(units = best_run1$flag_neurons1, activation = "relu",
              input_shape = ncol(x_train),
              constraint_maxnorm(max_value = best_run1$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run1$flag_dropout1) %>%
  layer_dense(units = 1, activation = "linear")
layer1_model

layer1_model %>% compile(loss = "mse",
                   optimizer = optimizer_rmsprop(learning_rate = best_run1$flag_lr),
                   metrics = list("mean_squared_error"))

# define early stop monitor
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

# Fit the model and store training stats
history <- layer1_model %>%
  fit(x_train, y_train, epochs = 300, batch_size = 128,
      validation_split = 0.2,
      verbose = 1,
      callbacks = list(early_stop))

# predict on validation set
nn_run1_pred_val <- layer1_model %>% predict(x_val)
score1 <- sqrt(mean((nn_run1_pred_val - val_y)^2))
score1

# save
save(layer1_model, file = './nn_models/layer1.Rdata')

#####################################################################
#####################################################################
# 2. TWO LAYER MODEL
# build a wide model that overfits the data and regularize with dropout and maxnorm
#####################################################################
#####################################################################
par <- list(
  dropout1 = c(0.3,0.4),
  neurons1 = c(64,128,256, 384, 512),
  neurons2 = c(32,64,128, 256),
  lr = c(0.001,0.01),
  maxnorm1 = c(0.5,1,2, 3)
)

# perform runs
runs2 <- tuning_run('./src/layer2_model.R', sample = 0.3, runs_dir = '_tuning2', flags = par)


# Finally, I simply list all the runs, by referring to its running directory, where all the information 
# from the run is stored and I ask for it to be ordered according to the mean squared error.
ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning2')

# Finally, I select the best model parameters and I train the model with it.
# The best model has a dropout value of 0.3, 256 neurons in the first layer, 64 in the second and a learning rate of 0.001
best_run2 <- ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning2')[1,]

# hier nog probleem: run the best model again lukt nog niet
# Run the best model again and save the model
layer2_model <- keras_model_sequential()
layer2_model %>%
  layer_dense(units = best_run2$flag_neurons1, activation = "relu",
              input_shape = ncol(x_train),
              constraint_maxnorm(max_value = best_run2$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run2$flag_dropout1) %>%
  layer_dense(units = best_run2$flag_neurons2, activation = "relu",
              constraint_maxnorm(max_value = best_run2$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run2$flag_dropout1) %>%
  layer_dense(units = 1, activation = "linear")
layer2_model

layer2_model %>% compile(loss = "mse",
                         optimizer = optimizer_rmsprop(learning_rate = best_run2$flag_lr),
                         metrics = list("mean_squared_error"))
# define early stop monitor
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

# Fit the model and store training stats
history <- layer2_model %>%
  fit(x_train, y_train, epochs = 300, batch_size = 128,
      validation_split = 0.2,
      verbose = 1,
      callbacks = list(early_stop))



# predict on validation set
nn_layer2_pred_val <- layer2_model %>% predict(x_val)
score2 <- sqrt(mean((nn_layer2_pred_val - val_y)^2))
score2

# save
save(layer2_model, file = './nn_models/layer2.Rdata')

#####################################################################
#####################################################################
# 3. THREE LAYER MODEL
# build a model that is balanced between width and depth
#####################################################################
#####################################################################
par <- list(
  dropout1 = c(0.3,0.4),
  neurons1 = c(64,128, 256, 384),
  neurons2 = c(16, 32,64, 128),
  neurons3 = c(8, 16, 32, 64),
  lr = c(0.001,0.01),
  maxnorm1 = c(0.5,1,2, 3)
)

# perform runs
runs3 <- tuning_run('./src/layer3_model.R', sample = 0.3, runs_dir = '_tuning3', flags = par)


# Finally, I simply list all the runs, by referring to its running directory, where all the information 
# from the run is stored and I ask for it to be ordered according to the mean squared error.
ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning3')

# Finally, I select the best model parameters and I train the model with it.
# The best model has a dropout value, 512 neurons and a learning rate of 0.001
best_run3 <- ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning3')[1,]

# hier nog probleem: run the best model again lukt nog niet
# Run the best model again and save the model
layer3_model <- keras_model_sequential()
layer3_model %>%
  layer_dense(units = best_run3$flag_neurons1, activation = "relu",
              input_shape = ncol(x_train),
              constraint_maxnorm(max_value = best_run3$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run3$flag_dropout1) %>%
  layer_dense(units = best_run3$flag_neurons2, activation = "relu",
              constraint_maxnorm(max_value = best_run3$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run3$flag_dropout1) %>%
  layer_dense(units = best_run3$flag_neurons2, activation = "relu",
              constraint_maxnorm(max_value = best_run3$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run3$flag_dropout1) %>%
  layer_dense(units = 1, activation = "linear")
layer3_model

layer3_model %>% compile(loss = "mse",
                         optimizer = optimizer_rmsprop(learning_rate = best_run3$flag_lr),
                         metrics = list("mean_squared_error"))
# define early stop monitor
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

# Fit the model and store training stats
history <- layer3_model %>%
  fit(x_train, y_train, epochs = 300, batch_size = 128,
      validation_split = 0.2,
      verbose = 1,
      callbacks = list(early_stop))


# predict on validation set
nn_layer3_pred_val <- layer3_model %>% predict(x_val)
score3 <- sqrt(mean((nn_layer3_pred_val - val_y)^2))
score3


# save
save(layer3_model, file = './nn_models/layer3.Rdata')


#####################################################################
#####################################################################
# 4. FOUR LAYER MODEL
# build a less wide model with more depth
#####################################################################
#####################################################################
par <- list(
  dropout1 = c(0.3,0.4),
  neurons1 = c(64,128, 256, 512),
  neurons2 = c(32,64, 128),
  neurons3 = c(8,16, 32, 64),
  neurons4 = c(8, 16, 32),
  lr = c(0.001,0.01),
  maxnorm1 = c(0.5,1,2, 3)
)


# perform runs
runs4 <- tuning_run('./src/layer4_model.R', sample = 0.2, runs_dir = '_tuning4.2', flags = par)


# Finally, I simply list all the runs, by referring to its running directory, where all the information 
# from the run is stored and I ask for it to be ordered according to the mean squared error.
ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning4.2')

# Finally, I select the best model parameters and I train the model with it.
# The best model has a dropout value, 512 neurons and a learning rate of 0.001
best_run4 <- ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning4.2')[1,]

# hier nog probleem: run the best model again lukt nog niet
# Run the best model again and save the model
layer4_model <- keras_model_sequential()
layer4_model %>%
  layer_dense(units = best_run4$flag_neurons1, activation = "relu",
              input_shape = ncol(x_train),
              constraint_maxnorm(max_value = best_run4$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run4$flag_dropout1) %>%
  layer_dense(units = best_run4$flag_neurons2, activation = "relu",
              constraint_maxnorm(max_value = best_run4$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run4$flag_dropout1) %>%
  layer_dense(units = best_run4$flag_neurons3, activation = "relu",
              constraint_maxnorm(max_value = best_run4$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run4$flag_dropout1) %>%
  layer_dense(units = best_run4$flag_neurons4, activation = "relu",
              constraint_maxnorm(max_value = best_run4$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run4$flag_dropout1) %>%
  layer_dense(units = 1, activation = "linear")
layer4_model

layer4_model %>% compile(loss = "mse",
                         optimizer = optimizer_rmsprop(learning_rate = best_run4$flag_lr),
                         metrics = list("mean_squared_error"))
# define early stop monitor
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

# Fit the model and store training stats
history <- layer4_model %>%
  fit(x_train, y_train, epochs = 300, batch_size = 128,
      validation_split = 0.2,
      verbose = 1,
      callbacks = list(early_stop))


# predict on validation set
nn_layer4_pred_val <- layer4_model %>% predict(x_val)
score4 <- sqrt(mean((nn_layer4_pred_val - val_y)^2))
score4


# save
save(layer4_model, file = './nn_models/layer4_run2.Rdata')

#####################################################################
#####################################################################
# 5. FIVE LAYER MODEL
# build a less wide model with more depth
#####################################################################
#####################################################################
par <- list(
  dropout1 = c(0.3,0.4),
  neurons1 = c(64,128, 256),
  neurons2 = c(32,64, 128),
  neurons3 = c(16, 32, 64),
  neurons4 = c(8, 16, 32),
  neurons4 = c(4, 8, 16),
  lr = c(0.001,0.01),
  maxnorm1 = c(0.5,1,2, 3)
)

# perform runs
runs5 <- tuning_run('./src/layer5_model.R', sample = 0.2, runs_dir = '_tuning5.2', flags = par)


# Finally, I simply list all the runs, by referring to its running directory, where all the information 
# from the run is stored and I ask for it to be ordered according to the mean squared error.
ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning5.2')

# Finally, I select the best model parameters and I train the model with it.
# The best model has a dropout value, 512 neurons and a learning rate of 0.001
best_run5 <- ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning5.2')[1,]

# hier nog probleem: run the best model again lukt nog niet
# Run the best model again and save the model
layer5_model <- keras_model_sequential()
layer5_model %>%
  layer_dense(units = best_run5$flag_neurons1, activation = "relu",
              input_shape = ncol(x_train),
              constraint_maxnorm(max_value = best_run5$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run5$flag_dropout1) %>%
  layer_dense(units = best_run5$flag_neurons2, activation = "relu",
              constraint_maxnorm(max_value = best_run5$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run5$flag_dropout1) %>%
  layer_dense(units = best_run5$flag_neurons3, activation = "relu",
              constraint_maxnorm(max_value = best_run5$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run5$flag_dropout1) %>%
  layer_dense(units = best_run5$flag_neurons4, activation = "relu",
              constraint_maxnorm(max_value = best_run5$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run5$flag_dropout1) %>%
  layer_dense(units = best_run5$flag_neurons5, activation = "relu",
              constraint_maxnorm(max_value = best_run5$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run5$flag_dropout1) %>%
  layer_dense(units = 1, activation = "linear")
layer5_model

layer5_model %>% compile(loss = "mse",
                         optimizer = optimizer_rmsprop(learning_rate = best_run5$flag_lr),
                         metrics = list("mean_squared_error"))
# define early stop monitor
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

# Fit the model and store training stats
history <- layer5_model %>%
  fit(x_train, y_train, epochs = 300, batch_size = 128,
      validation_split = 0.2,
      verbose = 1,
      callbacks = list(early_stop))



# predict on validation set
nn_layer5_pred_val <- layer5_model %>% predict(x_val)
score5 <- sqrt(mean((nn_layer5_pred_val - val_y)^2))
score5


# save
save(layer5_model, file = './nn_models/layer5_run2.Rdata')



#######################
# to test the best model
#######################

# Run the best model again and save the model
layer1_model <- keras_model_sequential()
layer1_model %>%
  layer_dense(units = best_run1$flag_neurons1, activation = "relu",
              input_shape = ncol(x_train),
              constraint_maxnorm(max_value = best_run1$flag_maxnorm, axis = 0)) %>%
  layer_dropout(rate = best_run1$flag_dropout1) %>%
  layer_dense(units = 1, activation = "linear")
layer1_model

layer1_model %>% compile(loss = "mse",
                         optimizer = optimizer_rmsprop(learning_rate = best_run1$flag_lr),
                         metrics = list("mean_squared_error"))

# define early stop monitor
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

# Fit the model and store training stats
history <- layer1_model %>%
  fit(x_train_and_val, y_train_and_val, epochs = 500, batch_size = 128,
      validation_split = 0.2,
      verbose = 1,
      callbacks = list(early_stop))



# predict on test set
x_test <- as.matrix(test_X[, -1])
nn_layer1_pred_test <- layer1_model %>% predict(x_test)
nn_layer1_pred_test

nn_df <- data.frame(id = as.integer(test_X$id),
                     average_daily_rate= nn_layer1_pred_test)
colnames(nn_df)[2] <- 'average_daily_rate'
str(nn_df)
# save submission file
write.csv(nn_df, file = "./data/sample_submission_nn.csv", row.names = F)


