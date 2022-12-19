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
if(!require('tfruns')) install.packages('tfruns')
library(tfruns)

# import data
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


# Transform the data in the right format to train neural networks
# for training set
x_train <- model.matrix(average_daily_rate ~ ., data = train)[, -1]
y_train <- train$average_daily_rate

# for validation set
x_val <- model.matrix(average_daily_rate ~ ., data = val)[, -1]
y_val <- val$average_daily_rate

# for training and validation set
x_train_and_val <-model.matrix(average_daily_rate ~ ., data = train_and_val)[, -1]
y_train_and_val <- train_and_val_y


#####################################################################
#####################################################################
# STRUCTURE:
# In this file, we build four different neural networks. From neural networks with one layer
# up until four layers. For each layer, we perform a grid search with the values that are mentioned
# in the beginning of each model. The function tuning run is used to tune the different models.
# This function uses the models defined in other scripts to find the optimal set of parameters. It also
# writes away the tuned models for each layer.
#####################################################################
#####################################################################



# For more information, consult this blog on how to hypertune a neural network in R
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
# build a wide model one layer model that overfits the data and regularize with dropout and maxnorm
#####################################################################
#####################################################################
# In the following lines of code I define the possible values of all the parameters I want to hypertune. 
# This will produce a lot of possible combinations of parameters. 
# That’s why in the tuning_run() function, I specify I only want to try 40% of the possible combinations (sampled).
par <- list(
  dropout1 = c(0.3,0.4),
  neurons1 = c(128,256, 384, 512, 640),
  lr = c(0.001,0.01),
  maxnorm1 = c(0.5, 1,2, 3)
)

# This code accesses the model written in R-script layer1_model and writes away the results
# of tuning the one layer model in the '_tuning' directory.
set.seed(1)
runs1 <- tuning_run('./src/layer1_model.R', runs_dir = '_tuning', flags = par, sample = 0.4)

# Finally, I simply list all the tuning runs, by referring to its running directory, where all the information 
# from the run is stored and I ask for it to be ordered according to the mean squared error.
ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning')

# Finally, I select the best model parameters and I train the model with it.
# The best model has a dropout value of 0.4, 512 neurons and a learning rate of 0.001 and a maxnorm of 1
best_run1 <- ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning')[1,]

# Run the model with the optimal parameters again to make predictions on the validation set
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
# RMSE: 22.26
nn_run1_pred_val <- layer1_model %>% predict(x_val)
score1 <- sqrt(mean((nn_run1_pred_val - val_y)^2))
score1


#####################################################################
#####################################################################
# 2. TWO LAYER MODEL
# In our second model, we again go wide, but not wider than the optimal solution of the 
# previous layer. Besides, we also parameters for the number of neurons in the second layer.
#####################################################################
#####################################################################

# In the following lines of code I define the possible values of all the parameters I want to hypertune. 
# This will produce a lot of possible combinations of parameters (more than in the one layer model)
# That’s why in the tuning_run() function, I specify I only want to try 30% of the possible combinations (sampled).
par <- list(
  dropout1 = c(0.3,0.4),
  neurons1 = c(64,128,256, 384, 512),
  neurons2 = c(32,64,128, 256),
  lr = c(0.001,0.01),
  maxnorm1 = c(0.5,1,2, 3)
)

# This code accesses the model written in R-script layer2_model and writes away the results
# of tuning the two layer model in the '_tuning2' directory.
set.seed(1)
runs2 <- tuning_run('./src/layer2_model.R', sample = 0.3, runs_dir = '_tuning2', flags = par)


# Finally, I simply list all the runs, by referring to its running directory, where all the information 
# from the run is stored and I ask for it to be ordered according to the mean squared error.
ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning2')

# Finally, I select the best model parameters and I train the model with it.
# The best model has a dropout value of 0.3, 384 neurons in the first layer and 128 in the second.
# Besides, it has a learning rate of 0.001 and a maxnorm value of 1
best_run2 <- ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning2')[1,]

# Run the model with the optimal parameters again to make predictions on the validation set
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
# RMSE: 51.49
nn_layer2_pred_val <- layer2_model %>% predict(x_val)
score2 <- sqrt(mean((nn_layer2_pred_val - val_y)^2))
score2


#####################################################################
#####################################################################
# 3. THREE LAYER MODEL
# In our third model, we again go wide, but not wider than the optimal solution of the 
# previous layer. We use less neurons per layer as we already increase the model complexity by adding a new layer.
#####################################################################
#####################################################################

# In the following lines of code I define the possible values of all the parameters I want to hypertune. 
# As we add a third layer, the number of possible combinations grows rapidly.
# Therefore, I specify I only want to try 20% of the possible combinations (sampled).
par <- list(
  dropout1 = c(0.3,0.4),
  neurons1 = c(64,128, 256, 384),
  neurons2 = c(16, 32,64, 128),
  neurons3 = c(8, 16, 32, 64),
  lr = c(0.001,0.01),
  maxnorm1 = c(0.5,1,2, 3)
)

# This code accesses the model written in R-script layer3_model and writes away the results
# of tuning the three layer model in the '_tuning3' directory.
set.seed(1)
runs3 <- tuning_run('./src/layer3_model.R', sample = 0.2, runs_dir = '_tuning3', flags = par)


# Finally, I simply list all the runs, by referring to its running directory, where all the information 
# from the run is stored and I ask for it to be ordered according to the mean squared error.
ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning3')

# Finally, I select the best model parameters and I train the model with it.
# The best model has a dropout value of 0.3, 384 neurons in the first layer, 64 in the second and 64 in the third layer.
# Besides, it has a learning rate of 0.001 and a maxnorm value of 0.5.
best_run3 <- ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning3')[1,]

# Run the model with the optimal parameters again to make predictions on the validation set
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
# RMSE: 35.88
nn_layer3_pred_val <- layer3_model %>% predict(x_val)
score3 <- sqrt(mean((nn_layer3_pred_val - val_y)^2))
score3


#####################################################################
#####################################################################
# 4. FOUR LAYER MODEL
# Finally, we build a model with four layers. Here, we used less neurons per layer,
# but complexity is adde by using multiple layers.
#####################################################################
#####################################################################
par <- list(
  dropout1 = c(0.3,0.4),
  neurons1 = c(32, 64,128),
  neurons2 = c(16,32, 64),
  neurons3 = c(8,16, 32, 64),
  neurons4 = c(8, 16, 32),
  lr = c(0.001,0.01),
  maxnorm1 = c(0.5,1,2, 3)
)


# This code accesses the model written in R-script layer4_model and writes away the results
# of tuning the four layer model in the '_tuning4' directory.
set.seed(1)
runs4 <- tuning_run('./src/layer4_model.R', sample = 0.2, runs_dir = '_tuning4', flags = par)


# Finally, I simply list all the runs, by referring to its running directory, where all the information 
# from the run is stored and I ask for it to be ordered according to the mean squared error.
ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning4')

# Finally, I select the best model parameters and I train the model with it.
# The best model has a dropout value of 0.3, 128 neurons in the first layer, 64 in the second and 32 in the third layer
# and 32 neurons in the fourth layer. Besides, it has a learning rate of 0.001 and a maxnorm value of 1.
best_run4 <- ls_runs(order = metric_val_mean_squared_error, decreasing= F, runs_dir = '_tuning4')[1,]

# Run the model with the optimal parameters again to make predictions on the validation set
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
# RMSE: 32.199
nn_layer4_pred_val <- layer4_model %>% predict(x_val)
score4 <- sqrt(mean((nn_layer4_pred_val - val_y)^2))
score4


#####################################################################
#####################################################################
# 5. Train the best model on all the available data
# When evaluating our models, we saw that the one layer model with its optimal parameters
# had the lowest RMSE on the validation set. Therefore, we retrain this model on
# all the available data (training and validation set). Then, we can use this model
# to make predictions on the test set.
#####################################################################
#####################################################################

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


