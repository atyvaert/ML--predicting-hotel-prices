# In essence, hypertuning is done through flags. Anywhere in your neural network, 
# you can replace a parameter with a flag
FLAGS <- flags(
  flag_numeric('dropout1', 0.3),
  flag_integer('neurons1', 128),
  flag_integer('neurons2', 128),
  flag_numeric('lr', 0.001),
  flag_numeric('maxnorm1', 3)
)
#

# build the neural network
two.layer.model <- function() {
  model <- keras_model_sequential() 
  model %>%
    layer_dense(units = FLAGS$neurons1, activation = "relu",
                input_shape = ncol(x_train),
                constraint_maxnorm(max_value = FLAGS$maxnorm1, axis = 0)) %>%
    layer_dropout(rate = FLAGS$dropout1) %>%
    layer_dense(units = FLAGS$neurons2, activation = "relu",
                constraint_maxnorm(max_value = FLAGS$maxnorm1, axis = 0)) %>%
    layer_dropout(rate = FLAGS$dropout1) %>%
    layer_dense(units = 1, activation = "linear")
  
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_rmsprop(learning_rate = FLAGS$lr),
    metrics = list("mean_squared_error")
  )
  model
}

layer2_model <- two.layer.model()
layer2_model %>% summary()


# define early stop monitor
early_stop <- callback_early_stopping(monitor = "val_loss", patience = 20)

# define the number of epochs
epochs <- 200


# Fit the model and store training stats
history <- layer2_model %>%
  fit(x_train, y_train, epochs = epochs, batch_size = 128,
      validation_split = 0.2,
      verbose = 1,
      callbacks = list(early_stop))



