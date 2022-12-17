
# In essence, hypertuning is done through flags. Anywhere in your neural network, 
# you can replace a parameter with a flag
FLAGS <- flags(
  flag_numeric('dropout1', 0.3),
  flag_integer('neurons1', 128),
  flag_numeric('lr', 0.001),
  flag_numeric('maxnorm1', 3)
)



##############
# wide model with 1 layer
#############


# build the neural network
one.layer.model <- function() {
  model <- keras_model_sequential() 
  model %>%
    layer_dense(units = FLAGS$neurons1, activation = "relu",
                input_shape = ncol(x_train),
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

layer1_model <- one.layer.model()
layer1_model %>% summary()

# define number of epochs
epochs <- 300


# Fit the model and store training stats
history <- layer1_model %>%
  fit(x_train, y_train, epochs = epochs, batch_size = 128,
      validation_split = 0.2,
      verbose = 1)




# save
# save(layer1_model, file = './nn_models/layer1.Rdata')







