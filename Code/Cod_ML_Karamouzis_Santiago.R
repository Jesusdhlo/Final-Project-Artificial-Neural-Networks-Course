
######### Particionando los datos para el modelo que no tiene en cuenta la variable Grade Point Average (GPA)
##################################

dat1 <- dat2 %>% select(-GPA_1ER_ANO) %>% select(-Rel_School_GPA)

set.seed(100)
train_test_split1 <- initial_split(dat1, prop = 0.8) 
train_tbl1 <- training(train_test_split1)
test_tbl1  <- testing(train_test_split1)

x_train_tbl1 <- train_tbl1 %>% select(-GRAD)
y_train_vec1 <- ifelse(pull(train_tbl1, GRAD) == "Y", 1, 0)

x_test_tbl1 <- test_tbl1 %>% select(-GRAD)
y_test_vec1  <- ifelse(pull(test_tbl1, GRAD) == "Y", 1, 0)


# Construyendo mi Red Neuronal artificial 
model_keras1 <- keras_model_sequential()

model_keras1 %>% 
  
  # Primer layer oculto 
  layer_dense(
    units              = 4, 
    kernel_initializer = "uniform", #to initialize weights. Normal dist and other possibilities
    activation         = "relu", 
    input_shape        = ncol(x_train_tbl1)) %>% 
  
  # Dropout para prevenir overfitting
  layer_dropout(rate = 0.3) %>%
  
  
  # Layer de salida
  layer_dense(
    units              = 1, 
    kernel_initializer = "uniform", 
    activation         = "sigmoid") %>% 
  
  # Compila ANN
  compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')
  )

t1 <- proc.time() 
# Entrenando la ANN
history1 <- fit(
  object           = model_keras1, 
  x                = as.matrix(x_train_tbl1), 
  y                = y_train_vec1,
  batch_size       = 32, 
  epochs           = 6000, 
  validation_split = 0.30 
)

# Midiendo el tiempo de entrenamiento  
proc.time()-t1 


plot(history1)

# Predicciones
yhat_keras_class_vec1 <- predict_classes(object = model_keras1, x = as.matrix(x_test_tbl1)) %>%
  as.vector()

yhat_keras_prob_vec1  <- predict_proba(object = model_keras1, x = as.matrix(x_test_tbl1)) %>%
  as.vector()

estimates_keras_tbl1 <- tibble(
  truth      = as.factor(y_test_vec1) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.factor(yhat_keras_class_vec1) %>% fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec1
)


options(yardstick.event_first = FALSE)

# Métricas

#Recall
re1 <- estimates_keras_tbl1 %>% conf_mat(truth, estimate)
recall1 <- re1$table[1,1]/(re1$table[1,1]+re1$table[2,1])
recall1

# AUC
estimates_keras_tbl1 %>% roc_auc(truth, class_prob)


######### Particionando los datos para el modelo que tiene en cuenta la variable Grade Point Average (GPA)
##################################
set.seed(100)
train_test_split <- initial_split(dat2, prop = 0.8) 
train_tbl <- training(train_test_split)
test_tbl  <- testing(train_test_split)

x_train_tbl <- train_tbl %>% select(-GRAD)
y_train_vec <- ifelse(pull(train_tbl, GRAD) == "Y", 1, 0)

x_test_tbl <- test_tbl %>% select(-GRAD)
y_test_vec  <- ifelse(pull(test_tbl, GRAD) == "Y", 1, 0)


# Construyendo mi Red Neuronal artificial 
model_keras <- keras_model_sequential()

model_keras %>% 
  
  # Primer layer oculto 
  layer_dense(
    units              = 4, 
    kernel_initializer = "uniform", #to initialize weights. Normal dist and other possibilities
    activation         = "relu", 
    input_shape        = ncol(x_train_tbl)) %>% 
  
  # Dropout para prevenir overfitting
  layer_dropout(rate = 0.3) %>%
  
  
  # Layer de salida
  layer_dense(
    units              = 1, 
    kernel_initializer = "uniform", 
    activation         = "sigmoid") %>% 
  
  # Compila ANN
  compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')
  )

t <- proc.time() 
# Entrenando la ANN
history <- fit(
  object           = model_keras, 
  x                = as.matrix(x_train_tbl), 
  y                = y_train_vec,
  batch_size       = 32, 
  epochs           = 6000,
  validation_split = 0.30 
)


# Midiendo el tiempo de entrenamiento 
proc.time()-t 

plot(history)

# Predicciones
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()


yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

estimates_keras_tbl <- tibble(
  truth      = as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec
)

options(yardstick.event_first = FALSE)

## Métricas 

# Recall
re<-estimates_keras_tbl %>% conf_mat(truth, estimate)
recall=re$table[1,1]/(re$table[1,1]+re$table[2,1])
recall

# AUC
estimates_keras_tbl %>% roc_auc(truth, class_prob)

#######################################################################Roberto rivera


######### Particionando los datos para el modelo que no tiene en cuenta la variable Grade Point Average (GPA)
##################################

dat1 <- dat2 %>% select(-GPA_1ER_ANO) %>% select(-Rel_School_GPA)

set.seed(100)
train_test_split1 <- initial_split(dat1, prop = 0.8) 
train_tbl1 <- training(train_test_split1)
test_tbl1  <- testing(train_test_split1)

x_train_tbl1 <- train_tbl1 %>% select(-GRAD)
y_train_vec1 <- ifelse(pull(train_tbl1, GRAD) == "Y", 1, 0)

x_test_tbl1 <- test_tbl1 %>% select(-GRAD)
y_test_vec1  <- ifelse(pull(test_tbl1, GRAD) == "Y", 1, 0)


# Construyendo mi Red Neuronal artificial 
model_keras1 <- keras_model_sequential()

model_keras1 %>% 
  
  # Primer layer oculto 
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", #to initialize weights. Normal dist and other possibilities
    activation         = "relu", 
    input_shape        = ncol(x_train_tbl1)) %>% 
  
  # Dropout para prevenir overfitting
  layer_dropout(rate = 0.3) %>%
  
  # Second hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu") %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.3) %>%   
  
  # Layer de salida
  layer_dense(
    units              = 1, 
    kernel_initializer = "uniform", 
    activation         = "sigmoid") %>% 
  
  # Compila ANN
  compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')
  )

t1 <- proc.time() 
# Entrenando la ANN
history1 <- fit(
  object           = model_keras1, 
  x                = as.matrix(x_train_tbl1), 
  y                = y_train_vec1,
  batch_size       = 600, 
  epochs           = 35, 
  validation_split = 0.30 
)

# Midiendo el tiempo de entrenamiento  
proc.time()-t1 


plot(history1)

# Predicciones
yhat_keras_class_vec1 <- predict_classes(object = model_keras1, x = as.matrix(x_test_tbl1)) %>%
  as.vector()

yhat_keras_prob_vec1  <- predict_proba(object = model_keras1, x = as.matrix(x_test_tbl1)) %>%
  as.vector()

estimates_keras_tbl1 <- tibble(
  truth      = as.factor(y_test_vec1) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.factor(yhat_keras_class_vec1) %>% fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec1
)


options(yardstick.event_first = FALSE)

# Métricas

#Recall
re1 <- estimates_keras_tbl1 %>% conf_mat(truth, estimate)
recall1 <- re1$table[1,1]/(re1$table[1,1]+re1$table[2,1])
recall1

# AUC
estimates_keras_tbl1 %>% roc_auc(truth, class_prob)


######### Particionando los datos para el modelo que tiene en cuenta la variable Grade Point Average (GPA)
##################################
set.seed(100)
train_test_split <- initial_split(dat2, prop = 0.8) 
train_tbl <- training(train_test_split)
test_tbl  <- testing(train_test_split)

x_train_tbl <- train_tbl %>% select(-GRAD)
y_train_vec <- ifelse(pull(train_tbl, GRAD) == "Y", 1, 0)

x_test_tbl <- test_tbl %>% select(-GRAD)
y_test_vec  <- ifelse(pull(test_tbl, GRAD) == "Y", 1, 0)


# Construyendo mi Red Neuronal artificial 
model_keras <- keras_model_sequential()

model_keras %>% 
  
  # Primer layer oculto 
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", #to initialize weights. Normal dist and other possibilities
    activation         = "relu", 
    input_shape        = ncol(x_train_tbl)) %>% 
  
  # Dropout para prevenir overfitting
  layer_dropout(rate = 0.3) %>%
  
  # Second hidden layer
  layer_dense(
    units              = 16, 
    kernel_initializer = "uniform", 
    activation         = "relu") %>% 
  
  # Dropout to prevent overfitting
  layer_dropout(rate = 0.3) %>% 
  
  # Layer de salida
  layer_dense(
    units              = 1, 
    kernel_initializer = "uniform", 
    activation         = "sigmoid") %>% 
  
  # Compila ANN
  compile(
    optimizer = 'adam',
    loss      = 'binary_crossentropy',
    metrics   = c('accuracy')
  )

t <- proc.time() 
# Entrenando la ANN
history <- fit(
  object           = model_keras, 
  x                = as.matrix(x_train_tbl), 
  y                = y_train_vec,
  batch_size       = 600, 
  epochs           = 35,
  validation_split = 0.30 
)

# Midiendo el tiempo de entrenamiento 
proc.time()-t 

plot(history)

# Predicciones
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()


yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(x_test_tbl)) %>%
  as.vector()

estimates_keras_tbl <- tibble(
  truth      = as.factor(y_test_vec) %>% fct_recode(yes = "1", no = "0"),
  estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"),
  class_prob = yhat_keras_prob_vec
)

options(yardstick.event_first = FALSE)

## Métricas 

# Recall
re<-estimates_keras_tbl %>% conf_mat(truth, estimate)
recall=re$table[1,1]/(re$table[1,1]+re$table[2,1])
recall

# AUC
estimates_keras_tbl %>% roc_auc(truth, class_prob)
