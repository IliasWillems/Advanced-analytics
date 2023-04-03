# Clear work space
rm(list = ls())

# Load in data
train <- read.csv("data/preprocessed_train.csv", header=TRUE)
validation <- read.csv("data/preprocessed_validation.csv", header=TRUE)

# Load necessary packages
library(dplyr)
library(ODRF)
library('corrr')
library(ggcorrplot)
library(class)
library(xgboost)

# Convert all character variables to factors
train <- train %>% mutate_if(is.character, as.factor)
validation <- validation %>% mutate_if(is.character, as.factor)

################################################################################

# !!!
# We still need to handle missing values for reviews
# !!!

# I made review_period 0 if there were no reviews.
for (i in 1:5196){
  if (is.na(preprocessed_train$review_period[i])){
    preprocessed_train$review_period[i] = 0
  }
}
# For reviews_per_month the same thing
for (i in 1:5196){
  if (is.na(preprocessed_train$reviews_per_month[i])){
    preprocessed_train$review_period[i] = 0
  }
}

# for the review scores I'm not sure what to do
# they're not missing at random
# there's 1290 missing values, which is a lot

#for (i in 1:length(train$reviews_acc)){
  #if (is.na(train$reviews_acc[i])){train$reviews_acc[i] = }
#}

################################################################################

# Define some useful data frames
train_normal <- select(train, -c("bc_target"))
train_bc <- select(train, -c("target"))

train_normal_ppp <- select(train_normal, -c("booking_price_covers"))
train_normal_ppp$target <- train_normal_ppp$target/train_normal$booking_price_covers

train_bc_ppp <- select(train_bc, -c("booking_price_covers"))
train_bc_ppp$bc_target <- train_bc_ppp$bc_target/train_bc$booking_price_covers

# Also for validation set
validation_normal <- select(validation, -c("bc_target"))
validation_bc <- select(validation, -c("target"))

validation_normal_ppp <- select(validation_normal, -c("booking_price_covers"))
validation_normal_ppp$target <- validation_normal_ppp$target/validation_normal$booking_price_covers

validation_bc_ppp <- select(validation_bc, -c("booking_price_covers"))
validation_bc_ppp$bc_target <- validation_bc_ppp$bc_target/validation_bc$booking_price_covers

################################################################################

# Function to compute RMSE
compute_RMSE <- function(predictions, target) {
  sqrt(1/length(predictions) * sum((predictions - target)^2))
}

# Revert the box-cox transformed target variable back to normal
bc2normal <- function(bc) {
  lambda_opt <- -18/99
  
  (bc*lambda_opt + 1)^(1/lambda_opt)
}

# Compute RMSE on training set
compute_RMSE_train <- function(model, model_variables, pred_for_ppp, pred_for_bc_target) {
  
  if (pred_for_ppp & pred_for_bc_target) {
    
    pred <- predict(model, Xnew = data.frame(train_bc_ppp[, model_variables]))
    pred <- pred*train$booking_price_covers
    pred <- bc2normal(pred)
    
  } else if (pred_for_ppp & !pred_for_bc_target) {
    
    pred <- predict(model, Xnew = data.frame(train_normal_ppp[, model_variables]))
    pred <- pred*validation$booking_price_covers
    
  } else if (!pred_for_ppp & pred_for_bc_target) {
    
    pred <- predict(model, Xnew = data.frame(train_bc[, model_variables]))
    pred <- bc2normal(pred)
    
  } else {
    
    pred <- predict(model, Xnew = data.frame(train[, model_variables]))
    
  }
  
  compute_RMSE(pred, validation$target)
}


# Compute RMSE on validation set
compute_RMSE_validation <- function(model, model_variables, pred_for_ppp, pred_for_bc_target) {
  
  if (pred_for_ppp & pred_for_bc_target) {
    
    pred <- predict(model, Xnew = data.frame(validation_bc_ppp[, model_variables]))
    pred <- pred*train$booking_price_covers
    pred <- bc2normal(pred)
    
  } else if (pred_for_ppp & !pred_for_bc_target) {
    
    pred <- predict(model, Xnew = data.frame(validation_normal_ppp[, model_variables]))
    pred <- pred*validation$booking_price_covers
    
  } else if (!pred_for_ppp & pred_for_bc_target) {
    
    pred <- predict(model, Xnew = data.frame(validation_bc[, model_variables]))
    pred <- bc2normal(pred)
    
  } else {
    
    pred <- predict(model, Xnew = data.frame(validation[, model_variables]))
  }
  
  compute_RMSE(pred, validation$target)
}

################################################################################

# Baseline model: global mean of target variable.
# If your model has an RMSE that is larger than this model, something is really
# wrong.

mean_target <- mean(train$target)
compute_RMSE(mean_target, train$target)

################################################################################

# Oblique decision tree-based random forest

# We train the tree on the true target variable but taking booking_price_covers
# into account. Because this function gives an error otherwise, we only fit the
# model using 4 variables.
forest <- ODRF(target ~ superhost + zipcode_class + type_class + dist_nearest_city_center,
               data = train_normal_ppp, split = "mse",
               parallel = TRUE)

# Put all variables used in the model in a vector
model_variables <- c("superhost", "zipcode_class", "type_class", "dist_nearest_city_center")

# RMSE on training set
# Since the model predicts the price per person, we set 'pred_for_ppp = TRUE'.
# Since the model does not work with the box-cox tranformed target variable, we
#   set 'pred_for_bc_target = FALSE'.
compute_RMSE_train(forest, model_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)

# RMSE on validation set
# Since the model predicts the price per person, we set 'pred_for_ppp = TRUE'.
# Since the model does not work with the box-cox tranformed target variable, we
#   set 'pred_for_bc_target = FALSE'.
compute_RMSE_validation(forest, model_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)

################################################################################

# Other models?

# Regular linear regression, but...
#   - Find out which variables are important using stepwise selection methods
#   - Find out which variables to transform and how
#     - In a any case, the target should be transformed (also allow negatives)
#     - For example, host_nr_listings is very skew. Maybe a log-transformation
#       applied to it improves the results of the model?
#     - Same thing for reviews_num
#     - ...
#   - Are the assumptions of the linear model satisfied?
#   - Think about useful interactions
#     - Or just include all and select automatically?

# selecting by hand
#PCA -> only for numeric
train_numerical <- train
for (i in 1:57){
  if  (!(is.numeric(preprocessed_train[,i]))){
    train_numerical<- train_numerical[,-i]}
}
corr_matrix <- cor(train_numerical)
ggcorrplot(corr_matrix)

# cor
#I'm looking at the correlations between the target and the other numeric columns
# 0 if not numeric, NA if there are missing values
correlations = rep(0,57)
for (i in 1:57){
  if  (is.numeric(preprocessed_train[,i])){
    correlations[i] = cor(preprocessed_train$target,preprocessed_train[,i])}
}
#colnames(preprocessed_train)

#linear regression
# These chosen variables can be changed!!!
lr_model_variables <- c("superhost", "zipcode_class", "booking_price_covers")
lr_model <- lm(train$target ~ train$superhost + train$zipcode_class + train$booking_price_covers , data = train)
#compute rmse
compute_RMSE_train(lr_model, lr_model_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)
compute_RMSE_validation(lr_model, lr_model_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)

# Ridge/Lasso
#   - Which variables are selected?
# Load the glmnet package
library(glmnet)

# Create the predictor matrix and response vector
x <- as.matrix(train[, -1])
y <- train$target

# Perform ridge regression
ridgeModel <- glmnet(x, y, alpha = 0, lambda = 0.1)

# Print the ridge coefficients
print(coef(ridgeModel))

# Perform lasso regression
lassoModel <- glmnet(x, y, alpha = 1, lambda = 0.1)

# Print the lasso coefficients
print(coef(lassoModel))


# (Generalized additive models)

# For the above three models, maybe an improvement is possible by first fitting
# a classifier on a categorized target variable and then fitting the regression
# models for each of the categories. (maybe leads to overfitting)

# Random forests
# Load the randomForest package
library(randomForest)

# Fit the random forest model
rfModel <- randomForest(train$target ~ train$superhost + train$zipcode_class + train$booking_price_covers, data = train, ntree = 500, importance = TRUE)

# Print the model summary
print(rfModel)

# Make predictions on the test set
predictions <- predict(rfModel, validation)

# Evaluate the model accuracy
accuracy <- mean(predictions == validation$target)
print(paste("Accuracy:", accuracy))


# Gradient boosting
# Extreme Gradient Boosting: XGBoost
# These variables can be chosen!!!!
#xgboost_model_variables <- c(32,34,45,55)
#xgboost_model <- xgboost(data = train[,xgboost_model_variables], label = HIER, nrounds = 10, objective = "reg:linear")

# Use clustering methods like Kmeans
# k-nearest neighbours
# I used 4 variables (We could change this!! We could select others!)
# number of neighbours k can also be changed

knn_model_variables <- c(32,34,45,55)
knn_model <- knn(train = train[,knn_model_variables], test = validation[,knn_model_variables], cl = train$target, k = 5)

#compute RMSE on training and validation set for knn
compute_RMSE_train(knn_model, knn_model_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)

compute_RMSE_validation(knn_model, knn_model_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)


# Make models that predict price based on different clusters of variables (like
# property_..., host_..., booking_..., reviews_.., etc.) and then combine these
# predictions using a decision tree

# If everyone has a model --> Put them together in an ensemble.

# Some other things to do:

# - Think about more models? :)

# - model evaluation
# - ensemble modelling
