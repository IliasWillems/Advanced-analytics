# Clear work space
rm(list = ls())

# Load in data
train <- read.csv("data/preprocessed_train.csv", header=TRUE)
validation <- read.csv("data/preprocessed_validation.csv", header=TRUE)


# Load necessary packages
library(dplyr)
library(ODRF)
library(corrr)
library(ggcorrplot)
library(class)
library(xgboost)
library(glmnet)

# Convert all character variables to factors
train <- train %>% mutate_if(is.character, as.factor)
validation <- validation %>% mutate_if(is.character, as.factor)

# If you imported the datasets by hand:
#train <- preprocessed_train
#validation<- preprocessed_validation


################################################################################
# We still need to handle missing values for reviews


# I made review_period 0 if there were no reviews.
for (i in 1:5196){
  if (is.na(train$review_period[i])){
    train$review_period[i] = 0
  }
}

for (i in 1:1299){
  if (is.na(validation$review_period[i])){
    validation$review_period[i] = 0
  }
}
# For reviews_per_month the same thing
for (i in 1:5196){
  if (is.na(train$reviews_per_month[i])){
    train$reviews_per_month[i] = 0
  }
}

for (i in 1:1299){
  if (is.na(validation$reviews_per_month[i])){
    validation$reviews_per_month[i] = 0
  }
}
# for the review scores I'm not sure what to do
# they're not missing at random
# there's 1290 missing values, which is a lot
# We also put them to 0
# for reviews_acc, reviews_rating, reviews_cleanliness, reviews_checkin, reviews_communication, reviews_location

for (i in 1:length(train$reviews_acc)){
  if (is.na(train$reviews_acc[i])){train$reviews_acc[i] = 0}
}
for (i in 1:length(validation$reviews_acc)){
  if (is.na(validation$reviews_acc[i])){validation$reviews_acc[i] = 0}
}

for (i in 1:length(train$reviews_rating)){
  if (is.na(train$reviews_rating[i])){train$reviews_rating[i] = 0}
}
for (i in 1:length(validation$reviews_rating)){
  if (is.na(validation$reviews_rating[i])){validation$reviews_rating[i] = 0}
}

for (i in 1:length(train$reviews_cleanliness)){
  if (is.na(train$reviews_cleanliness[i])){train$reviews_cleanliness[i] = 0}
}
for (i in 1:length(validation$reviews_cleanliness)){
  if (is.na(validation$reviews_cleanliness[i])){validation$reviews_cleanliness[i] = 0}
}

for (i in 1:length(train$reviews_checkin)){
  if (is.na(train$reviews_checkin[i])){train$reviews_checkin[i] = 0}
}
for (i in 1:length(validation$reviews_checkin)){
  if (is.na(validation$reviews_checkin[i])){validation$reviews_checkin[i] = 0}
}

for (i in 1:length(train$reviews_communication)){
  if (is.na(train$reviews_communication[i])){train$reviews_communication[i] = 0}
}
for (i in 1:length(validation$reviews_communication)){
  if (is.na(validation$reviews_communication[i])){validation$reviews_communication[i] = 0}
}

for (i in 1:length(train$reviews_location)){
  if (is.na(train$reviews_location[i])){train$reviews_location[i] = 0}
}
for (i in 1:length(validation$reviews_location)){
  if (is.na(validation$reviews_location[i])){validation$reviews_location[i] = 0}
}
################################################################################

# Load necessary packages
library(dplyr)
library(ODRF)
library('corrr')
library(ggcorrplot)
library(class)
library(xgboost)
library(caret)
library(FNN)
library(randomForest)

# Convert all character variables to factors
train <- train %>% mutate_if(is.character, as.factor)
validation <- validation %>% mutate_if(is.character, as.factor)

# Define some useful data frames
train_normal <- dplyr::select(train, -c("bc_target"))
train_bc <- dplyr::select(train, -c("target"))

train_normal_ppp <- dplyr::select(train_normal, -c("booking_price_covers"))
train_normal_ppp$target <- train_normal_ppp$target/train_normal$booking_price_covers

train_bc_ppp <- dplyr::select(train_bc, -c("booking_price_covers"))
train_bc_ppp$bc_target <- train_bc_ppp$bc_target/train_bc$booking_price_covers

# Also for validation set
validation_normal <- dplyr::select(validation, -c("bc_target"))
validation_bc <- dplyr::select(validation, -c("target"))

validation_normal_ppp <- dplyr::select(validation_normal, -c("booking_price_covers"))
validation_normal_ppp$target <- validation_normal_ppp$target/validation_normal$booking_price_covers

validation_bc_ppp <- dplyr::select(validation_bc, -c("booking_price_covers"))
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
    pred <- pred*train$booking_price_covers
    
  } else if (!pred_for_ppp & pred_for_bc_target) {
    
    pred <- predict(model, Xnew = data.frame(train_bc[, model_variables]))
    pred <- bc2normal(pred)
    
  } else {
    
    pred <- predict(model, Xnew = data.frame(train[, model_variables]))
    
  }
  
  compute_RMSE(pred, train$target)
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


# Compute RMSE on training set
compute_RMSE_train_glmnet <- function(model, model_variables, pred_for_ppp, pred_for_bc_target) {
  
  if (pred_for_ppp & pred_for_bc_target) {
    
    X <- train_bc_ppp[, model_variables]
    X <- X %>% mutate_if(is.factor, as.numeric)
    X <- as.matrix(X)
    
    pred <- predict(model, newx = X)
    pred <- pred*train$booking_price_covers
    pred <- bc2normal(pred)
    
  } else if (pred_for_ppp & !pred_for_bc_target) {
    
    X <- train_normal_ppp[, model_variables]
    X <- X %>% mutate_if(is.factor, as.numeric)
    X <- as.matrix(X)
    
    pred <- predict(model, newx = X)
    pred <- pred*train$booking_price_covers
    
  } else if (!pred_for_ppp & pred_for_bc_target) {
    
    X <- train_bc[, model_variables]
    X <- X %>% mutate_if(is.factor, as.numeric)
    X <- as.matrix(X)
    
    pred <- predict(model, newx = X)
    pred <- bc2normal(pred)
    
  } else {
    
    X <- train[, model_variables]
    X <- X %>% mutate_if(is.factor, as.numeric)
    X <- as.matrix(X)
    
    pred <- predict(model, newx = X)
    
  }
  
  compute_RMSE(pred, train$target)
}


# Compute RMSE on validation set
compute_RMSE_validation_glmnet <- function(model, model_variables, pred_for_ppp, pred_for_bc_target) {
  
  if (pred_for_ppp & pred_for_bc_target) {
    
    X <- validation_bc_ppp[, model_variables]
    X <- X %>% mutate_if(is.factor, as.numeric)
    X <- as.matrix(X)
    
    pred <- predict(model, newx = X)
    pred <- pred*train$booking_price_covers
    pred <- bc2normal(pred)
    
  } else if (pred_for_ppp & !pred_for_bc_target) {
    
    X <- validation_normal_ppp[, model_variables]
    X <- X %>% mutate_if(is.factor, as.numeric)
    X <- as.matrix(X)
    
    pred <- predict(model, newx = X)
    pred <- pred*validation$booking_price_covers
    
  } else if (!pred_for_ppp & pred_for_bc_target) {
    
    X <- validation_bc[, model_variables]
    X <- X %>% mutate_if(is.factor, as.numeric)
    X <- as.matrix(X)
    
    pred <- predict(model, newx = X)
    pred <- bc2normal(pred)
    
  } else {
    
    X <- validation[, model_variables]
    X <- X %>% mutate_if(is.factor, as.numeric)
    X <- as.matrix(X)
    
    pred <- predict(model, newx = X)
  }
  
  compute_RMSE(pred, validation$target)
}


################################################################################

# Baseline model: global mean of target variable.
# If your model has an RMSE that is larger than this model, something is really
# wrong.

mean_target <- mean(train$target)
compute_RMSE(rep(mean_target, length(train$target)), train$target)

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

# Random forests

forest2 <- randomForest(target ~ property_room_type + property_max_guests + property_bathrooms +
                          property_bedrooms + property_beds + property_bed_type +
                          host_response_time + host_response_rate + host_nr_listings +
                          booking_price_covers + booking_min_nights + booking_max_nights + 
                          booking_availability_30 + booking_availability_60 + booking_availability_90 +
                          booking_availability_365 + booking_cancel_policy + reviews_num +
                          reviews_rating + reviews_acc + reviews_cleanliness + reviews_checkin +
                          reviews_communication + reviews_location + reviews_per_month +
                          zipcode_class + dist_nearest_city_center + type_class +
                          host_location_country + host_verified_amount + years_as_host +
                          review_period + review_period_was_missing + profile_pic +
                          exact_location + instant_bookable + superhost + identified_host,
                        data = train_normal, ntree = 1000, mtry = 3, importance = TRUE)

# Put all variables used in the model in a vector
model_variables <- c("property_room_type", "property_max_guests", "property_bathrooms", 
                       "property_bedrooms", "property_beds", "property_bed_type", 
                       "host_response_time", "host_response_rate", "host_nr_listings", 
                       "booking_price_covers", "booking_min_nights", "booking_max_nights",  
                       "booking_availability_30", "booking_availability_60", "booking_availability_90", 
                       "booking_availability_365", "booking_cancel_policy", "reviews_num", 
                       "reviews_rating", "reviews_acc", "reviews_cleanliness", "reviews_checkin", 
                       "reviews_communication", "reviews_location", "reviews_per_month",
                       "zipcode_class", "dist_nearest_city_center", "type_class", 
                       "host_location_country", "host_verified_amount", "years_as_host", 
                       "review_period", "review_period_was_missing", "profile_pic", 
                       "exact_location", "instant_bookable", "superhost", "identified_host")

# RMSE on training set
compute_RMSE_train(forest2, model_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)

# RMSE on validation set
compute_RMSE_validation(forest2, model_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)

################################################################################

# Linear model

lin_model <- lm(target ~ property_room_type + property_max_guests + property_bathrooms +
                  property_bedrooms + property_beds + property_bed_type +
                  host_response_time + host_response_rate + host_nr_listings +
                  booking_price_covers + booking_min_nights + booking_max_nights + 
                  booking_availability_30 + booking_availability_60 + booking_availability_90 +
                  booking_availability_365 + booking_cancel_policy + reviews_num +
                  reviews_rating + reviews_acc + reviews_cleanliness + reviews_checkin +
                  reviews_communication + reviews_location + reviews_per_month +
                  zipcode_class + dist_nearest_city_center + type_class +
                  host_location_country + host_verified_amount + years_as_host +
                  review_period + review_period_was_missing + profile_pic +
                  exact_location + instant_bookable + superhost + identified_host,
                data = train_normal)

model_variables <- c("property_room_type", "property_max_guests", "property_bathrooms", 
                     "property_bedrooms", "property_beds", "property_bed_type", 
                     "host_response_time", "host_response_rate", "host_nr_listings", 
                     "booking_price_covers", "booking_min_nights", "booking_max_nights",  
                     "booking_availability_30", "booking_availability_60", "booking_availability_90", 
                     "booking_availability_365", "booking_cancel_policy", "reviews_num", 
                     "reviews_rating", "reviews_acc", "reviews_cleanliness", "reviews_checkin", 
                     "reviews_communication", "reviews_location", "reviews_per_month",
                     "zipcode_class", "dist_nearest_city_center", "type_class", 
                     "host_location_country", "host_verified_amount", "years_as_host", 
                     "review_period", "review_period_was_missing", "profile_pic", 
                     "exact_location", "instant_bookable", "superhost", "identified_host")

compute_RMSE_train(lin_model, model_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)
compute_RMSE_validation(lin_model, model_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)

################################################################################

# Ridge model

model_variables <- c("property_room_type", "property_max_guests", "property_bathrooms", 
                     "property_bedrooms", "property_beds", "property_bed_type", 
                     "host_response_time", "host_response_rate", "host_nr_listings", 
                     "booking_price_covers", "booking_min_nights", "booking_max_nights",  
                     "booking_availability_30", "booking_availability_60", "booking_availability_90", 
                     "booking_availability_365", "booking_cancel_policy", "reviews_num", 
                     "reviews_rating", "reviews_acc", "reviews_cleanliness", "reviews_checkin", 
                     "reviews_communication", "reviews_location", "reviews_per_month",
                     "zipcode_class", "dist_nearest_city_center", "type_class", 
                     "host_location_country", "host_verified_amount", "years_as_host", 
                     "review_period", "review_period_was_missing", "profile_pic", 
                     "exact_location", "instant_bookable", "superhost", "identified_host")

# Create the predictor matrix
X <- train_normal[, model_variables]
X <- X %>% mutate_if(is.factor, as.numeric)
X <- as.matrix(X)

# Perform ridge regression
cv_model <- cv.glmnet(X, train$target, alpha = 0)
lambda_opt <- cv_model$lambda.min
plot(cv_model)
lambda_opt

ridge_reg <- glmnet(X, train$target, alpha = 0, lambda = lambda_opt)

compute_RMSE_train_glmnet(ridge_reg, model_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)
compute_RMSE_validation_glmnet(ridge_reg, model_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)

################################################################################

# Lasso model

model_variables <- c("property_room_type", "property_max_guests", "property_bathrooms", 
                     "property_bedrooms", "property_beds", "property_bed_type", 
                     "host_response_time", "host_response_rate", "host_nr_listings", 
                     "booking_price_covers", "booking_min_nights", "booking_max_nights",  
                     "booking_availability_30", "booking_availability_60", "booking_availability_90", 
                     "booking_availability_365", "booking_cancel_policy", "reviews_num", 
                     "reviews_rating", "reviews_acc", "reviews_cleanliness", "reviews_checkin", 
                     "reviews_communication", "reviews_location", "reviews_per_month",
                     "zipcode_class", "dist_nearest_city_center", "type_class", 
                     "host_location_country", "host_verified_amount", "years_as_host", 
                     "review_period", "review_period_was_missing", "profile_pic", 
                     "exact_location", "instant_bookable", "superhost", "identified_host")

# Create the predictor matrix
X <- train_normal[, model_variables]
X <- X %>% mutate_if(is.factor, as.numeric)
X <- as.matrix(X)

# Perform ridge regression
cv_model <- cv.glmnet(X, train$target, alpha = 1)
lambda_opt <- cv_model$lambda.min
plot(cv_model)
lambda_opt

lasso_reg <- glmnet(X, train$target, alpha = 1, lambda = lambda_opt)
print(coef(lasso_reg))

compute_RMSE_train_glmnet(lasso_reg, model_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)
compute_RMSE_validation_glmnet(lasso_reg, model_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)

################################################################################

# Lasso model with interactions

model_variables <- c("property_room_type", "property_max_guests", "property_bathrooms", 
                     "property_bedrooms", "property_beds", "property_bed_type", 
                     "host_response_time", "host_response_rate", "host_nr_listings", 
                     "booking_price_covers", "booking_min_nights", "booking_max_nights",  
                     "booking_availability_30", "booking_availability_60", "booking_availability_90", 
                     "booking_availability_365", "booking_cancel_policy", "reviews_num", 
                     "reviews_rating", "reviews_acc", "reviews_cleanliness", "reviews_checkin", 
                     "reviews_communication", "reviews_location", "reviews_per_month",
                     "zipcode_class", "dist_nearest_city_center", "type_class", 
                     "host_location_country", "host_verified_amount", "years_as_host", 
                     "review_period", "review_period_was_missing", "profile_pic", 
                     "exact_location", "instant_bookable", "superhost", "identified_host")

num_cols <- unlist(lapply(train_normal, is.numeric))
means <- apply(train_normal[,num_cols], 2, mean)
sds <- apply(train_normal[,num_cols], 2, sd)

train_normal_normalized <- train_normal
train_normal_normalized[,num_cols] <- (train_normal[,num_cols] - means)/sds

# Create the predictor matrix
X <- model.matrix(target ~ (property_room_type + property_max_guests + property_bathrooms +
                    property_bedrooms + property_beds + property_bed_type +
                    host_response_time + host_response_rate + host_nr_listings +
                    booking_price_covers + booking_min_nights + booking_max_nights + 
                    booking_availability_30 + booking_availability_60 + booking_availability_90 +
                    booking_availability_365 + booking_cancel_policy + reviews_num +
                    reviews_rating + reviews_acc + reviews_cleanliness + reviews_checkin +
                    reviews_communication + reviews_location + reviews_per_month +
                    zipcode_class + dist_nearest_city_center + type_class +
                    host_location_country + host_verified_amount + years_as_host +
                    review_period + review_period_was_missing + profile_pic +
                    exact_location + instant_bookable + superhost + identified_host)^2,
                  data = train_normal_normalized)
X <- X[,-1]

head(X)

# Perform ridge regression
cv_model <- cv.glmnet(X, train$target, alpha = 1)
lambda_opt <- cv_model$lambda.min
plot(cv_model)
lambda_opt

lasso_reg_interactions <- glmnet(X, train$target, alpha = 1, lambda = lambda_opt)
print(coef(lasso_reg_interactions))


pred <- predict(lasso_reg_interactions, newx = X)
compute_RMSE(pred, train$target)

Xval <- model.matrix(target ~ (property_room_type + property_max_guests + property_bathrooms +
                              property_bedrooms + property_beds + property_bed_type +
                              host_response_time + host_response_rate + host_nr_listings +
                              booking_price_covers + booking_min_nights + booking_max_nights + 
                              booking_availability_30 + booking_availability_60 + booking_availability_90 +
                              booking_availability_365 + booking_cancel_policy + reviews_num +
                              reviews_rating + reviews_acc + reviews_cleanliness + reviews_checkin +
                              reviews_communication + reviews_location + reviews_per_month +
                              zipcode_class + dist_nearest_city_center + type_class +
                              host_location_country + host_verified_amount + years_as_host +
                              review_period + review_period_was_missing + profile_pic +
                              exact_location + instant_bookable + superhost + identified_host)^2,
                  data = validation)
Xval <- Xval[,-1]

pred <- predict(lasso_reg_interactions, newx = Xval)
compute_RMSE(pred, train$target)


################################################################################

#example variables
lassoModel_variables <- c("superhost", "zipcode_class", "booking_price_covers")

#compute rmse
compute_RMSE_train(ridgeModel, ridgeModel_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)
compute_RMSE_validation(ridgeModel, ridgeModel_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)


# Perform lasso regression
lassoModel <- glmnet(x, y, alpha = 1, lambda = 0.1)

# Print the lasso coefficients
print(coef(lassoModel))

# Example variables
lassoModel_variables <- c("superhost", "zipcode_class", "booking_price_covers")

#compute rmse
compute_RMSE_train(lassoModel, lassoModel_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)
compute_RMSE_validation(lassoModel, lassoModel_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)




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
# make a dataset of only the numeric variables
indices_list <- c()
for (i in 1:57){
  if  (!(is.numeric(train[,i]))){
    indices_list = append(indices_list, i)}
}
train_numerical <- train[,-indices_list]


corr_matrix <- cor(train_numerical)
ggcorrplot(corr_matrix)

data.pca <- princomp(corr_matrix)
summary(data.pca)

# cor
#I'm looking at the correlations between the target and the other numeric columns
# 0 if not numeric, NA if there are missing values
correlations = rep(0,57)
for (i in 1:57){
  if  (is.numeric(train[,i])){
    correlations[i] = cor(train$target,train[,i])}
}
#colnames(train)

#linear regression
# These chosen variables can be changed!!!
lr_model_variables <- c("superhost", "zipcode_class", "booking_price_covers")
lr_model <- lm(train$target ~ train$superhost + train$zipcode_class + train$booking_price_covers , data = train_normal)
#compute rmse
compute_RMSE_train(lr_model, lr_model_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)
compute_RMSE_validation(lr_model, lr_model_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)

#only one variable
lr_model_variables <- c("superhost")
lr_model <- lm(train$target ~ train$superhost , data = train_normal)
#compute rmse
compute_RMSE_train(lr_model, lr_model_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)
compute_RMSE_validation(lr_model, lr_model_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)

# more variables
lr_model_variables <- c("superhost", "zipcode_class", "booking_price_covers", "property_max_guests", "property_beds", "host_id")
lr_model <- lm(train$target ~ train$superhost + train$zipcode_class + train$booking_price_covers + train$property_max_guests + train$property_beds + train$host_id , data = train_normal)
#compute rmse
compute_RMSE_train(lr_model, lr_model_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)
compute_RMSE_validation(lr_model, lr_model_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)

#linear regression with all variables
lr_model2 <- lm(train$target ~., data = train_normal)
lr_model2_variables <- c(1:57)
compute_RMSE_train(lr_model2, lr_model2_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)
compute_RMSE_validation(lr_model2, lr_model2_variables, pred_for_ppp = FALSE, pred_for_bc_target = FALSE)
#result: long running time, bigger error

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

#example variables
lassoModel_variables <- c("superhost", "zipcode_class", "booking_price_covers")

#compute rmse
compute_RMSE_train(ridgeModel, ridgeModel_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)
compute_RMSE_validation(ridgeModel, ridgeModel_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)


# Perform lasso regression
lassoModel <- glmnet(x, y, alpha = 1, lambda = 0.1)

# Print the lasso coefficients
print(coef(lassoModel))

# Example variables
lassoModel_variables <- c("superhost", "zipcode_class", "booking_price_covers")

#compute rmse
compute_RMSE_train(lassoModel, lassoModel_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)
compute_RMSE_validation(lassoModel, lassoModel_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)


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
predictions <- predict(rfModel, Xnew = validation)

# Example variables 
rfModel_variables <- c("superhost", "zipcode_class", "booking_price_covers")

#compute rmse
compute_RMSE_train(rfModel, rfModel_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)
compute_RMSE_validation(rfModel, rfModel_variables, pred_for_ppp = TRUE, pred_for_bc_target = FALSE)


# Gradient boosting
# Extreme Gradient Boosting: XGBoost
# These variables can be chosen!!!!

#XGBOOST
# Define the features to use in the model
features <- c("superhost", "zipcode_class", "booking_price_covers")

# Convert the training and test sets into xgb.DMatrix objects
dtrain <- xgb.DMatrix(as.matrix(train[, features]), label = train$target)
dtest <- xgb.DMatrix(as.matrix(validation[, features]), label = validation$target)

# Set the model parameters
params <- list(
  objective = "reg:squarederror",  # regression task with mean squared error
  max_depth = 6,  # maximum tree depth
  eta = 0.1,  # learning rate
  gamma = 0,  # minimum loss reduction to create a new tree split
  subsample = 1,  # subsample ratio of the training instances
  colsample_bytree = 1  # subsample ratio of columns when constructing each tree
)

# Train the xgboost model
bst <- xgb.train(params, dtrain, nrounds = 500)

# Make predictions on the test set
predictions <- predict(bst, dtest)

# Compute the RMSE
RMSE <- sqrt(mean((predictions - validation$target)^2))
print(paste0("RMSE: ", RMSE))



# Use clustering methods like Kmeans
# k-nearest neighbours
# variables can be changed
# number of neighbours k can also be changed

knn_model_variables <- c(14,32,34,45,55)

knn_model <- knn.reg(train = train[, knn_model_variables], y = train$target, test = validation[,knn_model_variables], k = 100)

#compute RMSE on training and validation set for knn
sqrt(1/length(train$target) * sum((knn_model$pred - train$target)^2))
sqrt(1/length(validation$target) * sum((knn_model$pred - validation$target)^2))

#version with all variables (that are numeric)
b<- c(1:57)
knn_model_variables <- b[-indices_list]

knn_model <- knn.reg(train = train[, knn_model_variables], y = train$target, test = validation[,knn_model_variables], k = 5)

sqrt(1/length(train$target) * sum((knn_model$pred - train$target)^2))
sqrt(1/length(validation$target) * sum((knn_model$pred - validation$target)^2))

# Make models that predict price based on different clusters of variables (like
# property_..., host_..., booking_..., reviews_.., etc.) and then combine these
# predictions using a decision tree

# If everyone has a model --> Put them together in an ensemble.

# Some other things to do:

# - Think about more models? :)

# - model evaluation
# - ensemble modelling
