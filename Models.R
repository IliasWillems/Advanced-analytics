
# Clear work space
rm(list = ls())

# Load in data
train <- read.csv("data/preprocessed_train.csv", header=TRUE)

# Load necessary packages
library(dplyr)
library(ODRF)

# Convert all character variables to factors
train <- train %>% mutate_if(is.character, as.factor)

# Drop a few more variables, for now
train <- select(train, -c("property_last_updated", "host_id", "reviews_first",
                          "reviews_last"))

################################################################################

# We still need to handle missing values for reviews

################################################################################

# Define some useful data frames
train_normal <- select(train, -c("bc_target"))
train_bc <- select(train, -c("target"))

train_normal_ppp <- select(train_normal, -c("booking_price_covers"))
train_normal_ppp$target <- train_normal_ppp$target/train_normal$booking_price_covers

train_bc_ppp <- select(train_bc, -c("booking_price_covers"))
train_bc_ppp$bc_target <- train_bc_ppp$bc_target/train_bc$booking_price_covers

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

################################################################################

# Baseline model: global mean of target variable.
# If your model has an RMSE that is larger than this model, something is really
# wrong.

mean_target <- mean(train$target)
compute_RMSE(mean_target, train$target)

################################################################################

# Oblique decision tree-based random forest

# We train the tree on the true target variable but taking booking_price_covers
# into account.
forest <- ODRF(target ~ superhost + zipcode_class + type_class + dist_nearest_city_center,
               data = train_normal_ppp, split = "mse",
               parallel = TRUE)

predictions <- predict(forest, Xnew = data.frame(train[, c("superhost", "zipcode_class", "type_class", "dist_nearest_city_center")]))
compute_RMSE(predictions*train_normal$booking_price_covers, train$target)

# Other approaches?

# Regular linear regression, but...
#   - Find out which variables are important using stepwise selection methods
#   - Find out which variables to transform and how
#     - In a any case, the target should be transformed (also allow negatives)
#   - Are the assumptions of the linear model satisfied?
#   - Think about useful interactions
#     - Or just include all and select automatically?

# Ridge/Lasso

# Generalized additive models

# For the above three models, maybe an improvement is possible by first fitting
# a classifier on a categorized target variable and then fitting the regression
# models for each of the categories. (maybe leads to overfitting)

# Random forests

# XGBoost

# Use clustering methods like Kmeans

# Make models that predict price based on different clusters of variables (like
# property_..., host_..., booking_..., reviews_.., etc.) and then combine these
# predictions using a decision tree

# If everyone has a model --> Put them together in an ensemble.

# Some other things to do:

# - If we agree on all the preprocessing steps, someone should extract all of
#   the models and parameters in that file and make a new script with which we
#   can preprocess the test data in the same way.

# - Think about more models? :)









