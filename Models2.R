# Clear work space
rm(list = ls())

# Load in data
train <- read.csv("data/preprocessed_train2.csv", header=TRUE)
validation <- read.csv("data/preprocessed_validation2.csv", header=TRUE)
test <- read.csv("data/preprocessed_test2.csv", header=TRUE)
df_all <- rbind(train, dplyr::select(validation, -c("property_id")))


# Load necessary packages
library(xgboost)
library(MASS)
library(readr)
library(stringr)
library(caret)
library(car)
library(dplyr)
library(Boruta)
library(reshape)
library(SHAPforxgboost)


#### Convert all character variables to factors ####

train <- train %>% mutate_if(is.character, as.factor)
validation <- validation %>% mutate_if(is.character, as.factor)
test <- test %>% mutate_if(is.character, as.factor)
df_all <- df_all %>% mutate_if(is.character, as.factor)

#
# XGboost models
#

#### XGboost on minimally preprocessed data ####

# Here we use only the original variables of the data set (that are pre-
# processed in the sense of interpolating missing values) and don't use any
# engineered variables. Note that we also exclude "property_zipcode" and
# "property_type" (since I forgot to put them in, apparently).

#
# Select variables to work on
#

# Train + validation
data_minimal_prep_idx <- c(1, 3:8, 11:22, 25:31, 36)
mp_df_all <- df_all[,data_minimal_prep_idx] 

dummies <- dummyVars(~ property_bed_type +  host_response_time +
                       booking_cancel_policy, data = mp_df_all)
dummy_var_col_idx <- which(colnames(mp_df_all) %in% c("property_bed_type",
                                                      "host_response_time",
                                                      "booking_cancel_policy"))
mp_df_all_ohe <- as.data.frame(predict(dummies, newdata = mp_df_all))
mp_df_all_combined <- cbind(mp_df_all[, -dummy_var_col_idx], mp_df_all_ohe)
mp_train <- mp_df_all_combined[1:nrow(train), ]
mp_validation <- mp_df_all_combined[(nrow(train) + 1):nrow(mp_df_all_combined), ]

# test set
colnames_minimal_prep <- colnames(train)[data_minimal_prep_idx]
mp_test_all <- test[,c("property_id", colnames_minimal_prep[-1])]

dummy_var_col_idx <- which(colnames(mp_test_all) %in% c("property_bed_type",
                                                        "host_response_time",
                                                        "booking_cancel_policy"))
mp_test_all_ohe <- as.data.frame(predict(dummies, newdata = mp_test_all))
mp_test <- cbind(mp_test_all[, -dummy_var_col_idx], mp_test_all_ohe)
mpxgb_test = xgb.DMatrix(data = data.matrix(mp_test[,-1]))

#
# Setup for tuning xgboost
#

mpxgb_train = xgb.DMatrix(data = data.matrix(mp_train[,-1]), label = mp_train[,1])
mpxgb_validation = xgb.DMatrix(data = data.matrix(mp_validation[,-1]))
mpxgb_all <- xgb.DMatrix(data = data.matrix(mp_df_all_combined[,-1]),
                         label = mp_df_all_combined[,1])

etas <- c(0.05, 0.1, 0.2, 0.3, 0.4)
gammas <- c(0, 1, 10, 100, 500, 1000)
max_depths <- c(6, 10, 15, 20)
subsamples <- c(0.1, 0.5, 1)
colsamples_bytree <- c(0.1, 0.5, 1)

# for testing
# eta <- 0.1; gamma <- 10; max_depth <- 10; subsample <- 0.5; colsample_bytree <- 0.1

#
# Grid search over hyperparameter grid and store the 10 best models
#

mp_min_test_rmse <- rep(Inf, 10)
mp_optimal_par_set <- list(list(), list(), list(), list(), list(),
                           list(), list(), list(), list(), list())
for (eta in etas) {
  for (gamma in gammas) {
    for (max_depth in max_depths) {
      for (subsample in subsamples) {
        for (colsample_bytree in colsamples_bytree) {
          params <- list(booster = "gbtree",
                         eta = eta,
                         gamma = gamma,
                         max_depth = max_depth, 
                         nround = 200, 
                         subsample = subsample,
                         colsample_bytree = colsample_bytree,
                         objective = "reg:squarederror")
          
          mp_xgbcv <- xgb.cv(params = params, data = mpxgb_train, nrounds = 100, nfold = 5,
                             showsd = T)
          
          if (max(mp_min_test_rmse) > min(mp_xgbcv$evaluation_log$test_rmse_mean)) {
            worst_model_idx <- which.max(mp_min_test_rmse)
            mp_min_test_rmse[worst_model_idx] <- min(mp_xgbcv$evaluation_log$test_rmse_mean)
            mp_optimal_par_set[[worst_model_idx]] <- list("eta" = eta,
                                                          "gamma" = gamma,
                                                          "max_depth" = max_depth,
                                                          "subsample" = subsample,
                                                          "colsample_bytree" = colsample_bytree,
                                                          "nrounds" = which.min(mp_xgbcv$evaluation_log$test_rmse_mean))
          }
        }
      }
    }
  }
}

#
# For these 10 best models, select a final model based on the validation set.
#

mp_validation_rmses <- rep(0, 10)
for (i in 1:10) {
  
  # Retrieve model parameters
  par_set <- mp_optimal_par_set[[i]]
  
  # Create model
  val_mp_xgb <- xgboost(data = mpxgb_train,
                        booster = "gbtree",
                        eta = par_set$eta,
                        gamma = par_set$gamma,
                        max_depth = par_set$max_depth, 
                        subsample = par_set$subsample,
                        colsample_bytree = par_set$colsample_bytree,
                        objective = "reg:squarederror",
                        nrounds = par_set$nrounds)
  
  # Make predictions on validation set
  val_pred <- predict(val_mp_xgb, mpxgb_validation)
  
  # Compute and store the validation set rmse
  mp_validation_rmses[i] <- sqrt((1/length(val_pred)) * sum((val_pred - mp_validation[,1])^2))
}

mp_final_par <- mp_optimal_par_set[[which.min(mp_validation_rmses)]]

#
# Retrain best model on all data (train + validation)
#

if (!exists("opt_mb_xgb")) {
  mp_final_par <- read.csv("mp_pars.csv")
}

# ... Without re-estimating the optimal number of rounds
opt_mp_xgb <- xgboost(data = mpxgb_all,
                      booster = "gbtree",
                      eta = mp_final_par$eta,
                      gamma = mp_final_par$gamma,
                      max_depth = mp_final_par$max_depth, 
                      subsample = mp_final_par$subsample,
                      colsample_bytree = mp_final_par$colsample_bytree,
                      objective = "reg:squarederror",
                      nrounds = mp_final_par$nrounds)

# ... with re-estimating the optimal number of rounds
mp_opt_params <- list(booster = "gbtree",
                      eta = mp_final_par$eta,
                      gamma = mp_final_par$gamma,
                      max_depth = mp_final_par$max_depth, 
                      subsample = mp_final_par$subsample,
                      colsample_bytree = mp_final_par$colsample_bytree,
                      objective = "reg:squarederror",
                      nrounds = 200)

mp_opt_xgbcv <- xgb.cv(params = mp_opt_params, data = mpxgb_train, nrounds = 100, nfold = 5,
                       showsd = T)
opt_mp_xgb2 <- xgboost(data = mpxgb_all,
                       booster = "gbtree",
                       eta = mp_final_par$eta,
                       gamma = mp_final_par$gamma,
                       max_depth = mp_final_par$max_depth, 
                       subsample = mp_final_par$subsample,
                       colsample_bytree = mp_final_par$colsample_bytree,
                       objective = "reg:squarederror",
                       nrounds = which.min(mp_opt_xgbcv$evaluation_log$test_rmse_mean))

#
# Make predictions for test set
#

mp_test_pred1 <- data.frame("ID" = mp_test_all[,"property_id"], "PRED" = predict(opt_mp_xgb, mpxgb_test))
mp_test_pred2 <- data.frame("ID" = mp_test_all[,"property_id"], "PRED" = predict(opt_mp_xgb2, mpxgb_test))

write.csv(mp_test_pred1, file = "mp_test_pred1.csv", row.names = FALSE)
write.csv(mp_test_pred2, file = "mp_test_pred2.csv", row.names = FALSE)
write.csv(as.data.frame(mp_final_par), "mp_pars.csv", row.names = FALSE)

#
# Model interpretation: variable importance plot
#

# Variable importance plot (based on increase in rmse per split)
model <- xgb.dump(opt_mp_xgb, with_stats = T)
names <- dimnames(data.matrix(mp_train[,-1]))[[2]]
importance_matrix <- xgb.importance(names, model = opt_mp_xgb)
xgb.plot.importance(importance_matrix[1:20,])

# Using Boruta package: Find important variables based on permutation based
# feature importance.
# IMPORTANT: The results seem to change each time you run this.

# helpful website:
# https://amirali-n.github.io/BorutaFeatureSelectionWithShapAnalysis/

xgb.boruta = Boruta(data.matrix(mp_df_all_combined[,-1]),
                    y = mp_df_all_combined[,1],
                    maxRuns = 100, 
                    doTrace = 2,
                    holdHistory = TRUE,
                    getImp = getImpXgboost,
                    eta = mp_final_par$eta,
                    gamma = mp_final_par$gamma,
                    max.depth = mp_final_par$max_depth, 
                    subsample = mp_final_par$subsample,
                    colsample_bytree = mp_final_par$colsample_bytree,
                    objective = "reg:squarederror",
                    nrounds = which.min(mp_opt_xgbcv$evaluation_log$test_rmse_mean), 
                    eval_metric = "rmse",
                    tree_method = "hist")

boruta_dec = attStats(xgb.boruta)
boruta_dec[boruta_dec$decision != "Rejected",]
# Only reviews_rating not rejected.

imp_features=row.names(boruta_dec)
boruta.imp.df=as.data.frame(xgb.boruta$ImpHistory)
boruta.imp.df=boruta.imp.df[,names(boruta.imp.df)%in%imp_features]
boruta.imp.df=melt(boruta.imp.df)
feature_order=with(boruta.imp.df, reorder(variable, value, median, order = TRUE))
boruta.imp.df$variable=factor(boruta.imp.df$variable, levels = levels(feature_order))

ggplot(boruta.imp.df, aes(y = variable, x = value)) + geom_boxplot()

# Using Shapley values
library(SHAPforxgboost)
shap_values=shap.values(xgb_model = opt_mp_xgb2, X_train = mpxgb_test)
shap_values$mean_shap_score

#### XGboost on 'all' data  ####

#
# Select variables to work on
#

# Train + validation set
data_extended_idx <- c(1, 3:8, 11:22, 25:49, 63:70)
ext_df_all <- df_all[,data_extended_idx] 

dummies <- dummyVars(~ property_bed_type + host_response_time +
                       booking_cancel_policy + host_location_country,
                     data = ext_df_all)
dummy_var_col_idx <- which(colnames(ext_df_all) %in% c("property_bed_type",
                                                       "host_response_time",
                                                       "booking_cancel_policy",
                                                       "host_location_country"))
ext_df_all_ohe <- as.data.frame(predict(dummies, newdata = ext_df_all))
ext_df_all_combined <- cbind(ext_df_all[, -dummy_var_col_idx], ext_df_all_ohe)
ext_train <- ext_df_all_combined[1:nrow(train), ]
ext_validation <- ext_df_all_combined[(nrow(train) + 1):nrow(ext_df_all_combined), ]

# test set
colnames_ext <- colnames(train)[data_extended_idx]
ext_test_all <- test[,c("property_id", colnames_ext[-1])]

dummy_var_col_idx <- which(colnames(ext_df_all) %in% c("property_bed_type",
                                                       "host_response_time",
                                                       "booking_cancel_policy",
                                                       "host_location_country"))
ext_test_all_ohe <- as.data.frame(predict(dummies, newdata = ext_test_all))
ext_test <- cbind(ext_test_all[, -dummy_var_col_idx], ext_test_all_ohe)
ext_xgb_test = xgb.DMatrix(data = data.matrix(ext_test[,-1]))

#
# Setup for tuning XGboost
#

ext_xgb_train = xgb.DMatrix(data = data.matrix(ext_train[,-1]), label = ext_train[,1])
ext_xgb_validation = xgb.DMatrix(data = data.matrix(ext_validation[,-1]))
ext_xgb_all <- xgb.DMatrix(data = data.matrix(ext_df_all_combined[,-1]),
                           label = ext_df_all_combined[,1])

etas <- c(0.05, 0.1, 0.2, 0.3, 0.4)
gammas <- c(0, 1, 10, 100, 500, 1000)
max_depths <- c(6, 10, 15, 20)
subsamples <- c(0.1, 0.5, 1)
colsamples_bytree <- c(0.1, 0.5, 1)

# Grid search over hyperparameter grid and store the 10 best models
ext_min_test_rmse <- rep(Inf, 10)
ext_optimal_par_set <- list(list(), list(), list(), list(), list(),
                            list(), list(), list(), list(), list())
for (eta in etas) {
  for (gamma in gammas) {
    for (max_depth in max_depths) {
      for (subsample in subsamples) {
        for (colsample_bytree in colsamples_bytree) {
          params <- list(booster = "gbtree",
                         eta = eta,
                         gamma = gamma,
                         max_depth = max_depth, 
                         nround = 200, 
                         subsample = subsample,
                         colsample_bytree = colsample_bytree,
                         objective = "reg:squarederror")
          
          ext_xgbcv <- xgb.cv(params = params, data = ext_xgb_train, nrounds = 100, nfold = 5)
          
          # If the worst model of the 5 best models is worse than this model,
          # replace it with this model.
          if (max(ext_min_test_rmse) > min(ext_xgbcv$evaluation_log$test_rmse_mean)) {
            worst_model_idx <- which.max(ext_min_test_rmse)
            ext_min_test_rmse[worst_model_idx] <- min(ext_xgbcv$evaluation_log$test_rmse_mean)
            ext_optimal_par_set[[worst_model_idx]] <- list("eta" = eta,
                                                         "gamma" = gamma,
                                                         "max_depth" = max_depth,
                                                         "subsample" = subsample,
                                                         "colsample_bytree" = colsample_bytree,
                                                         "nrounds" = which.min(ext_xgbcv$evaluation_log$test_rmse_mean))
          }
        }
      }
    }
  }
}


#
# For these 10 best models, select a final model based on the validation set.
#

ext_validation_rmses <- rep(0, 10)
for (i in 1:10) {
  
  # Retrieve model parameters
  par_set <- ext_optimal_par_set[[i]]
  
  # Create model
  val_ext_xgb <- xgboost(data = ext_xgb_train,
                        booster = "gbtree",
                        eta = par_set$eta,
                        gamma = par_set$gamma,
                        max_depth = par_set$max_depth, 
                        subsample = par_set$subsample,
                        colsample_bytree = par_set$colsample_bytree,
                        objective = "reg:squarederror",
                        nrounds = par_set$nrounds)
  
  # Make predictions on validation set
  val_pred <- predict(val_ext_xgb, ext_xgb_validation)
  
  # Compute and store the validation set rmse
  ext_validation_rmses[i] <- sqrt((1/length(val_pred)) * sum((val_pred - ext_validation[,1])^2))
}

ext_final_par <- ext_optimal_par_set[[which.min(ext_validation_rmses)]]

#
# Retrain best model on all data (train + validation)
#

if (!exists("ext_final_par")) {
  ext_final_par <- read.csv("ext_pars")
} 

# ... Without re-estimating the optimal number of rounds
opt_ext_xgb <- xgboost(data = ext_xgb_all,
                       booster = "gbtree",
                       eta = ext_final_par$eta,
                       gamma = ext_final_par$gamma,
                       max_depth = ext_final_par$max_depth, 
                       subsample = ext_final_par$subsample,
                       colsample_bytree = ext_final_par$colsample_bytree,
                       objective = "reg:squarederror",
                       nrounds = ext_final_par$nrounds)

# ... with re-estimating the optimal number of rounds
ext_opt_params <- list(booster = "gbtree",
                       eta = ext_final_par$eta,
                       gamma = ext_final_par$gamma,
                       max_depth = ext_final_par$max_depth, 
                       subsample = ext_final_par$subsample,
                       colsample_bytree = ext_final_par$colsample_bytree,
                       objective = "reg:squarederror",
                       nrounds = 200)

ext_opt_xgbcv <- xgb.cv(params = ext_opt_params, data = ext_xgb_train, nrounds = 100, nfold = 5,
                        showsd = T)
opt_ext_xgb2 <- xgboost(data = ext_xgb_all,
                       booster = "gbtree",
                       eta = ext_final_par$eta,
                       gamma = ext_final_par$gamma,
                       max_depth = ext_final_par$max_depth, 
                       subsample = ext_final_par$subsample,
                       colsample_bytree = ext_final_par$colsample_bytree,
                       objective = "reg:squarederror",
                       nrounds = which.min(ext_opt_xgbcv$evaluation_log$test_rmse_mean))

#
# Make predictions for test set
#

ext_test_pred1 <- data.frame("ID" = ext_test_all[,"property_id"], "PRED" = predict(opt_ext_xgb, ext_xgb_test))
ext_test_pred2 <- data.frame("ID" = ext_test_all[,"property_id"], "PRED" = predict(opt_ext_xgb2, ext_xgb_test))

write.csv(ext_test_pred1, file = "ext_test_pred1.csv", row.names = FALSE)
write.csv(ext_test_pred2, file = "ext_test_pred2.csv", row.names = FALSE)
write.csv(as.data.frame(ext_final_par), "ext_pars.csv", row.names = FALSE)

#
# Model interpretation: variable importance plot
#

model <- xgb.dump(opt_ext_xgb, with.stats = T)
names <- dimnames(data.matrix(ext_train[,-1]))[[2]]
importance_matrix <- xgb.importance(names, model = opt_ext_xgb)
xgb.plot.importance(importance_matrix[1:20,])

#### XGboost on most important data ####

# Since from the previous optimal parameters it turned out that the hyperparam-
# eter tuning is doing everything it can to prevent overfitting, it might be a
# good idea to only work with a select amount of parameters.

# Intuitively, this would mean that we also have to include property_sqfeet into
# the model

#
# Select variables to work on
#

# Train + validation set
data_important_colnames <- c("target",
                             "property_room_type",
                             "property_bathrooms",
                             "host_response_rate",
                             "booking_price_covers",
                             "booking_availability_90",
                             "booking_cancel_policy",
                             "reviews_num",
                             "reviews_rating",
                             "zipcode_class",
                             "dist_nearest_city_center",
                             "type_class",
                             "property_last_updated_numerical",
                             "review_period",
                             "summary_score",
                             "property_sqfeet")
data_important_idx <- which(colnames(train) %in% data_important_colnames)
imp_df_all <- df_all[,data_important_idx] 

dummies <- dummyVars(~ booking_cancel_policy, data = imp_df_all)
dummy_var_col_idx <- which(colnames(imp_df_all) %in% c("booking_cancel_policy"))
imp_df_all_ohe <- as.data.frame(predict(dummies, newdata = imp_df_all))
imp_df_all_combined <- cbind(imp_df_all[, -dummy_var_col_idx], imp_df_all_ohe)
imp_train <- imp_df_all_combined[1:nrow(train), ]
imp_validation <- imp_df_all_combined[(nrow(train) + 1):nrow(imp_df_all_combined), ]

# test set
colnames_imp <- colnames(train)[data_important_idx]
imp_test_all <- test[,c("property_id", colnames_imp[-1])]

dummy_var_col_idx <- which(colnames(imp_df_all) %in% c("booking_cancel_policy"))
imp_test_all_ohe <- as.data.frame(predict(dummies, newdata = imp_test_all))
imp_test <- cbind(imp_test_all[, -dummy_var_col_idx], imp_test_all_ohe)
imp_xgb_test = xgb.DMatrix(data = data.matrix(imp_test[,-1]))

#
# Setup for tuning XGboost
#

imp_xgb_train = xgb.DMatrix(data = data.matrix(imp_train[,-1]), label = imp_train[,1])
imp_xgb_validation = xgb.DMatrix(data = data.matrix(imp_validation[,-1]))
imp_xgb_all <- xgb.DMatrix(data = data.matrix(imp_df_all_combined[,-1]),
                           label = imp_df_all_combined[,1])

etas <- c(0.05, 0.1, 0.2, 0.3, 0.4)
gammas <- c(0, 1, 10, 100, 500, 1000)
max_depths <- c(6, 10, 15, 20)
subsamples <- c(0.1, 0.5, 1)
colsamples_bytree <- c(0.1, 0.5, 1)

# Grid search over hyperparameter grid and store the 10 best models
imp_min_test_rmse <- rep(Inf, 10)
imp_optimal_par_set <- list(list(), list(), list(), list(), list(),
                            list(), list(), list(), list(), list())
for (eta in etas) {
  for (gamma in gammas) {
    for (max_depth in max_depths) {
      for (subsample in subsamples) {
        for (colsample_bytree in colsamples_bytree) {
          params <- list(booster = "gbtree",
                         eta = eta,
                         gamma = gamma,
                         max_depth = max_depth, 
                         nround = 200, 
                         subsample = subsample,
                         colsample_bytree = colsample_bytree,
                         objective = "reg:squarederror")
          
          imp_xgbcv <- xgb.cv(params = params, data = imp_xgb_train, nrounds = 100, nfold = 5)
          
          # If the worst model of the 5 best models is worse than this model,
          # replace it with this model.
          if (max(imp_min_test_rmse) > min(imp_xgbcv$evaluation_log$test_rmse_mean)) {
            worst_model_idx <- which.max(imp_min_test_rmse)
            imp_min_test_rmse[worst_model_idx] <- min(imp_xgbcv$evaluation_log$test_rmse_mean)
            imp_optimal_par_set[[worst_model_idx]] <- list("eta" = eta,
                                                           "gamma" = gamma,
                                                           "max_depth" = max_depth,
                                                           "subsample" = subsample,
                                                           "colsample_bytree" = colsample_bytree,
                                                           "nrounds" = which.min(imp_xgbcv$evaluation_log$test_rmse_mean))
          }
        }
      }
    }
  }
}


#
# For these 10 best models, select a final model based on the validation set.
#

imp_validation_rmses <- rep(0, 10)
for (i in 1:10) {
  
  # Retrieve model parameters
  par_set <- imp_optimal_par_set[[i]]
  
  # Create model
  val_imp_xgb <- xgboost(data = imp_xgb_train,
                         booster = "gbtree",
                         eta = par_set$eta,
                         gamma = par_set$gamma,
                         max_depth = par_set$max_depth, 
                         subsample = par_set$subsample,
                         colsample_bytree = par_set$colsample_bytree,
                         objective = "reg:squarederror",
                         nrounds = par_set$nrounds)
  
  # Make predictions on validation set
  val_pred <- predict(val_imp_xgb, imp_xgb_validation)
  
  # Compute and store the validation set rmse
  imp_validation_rmses[i] <- sqrt((1/length(val_pred)) * sum((val_pred - imp_validation[,1])^2))
}

imp_final_par <- imp_optimal_par_set[[which.min(imp_validation_rmses)]]

#
# Retrain best model on all data (train + validation)
#

# ... Without re-estimating the optimal number of rounds
opt_imp_xgb <- xgboost(data = imp_xgb_all,
                       booster = "gbtree",
                       eta = imp_final_par$eta,
                       gamma = imp_final_par$gamma,
                       max_depth = imp_final_par$max_depth, 
                       subsample = imp_final_par$subsample,
                       colsample_bytree = imp_final_par$colsample_bytree,
                       objective = "reg:squarederror",
                       nrounds = imp_final_par$nrounds)

# ... with re-estimating the optimal number of rounds
imp_opt_params <- list(booster = "gbtree",
                       eta = imp_final_par$eta,
                       gamma = imp_final_par$gamma,
                       max_depth = imp_final_par$max_depth, 
                       subsample = imp_final_par$subsample,
                       colsample_bytree = imp_final_par$colsample_bytree,
                       objective = "reg:squarederror",
                       nrounds = 200)

imp_opt_xgbcv <- xgb.cv(params = imp_opt_params, data = imp_xgb_train, nrounds = 100, nfold = 5,
                        showsd = T)
opt_imp_xgb2 <- xgboost(data = imp_xgb_all,
                        booster = "gbtree",
                        eta = imp_final_par$eta,
                        gamma = imp_final_par$gamma,
                        max_depth = imp_final_par$max_depth, 
                        subsample = imp_final_par$subsample,
                        colsample_bytree = imp_final_par$colsample_bytree,
                        objective = "reg:squarederror",
                        nrounds = which.min(imp_opt_xgbcv$evaluation_log$test_rmse_mean))

#
# Make predictions for test set
#

imp_test_pred1 <- data.frame("ID" = imp_test_all[,"property_id"], "PRED" = predict(opt_imp_xgb, imp_xgb_test))
imp_test_pred2 <- data.frame("ID" = imp_test_all[,"property_id"], "PRED" = predict(opt_imp_xgb2, imp_xgb_test))

write.csv(imp_test_pred1, file = "imp_test_pred1.csv", row.names = FALSE)
write.csv(imp_test_pred2, file = "imp_test_pred2.csv", row.names = FALSE)
write.csv(as.data.frame(imp_final_par), "imp_pars.csv", row.names = FALSE)

#
# Model interpretation: variable importance plot
#

model <- xgb.dump(opt_imp_xgb, with.stats = T)
names <- dimnames(data.matrix(imp_train[,-1]))[[2]]
importance_matrix <- xgb.importance(names, model = opt_imp_xgb)
xgb.plot.importance(importance_matrix[1:20,])

#### XGboost (dart) on 'all' data  ####

#
# Select variables to work on
#

# Train + validation set
data_dart_idx <- c(1, 3:8, 11:22, 25:49, 63:70)
dart_df_all <- df_all[,data_dart_idx] 

dummies <- dummyVars(~ property_bed_type + host_response_time +
                       booking_cancel_policy + host_location_country,
                     data = dart_df_all)
dummy_var_col_idx <- which(colnames(dart_df_all) %in% c("property_bed_type",
                                                       "host_response_time",
                                                       "booking_cancel_policy",
                                                       "host_location_country"))
dart_df_all_ohe <- as.data.frame(predict(dummies, newdata = dart_df_all))
dart_df_all_combined <- cbind(dart_df_all[, -dummy_var_col_idx], dart_df_all_ohe)
dart_train <- dart_df_all_combined[1:nrow(train), ]
dart_validation <- dart_df_all_combined[(nrow(train) + 1):nrow(dart_df_all_combined), ]

# test set
colnames_dart <- colnames(train)[data_dart_idx]
dart_test_all <- test[,c("property_id", colnames_dart[-1])]

dummy_var_col_idx <- which(colnames(dart_df_all) %in% c("property_bed_type",
                                                       "host_response_time",
                                                       "booking_cancel_policy",
                                                       "host_location_country"))
dart_test_all_ohe <- as.data.frame(predict(dummies, newdata = dart_test_all))
dart_test <- cbind(dart_test_all[, -dummy_var_col_idx], dart_test_all_ohe)
dart_xgb_test = xgb.DMatrix(data = data.matrix(dart_test[,-1]))

#
# Setup for tuning XGboost
#

dart_xgb_train = xgb.DMatrix(data = data.matrix(dart_train[,-1]), label = dart_train[,1])
dart_xgb_validation = xgb.DMatrix(data = data.matrix(dart_validation[,-1]))
dart_xgb_all <- xgb.DMatrix(data = data.matrix(dart_df_all_combined[,-1]),
                           label = dart_df_all_combined[,1])

etas <- c(0.05, 0.1, 0.2, 0.3, 0.4)
gammas <- c(0, 1, 10, 100, 500, 1000)
max_depths <- c(6, 10, 15, 20)
subsamples <- c(0.1, 0.5, 1)
colsamples_bytree <- c(0.1, 0.5, 1)

# Grid search over hyperparameter grid and store the 10 best models
dart_min_test_rmse <- rep(Inf, 10)
dart_optimal_par_set <- list(list(), list(), list(), list(), list(),
                            list(), list(), list(), list(), list())
for (eta in etas) {
  for (gamma in gammas) {
    for (max_depth in max_depths) {
      for (subsample in subsamples) {
        for (colsample_bytree in colsamples_bytree) {
          params <- list(booster = "dart",
                         eta = eta,
                         gamma = gamma,
                         max_depth = max_depth, 
                         nround = 200, 
                         subsample = subsample,
                         colsample_bytree = colsample_bytree,
                         objective = "reg:squarederror")
          
          dart_xgbcv <- xgb.cv(params = params, data = dart_xgb_train, nrounds = 100, nfold = 5)
          
          # If the worst model of the 5 best models is worse than this model,
          # replace it with this model.
          if (max(dart_min_test_rmse) > min(dart_xgbcv$evaluation_log$test_rmse_mean)) {
            worst_model_idx <- which.max(dart_min_test_rmse)
            dart_min_test_rmse[worst_model_idx] <- min(dart_xgbcv$evaluation_log$test_rmse_mean)
            dart_optimal_par_set[[worst_model_idx]] <- list("eta" = eta,
                                                           "gamma" = gamma,
                                                           "max_depth" = max_depth,
                                                           "subsample" = subsample,
                                                           "colsample_bytree" = colsample_bytree,
                                                           "nrounds" = which.min(dart_xgbcv$evaluation_log$test_rmse_mean))
          }
        }
      }
    }
  }
}


#
# For these 10 best models, select a final model based on the validation set.
#

dart_validation_rmses <- rep(0, 10)
for (i in 1:10) {
  
  # Retrieve model parameters
  par_set <- dart_optimal_par_set[[i]]
  
  # Create model
  val_dart_xgb <- xgboost(data = dart_xgb_train,
                         booster = "dart",
                         eta = par_set$eta,
                         gamma = par_set$gamma,
                         max_depth = par_set$max_depth, 
                         subsample = par_set$subsample,
                         colsample_bytree = par_set$colsample_bytree,
                         objective = "reg:squarederror",
                         nrounds = par_set$nrounds)
  
  # Make predictions on validation set
  val_pred <- predict(val_dart_xgb, dart_xgb_validation)
  
  # Compute and store the validation set rmse
  dart_validation_rmses[i] <- sqrt((1/length(val_pred)) * sum((val_pred - dart_validation[,1])^2))
}

dart_final_par <- dart_optimal_par_set[[which.min(dart_validation_rmses)]]

#
# Retrain best model on all data (train + validation)
#

if (!exists("dart_final_par")) {
  dart_final_par <- read.csv("dart_pars")
} 

# ... Without re-estimating the optimal number of rounds
opt_dart_xgb <- xgboost(data = dart_xgb_all,
                       booster = "dart",
                       eta = dart_final_par$eta,
                       gamma = dart_final_par$gamma,
                       max_depth = dart_final_par$max_depth, 
                       subsample = dart_final_par$subsample,
                       colsample_bytree = dart_final_par$colsample_bytree,
                       objective = "reg:squarederror",
                       nrounds = dart_final_par$nrounds)

# ... with re-estimating the optimal number of rounds
dart_opt_params <- list(booster = "dart",
                       eta = dart_final_par$eta,
                       gamma = dart_final_par$gamma,
                       max_depth = dart_final_par$max_depth, 
                       subsample = dart_final_par$subsample,
                       colsample_bytree = dart_final_par$colsample_bytree,
                       objective = "reg:squarederror",
                       nrounds = 200)

dart_opt_xgbcv <- xgb.cv(params = dart_opt_params, data = dart_xgb_train, nrounds = 100, nfold = 5,
                        showsd = T)
opt_dart_xgb2 <- xgboost(data = dart_xgb_all,
                        booster = "dart",
                        eta = dart_final_par$eta,
                        gamma = dart_final_par$gamma,
                        max_depth = dart_final_par$max_depth, 
                        subsample = dart_final_par$subsample,
                        colsample_bytree = dart_final_par$colsample_bytree,
                        objective = "reg:squarederror",
                        nrounds = which.min(dart_opt_xgbcv$evaluation_log$test_rmse_mean))

#
# Make predictions for test set
#

dart_test_pred1 <- data.frame("ID" = dart_test_all[,"property_id"], "PRED" = predict(opt_dart_xgb, dart_xgb_test))
dart_test_pred2 <- data.frame("ID" = dart_test_all[,"property_id"], "PRED" = predict(opt_dart_xgb2, dart_xgb_test))

write.csv(dart_test_pred1, file = "dart_test_pred1.csv", row.names = FALSE)
write.csv(dart_test_pred2, file = "dart_test_pred2.csv", row.names = FALSE)
write.csv(as.data.frame(dart_final_par), "dart_pars.csv", row.names = FALSE)

#
# Model interpretation: variable importance plot
#

model <- xgb.dump(opt_dart_xgb, with.stats = T)
names <- dimnames(data.matrix(dart_train[,-1]))[[2]]
importance_matrix <- xgb.importance(names, model = opt_dart_xgb)
xgb.plot.importance(importance_matrix[1:20,])

#### XGboost on most important data (minor changes) ####

# Same as "XGboost on most important data" but this time we use the original
# variables 'property_zipcode' and 'property_type' instead of the releveled ones

# Also, to reduce compute we limit the grid only search over the seemingly most
# promising optimal parameters.

#
# Select variables to work on
#

# Train + validation set
data_important2_colnames <- c("target",
                             "property_room_type",
                             "property_bathrooms",
                             "host_response_rate",
                             "booking_price_covers",
                             "booking_availability_90",
                             "booking_cancel_policy",
                             "reviews_num",
                             "reviews_rating",
                             "property_zipcode",
                             "dist_nearest_city_center",
                             "property_type",
                             "property_last_updated_numerical",
                             "review_period",
                             "summary_score",
                             "property_sqfeet")
data_important2_idx <- which(colnames(train) %in% data_important2_colnames)
imp2_df_all <- df_all[,data_important2_idx] 

dummies <- dummyVars(~ booking_cancel_policy, data = imp2_df_all)
dummy_var_col_idx <- which(colnames(imp2_df_all) %in% c("booking_cancel_policy"))
imp2_df_all_ohe <- as.data.frame(predict(dummies, newdata = imp2_df_all))
imp2_df_all_combined <- cbind(imp2_df_all[, -dummy_var_col_idx], imp2_df_all_ohe)
imp2_train <- imp2_df_all_combined[1:nrow(train), ]
imp2_validation <- imp2_df_all_combined[(nrow(train) + 1):nrow(imp2_df_all_combined), ]

# test set
colnames_imp2 <- colnames(train)[data_important2_idx]
imp2_test_all <- test[,c("property_id", colnames_imp2[-1])]

dummy_var_col_idx <- which(colnames(imp2_df_all) %in% c("booking_cancel_policy"))
imp2_test_all_ohe <- as.data.frame(predict(dummies, newdata = imp2_test_all))
imp2_test <- cbind(imp2_test_all[, -dummy_var_col_idx], imp2_test_all_ohe)
imp2_xgb_test = xgb.DMatrix(data = data.matrix(imp2_test[,-1]))

#
# Setup for tuning XGboost
#

imp2_xgb_train = xgb.DMatrix(data = data.matrix(imp2_train[,-1]), label = imp2_train[,1])
imp2_xgb_validation = xgb.DMatrix(data = data.matrix(imp2_validation[,-1]))
imp2_xgb_all <- xgb.DMatrix(data = data.matrix(imp2_df_all_combined[,-1]),
                           label = imp2_df_all_combined[,1])

etas <- c(0.05, 0.1)
gammas <- c(0, 1, 10, 100, 500)
max_depths <- c(6, 10, 15)
subsamples <- c(0.1, 0.5)
colsamples_bytree <- c(0.1, 0.5)

# Grid search over hyperparameter grid and store the 10 best models
imp2_min_test_rmse <- rep(Inf, 10)
imp2_optimal_par_set <- list(list(), list(), list(), list(), list(),
                            list(), list(), list(), list(), list())
for (eta in etas) {
  for (gamma in gammas) {
    for (max_depth in max_depths) {
      for (subsample in subsamples) {
        for (colsample_bytree in colsamples_bytree) {
          params <- list(booster = "gbtree",
                         eta = eta,
                         gamma = gamma,
                         max_depth = max_depth, 
                         nround = 200, 
                         subsample = subsample,
                         colsample_bytree = colsample_bytree,
                         objective = "reg:squarederror")
          
          imp2_xgbcv <- xgb.cv(params = params, data = imp2_xgb_train, nrounds = 100, nfold = 5)
          
          # If the worst model of the 5 best models is worse than this model,
          # replace it with this model.
          if (max(imp2_min_test_rmse) > min(imp2_xgbcv$evaluation_log$test_rmse_mean)) {
            worst_model_idx <- which.max(imp2_min_test_rmse)
            imp2_min_test_rmse[worst_model_idx] <- min(imp2_xgbcv$evaluation_log$test_rmse_mean)
            imp2_optimal_par_set[[worst_model_idx]] <- list("eta" = eta,
                                                           "gamma" = gamma,
                                                           "max_depth" = max_depth,
                                                           "subsample" = subsample,
                                                           "colsample_bytree" = colsample_bytree,
                                                           "nrounds" = which.min(imp2_xgbcv$evaluation_log$test_rmse_mean))
          }
        }
      }
    }
  }
}


#
# For these 10 best models, select a final model based on the validation set.
#

imp2_validation_rmses <- rep(0, 10)
for (i in 1:10) {
  
  # Retrieve model parameters
  par_set <- imp2_optimal_par_set[[i]]
  
  # Create model
  val_imp2_xgb <- xgboost(data = imp2_xgb_train,
                         booster = "gbtree",
                         eta = par_set$eta,
                         gamma = par_set$gamma,
                         max_depth = par_set$max_depth, 
                         subsample = par_set$subsample,
                         colsample_bytree = par_set$colsample_bytree,
                         objective = "reg:squarederror",
                         nrounds = par_set$nrounds)
  
  # Make predictions on validation set
  val_pred <- predict(val_imp2_xgb, imp2_xgb_validation)
  
  # Compute and store the validation set rmse
  imp2_validation_rmses[i] <- sqrt((1/length(val_pred)) * sum((val_pred - imp2_validation[,1])^2))
}

imp2_final_par <- imp2_optimal_par_set[[which.min(imp2_validation_rmses)]]

#
# Retrain best model on all data (train + validation)
#

# ... Without re-estimating the optimal number of rounds
opt_imp2_xgb <- xgboost(data = imp2_xgb_all,
                       booster = "gbtree",
                       eta = imp2_final_par$eta,
                       gamma = imp2_final_par$gamma,
                       max_depth = imp2_final_par$max_depth, 
                       subsample = imp2_final_par$subsample,
                       colsample_bytree = imp2_final_par$colsample_bytree,
                       objective = "reg:squarederror",
                       nrounds = imp2_final_par$nrounds)

# ... with re-estimating the optimal number of rounds
imp2_opt_params <- list(booster = "gbtree",
                       eta = imp2_final_par$eta,
                       gamma = imp2_final_par$gamma,
                       max_depth = imp2_final_par$max_depth, 
                       subsample = imp2_final_par$subsample,
                       colsample_bytree = imp2_final_par$colsample_bytree,
                       objective = "reg:squarederror",
                       nrounds = 200)

imp2_opt_xgbcv <- xgb.cv(params = imp2_opt_params, data = imp2_xgb_train, nrounds = 100, nfold = 5,
                        showsd = T)
opt_imp2_xgb2 <- xgboost(data = imp2_xgb_all,
                        booster = "gbtree",
                        eta = imp2_final_par$eta,
                        gamma = imp2_final_par$gamma,
                        max_depth = imp2_final_par$max_depth, 
                        subsample = imp2_final_par$subsample,
                        colsample_bytree = imp2_final_par$colsample_bytree,
                        objective = "reg:squarederror",
                        nrounds = which.min(imp2_opt_xgbcv$evaluation_log$test_rmse_mean))

#
# Make predictions for test set
#

imp2_test_pred1 <- data.frame("ID" = imp2_test_all[,"property_id"], "PRED" = predict(opt_imp2_xgb, imp2_xgb_test))
imp2_test_pred2 <- data.frame("ID" = imp2_test_all[,"property_id"], "PRED" = predict(opt_imp2_xgb2, imp2_xgb_test))

write.csv(imp2_test_pred1, file = "imp2_test_pred1.csv", row.names = FALSE)
write.csv(imp2_test_pred2, file = "imp2_test_pred2.csv", row.names = FALSE)
write.csv(as.data.frame(imp2_final_par), "imp2_pars.csv", row.names = FALSE)

#
# Model interpretation: variable importance plot
#

model <- xgb.dump(opt_imp2_xgb, with_stats = T)
names <- dimnames(data.matrix(imp2_train[,-1]))[[2]]
importance_matrix <- xgb.importance(names, model = opt_imp2_xgb)
xgb.plot.importance(importance_matrix[1:20,])

#### XGboost on minimally preprocessed data (some changes) ####

# This is the same as "XGboost on minimally preprocessed data" but now using the
# original variables 'property_type', 'property_zipcode' and 'property_sqfeet'.
# Additionally, we increase the grid search to also change booster to "dart",
# but decrease the search over other parameters when booster "gbtree" is
# selected. The latter is done to decrease the lenghty compute time of this
# parameter tuning.

# Note that "property_sqfeet" has a lot of missing values but luckily, decision
# trees can deal with that.

# Since the variable "dist_nearest_city_center" turned out to be very important
# in predicting the price, it might be a good idea to include it in the model.
# Therefore in the code below, we allow for the option to do so.

#
# Select variables to work on
#

# Train + validation
include_distances_to_city_center = FALSE
if (include_distances_to_city_center) {
  data_minimal_prep2_idx <- c(1, 3:8, 11:22, 25:31, 36, 71:73, 34)
} else {
  data_minimal_prep2_idx <- c(1, 3:8, 11:22, 25:31, 36, 71:73)
}

mp2_df_all <- df_all[,data_minimal_prep2_idx] 

dummies <- dummyVars(~ property_bed_type +  host_response_time +
                       booking_cancel_policy, data = mp2_df_all)
dummy_var_col_idx <- which(colnames(mp2_df_all) %in% c("property_bed_type",
                                                      "host_response_time",
                                                      "booking_cancel_policy"))
mp2_df_all_ohe <- as.data.frame(predict(dummies, newdata = mp2_df_all))
mp2_df_all_combined <- cbind(mp2_df_all[, -dummy_var_col_idx], mp2_df_all_ohe)
mp2_train <- mp2_df_all_combined[1:nrow(train), ]
mp2_validation <- mp2_df_all_combined[(nrow(train) + 1):nrow(mp2_df_all_combined), ]

# test set
colnames_minimal_prep2 <- colnames(train)[data_minimal_prep2_idx]
mp2_test_all <- test[,c("property_id", colnames_minimal_prep2[-1])]

dummy_var_col_idx <- which(colnames(mp2_test_all) %in% c("property_bed_type",
                                                        "host_response_time",
                                                        "booking_cancel_policy"))
mp2_test_all_ohe <- as.data.frame(predict(dummies, newdata = mp2_test_all))
mp2_test <- cbind(mp2_test_all[, -dummy_var_col_idx], mp2_test_all_ohe)
mp2xgb_test = xgb.DMatrix(data = data.matrix(mp2_test[,-1]))

#
# Setup for tuning xgboost
#

mp2xgb_train = xgb.DMatrix(data = data.matrix(mp2_train[,-1]), label = mp2_train[,1])
mp2xgb_validation = xgb.DMatrix(data = data.matrix(mp2_validation[,-1]))
mp2xgb_all <- xgb.DMatrix(data = data.matrix(mp2_df_all_combined[,-1]),
                         label = mp2_df_all_combined[,1])

etas <- c(0.05, 0.1, 0.2, 0.3, 0.4)
gammas <- c(0, 1, 10, 100, 500, 1000)
max_depths <- c(6, 10, 15, 20)
subsamples <- c(0.1, 0.5, 1)
colsamples_bytree <- c(0.1, 0.5, 1)
boosters <- c("gbtree", "dart")

#
# Grid search over hyperparameter grid and store the 10 best models
#

mp2_min_test_rmse <- rep(Inf, 10)
mp2_optimal_par_set <- list(list(), list(), list(), list(), list(),
                           list(), list(), list(), list(), list())
for (booster in boosters) {
  for (eta in etas) {
    for (gamma in gammas) {
      for (max_depth in max_depths) {
        for (subsample in subsamples) {
          for (colsample_bytree in colsamples_bytree) {
            
            if (eta > 0.2 | subsample == 1 | colsample_bytree == 1 | max_depth > 15) {
              next
            }
            
            params <- list(booster = booster,
                           eta = eta,
                           gamma = gamma,
                           max_depth = max_depth, 
                           nround = 200, 
                           subsample = subsample,
                           colsample_bytree = colsample_bytree,
                           objective = "reg:squarederror")
            
            mp2_xgbcv <- xgb.cv(params = params, data = mp2xgb_train, nrounds = 100, nfold = 5,
                               showsd = T)
            
            if (max(mp2_min_test_rmse) > min(mp2_xgbcv$evaluation_log$test_rmse_mean)) {
              worst_model_idx <- which.max(mp2_min_test_rmse)
              mp2_min_test_rmse[worst_model_idx] <- min(mp2_xgbcv$evaluation_log$test_rmse_mean)
              mp2_optimal_par_set[[worst_model_idx]] <- list("booster" = booster,
                                                            "eta" = eta,
                                                            "gamma" = gamma,
                                                            "max_depth" = max_depth,
                                                            "subsample" = subsample,
                                                            "colsample_bytree" = colsample_bytree,
                                                            "nrounds" = which.min(mp2_xgbcv$evaluation_log$test_rmse_mean))
            }
          }
        }
      }
    }
  }
}

#
# For these 10 best models, select a final model based on the validation set.
#

mp2_validation_rmses <- rep(0, 10)
for (i in 1:10) {
  
  # Retrieve model parameters
  par_set <- mp2_optimal_par_set[[i]]
  
  # Create model
  val_mp2_xgb <- xgboost(data = mp2xgb_train,
                        booster = par_set$booster,
                        eta = par_set$eta,
                        gamma = par_set$gamma,
                        max_depth = par_set$max_depth, 
                        subsample = par_set$subsample,
                        colsample_bytree = par_set$colsample_bytree,
                        objective = "reg:squarederror",
                        nrounds = par_set$nrounds)
  
  # Make predictions on validation set
  val_pred <- predict(val_mp2_xgb, mp2xgb_validation)
  
  # Compute and store the validation set rmse
  mp2_validation_rmses[i] <- sqrt((1/length(val_pred)) * sum((val_pred - mp2_validation[,1])^2))
}

if (!exists("mp2_final_par")) {
  mp2_final_par <- read.csv("mp2_pars_incl.csv")
  # mp2_final_par <- read.csv("mp2_pars_incl.csv")
}

mp2_final_par <- mp2_optimal_par_set[[which.min(mp2_validation_rmses)]]

#
# Retrain best model on all data (train + validation)
#

# ... Without re-estimating the optimal number of rounds
opt_mp2_xgb <- xgboost(data = mp2xgb_all,
                      booster = mp2_final_par$booster,
                      eta = mp2_final_par$eta,
                      gamma = mp2_final_par$gamma,
                      max_depth = mp2_final_par$max_depth, 
                      subsample = mp2_final_par$subsample,
                      colsample_bytree = mp2_final_par$colsample_bytree,
                      objective = "reg:squarederror",
                      nrounds = mp2_final_par$nrounds)

# ... with re-estimating the optimal number of rounds
mp2_opt_params <- list(booster = mp2_final_par$booster,
                      eta = mp2_final_par$eta,
                      gamma = mp2_final_par$gamma,
                      max_depth = mp2_final_par$max_depth, 
                      subsample = mp2_final_par$subsample,
                      colsample_bytree = mp2_final_par$colsample_bytree,
                      objective = "reg:squarederror",
                      nrounds = 200)

mp2_opt_xgbcv <- xgb.cv(params = mp2_opt_params, data = mp2xgb_train, nrounds = 100, nfold = 5,
                       showsd = T)
opt_mp2_xgb2 <- xgboost(data = mp2xgb_all,
                       booster = mp2_final_par$booster,
                       eta = mp2_final_par$eta,
                       gamma = mp2_final_par$gamma,
                       max_depth = mp2_final_par$max_depth, 
                       subsample = mp2_final_par$subsample,
                       colsample_bytree = mp2_final_par$colsample_bytree,
                       objective = "reg:squarederror",
                       nrounds = which.min(mp2_opt_xgbcv$evaluation_log$test_rmse_mean))

#
# Make predictions for test set
#

mp2_test_pred1 <- data.frame("ID" = mp2_test_all[,"property_id"], "PRED" = predict(opt_mp2_xgb, mp2xgb_test))
mp2_test_pred2 <- data.frame("ID" = mp2_test_all[,"property_id"], "PRED" = predict(opt_mp2_xgb2, mp2xgb_test))

if (include_distances_to_city_center) {
  filename1 <- "mp2_test_pred1_incl.csv"
  filename2 <- "mp2_test_pred2_incl.csv"
  filename3 <- "mp2_pars_incl.csv"
} else {
  filename1 <- "mp2_test_pred1.csv"
  filename2 <- "mp2_test_pred2.csv"
  filename3 <- "mp2_pars.csv"
}

write.csv(mp2_test_pred1, file = filename1, row.names = FALSE)
write.csv(mp2_test_pred2, file = filename2, row.names = FALSE)
write.csv(as.data.frame(mp2_final_par), filename3, row.names = FALSE)

#
# Model interpretation: variable importance plot
#

model <- xgb.dump(opt_mp2_xgb, with.stats = T)
names <- dimnames(data.matrix(mp2_train[,-1]))[[2]]
importance_matrix <- xgb.importance(names, model = opt_mp2_xgb)
xgb.plot.importance(importance_matrix[1:20,])


#### Feature selection for the model (including "dist_nearest_city_center") above ####

# Data set creation
data_minimal_prep2_idx <- c(1, 3:8, 11:22, 25:31, 36, 71:73, 34)
mp2_df_all <- df_all[,data_minimal_prep2_idx] 
dummies <- dummyVars(~ property_bed_type +  host_response_time +
                       booking_cancel_policy, data = mp2_df_all)
dummy_var_col_idx <- which(colnames(mp2_df_all) %in% c("property_bed_type",
                                                       "host_response_time",
                                                       "booking_cancel_policy"))
mp2_df_all_ohe <- as.data.frame(predict(dummies, newdata = mp2_df_all))
mp2_df_all_combined <- cbind(mp2_df_all[, -dummy_var_col_idx], mp2_df_all_ohe)
mp2_train <- mp2_df_all_combined[1:nrow(train), ]
mp2_validation <- mp2_df_all_combined[(nrow(train) + 1):nrow(mp2_df_all_combined), ]
colnames_minimal_prep2 <- colnames(train)[data_minimal_prep2_idx]
mp2_test_all <- test[,c("property_id", colnames_minimal_prep2[-1])]

dummy_var_col_idx <- which(colnames(mp2_test_all) %in% c("property_bed_type",
                                                         "host_response_time",
                                                         "booking_cancel_policy"))
mp2_test_all_ohe <- as.data.frame(predict(dummies, newdata = mp2_test_all))
mp2_test <- cbind(mp2_test_all[, -dummy_var_col_idx], mp2_test_all_ohe)
mp2xgb_test = xgb.DMatrix(data = data.matrix(mp2_test[,-1]))
mp2xgb_train = xgb.DMatrix(data = data.matrix(mp2_train[,-1]), label = mp2_train[,1])
mp2xgb_validation = xgb.DMatrix(data = data.matrix(mp2_validation[,-1]))
mp2xgb_all <- xgb.DMatrix(data = data.matrix(mp2_df_all_combined[,-1]),
                          label = mp2_df_all_combined[,1])

if (!exists("mp2_final_par") | !exists("opt_mp2_xgb") | !exists("mp2_opt_params") |
    !exists("mp2_opt_xgbcv") | !exists("opt_mp2_xgb2")) {
  mp2_final_par <- read.csv("mp2_pars_incl.csv")
  
  
  #
  # Retrain best model on all data (train + validation)
  #
  
  # ... Without re-estimating the optimal number of rounds
  opt_mp2_xgb <- xgboost(data = mp2xgb_all,
                         booster = mp2_final_par$booster,
                         eta = mp2_final_par$eta,
                         gamma = mp2_final_par$gamma,
                         max_depth = mp2_final_par$max_depth, 
                         subsample = mp2_final_par$subsample,
                         colsample_bytree = mp2_final_par$colsample_bytree,
                         objective = "reg:squarederror",
                         nrounds = mp2_final_par$nrounds)
  
  # ... with re-estimating the optimal number of rounds
  mp2_opt_params <- list(booster = mp2_final_par$booster,
                         eta = mp2_final_par$eta,
                         gamma = mp2_final_par$gamma,
                         max_depth = mp2_final_par$max_depth, 
                         subsample = mp2_final_par$subsample,
                         colsample_bytree = mp2_final_par$colsample_bytree,
                         objective = "reg:squarederror",
                         nrounds = 200)
  
  mp2_opt_xgbcv <- xgb.cv(params = mp2_opt_params, data = mp2xgb_train, nrounds = 100, nfold = 5,
                          showsd = T)
  opt_mp2_xgb2 <- xgboost(data = mp2xgb_all,
                          booster = mp2_final_par$booster,
                          eta = mp2_final_par$eta,
                          gamma = mp2_final_par$gamma,
                          max_depth = mp2_final_par$max_depth, 
                          subsample = mp2_final_par$subsample,
                          colsample_bytree = mp2_final_par$colsample_bytree,
                          objective = "reg:squarederror",
                          nrounds = which.min(mp2_opt_xgbcv$evaluation_log$test_rmse_mean))
}


# Variable importance plot (based on increase in rmse per split). Also note 
# that we work with the model with re-optimized number of training iterations.
model <- xgb.dump(opt_mp2_xgb2, with_stats = T)
names <- dimnames(data.matrix(mp2_train[,-1]))[[2]]
importance_matrix <- xgb.importance(names, model = opt_mp2_xgb2)
xgb.plot.importance(importance_matrix[1:20,])

# Using Boruta package: Find important variables based on permutation based
# feature importance.
# IMPORTANT: The results seem to change each time you run this.

# helpful website:
# https://amirali-n.github.io/BorutaFeatureSelectionWithShapAnalysis/

xgb.boruta = Boruta(data.matrix(mp2_df_all_combined[,-1]),
                    y = mp2_df_all_combined[,1],
                    maxRuns = 100, 
                    doTrace = 2,
                    holdHistory = TRUE,
                    getImp = getImpXgboost,
                    eta = mp2_final_par$eta,
                    gamma = mp2_final_par$gamma,
                    max.depth = mp2_final_par$max_depth, 
                    subsample = mp2_final_par$subsample,
                    colsample_bytree = mp2_final_par$colsample_bytree,
                    objective = "reg:squarederror",
                    nrounds = which.min(mp2_opt_xgbcv$evaluation_log$test_rmse_mean), 
                    eval_metric = "rmse",
                    tree_method = "hist")
# The top 4 variables are "dist_nearest_city_center", "booking_availablity_365",
# "reviews_num" and "reviews_per_month"

boruta_dec = attStats(xgb.boruta)
boruta_dec[boruta_dec$decision != "Rejected",]
# Every variable gets rejected, but "reviews_per_month", "dist_nearest_city_center",
# and "reviews_num" get rejected last.

imp_features=row.names(boruta_dec)
boruta.imp.df=as.data.frame(xgb.boruta$ImpHistory)
boruta.imp.df=boruta.imp.df[,names(boruta.imp.df)%in%imp_features]
boruta.imp.df=melt(boruta.imp.df)
feature_order=with(boruta.imp.df, reorder(variable, value, median, order = TRUE))
boruta.imp.df$variable=factor(boruta.imp.df$variable, levels = levels(feature_order))

ggplot(boruta.imp.df, aes(y = variable, x = value)) + geom_boxplot()

# Using Shapley values. The returned values are the mean of the absolute values
# of the instance level SHAP scores. Hence the larger this mean, the more
# important the variable according to the shapley score.
shap_values=shap.values(xgb_model = opt_mp2_xgb2, X_train = mp2xgb_all)
shap_values$mean_shap_score
# The top 4 variables are "dist_nearest_city_center", "property_max_guests",
# "booking_availablity_365" and "reviews_num".

#### XGboost using only 5 most important features ####

#
# Select variables to work on
#

# Train + validation
data_5_most_imp_idx <- which(colnames(df_all) %in% c("dist_nearest_city_center",
                                                     "property_max_guests",
                                                     "booking_availablity_365",
                                                     "reviews_num",
                                                     "reviews_per_month"))
data_5_most_imp_idx <- c(1, data_5_most_imp_idx)

best5_df_all <- df_all[,data_5_most_imp_idx] 

best5_df_all_combined <- best5_df_all
best5_train <- best5_df_all_combined[1:nrow(train), ]
best5_validation <- best5_df_all_combined[(nrow(train) + 1):nrow(best5_df_all_combined), ]

# test set
colnames_best5 <- colnames(train)[data_5_most_imp_idx]
best5_test_all <- test[,c("property_id", colnames_best5[-1])]

best5_test <- best5_test_all
best5_xgb_test = xgb.DMatrix(data = data.matrix(best5_test[,-1]))

#
# Setup for tuning xgboost
#

best5_xgb_train = xgb.DMatrix(data = data.matrix(best5_train[,-1]), label = best5_train[,1])
best5_xgb_validation = xgb.DMatrix(data = data.matrix(best5_validation[,-1]))
best5_xgb_all <- xgb.DMatrix(data = data.matrix(best5_df_all_combined[,-1]),
                          label = best5_df_all_combined[,1])

etas <- c(0.05, 0.1, 0.2, 0.3, 0.4)
gammas <- c(0, 1, 10, 100, 500, 1000)
max_depths <- c(6, 10, 15, 20)
subsamples <- c(0.1, 0.5, 1)
colsamples_bytree <- c(0.1, 0.5, 1)
boosters <- c("gbtree", "dart")

#
# Grid search over hyperparameter grid and store the 10 best models
#

best5_min_test_rmse <- rep(Inf, 10)
best5_optimal_par_set <- list(list(), list(), list(), list(), list(),
                            list(), list(), list(), list(), list())
for (booster in boosters) {
  for (eta in etas) {
    for (gamma in gammas) {
      for (max_depth in max_depths) {
        for (subsample in subsamples) {
          for (colsample_bytree in colsamples_bytree) {
            
            if (eta > 0.2 | subsample == 1 | colsample_bytree == 1 | max_depth > 15) {
              next
            }
            
            params <- list(booster = booster,
                           eta = eta,
                           gamma = gamma,
                           max_depth = max_depth, 
                           nround = 200, 
                           subsample = subsample,
                           colsample_bytree = colsample_bytree,
                           objective = "reg:squarederror")
            
            best5_xgbcv <- xgb.cv(params = params, data = best5_xgb_train, nrounds = 100, nfold = 5,
                                showsd = T)
            
            if (max(best5_min_test_rmse) > min(best5_xgbcv$evaluation_log$test_rmse_mean)) {
              worst_model_idx <- which.max(best5_min_test_rmse)
              best5_min_test_rmse[worst_model_idx] <- min(best5_xgbcv$evaluation_log$test_rmse_mean)
              best5_optimal_par_set[[worst_model_idx]] <- list("booster" = booster,
                                                             "eta" = eta,
                                                             "gamma" = gamma,
                                                             "max_depth" = max_depth,
                                                             "subsample" = subsample,
                                                             "colsample_bytree" = colsample_bytree,
                                                             "nrounds" = which.min(best5_xgbcv$evaluation_log$test_rmse_mean))
            }
          }
        }
      }
    }
  }
}

#
# For these 10 best models, select a final model based on the validation set.
#

best5_validation_rmses <- rep(0, 10)
for (i in 1:10) {
  
  # Retrieve model parameters
  par_set <- best5_optimal_par_set[[i]]
  
  # Create model
  val_best5_xgb <- xgboost(data = best5_xgb_train,
                         booster = par_set$booster,
                         eta = par_set$eta,
                         gamma = par_set$gamma,
                         max_depth = par_set$max_depth, 
                         subsample = par_set$subsample,
                         colsample_bytree = par_set$colsample_bytree,
                         objective = "reg:squarederror",
                         nrounds = par_set$nrounds)
  
  # Make predictions on validation set
  val_pred <- predict(val_best5_xgb, best5_xgb_validation)
  
  # Compute and store the validation set rmse
  best5_validation_rmses[i] <- sqrt((1/length(val_pred)) * sum((val_pred - best5_validation[,1])^2))
}

best5_final_par <- best5_optimal_par_set[[which.min(best5_validation_rmses)]]

if (!exists("best5_final_par")) {
  best5_final_par <- read.csv("best5_pars.csv")
}

#
# Retrain best model on all data (train + validation)
#

# ... Without re-estimating the optimal number of rounds
opt_best5_xgb <- xgboost(data = best5_xgb_all,
                       booster = best5_final_par$booster,
                       eta = best5_final_par$eta,
                       gamma = best5_final_par$gamma,
                       max_depth = best5_final_par$max_depth, 
                       subsample = best5_final_par$subsample,
                       colsample_bytree = best5_final_par$colsample_bytree,
                       objective = "reg:squarederror",
                       nrounds = best5_final_par$nrounds)

# ... with re-estimating the optimal number of rounds
best5_opt_params <- list(booster = best5_final_par$booster,
                       eta = best5_final_par$eta,
                       gamma = best5_final_par$gamma,
                       max_depth = best5_final_par$max_depth, 
                       subsample = best5_final_par$subsample,
                       colsample_bytree = best5_final_par$colsample_bytree,
                       objective = "reg:squarederror",
                       nrounds = 200)

best5_opt_xgbcv <- xgb.cv(params = best5_opt_params, data = best5_xgb_train, nrounds = 100, nfold = 5,
                        showsd = T)
opt_best5_xgb2 <- xgboost(data = best5_xgb_all,
                        booster = best5_final_par$booster,
                        eta = best5_final_par$eta,
                        gamma = best5_final_par$gamma,
                        max_depth = best5_final_par$max_depth, 
                        subsample = best5_final_par$subsample,
                        colsample_bytree = best5_final_par$colsample_bytree,
                        objective = "reg:squarederror",
                        nrounds = which.min(best5_opt_xgbcv$evaluation_log$test_rmse_mean))

#
# Make predictions for test set
#

best5_test_pred1 <- data.frame("ID" = best5_test_all[,"property_id"], "PRED" = predict(opt_best5_xgb, best5_xgb_test))
best5_test_pred2 <- data.frame("ID" = best5_test_all[,"property_id"], "PRED" = predict(opt_best5_xgb2, best5_xgb_test))

write.csv(best5_test_pred1, file = "best5_test_pred1.csv", row.names = FALSE)
write.csv(best5_test_pred2, file = "best5_test_pred2.csv", row.names = FALSE)
write.csv(as.data.frame(best5_final_par), "best5_pars.csv", row.names = FALSE)

#
# Variable importance
#

# Variable importance plot (based on increase in rmse per split). Also note 
# that we work with the model with re-optimized number of training iterations.
model <- xgb.dump(opt_best5_xgb2, with_stats = T)
names <- dimnames(data.matrix(best5_train[,-1]))[[2]]
importance_matrix <- xgb.importance(names, model = opt_best5_xgb2)
xgb.plot.importance(importance_matrix)

# Using Boruta package: Find important variables based on permutation based
# feature importance.
# IMPORTANT: The results seem to change each time you run this.

# helpful website:
# https://amirali-n.github.io/BorutaFeatureSelectionWithShapAnalysis/

xgb.boruta = Boruta(data.matrix(best5_df_all_combined[,-1]),
                    y = best5_df_all_combined[,1],
                    maxRuns = 100, 
                    doTrace = 2,
                    holdHistory = TRUE,
                    getImp = getImpXgboost,
                    eta = best5_final_par$eta,
                    gamma = best5_final_par$gamma,
                    max.depth = best5_final_par$max_depth, 
                    subsample = best5_final_par$subsample,
                    colsample_bytree = best5_final_par$colsample_bytree,
                    objective = "reg:squarederror",
                    nrounds = which.min(best5_opt_xgbcv$evaluation_log$test_rmse_mean), 
                    eval_metric = "rmse",
                    tree_method = "hist")


boruta_dec = attStats(xgb.boruta)
boruta_dec[boruta_dec$decision != "Rejected",]
# Every variable gets rejected, but "reviews_per_month", "dist_nearest_city_center",
# and "reviews_num" get rejected last.

imp_features=row.names(boruta_dec)
boruta.imp.df=as.data.frame(xgb.boruta$ImpHistory)
boruta.imp.df=boruta.imp.df[,names(boruta.imp.df)%in%imp_features]
boruta.imp.df=melt(boruta.imp.df)
feature_order=with(boruta.imp.df, reorder(variable, value, median, order = TRUE))
boruta.imp.df$variable=factor(boruta.imp.df$variable, levels = levels(feature_order))

ggplot(boruta.imp.df, aes(y = variable, x = value)) + geom_boxplot()

# Using Shapley values. The returned values are the mean of the absolute values
# of the instance level SHAP scores. Hence the larger this mean, the more
# important the variable according to the shapley score.
shap_values=shap.values(xgb_model = opt_best5_xgb2, X_train = best5_xgb_all)
shap_values$mean_shap_score
# The top 4 variables are "dist_nearest_city_center", "property_max_guests",
# "booking_availablity_365" and "reviews_num".


#### linear model ####

# Since all of the XGboost models seem to be overfitting on the training data,
# we try a model with less flexibility, namely a linear model. Moreover, we will
# continue to work with only the top 5 best predictors found in the previous
# section.

data_5_most_imp_idx <- which(colnames(df_all) %in% c("dist_nearest_city_center",
                                                     "property_max_guests",
                                                     "booking_availablity_365",
                                                     "reviews_num",
                                                     "reviews_per_month"))
data_5_most_imp_idx <- c(1, data_5_most_imp_idx)

best5_df_all <- df_all[,data_5_most_imp_idx]

#
# Select data and create training and validation set
# 

best5_train <- best5_df_all[1:nrow(train),]
best5_validation <- best5_df_all[(nrow(train) + 1):nrow(best5_df_all),]

# Unfortunately, reviews_per_month contains a lot of missing values. We impute
# them with the mean and keep a column indicating if the value was originally
# missing.
colSums(is.na(best5_train))
missing_idx <- which(is.na(best5_train$reviews_per_month))
mean_reviews_per_month <- mean(best5_train$reviews_per_month, na.rm = TRUE)
best5_train[missing_idx, "reviews_per_month"] <- mean_reviews_per_month
best5_train[missing_idx, "rpm_originally_missing"] <- 1
best5_train[-missing_idx, "rpm_originally_missing"] <- 0

missing_idx_val <- which(is.na(best5_validation$reviews_per_month))
best5_validation[missing_idx_val, "reviews_per_month"] <- mean_reviews_per_month
best5_validation[missing_idx_val, "rpm_originally_missing"] <- 1
best5_validation[-missing_idx_val, "rpm_originally_missing"] <- 0

best5_df_all <- rbind(best5_train, best5_validation)

lin.mod <- lm(target ~ ., data = best5_train)

# stepwise selection
step.mod <- stepAIC(lin.mod, direction = 'both')
summary(step.mod)
# Only the variables "dist_nearest_city_center" and "rpm_originally_missing" are
# retained.

# To check, also make a model with no predictors
simple.mod <- lm(target ~ 1, data = best5_train)
summary(simple.mod)

pred.lin.mod <- predict(lin.mod, newdata = best5_validation)
pred.step.mod <- predict(step.mod, newdata = best5_validation)
pred.simple.mod <- predict(simple.mod, newdata = best5_validation)

rmse.lin.mod <- sqrt((1/length(pred.lin.mod)) * sum((pred.lin.mod - best5_validation$target)^2))
rmse.step.mod <- sqrt((1/length(pred.step.mod)) * sum((pred.step.mod - best5_validation$target)^2))
rmse.simple.mod <- sqrt((1/length(pred.simple.mod)) * sum((pred.simple.mod - best5_validation$target)^2))

rmse.lin.mod
rmse.step.mod
rmse.simple.mod

# The step model performs slightly better than the simple model. Therefore, we
# use it to make predictions on the test set (which we first have to preprocess
# to deal with the missing values in "reviews_per_month"


data_5_most_imp_idx_test <- which(colnames(test) %in% c("dist_nearest_city_center",
                                                        "property_max_guests",
                                                        "booking_availablity_365",
                                                        "reviews_num",
                                                        "reviews_per_month"))
data_5_most_imp_idx_test <- c(1, data_5_most_imp_idx_test)
best5_test <- test[,data_5_most_imp_idx_test]

missing_test_idx <- which(is.na(best5_test$reviews_per_month))
best5_test[missing_test_idx, "reviews_per_month"] <- mean_reviews_per_month
best5_test[missing_test_idx, "rpm_originally_missing"] <- 1
best5_test[-missing_test_idx, "rpm_originally_missing"] <- 0
colSums(is.na(best5_test))

# Retrain model on full data set
step.mod.full <- lm(target ~ dist_nearest_city_center + rpm_originally_missing, data = best5_df_all)
summary(step.mod.full)

# Make predictions
test.pred.step <- data.frame("ID" = best5_test[, "property_id"], "PRED" = predict(step.mod.full, newdata = best5_test))
write.csv(test.pred.step, file = "test_pred_step.csv", row.names = FALSE)

# For some reason, this does not give good results on the test set
par(mfrow = c(1, 2))
hist(best5_test$dist_nearest_city_center)
hist(best5_train$dist_nearest_city_center)

table(best5_train$rpm_originally_missing)
table(best5_test$rpm_originally_missing)



#### linear model V2 ####

# Since the variable "dist_to_nearest_city_center" is skewed, which is a violation
# of the linear regression assumptions, we make it more normal by applying a
# transformation.

data_5_most_imp_idx <- which(colnames(df_all) %in% c("dist_nearest_city_center",
                                                     "property_max_guests",
                                                     "booking_availablity_365",
                                                     "reviews_num",
                                                     "reviews_per_month"))
data_5_most_imp_idx <- c(1, data_5_most_imp_idx)

best5_df_all <- df_all[,data_5_most_imp_idx]

#
# Select data and create training and validation set
# 

# Turns out that the data is so noisy that different models are selected each
# time you rerun this section of code (resampling training and test set). 

# rmse's differ a lot based on which training and validation set is selected.
# Hence if you can't really trust your validation set, how would you even build
# a model?

# --> Just use sample mean

train_idx <- sample(1:nrow(df_all), nrow(train))
best5_train <- best5_df_all[train_idx,]
best5_validation <- best5_df_all[-train_idx,]

# Fix missings in reviews_per_month and keep a column indicating if the value
# was originally missing.
colSums(is.na(best5_train))
missing_idx <- which(is.na(best5_train$reviews_per_month))
mean_reviews_per_month <- mean(best5_train$reviews_per_month, na.rm = TRUE)
best5_train[missing_idx, "reviews_per_month"] <- mean_reviews_per_month
best5_train[missing_idx, "rpm_originally_missing"] <- 1
best5_train[-missing_idx, "rpm_originally_missing"] <- 0

bc_trans <- boxcox(lm(best5_train$dist_nearest_city_center ~ 1))
lambda_opt <- bc_trans$x[which.max(bc_trans$y)]
bc_dncc <- (best5_train$dist_nearest_city_center^lambda_opt - 1)/lambda_opt
best5_train$bc_dncc <- bc_dncc
best5_train <- dplyr::select(best5_train, -"dist_nearest_city_center")

par(mfrow = c(1, 1))
hist(best5_train$bc_dncc)

missing_idx_val <- which(is.na(best5_validation$reviews_per_month))
best5_validation[missing_idx_val, "reviews_per_month"] <- mean_reviews_per_month
best5_validation[missing_idx_val, "rpm_originally_missing"] <- 1
best5_validation[-missing_idx_val, "rpm_originally_missing"] <- 0

lambda_opt <- bc_trans$x[which.max(bc_trans$y)]
bc_dncc <- (best5_validation$dist_nearest_city_center^lambda_opt - 1)/lambda_opt
best5_validation$bc_dncc <- bc_dncc
best5_validation <- dplyr::select(best5_validation, -"dist_nearest_city_center")

#
# Train model
#

lin.mod <- lm(target ~ ., data = best5_train)

# stepwise selection
step.mod <- stepAIC(lin.mod, direction = 'both')
summary(step.mod)
# Only the variables "dist_nearest_city_center" and "rpm_originally_missing" are
# retained.

step2.mod <- lm(target ~ bc_dncc, data = best5_train)
summary(step2.mod)

# To check, also make a model with no predictors
simple.mod <- lm(target ~ 1, data = best5_train)
summary(simple.mod)

pred.lin.mod <- predict(lin.mod, newdata = best5_validation)
pred.step.mod <- predict(step.mod, newdata = best5_validation)
pred.step2.mod <- predict(step2.mod, newdata = best5_validation)
pred.simple.mod <- predict(simple.mod, newdata = best5_validation)

rmse.lin.mod <- sqrt((1/length(pred.lin.mod)) * sum((pred.lin.mod - best5_validation$target)^2))
rmse.step.mod <- sqrt((1/length(pred.step.mod)) * sum((pred.step.mod - best5_validation$target)^2))
rmse.step2.mod <- sqrt((1/length(pred.step2.mod)) * sum((pred.step2.mod - best5_validation$target)^2))
rmse.simple.mod <- sqrt((1/length(pred.simple.mod)) * sum((pred.simple.mod - best5_validation$target)^2))

rmse.lin.mod
rmse.step.mod
rmse.step2.mod
rmse.simple.mod
# We can see that the rmse of step.mod is slightly lower than in the previous
# section. Hence applying the transformation helped.

# The step model performs slightly better than the simple model. Therefore, we
# use it to make predictions on the test set (which we first have to preprocess
# to deal with the missing values in "reviews_per_month"


data_5_most_imp_idx_test <- which(colnames(test) %in% c("dist_nearest_city_center",
                                                        "property_max_guests",
                                                        "booking_availablity_365",
                                                        "reviews_num",
                                                        "reviews_per_month"))
data_5_most_imp_idx_test <- c(1, data_5_most_imp_idx_test)
best5_test <- test[,data_5_most_imp_idx_test]

missing_test_idx <- which(is.na(best5_test$reviews_per_month))
best5_test[missing_test_idx, "reviews_per_month"] <- mean_reviews_per_month
best5_test[missing_test_idx, "rpm_originally_missing"] <- 1
best5_test[-missing_test_idx, "rpm_originally_missing"] <- 0

lambda_opt <- bc_trans$x[which.max(bc_trans$y)]
bc_dncc <- (best5_test$dist_nearest_city_center^lambda_opt - 1)/lambda_opt
best5_test$bc_dncc <- bc_dncc
best5_test <- dplyr::select(best5_test, -"dist_nearest_city_center")

test.pred.step.bc <- data.frame("ID" = best5_test[, "property_id"], "PRED" = predict(step.mod, newdata = best5_test))
write.csv(test.pred.step.bc, file = "test_pred_step_bc.csv", row.names = FALSE)

# For some reason, this does not give good results on the test set
par(mfrow = c(1, 2))
hist(best5_test$dist_nearest_city_center)
hist(best5_train$dist_nearest_city_center)

table(best5_train$rpm_originally_missing)
table(best5_test$rpm_originally_missing)


