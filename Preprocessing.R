
# Clear work space
rm(list = ls())

# Load in data
data <- read.csv("data/train.csv", header=TRUE)
test <- read.csv("data/test.csv", header=TRUE)

# Load necessary packages
library(MASS)
library(tree)
library(rpart)
library(rpart.plot)
library(dplyr)
library(ggplot2)
library(stringr)

source("Useful functions.R")

################################################################################
#                  Split into train and validation set                         #
################################################################################

set.seed(123456)
train_size <- floor(0.80*nrow(data))
train_idx <- sample(1:nrow(data), train_size, replace = FALSE)
train <- data[train_idx, ]
validation <- data[-train_idx, ]

################################################################################
#                        Data inspection + cleaning                            #
################################################################################

# All data cleaning will be done first on the training data and only later
# applied to the validation data and test data set.

summary(data)

# In the following, we will create some features of our own. It might be useful
# to have a vector that stores these features names.
features <- c()


#
# target
#

# We see that the prices show a lot of skewness to the right. Maybe apply a
# transformation to them?
plot(density(train$target))

# Log(prices) seem to be a lot better behaved, though not perfect
plot(density(log(train$target)))
boxplot(log(train$target))

# Box-cox transformation of prices does a bit better than log(prices).
bc_trans <- boxcox(lm(train$target ~ 1))
lambda_opt <- bc_trans$x[which.max(bc_trans$y)]
bc_target <- (train$target^lambda_opt - 1)/lambda_opt
plot(density(bc_target))
boxplot(bc_target)

# lambda_opt = -18/99

train$bc_target <- bc_target

#
# Property_id
#

# Are all properties unique?
# Yes
length(train$property_id) == length(unique(train$property_id))

#
# property_summary, property_space, property_desc, property_neighbourhood, 
# property_notes, property_transit, property_access, property_interaction,
# property_rules
#

# Some of these values seem to be repeated. What is up with that?
length(train$property_summary) == length(unique(train$property_summary))
  # This is a more difficult to investigate properly. Maybe wait with this to
  # when we decide to investigate these variables in more detail.


#
# Property_zipcodes
#

# There are 134 missing zip codes, which are encoded as as empty string. Moreover,
# there are two zip codes miscoded as "11 20" instead of "1020".
summary(as.factor(train$property_zipcode))

train$property_zipcode[which(train$property_zipcode == "11 20")] <- "1120"
missing_zip_idx <- which(train$property_zipcode == "")

# Save info that zip code wass initially missing
train$property_zip_missing <- as.numeric(train$property_zipcode == "")
features <- c(features, "property_zip_missing")
  # How could we impute these?
  # 1) Search for clues in the description, neighbourhood, etc. columns
  # 2) Look at latitude and longitude values and find nearest neighbours <--

  # For all properties with a missing zip code...
  K_zipcode <- 5
  for (idx in missing_zip_idx) {
    
    # Store the geographical coordinates of that property
    lat <- train[idx, "property_lat"]
    lon <- train[idx, "property_lon"]
    
    # We will look at the K = 5 nearest neighbours.
    # Why 5? Doesn't it make more sense to just look at the value of the
    # closest observation?
    #   - Yes, but in this way, we avoid issues that arise if the nearest
    #     neighbour is again an observation with missing zip code. Since we put
    #     all the most uncommon zipcodes into one category later, this
    #     decision won't matter anyway.
    
    # Find K nearest neighbours (euclidean distance) in terms of coordinates
    distances <- sqrt((train$property_lat - lat)^2 + (train$property_lon - lon)^2)
    neirest_idxs <- sort(distances, index.return = TRUE)$ix[2:(K_zipcode+1)]
    
    # Look at their zip codes, select the most common one and assign it to the
    # previously missing value.
    train[idx, "property_zipcode"] <- names(which.max(table(train[neirest_idxs, "property_zipcode"])))
  }

# We should find a nice way to encode these. One idea could be to fit a decision
# tree on a categorized version of the target variable and classify zip codes 
# into cheap/intermediate/expensive 'zones'. Unfortunately, the "tree" function
# in R only accepts categorical variables with a set number of levels, which are
# exceeded in this case. Therefore, we group the least common zip codes into
# one new category "other". 
zipcode_data <- data.frame("zipcode" = train$property_zipcode, "price" = train$target)

# Price class categories: lower quartile, IQR and upper quartile.
zipcode_data$price_class <- ifelse(zipcode_data$price > quantile(zipcode_data$price, 0.75), "high",
                                   ifelse(zipcode_data$price > quantile(zipcode_data$price, 0.25), "medium", "low"))
uncommon_zipcodes <- names(which(table(zipcode_data$zipcode) <= 10))
zipcode_data$zipcode[which(zipcode_data$zipcode %in% uncommon_zipcodes)] <- "other"

zipcode_data$zipcode <- as.factor(zipcode_data$zipcode)
zipcode_data$price_class <- as.factor(zipcode_data$price_class)

# Fit classification tree on price_class or regression tree on price?
do_reg_tree <- TRUE

if (!do_reg_tree) {
  # takes a few minutes to run...
  zipcode_tree <- tree(price_class ~ zipcode, data = zipcode_data, split = "gini")
  plot(zipcode_tree)
  text(zipcode_tree)
  
  # Determine accuracy of the tree. The tree is not very accurate, but hopefully 
  # accurate enough. If it makes mistakes, the mistake will at least be not too
  # bad.
  good <- 0
  bad1 <- 0
  bad2 <- 0
  pred <- predict(zipcode_tree, type = "class")
  for (i in 1:length(pred)) {
    if (pred[i] == zipcode_data[i, "price_class"]) {
      good <- good + 1
    } else if ((pred[i] == as.factor("low") && 
                zipcode_data[i, "price_class"] == as.factor("high")) | (
                  pred[i] == as.factor("low") && 
                  zipcode_data[i, "price_class"] == as.factor("high")
                )
    ) {
      bad2 <- bad2 + 1
    } else {
      bad1 <- bad1 + 1
    }
  }
  data.frame("good" = good/length(pred), "somewhat bad" = bad1/length(pred),
             "bad" = bad2/length(pred))
  train$zipcode_class <- ifelse(pred == "high", "zipclass1",
                                ifelse(pred == "medium", "zipclass2", "zipclass3"))
  train$zipcode_class <- as.factor(train$zipcode_class)
  
} else {
  
  tree_zipcode <- rpart(target ~ property_zipcode, data = train, control=rpart.control(cp=.00001))
  printcp(tree_zipcode)
  
  #plot the pruned tree
  prp(tree_zipcode,
      faclen=0, #use full names for factor labels
      extra=1, #display number of obs. for each terminal node
      roundint=F, #don't round to integers in output
      digits=5) #display 5 decimal places in output
  
  train$zipcode_class <- predict(tree_zipcode)
}

features <- c(features, "zipcode_class")

#
# property_lat and property_lon
#

# Based on these coordinates, we could determine the distance to the center of
# the nearest city and use it as a covariate later. Coordinates are retrieved
# from Google maps.
antwerp_lat = 51.221359578942604; antwerp_lon = 4.398997929972499 # city hall
brussels_lat = 50.8465549517382; brussels_lon = 4.351922557884677 # city hall

# Since all of the zip codes indicate that the properties are in Antwerp or
# or brussels, these other coordinates shouldn't matter too much, but I'll leave
# them in anyway.
ghent_lat = 51.05446861792642; ghent_lon = 3.725283407738655 # city hall
leuven_lat = 50.87899695490237; leuven_lon = 4.701191689536359 # city hall
luik_lat = 50.64554868682975; luik_lon = 5.575502106620438 # city hall

train$dist_nearest_city_center <- rep(0, nrow(train))
for (i in 1:nrow(train)) {
  lat <- train[i, "property_lat"]
  lon <- train[i, "property_lon"]
  
  distance_to_antwerp = sqrt((lat - antwerp_lat)^2 + (lon - antwerp_lon)^2)
  distance_to_brussels = sqrt((lat - brussels_lat)^2 + (lon - brussels_lon)^2)
  distance_to_ghent = sqrt((lat - ghent_lat)^2 + (lon - ghent_lon)^2)
  distance_to_leuven = sqrt((lat - leuven_lat)^2 + (lon - leuven_lon)^2)
  distance_to_luik = sqrt((lat - luik_lat)^2 + (lon - luik_lon)^2)
  
  dists <- c(distance_to_antwerp, distance_to_brussels, distance_to_ghent,
             distance_to_leuven, distance_to_luik)
  
  train$dist_nearest_city_center[i] <- min(dists)
}

features <- c(features, "dist_nearest_city_center")


#
# property_type and property_room_type
#

# What are all of the property (room) types? How much of each in data set?
# While there are a lot of different property types, there are only three
# different property room types. Therefore, property_room_type does not seem
# to need preprocessing but property_type does. We use the same idea as we did
# for zip codes and reclassify based on a decision tree.
summary(as.factor(train$property_type))
summary(as.factor(train$property_room_type))
train$property_type <- as.factor(train$property_type)
train$property_room_type <- as.factor(train$property_room_type)

type_data <- data.frame("type" = train$property_type, "price" = train$target)
type_data$price_class <- ifelse(type_data$price > quantile(type_data$price, 0.75), "high",
                                   ifelse(type_data$price > quantile(type_data$price, 0.25), "medium", "low"))
type_data$price_class <- as.factor(type_data$price_class)

# Fit classification tree on price_class or regression tree on price?
do_reg_tree <- TRUE

if (!do_reg_tree) {
  
  # We fully grow the tree, as that makes sense in this case.
  type_tree <- tree(price_class ~ type, data = type_data, split = "gini",
                    control = tree.control(nrow(train), mincut = 1, minsize = 2))
  plot(type_tree)
  text(type_tree)
  
  # Determine accuracy of the tree. The tree is not very accurate, but hopefully 
  # accurate enough. If it makes mistakes, the mistake will at least be not too
  # bad.
  good <- 0
  bad1 <- 0
  bad2 <- 0
  pred <- predict(type_tree, type = "class")
  for (i in 1:length(pred)) {
    if (pred[i] == type_data[i, "price_class"]) {
      good <- good + 1
    } else if ((pred[i] == as.factor("low") && 
                type_data[i, "price_class"] == as.factor("high")) | (
                  pred[i] == as.factor("low") && 
                  type_data[i, "price_class"] == as.factor("high")
                )
    ) {
      bad2 <- bad2 + 1
    } else {
      bad1 <- bad1 + 1
    }
  }
  data.frame("good" = good/length(pred), "somewhat bad" = bad1/length(pred),
             "bad" = bad2/length(pred))
  train$type_class <- ifelse(pred == "high", "typeclass1",
                             ifelse(pred == "medium", "typeclass2", "typeclass3"))
  train$type_class <- as.factor(train$type_class)
  
  unique(train[which(train$type_class == "typeclass1"), "property_type"]) # all expensive types
  unique(train[which(train$type_class == "typeclass3"), "property_type"]) # all cheap types
  
} else {
  
  tree_prop_type <- rpart(target ~ property_type, data = train, control=rpart.control(cp=.00001))
  printcp(tree_prop_type)
  
  #plot the pruned tree
  prp(tree_prop_type,
      faclen=0, #use full names for factor labels
      extra=1, #display number of obs. for each terminal node
      roundint=F, #don't round to integers in output
      digits=5) #display 5 decimal places in output
  
  train$type_class <- predict(tree_prop_type)
}

features <- c(features, "type_class")

#
# Property_max_guests
#

# Based on the plots, there does not seem to be any correlation between the max
# amount of guests a property has and its price. 
table(train$property_max_guests)
plot(train$property_max_guests, bc_target)


#
# property_bathrooms
#

# Treat missing values. We could create a variable indicating whether the value
# was missing but since there are only 12, this is probably not very informative.

median_bathrooms <- median(train$property_bathrooms, na.rm = TRUE)
train[which(is.na(train$property_bathrooms)), "property_bathrooms"] <- median_bathrooms
  
# Some properties have 0 bathrooms and some have a non-integer amount of
# bathrooms. Since the number of bathrooms is usually an indication of the 
# luxury (and hence price) of the property, this will likely be a good variable
# to include in the model. Based on the plot, however, it only seems to matter
# whether there are 0 bathrooms or not.
table(train$property_bathrooms)
plot(train$property_bathrooms, bc_target)
train[which(train$property_bathrooms == 0), "property_type"]

#
# property_bedrooms
#

table(train$property_bedrooms)

# Treat missing values. Based on the property descriptions it is not always
# clear how many bedrooms there are, but most of the time, 1 seems a reasonable
# guess.
median_bedrooms <- median(train$property_bedrooms, na.rm = TRUE)
bedrooms_missing_idx <- which(is.na(train$property_bedrooms))
train[bedrooms_missing_idx, "property_desc"]
train[bedrooms_missing_idx, "property_bedrooms"] <- median_bedrooms

#
# property_beds
#

table(train$property_beds)

# Treat missing values. This is very difficult information to obtain from the
# descriptions, so we just make it equal to the median amount of beds
beds_missing_idx <- which(is.na(train$property_beds))
length(beds_missing_idx)

median_beds <- median(train$property_beds, na.rm = TRUE)
train[beds_missing_idx, "property_desc"]
train[beds_missing_idx, "property_beds"] <- median_beds

#
# host_id, host_location
#

## How many unique hosts are there?
num_unique_host_ids <- length(unique(train$host_id))
print(num_unique_host_ids)

## Can we interpolate host response rates from other properties of that host? ##

# First some data exploration
head(train[,c("host_response_rate", "host_location", "host_response_time",
           "host_nr_listings", "host_verified", "host_since")])

# There are a lot of different levels for the location of the host. Maybe boil
# it down to just country of origin.
unique(train$host_location)

# host_location_country <- unlist(lapply(train$host_location, get_host_country))

# Split strings by commas
split_data <- str_split(train$host_location, ",")

# Extract last element as country information
host_location_country <- sapply(split_data, function(x) trimws(x[length(x)]))

host_location_country[which(host_location_country %in% c("Belgie", "belgium", "belgique", "Belgique", "belgie", "Belgium"))] <- "BE"
host_location_country[which(host_location_country %in% c("France"))] <- "FR"
host_location_country[which(host_location_country %in% c("Netherlands", "The Netherlands", "Nederland"))] <- "NL"
host_location_country[which(!(host_location_country %in% c("BE", "NL", "FR")))] <- "other"
train$host_location_country <- host_location_country

features <- c(features, "host_location_country")

#
# host_nr_listings
#

# Only 1 missing value. Impute it by the median of host_nr_listings
length(which(is.na(train$host_nr_listings)))

median_host_nr_listings <- median(train$host_nr_listings, na.rm = TRUE)
train[which(is.na(train$host_nr_listings)), "host_nr_listings"] <- median_host_nr_listings

# Does a host really have 591 listings? This makes the variable very skewed.
hist(train$host_nr_listings)

# It might be a good idea to apply a log transformation to this variable, but
# this will give problems if a host has 0 listings. Since there are only 5 such
# hosts, we pretend like they have 1 listing as well. The variable does seem to
# have some predictive value, so we store it in the data set.
length(which(train$host_nr_listings == 0))
log_host_listings <- log(pmax(train$host_nr_listings, 1))
plot(log_host_listings, train$target/train$booking_price_covers)

#
# host_response_time
#

# Amount of levels (4) is okay. However, if host_response_time is missing, then
# host_response_rate is also missing, so we cannot use this in the predictions.
table(train$host_response_time)
table(train[which(is.na(train$host_response_rate)), "host_response_time"])

#
# host_verified
#

# Again a lot of different levels. Maybe just summarize this by counting the
# amount of verifications.
unique(train$host_verified)

train$host_verified_amount <- unlist(lapply(train$host_verified, count_verifications))

features <- c(features, "host_verified_amount")

#
# host_since
#

# Again a lot of levels. Let's transform this variable to show the amount of time
# (in years) that the host has been a host
unique(train$host_since)

train$years_as_host <- unlist(lapply(train$host_since, count_years_host))

median_years_as_host <- median(train$years_as_host, na.rm = TRUE)
train[which(is.na(train$years_as_host)), "years_as_host"] <- median_years_as_host

features <- c(features, "years_as_host")

use_new_model <- TRUE

if (use_new_model) {
  # Build a linear regression model to predict response rate from other properties
  model_response_rate <- lm(host_response_rate ~ host_location_country + 
                            host_nr_listings + host_verified_amount +
                              years_as_host, data = train)
  
  # for example make predictions for new data based on the trained model
  new_data <- data.frame(host_location_country = "BE",
                         host_nr_listings = 1,
                         host_verified_amount = 3,
                         years_as_host = 6)
  predicted_response_rate <- predict(model_response_rate, new_data)
  
  # Print the predicted response rate
  print(predicted_response_rate)
  
} else {
  # Build a linear regression model to predict response rate from other properties
  model_response_rate <- lm(host_response_rate ~ host_location_country + host_response_time + 
                            host_nr_listings + host_verified + host_since, data = train)
  
  # for example make predictions for new data based on the trained model
  new_data <- data.frame(host_location = "Brussels, Brussels, Belgium",
                         host_response_time = "within a few hours",
                         host_nr_listings = 1,
                         host_verified = "email, phone, reviews",
                         host_since = "2013-11-14")
  predicted_response_rate <- predict(model_response_rate, new_data)
  
  # Print the predicted response rate
  print(predicted_response_rate)
}

# use this model to impute missing values for host_response_rate
predictions <- predict(model_response_rate,
                       train[which(is.na(train$host_response_rate)),
                              c("host_location_country", "host_response_time",
                              "host_nr_listings", "host_verified_amount",
                              "years_as_host")])
predictions <- pmin(predictions, 100)
which(is.na(predictions))

# Impute the missing values with this model. Note that we do not have to store
# the information about which values were initially missing since this is
# represented by the empty strings in host_response_time.
train[which(is.na(train$host_response_rate)), "host_response_rate"] <- predictions


## what is the difference with host_nr_listings and host_nr_listings_total?
if(all(train$host_nr_listings == train$host_nr_listings_total, na.rm = TRUE)) {
  print("Column A and B are identical")
}
train <- select(train, -c("host_nr_listings_total"))

# Impact of the host values on the target variable 

# Convert host_since variable to date format
train$host_since <- as.Date(train$host_since, format = "%y-%m-%d")

# Subset only the relevant columns
host_vars <- c("host_response_rate", "host_location_country", "host_nr_listings",
               "host_response_time", "host_verified", "host_since")

# Create a new plot for each host variable
for (var in host_vars) {
  
  # Only makes sense to compute correlations with numeric variables
  if (is.numeric(train[,var])) {
    # Calculate the correlation between "price" and the host variable
    cor_val <- cor(train$target, train[, var], use = "complete.obs")
    
    # Create a scatter plot of the two variables
    plot(train[, var], train$target, main = paste("Correlation with Price:", round(cor_val, 5)), 
         xlab = var, ylab = "Price")
  } 
  
  # And a boxplot 
  boxplot(train$target ~ train[, var], main = paste("Correlation with Price:", round(cor_val, 5)), 
          xlab = var, ylab = "Price")
}

# Look at the model of the price 
model <- lm(target ~ host_response_rate + host_location_country + 
              host_nr_listings + host_verified_amount + years_as_host, data = train)

# for example make predictions for new data
new_data <- data.frame(host_response_rate = 100,
                       host_location_country = "BE",
                       host_nr_listings = 1,
                       host_verified_amount = 3,
                       years_as_host = 6)

predicted_response_rate <- predict(model, new_data)

# Print the predicted response rate
print(predicted_response_rate)



# What is up with booking_min_nights being 1000 sometimes? 

###booking_price_covers###
attach(train)
summary(as.factor(train$booking_price_covers))
#up to 16 people can be covered by the booking price which seems realistic


###booking_min_nights and booking_max_nights###

summary(as.factor(train$booking_min_nights))
#????????Property_id 5620 has min nights of 1000 nights, too much? also 2x 365 and 1x 360
summary(as.factor(train$booking_max_nights))

subset(train, booking_min_nights > booking_max_nights, select=c("property_id", "booking_min_nights", "booking_max_nights"))
# property_id (3792) has MIN nights(7) > MAX nights(1)... These should be switched:
train$booking_min_nights[train$property_id == 3792] <- 1
train$booking_max_nights[train$property_id == 3792] <- 7


###booking_availability_30  + 60 + 90 +365
# Check unique values and their frequencies for booking_availability_30
table(train$booking_availability_30)

# Check unique values and their frequencies for booking_availability_60
table(train$booking_availability_60)

# Check unique values and their frequencies for booking_availability_90
table(train$booking_availability_90)

# Check unique values and their frequencies for booking_availability_365
table(train$booking_availability_365)

#any unusual values, we will split the categories in 3, such that each categories contains as much rows:
# Create quantiles
q30 <- quantile(train$booking_availability_30, probs = c(0.33, 0.66))
q60 <- quantile(train$booking_availability_60, probs = c(0.33, 0.66))
q90 <- quantile(train$booking_availability_90, probs = c(0.33, 0.66))
q365 <- quantile(train$booking_availability_365, probs = c(0.33, 0.66))

# Divide into categories
train$category_30 <- cut(train$booking_availability_30, breaks = unique(c(-Inf, q30, Inf)), labels = c("low", "medium", "high"))
train$category_60 <- cut(train$booking_availability_60, breaks = unique(c(-Inf, q60, Inf)), labels = c("low", "medium", "high"))
train$category_90 <- cut(train$booking_availability_90, breaks = unique(c(-Inf, q90, Inf)), labels = c("low", "medium", "high"))
train$category_365 <- cut(train$booking_availability_365, breaks = unique(c(-Inf, q365, Inf)), labels = c("low", "medium", "high"))

# Create dummy columns for category_30
train$category_30_low <- ifelse(train$category_30 == "low", 1, 0)
train$category_30_medium <- ifelse(train$category_30 == "medium", 1, 0)
train$category_30_high <- ifelse(train$category_30 == "high", 1, 0)

# Create dummy columns for category_60
train$category_60_low <- ifelse(train$category_60 == "low", 1, 0)
train$category_60_medium <- ifelse(train$category_60 == "medium", 1, 0)
train$category_60_high <- ifelse(train$category_60 == "high", 1, 0)

# Create dummy columns for category_90
train$category_90_low <- ifelse(train$category_90 == "low", 1, 0)
train$category_90_medium <- ifelse(train$category_90 == "medium", 1, 0)
train$category_90_high <- ifelse(train$category_90 == "high", 1, 0)

# Create dummy columns for category_365
train$category_365_low <- ifelse(train$category_365 == "low", 1, 0)
train$category_365_medium <- ifelse(train$category_365 == "medium", 1, 0)
train$category_365_high <- ifelse(train$category_365 == "high", 1, 0)

features <- c(features, "category_30_low", "category_30_medium", "category_30_high",
              "category_60_low", "category_60_medium", "category_60_high",
              "category_90_low", "category_90_medium", "category_90_high",
              "category_365_low", "category_365_medium", "category_365_high")

###booking_cancel_policy
table(train$booking_cancel_policy)

train$booking_cancel_policy[train$booking_cancel_policy == "super_strict_30"] <- "strict"

###
# reviews
###

#
#reviews_num
#

#There is a lot of positive skewness
plot(density(train$reviews_num))
boxplot(train$reviews_num)

# I did the same thing as for the target variable, looking at log and box-cox transformation
plot(density(log(train$reviews_num)))
boxplot(log(train$reviews_num))

# Box-cox transformation
bc_trans <- boxcox(lm(train$reviews_num ~ 1))
lambda_opt <- bc_trans$x[which.max(bc_trans$y)]
bc_reviews_num <- (train$reviews_num^lambda_opt - 1)/lambda_opt
plot(density(bc_reviews_num))
boxplot(bc_reviews_num)

#We could group them into categories.
train$reviews_num_cat <- rep(0, nrow(train))

for (i in 1:length(train$reviews_num)){
  if (train$reviews_num[i] >= 1 & train$reviews_num[i]<=5){train$reviews_num_cat[i] = 1}
  else if (train$reviews_num[i] > 5 & train$reviews_num[i] <= 19){train$reviews_num_cat[i] = 2}
  else {train$reviews_num_cat[i] = 3}
}

hist(train$reviews_num_cat)

features <- c(features, "reviews_num_cat")


#
# reviews_first, reviews_last, reviews_per_month
#


#reviews_first and reviews_last: create new variable review_period. 
#reviews_last might still be valuable
train$review_period <- as.numeric(as.Date(train$reviews_last) - as.Date(train$reviews_first))

# Contains a lot of missing values. Store which values were missing.
train$review_period_was_missing <- as.numeric(is.na(train$review_period))

features <- c(features, c("review_period", "review_period_was_missing"))

#I want to check if the 1030  missing values are the 0 values for reviews_num
length(train$reviews_num[which(train$reviews_num==0)])
# yes. There are 1030 instances with no review information.This also gives us information about
# the airbnb (because there are no reviews) -> not random 
# keep the separate missing value indicator feature

#reviews_per_month outliers? skewed?
plot(density(train$reviews_per_month ,na.rm=TRUE))
boxplot(train$reviews_per_month)
# positively skewed. We could do a transformation again:

# log transformation
plot(density(log(train$reviews_per_month),na.rm=TRUE))
boxplot(log(train$reviews_per_month))

# Box-cox transformation
bc_trans <- boxcox(lm(train$reviews_per_month ~ 1))
lambda_opt <- bc_trans$x[which.max(bc_trans$y)]
bc_reviews_per_month <- (train$reviews_per_month^lambda_opt - 1)/lambda_opt
plot(density(bc_reviews_per_month),na.rm=TRUE)
boxplot(bc_reviews_per_month)

#
# reviews_acc, reviews_cleanliness, reviews_checkin, reviews_communication, reviews_location, reviews_value
# reviews_rating
#

# Check for missing values
sum(is.na(train$reviews_rating))

mean_reviews_acc <- mean(train$reviews_acc,na.rm=TRUE)
mean_reviews_cleanliness <- mean(train$reviews_cleanliness,na.rm=TRUE)
mean_reviews_checkin <- mean(train$reviews_checkin,na.rm=TRUE)
mean_reviews_communication <- mean(train$reviews_communication,na.rm=TRUE)
mean_reviews_location <- mean(train$reviews_location,na.rm=TRUE)
mean_reviews_value <- mean(train$reviews_value,na.rm=TRUE)
mean_reviews_rating <- mean(train$reviews_rating,na.rm=TRUE)

# Here we have some additional missing values in the data. -> random missing values
# We can replace these (but keep NA for the 1290 airbnb's with no reviews)
for (i in 1:length(train$reviews_acc)){
  if (is.na(train$reviews_acc[i]) & (is.na(train$reviews_per_month[i]) == FALSE)){train$reviews_acc[i] = mean_reviews_acc}
}

# do the same thing for the others

for (i in 1:length(train$reviews_cleanliness)){
  if (is.na(train$reviews_cleanliness[i]) & (is.na(train$reviews_per_month[i]) == FALSE)){train$reviews_cleanliness[i] = mean_reviews_cleanliness}
}

for (i in 1:length(train$reviews_checkin)){
  if (is.na(train$reviews_checkin[i]) & (is.na(train$reviews_per_month[i]) == FALSE)){train$reviews_checkin[i] = mean_reviews_checkin}
}

for (i in 1:length(train$reviews_communication)){
  if (is.na(train$reviews_communication[i]) & (is.na(train$reviews_per_month[i]) == FALSE)){train$reviews_communication[i] = mean_reviews_communication}
}

for (i in 1:length(train$reviews_location)){
  if (is.na(train$reviews_location[i]) & (is.na(train$reviews_per_month[i]) == FALSE)){train$reviews_location[i] = mean_reviews_location}
}

for (i in 1:length(train$reviews_value)){
  if (is.na(train$reviews_value[i]) & (is.na(train$reviews_per_month[i]) == FALSE)){train$reviews_value[i] = mean_reviews_value}
}

for (i in 1:length(train$reviews_rating)){
  if (is.na(train$reviews_rating[i]) & (is.na(train$reviews_per_month[i]) == FALSE)){train$reviews_rating[i] = mean_reviews_rating}
}

#
#extra
#

# we want to make different categories based on the strings inside this variable ...
#profile_pic
vector = numeric(nrow(train))
for (i in 1:nrow(train)){
  if (grepl("Host Has Profile Pic",  train$extra[i], fixed = TRUE)){
    vector[i] = 1
  }
}
train$profile_pic <- factor(vector)

features <- c(features, "profile_pic")

#exact_location
vector = numeric(nrow(train))
for (i in 1:nrow(train)){
  if (grepl("Is Location Exact",  train$extra[i], fixed = TRUE)){
    vector[i] = 1
  }
}
train$exact_location <- factor(vector)

features <- c(features, "exact_location")

#instant_bookable
vector = numeric(nrow(train))
for (i in 1:nrow(train)){
  if (grepl("Instant Bookable",  train$extra[i], fixed = TRUE)){
    vector[i] = 1
  }
}
train$instant_bookable <- factor(vector)

features <- c(features, "instant_bookable")

#superhost
vector = numeric(nrow(train))
for (i in 1:nrow(train)){
  if (grepl("Host Is Superhost",  train$extra[i], fixed = TRUE)){
    vector[i] = 1
  }
}
train$superhost <- factor(vector)

features <- c(features, "superhost")

#identified_host
vector = numeric(nrow(train))
for (i in 1:nrow(train)){
  if (grepl("Host Identity Verified",  train$extra[i], fixed = TRUE)){
    vector[i] = 1
  }
}
train$identified_host <- factor(vector)

features <- c(features, "identified_host")

targets <- c("target", "bc_target")

# Removing all of the variables listed below (for now):
# "property_type"              "property_zipcode"
# "property_id"                "property_name"              "property_summary"          
# "property_space"             "property_desc"              "property_neighborhood"     
# "property_notes"             "property_transit"           "property_access"           
# "property_interaction"       "property_rules"             "property_lat
# "property_lon"               "property_amenities"         "property_sqfeet"
# "property_scraped_at"        "host_about"                 "extra"
# "host_since"                 "host_location"              "host_verified"

features_to_keep <- c("property_room_type", 
                      "property_max_guests","property_bathrooms", "property_bedrooms",
                      "property_beds", "property_bed_type", "property_last_updated",
                      "host_id", "host_response_time",
                      "host_response_rate", "host_nr_listings",
                      "booking_price_covers", "booking_min_nights", "booking_max_nights",      
                      "booking_availability_30", "booking_availability_60", "booking_availability_90",   
                      "booking_availability_365", "booking_cancel_policy", "reviews_num",               
                      "reviews_first", "reviews_last", "reviews_rating", "reviews_acc",
                      "reviews_cleanliness", "reviews_checkin", "reviews_communication",
                      "reviews_location", "reviews_per_month")
keep <- c(targets, features_to_keep, features)

train_preprocessed <- select(train, all_of(keep))

colSums(is.na(train_preprocessed))

# Write final data set to csv
write.csv(train_preprocessed, "data/preprocessed_train.csv", row.names = FALSE)

params_and_models <- list(
  lambda_opt,
  K_zipcode,
  uncommon_zipcodes,
  tree_zipcode,
  median_bathrooms,
  median_bedrooms,
  median_beds,
  median_host_nr_listings,
  median_years_as_host,
  model_response_rate,
  mean_reviews_acc,
  mean_reviews_cleanliness,
  mean_reviews_checkin,
  mean_reviews_communication,
  mean_reviews_location,
  mean_reviews_value,
  mean_reviews_rating,
  keep
)


################################################################################
#                       Preprocess validation and test data                    #
################################################################################

source("Useful functions.R")

preprocess_data(validation, train, params_and_models, "preprocessed_validation")
preprocess_data(test, train, params_and_models, "preprocessed_test")

validation_preprocessed <- read.csv("data/preprocessed_validation.csv", header=TRUE)
colSums(is.na(validation_preprocessed))

# test_preprocessed <- read.csv("data/preprocessed_test.csv", header=TRUE)
# colSums(is.na(test_preprocessed))




