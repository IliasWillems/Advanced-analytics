
# Clear work space
rm(list = ls())

# Load in data
train <- read.csv("data/train.csv", header=TRUE)

# Load necessary packages
library(MASS)
library(tree)
library(rpart)
library(rpart.plot)
library(dplyr)

################################################################################
#                        Data inspection + cleaning                            #
################################################################################

# BELANGRIJK: Alle analyses hieronder nemen niet in rekening dat de prijs niet
#             altijd overeenstemt met prijs per nacht of prijs per persoon, maar
#             dat het kan zijn dat je zou moeten bijbetalen voor extra personen.
#             
#             --> Afspreken hoe we dit oplossen!

summary(train)


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
  # How could we impute these?
  # 1) Search for clues in the description, neighbourhood, etc. columns
  # 2) Look at latitude and longitude values and find nearest neighbours <--

  # For all properties with a missing zip code...
  for (idx in missing_zip_idx) {
    
    # Store the geographical coordinates of that property
    lat <- train[idx, "property_lat"]
    lon <- train[idx, "property_lon"]
    
    # We will look at the K = 10 nearest neighbours
    K <- 10
    
    # Find K nearest neighbours (euclidean distance) in terms of coordinates
    distances <- sqrt((train$property_lat - lat)^2 + (train$property_lon - lon)^2)
    neirest_idxs <- sort(distances, index.return = TRUE)$ix[2:(K+1)]
    
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
  
  tree_zipcode <- rpart(price ~ zipcode, data = zipcode_data, control=rpart.control(cp=.0003))
  printcp(tree_zipcode)
  
  #plot the pruned tree
  prp(tree_zipcode,
      faclen=0, #use full names for factor labels
      extra=1, #display number of obs. for each terminal node
      roundint=F, #don't round to integers in output
      digits=5) #display 5 decimal places in output
  
  train$zipcode_class <- predict(tree_zipcode)
}

#
# property_lat and property_lon
#

# Based on these coordinates, we could determine the distance to the center of
# the nearest city and use it as a covariate later. Coordinates are retrieved
# from Google maps.
antwerp_lat = 51.221359578942604; antwerp_lon = 4.398997929972499 # city hall
ghent_lat = 51.05446861792642; ghent_lon = 3.725283407738655 # city hall
leuven_lat = 50.87899695490237; leuven_lon = 4.701191689536359 # city hall
luik_lat = 50.64554868682975; luik_lon = 5.575502106620438 # city hall

train$dist_nearest_city_center <- rep(0, nrow(train))
for (i in 1:nrow(train)) {
  lat <- train[i, "property_lat"]
  lon <- train[i, "property_lon"]
  
  distance_to_antwerp = sqrt((lat - antwerp_lat)^2 + (lon - antwerp_lon)^2)
  distance_to_ghent = sqrt((lat - ghent_lat)^2 + (lon - ghent_lon)^2)
  distance_to_leuven = sqrt((lat - leuven_lat)^2 + (lon - leuven_lon)^2)
  distance_to_luik = sqrt((lat - luik_lat)^2 + (lon - luik_lon)^2)
  
  dists <- c(distance_to_antwerp, distance_to_ghent, distance_to_leuven,
             distance_to_luik)
  
  train$dist_nearest_city_center[i] <- min(dists)
}


#
# property_type and property_room_type
#

# What are all of the property (room) types? How much of each in data set?
# While there are a lot of different property types, there are only three
# different property room types. Therefore, property_room_type does not seem
# to need preprocessing but property_type does. We use the same idea as we did
# for zip codes and reclassyfy based on a decision tree.
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
  
  tree_prop_type <- rpart(price ~ type, data = type_data, control=rpart.control(cp=.0001))
  printcp(tree_prop_type)
  
  #plot the pruned tree
  prp(tree_prop_type,
      faclen=0, #use full names for factor labels
      extra=1, #display number of obs. for each terminal node
      roundint=F, #don't round to integers in output
      digits=5) #display 5 decimal places in output
  
  train$type_class <- predict(tree_prop_type)
}



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

# Treat missing values
train$property_bathrooms_missing <- rep(0, nrow(train))
train[which(is.na(train$property_bathrooms)), "property_bathrooms_missing"] <- 1
train[which(is.na(train$property_bathrooms)), "property_desc"]
  # Based on these descriptions, it seems that the number of bathrooms in each
  # of the properties with missing values for them could be
  # [1] 0 (shared)
  # [2] 0 (shared)
  # [3] 1
  # [4] 1 (probably)
  # [5] Not clear
  # [6] Not clear
  # [7] at least 1
  # [8] 1
  # [9] 1
  # [10] 1
  # [11] 1 
  # [12] 1
  #
  # Maybe just change these values manually?
  replacements_bathrooms <- c(0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1)
  bathroom_missing_idxs <- which(is.na(train$property_bathrooms))
  for (i in 1:length(bathroom_missing_idxs)) {
    train[bathroom_missing_idxs[i], "property_bathrooms"] <- replacements_bathrooms[i]
  }

# Some properties have 0 bathrooms and some have a non-integer amount of
# bathrooms. Since the number of bathrooms is usually an indication of the 
# luxury (and hence price) of the property, this will likely be a good variable
# to include in the model. Based on the plot, however, it only seems to matter
# whether there are 0 bathrooms or not.
table(train$property_bathrooms)
plot(train$property_bathrooms, bc_target)
train[which(train$property_bathrooms == 0), "property_type"]

# Is the 0 bedroom property an outlier?

# What is up with the 0 square feet appartments?

## How many unique hosts are there?
num_unique_host_ids <- length(unique(train$host_id))
print(num_unique_host_ids)

## Can we interpolate host response rates from other properties of that host? ##

# First some data exploration
head(train[,c("host_response_rate", "host_location", "host_response_time",
           "host_nr_listings", "host_verified", "host_since")])

# There are a lot of different levels. Maybe boil it down to just country of
# origin.
unique(train$host_location)

get_host_country <- function(x) {
  if (x == "") {
    "Other"
  } else {
    tail(trimws(unlist(strsplit(x, ","))), n=1)
  }
}

host_location_country <- lapply(train$host_location, get_host_country)
host_location_country <- unlist(lapply(train$host_location, get_host_country))
host_location_country[which(host_location_country %in% c("Belgie", "belgium", "belgique", "Belgique", "belgie", "Belgium"))] <- "BE"
host_location_country[which(host_location_country %in% c("France"))] <- "FR"
host_location_country[which(host_location_country %in% c("Netherlands", "The Netherlands", "Nederland"))] <- "NL"
host_location_country[which(!(host_location_country %in% c("BE", "NL", "FR")))] <- "other"
train$host_location_country <- host_location_country

# Amount of levels (4) is okay. However, if host_response_time is missing, then
# host_response_rate is also missing, so we cannot use this in the predictions.
table(train$host_response_time)
table(train[which(!is.na(train$host_response_rate)), "host_response_time"])

# Again a lot of different levels. Maybe just summarize this by counting the
# amount of verifications.
unique(train$host_verified)

count_verifications <- function(x) {
  length(unlist(strsplit(x, ",")))
}
train$host_verified_amount <- unlist(lapply(train$host_verified, count_verifications))

# Again a lot of levels. Let's transform this variable to show the amount of time
# (in years) that the host has been a host
unique(train$host_since)

count_years_host <- function(x) {
  this_year <- as.numeric(substr(Sys.Date(), 1, 4))
  host_since_year <- as.numeric(substr(x, 1, 4))
  max(0, this_year - host_since_year)
}

train$years_as_host <- unlist(lapply(train$host_since, count_years_host))

# Build a linear regression model to predict response rate from other properties
model <- lm(host_response_rate ~ host_location + host_response_time + 
              host_nr_listings + host_verified + host_since, data = train)

# for example make predictions for new data based on the trained model
new_data <- data.frame(host_location = "Brussels, Brussels, Belgium",
                       host_response_time = "within a few hours",
                       host_nr_listings = 1,
                       host_verified = "email, phone, reviews",
                       host_since = "2013-11-14")
predicted_response_rate <- predict(model, new_data)

# Print the predicted response rate
print(predicted_response_rate)

# Do the same but with new variables
model <- lm(host_response_rate ~ host_location_country + 
              host_nr_listings + host_verified_amount + years_as_host, data = train)

# for example make predictions for new data based on the trained model
new_data <- data.frame(host_location_country = "BE",
                       host_nr_listings = 1,
                       host_verified_amount = 3,
                       years_as_host = 6)
predicted_response_rate <- predict(model, new_data)

# Print the predicted response rate
print(predicted_response_rate)

# use this model to impute missing values for host_response_rate
predictions <- predict(model, train[which(is.na(train$host_response_rate)),
                                    c("host_location_country", "host_response_time",
                                      "host_nr_listings", "host_verified_amount",
                                      "years_as_host")])
predictions <- pmin(predictions, 100)

# Impute the missing values with this model. Note that we do not have to store
# the information about which values were initially missing since this is
# represented by the empty strings in host_response_time.
train[which(is.na(train$host_response_rate)), "host_response_rate"] <- predictions


## what is the difference with host_nr_listings and host_nr_listings_total?
if(all(train$host_nr_listings == train$host_nr_listings_total, na.rm = TRUE)) {
  print("Column A and B are identical")
}

# Does a host really have 591 properties?
boxplot(train$host_nr_listings)
  # Extremely skewed variable...

# What is up with booking_min_nights being 1000 sometimes? 


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
# I changed the variable values inside the train data set. (Maybe make a separate variable?)
for (i in 1:length(train$reviews_num)){
  if (train$reviews_num[i] >= 1 & train$reviews_num[i]<=5){train$reviews_num[i] = 1}
  if (train$reviews_num[i] >= 6 & train$reviews_num[i]<=19){train$reviews_num[i] = 2}
  if (train$reviews_num[i] > 19){train$reviews_num[i] = 3}
}

#
# reviews_first, reviews_last, reviews_per_month
#

#reviews_first and reviews_last: maybe create new variable reviews_period? The date 
# of the last review is still valuable. (NEVER use dates directly -> lecture)

#I want to check if the 1290  missing values are the 0 values for reviews_num
length(train$reviews_num[which(train$reviews_num==0)])
# yes. There are 1290 instances with no review information.This also gives us information about
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

# Here we have some additional missing values in the data. -> random missing values
# We can replace these (but keep NA for the 1290 airbnb's with no reviews)
for (i in 1:length(train$reviews_acc)){
  if (is.na(train$reviews_acc[i]) & (is.na(train$reviews_per_month[i]) == FALSE)){train$reviews_acc[i] = mean(train$reviews_acc,na.rm=TRUE)}
}

# do the same thing for the others

for (i in 1:length(train$reviews_cleanliness)){
  if (is.na(train$reviews_cleanliness[i]) & (is.na(train$reviews_per_month[i]) == FALSE)){train$reviews_cleanliness[i] = mean(train$reviews_cleanliness,na.rm=TRUE)}
}

for (i in 1:length(train$reviews_checkin)){
  if (is.na(train$reviews_checkin[i]) & (is.na(train$reviews_per_month[i]) == FALSE)){train$reviews_checkin[i] = mean(train$reviews_checkin,na.rm=TRUE)}
}

for (i in 1:length(train$reviews_communication)){
  if (is.na(train$reviews_communication[i]) & (is.na(train$reviews_per_month[i]) == FALSE)){train$reviews_communication[i] = mean(train$reviews_communication,na.rm=TRUE)}
}

for (i in 1:length(train$reviews_location)){
  if (is.na(train$reviews_location[i]) & (is.na(train$reviews_per_month[i]) == FALSE)){train$reviews_location[i] = mean(train$reviews_location,na.rm=TRUE)}
}

for (i in 1:length(train$reviews_value)){
  if (is.na(train$reviews_value[i]) & (is.na(train$reviews_per_month[i]) == FALSE)){train$reviews_value[i] = mean(train$reviews_value,na.rm=TRUE)}
}

for (i in 1:length(train$reviews_rating)){
  if (is.na(train$reviews_rating[i]) & (is.na(train$reviews_per_month[i]) == FALSE)){train$reviews_rating[i] = mean(train$reviews_rating,na.rm=TRUE)}
}

#
#extra
#

# we want to make different categories based on the strings inside this variable ...
#profile_pic
vector = numeric(6495)
for (i in 1:6495){
  if (grepl("Host Has Profile Pic",  train$extra[i], fixed = TRUE)){
    vector[i] = 1
  }
}
train$profile_pic <- factor(vector)

#exact_location
vector = numeric(6495)
for (i in 1:6495){
  if (grepl("Is Location Exact",  train$extra[i], fixed = TRUE)){
    vector[i] = 1
  }
}
train$exact_location <- factor(vector)

#instant_bookable
vector = numeric(6495)
for (i in 1:6495){
  if (grepl("Instant Bookable",  train$extra[i], fixed = TRUE)){
    vector[i] = 1
  }
}
train$instant_bookable <- factor(vector)

#superhost
vector = numeric(6495)
for (i in 1:6495){
  if (grepl("Host Is Superhost",  train$extra[i], fixed = TRUE)){
    vector[i] = 1
  }
}
train$superhost <- factor(vector)

#identified_host
vector = numeric(6495)
for (i in 1:6495){
  if (grepl("Host Identity Verified",  train$extra[i], fixed = TRUE)){
    vector[i] = 1
  }
}
train$identified_host <- factor(vector)



