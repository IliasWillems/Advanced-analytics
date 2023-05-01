# This script contains the functions used in the data preprocessing scripts.

#### Text mining functions ####

# Verwijder alle leestekens in een textje
remove_punctuation <- function (string) {
  gsub('[[:punct:] ]+',' ', string)
}

# Verander alle hoofdletters in kleine letters
set_lower_case <- function(string) {
  tolower(string)
}

# Splits een tekst op in de verschillende woorden. De waarde die deze functie
# terug geeft is bijgevolg een vector met daarin alle woorden van het tekstje
# dat je aan deze functie hebt gegeven.
get_words <- function(string) {
  strsplit(string, ' ')
}

# Verwijder alle spaties aan het begin of het einde van de woorden in de woord-
# vector die je hebt gekregen door "get_words" toe te passen (Het kan
# voorkomen dat er een woord " huis  " in de lijst zit).
trim_exces_whitespace <- function(word_vec) {
  unlist(lapply(word_vec, trimws))
}

# Verwijder alle 'lege woorden'. Dit wil zeggen: alle woorden van de vorm "".
remove_empty_words <- function(word_vec) {
  word_vec[which(nchar(word_vec) > 0)]
}

# Verwijder alle stopwoorden. Het heeft immers geen zin om te proberen bepalen
# of woorden zoals "hij", "ben", "soms", etc. informatief zijn. We weten al op
# voorhand dat dat niet zo is.
remove_unnessecary_words <- function(word_vec) {
  stopword_idx <- which(word_vec %in% stopwords("french") | word_vec %in% stopwords("dutch") | word_vec %in% stopwords("english"))
  
  word_vec[-stopword_idx]
}

# Maak woordenschat voor een bepaalde textuele variabele. Met 'de woordenschat'
# bedoelen we dat we een lijst van alle nuttige woorden maken die in de tekstjes
# van de variabele voorkomen en deze een score geven naargelang de prijs van het
# bijhorende eigendom.
get_vocabulary <- function(text_variable_colname) {
  
  vocabulary <- data.frame("word" = character(), "value" = numeric(),
                           "occurrence" = numeric())
  
  for (i in 1:nrow(train)) { # Deze for-loopt duurt wel een aantal minuutjes...
    
    if (i %% 1000 == 0) {
      message(round(i/nrow(train)*100, 2), "% completion of vocabulary")
    }
    
    summary <- train[i, text_variable_colname]
    
    summary <- remove_punctuation(summary)
    summary <- set_lower_case(summary)
    words <- get_words(summary)
    words <- trim_exces_whitespace(words)
    words <- remove_empty_words(words)
    words <- remove_unnessecary_words(words)
    
    for (word in words) {
      if (word %in% vocabulary$word) {
        word_idx <- which(vocabulary$word == word)
        vocabulary[word_idx, "value"] <- vocabulary[word_idx, "value"] + train[i, "target"]/train[i, "booking_price_covers"]
        vocabulary[word_idx, "occurrence"] <- vocabulary[word_idx, "occurrence"] + 1
      } else {
        vocabulary <- rbind(vocabulary, data.frame("word" = word,
                                                   "value" = train[i, "target"]/train[i, "booking_price_covers"],
                                                   "occurrence" = 1))
      }
    }
  }
  
  # Gemiddelde prijs van huizen waarvan een gegeven woord in de beschrijving staat.
  vocabulary$word_mean_value <- vocabulary$value/vocabulary$occurrence
  
  return(vocabulary)
}

# Nadat we de score van elk woord hebben berekend kunnen we de score van een
# tekst berekenen. Dit doen we door het aantal woorden te tellen in die tekst
# dat in de top 'score_threshold' van hoogst scorende woorden zit.
# Zonder meer zou dit een voordeel geven aan de langere teksten. We delen daarom
# door log(n) om dit in rekening te nemen
get_summary_score <- function(summary, vocabulary, score_threshold) {
  quartile_th <- quantile(vocabulary$word_score, score_threshold)
  
  summary <- remove_punctuation(summary)
  summary <- set_lower_case(summary)
  words <- get_words(summary)
  words <- trim_exces_whitespace(words)
  words <- remove_empty_words(words)
  words <- remove_unnessecary_words(words)
  
  score <- 0
  for (word in words) {
    if (word %in% vocabulary$word) {
      word_score <- vocabulary[which(vocabulary$word == word), "word_score"]
      if (word_score > quartile_th) {
        score <- score + 1
      }
    }
  }
  
  return(score)
}

get_summary_score_2 <- function(summary, high_price_indicative_words) {
  summary <- remove_punctuation(summary)
  summary <- set_lower_case(summary)
  words <- get_words(summary)
  words <- trim_exces_whitespace(words)
  words <- remove_empty_words(words)
  words <- remove_unnessecary_words(words)
  
  score <- 0
  for (word in words) {
    if (word %in% high_price_indicative_words) {
      score <- score + 1
    }
  }
  
  return(score)
}

get_summary_score_3 <- function(dataset, words_high, words_low, summary_list) {
  summary_score <- rep(0, nrow(dataset))
  
  for (i in 1:nrow(dataset)) {
    if (i %% 1000 == 0) {
      message(round(i/nrow(dataset)*100, 2), "% of scores obtained")
    }
    
    score <- 0
    for (text_variable_colname in summary_list) {
      summary <- dataset[i, text_variable_colname]
      
      summary <- remove_punctuation(summary)
      summary <- set_lower_case(summary)
      words <- get_words(summary)
      words <- trim_exces_whitespace(words)
      
      score <- score + (length(which(important_words_high %in% words)) - length(which(important_words_low %in% words)))
    }
    
    summary_score[i] <- score
  }
  return(summary_score)
}

get_important_word_ohe <- function(summary, important_words) {
  ohe <- matrix(rep(0, length(important_words)), nrow = 1)
  colnames(ohe) <- important_words
  for (word in important_words) {
    if (str_detect(summary, word)) {
      ohe[1, "word"] <- 1
    }
  }
  
  as.data.frame(ohe)
}

get_property_square_feet <- function(dataset) {
  
  property_sq_feet <- rep(0, nrow(dataset))
  for (i in 1:nrow(dataset)) {
    hits_desc <- str_extract_all(dataset[i, "property_desc"], regex("[0-9]+[ ]*+m2"))[[1]]
    hits_space <- str_extract_all(dataset[i, "property_space"], regex("[0-9]+[ ]*+m2"))[[1]]
    hits_sum <- str_extract_all(dataset[i, "property_summary"], regex("[0-9]+[ ]*+m2"))[[1]]
    
    hits_both <- union(union(hits_desc, hits_space), hits_sum)
    
    hits_both <- as.integer(gsub("m2", "", hits_both))
    
    if (length(hits_both) != 0) {
      property_sq_feet[i] <- sum(hits_both)
    }
  }
  
  return(property_sq_feet)
}


#### Other data preprocessing functions ####

get_host_country <- function(x) {
  if (x == "") {
    "Other"
  } else {
    tail(trimws(unlist(strsplit(x, ","))), n=1)
  }
}

count_verifications <- function(x) {
  length(unlist(strsplit(x, ",")))
}

count_years_host <- function(x) {
  this_year <- as.numeric(substr(Sys.Date(), 1, 4))
  host_since_year <- as.numeric(substr(x, 1, 4))
  max(0, this_year - host_since_year)
}

preprocess_data <- function(dataset, trainset, params_and_models, outfile) {
  
  lambda_opt <- params_and_models[[1]]
  K_zipcode <- params_and_models[[2]]
  uncommon_zipcodes <- params_and_models[[3]]
  tree_zipcode <- params_and_models[[4]]
  median_bathrooms <- params_and_models[[5]]
  median_bedrooms <- params_and_models[[6]]
  median_beds <- params_and_models[[7]]
  median_host_nr_listings <- params_and_models[[8]]
  median_years_as_host <- params_and_models[[9]]
  model_response_rate <- params_and_models[[10]]
  mean_reviews_acc <- params_and_models[[11]]
  mean_reviews_cleanliness <- params_and_models[[12]]
  mean_reviews_checkin <- params_and_models[[13]]
  mean_reviews_communication <- params_and_models[[14]]
  mean_reviews_location <- params_and_models[[15]]
  mean_reviews_value <- params_and_models[[16]]
  mean_reviews_rating <- params_and_models[[17]]
  keep <- params_and_models[[18]]
  selected_amenities <- params_and_models[[19]]
  summary_list <- params_and_models[[20]]
  important_words_high <- params_and_models[[21]]
  important_words_low <- params_and_models[[22]]
    
  
  #
  # Target
  #
  
  if ("target" %in% colnames(dataset)) {
    dataset$bc_target <- (dataset$target^lambda_opt - 1)/lambda_opt
  }
  
  #
  # Property_zipcode
  # 
  
  dataset$property_zipcode[which(dataset$property_zipcode == "11 20")] <- "1120"
  missing_zip_idx <- which((dataset$property_zipcode == "") | is.na(dataset$property_zipcode))
  dataset$property_zip_missing <- as.numeric(dataset$property_zipcode == "")
  
  for (idx in missing_zip_idx) {
    
    lat <- dataset[idx, "property_lat"]
    lon <- dataset[idx, "property_lon"]
    
    K <- K_zipcode
    
    # Note that we look for nearest neighbours in the training data!
    distances <- sqrt((trainset$property_lat - lat)^2 + (trainset$property_lon - lon)^2)
    neirest_idxs <- sort(distances, index.return = TRUE)$ix[2:(K+1)]
    
    dataset[idx, "property_zipcode"] <- names(which.max(table(trainset[neirest_idxs, "property_zipcode"])))
  }
  
  unseen_zipcodes_idx <- which(!(dataset$property_zipcode %in% train$property_zipcode))
  for (unseen_zip_idx in unseen_zipcodes_idx) {
    zip <- as.numeric(as.character(dataset[unseen_zip_idx, "property_zipcode"]))
    new_zip <-unique(as.numeric(train$property_zipcode))[which.min(abs(zip - unique(as.numeric(train$property_zipcode))))]
    dataset[unseen_zip_idx, "property_zipcode"] <- new_zip
  }
  
  
  dataset$property_zipcode <- as.factor(dataset$property_zipcode)
  zipcode_data$zipcode <- as.factor(zipcode_data$zipcode)
  
  dataset$zipcode_class <- predict(tree_zipcode, newdata = dataset)
  
  #
  # property_lat and property_lon
  #
  
  antwerp_lat = 51.221359578942604; antwerp_lon = 4.398997929972499
  brussels_lat = 50.8465549517382; brussels_lon = 4.351922557884677
  ghent_lat = 51.05446861792642; ghent_lon = 3.725283407738655
  leuven_lat = 50.87899695490237; leuven_lon = 4.701191689536359
  luik_lat = 50.64554868682975; luik_lon = 5.575502106620438
  
  dataset$dist_nearest_city_center <- rep(0, nrow(dataset))
  for (i in 1:nrow(dataset)) {
    lat <- dataset[i, "property_lat"]
    lon <- dataset[i, "property_lon"]
    
    distance_to_antwerp = sqrt((lat - antwerp_lat)^2 + (lon - antwerp_lon)^2)
    distance_to_brussels = sqrt((lat - brussels_lat)^2 + (lon - brussels_lon)^2)
    distance_to_ghent = sqrt((lat - ghent_lat)^2 + (lon - ghent_lon)^2)
    distance_to_leuven = sqrt((lat - leuven_lat)^2 + (lon - leuven_lon)^2)
    distance_to_luik = sqrt((lat - luik_lat)^2 + (lon - luik_lon)^2)
    
    dists <- c(distance_to_antwerp, distance_to_brussels, distance_to_ghent,
               distance_to_leuven, distance_to_luik)
    
    dataset$dist_nearest_city_center[i] <- min(dists)
  }
  
  #
  # property_type and property_room_type
  #
  
  type_data <- data.frame("type" = dataset$property_type)
  
  if (any(type_data$type == "Camper/RV" | type_data$type == "Yurt")) {
    type_data[which(type_data$type == "Camper/RV" | type_data$type == "Yurt"), "type"] <- "Tent"
    warning("Changed property type 'Camper/RV' and/or 'Yurt' to 'Tent'.")
  }
  if (any(!(type_data$type %in% levels(trainset$property_type)))) {
    type_data[which(!(type_data$type %in% levels(trainset$property_type))), "type"] <- "Tent"
    warning("Changed unknown property types to 'Apartment'")
  }
  
  colnames(type_data) <- c("property_type")
  
  dataset$property_type <- as.factor(dataset$property_type)
  dataset$property_room_type <- as.factor(dataset$property_room_type)
  dataset$type_class <- predict(tree_prop_type, newdata = type_data)
  
  #
  # property_bathrooms, property_bedrooms, property_beds
  #
  
  dataset[which(is.na(dataset$property_bathrooms)), "property_bathrooms"] <- median_bathrooms
  dataset[which(is.na(dataset$property_bedrooms)), "property_bedrooms"] <- median_bedrooms
  dataset[which(is.na(dataset$property_beds)), "property_beds"] <- median_beds
  
  #
  # property_last_updated
  #
  
  property_last_updated_numerical <- rep(0, nrow(dataset))
  for (i in 1:nrow(dataset)) {
    components <- unlist(strsplit(dataset[i, "property_last_updated"], split = " "))
    
    if (length(components) == 3) {
      multiplier <- 1
      if (components[2] == "months" | components[2] == "month") {
        multiplier <- 31
      } else if (components[2] == "weeks" | components[2] == "week") {
        multiplier <- 7
      } else {
        multiplier <- 365
      }
      
      if (components[1] == "a") {
        number <- 1
      } else {
        number <- as.numeric(components[1])
      }
      
      property_last_updated_numerical[i] <- multiplier*number
      
    } else if (components[1] == "today") {
      property_last_updated_numerical[i] <- 0
    } else if (components[1] == "yesterday") {
      property_last_updated_numerical[i] <- 1
    } else { # never
      property_last_updated_numerical[i] <- NA
    }
  }
  
  property_last_updated_numerical[which(is.na(property_last_updated_numerical))] <- median(property_last_updated_numerical, na.rm = TRUE)
  dataset$property_last_updated_numerical <- property_last_updated_numerical
  
  #
  # property_amenities
  #
  
  amenity_data <- data.frame(dummy = rep(0, nrow(dataset)))
  
  for (amenity in selected_amenities) {
    amenity_data[,amenity] <- rep(0, nrow(dataset))
    for (i in 1:nrow(dataset)) {
      if (amenity %in% unlist(strsplit(dataset[i, "property_amenities"], ", "))) {
        amenity_data[i, amenity] <- 1
      }
    }
  }
  
  amenity_data <- amenity_data[,-1]
  dataset <- cbind(dataset, amenity_data)
  
  #
  # host_location
  #
  
  host_location_country <- unlist(lapply(dataset$host_location, get_host_country))
  host_location_country[which(host_location_country %in% c("Belgie", "belgium", "belgique", "Belgique", "belgie", "Belgium"))] <- "BE"
  host_location_country[which(host_location_country %in% c("France"))] <- "FR"
  host_location_country[which(host_location_country %in% c("Netherlands", "The Netherlands", "Nederland"))] <- "NL"
  host_location_country[which(!(host_location_country %in% c("BE", "NL", "FR")))] <- "other"
  dataset$host_location_country <- host_location_country
  
  #
  # host_nr_listings, host_nr_listings_total
  #
  
  dataset[which(is.na(dataset$host_nr_listings)), "host_nr_listings"] <- median_host_nr_listings
  dataset <- dplyr::select(dataset, -c("host_nr_listings_total"))
  
  #
  # host_verified
  #
  
  dataset$host_verified_amount <- unlist(lapply(dataset$host_verified, count_verifications))
  
  #
  # host_since
  #
  
  dataset$years_as_host <- unlist(lapply(dataset$host_since, count_years_host))
  dataset[which(is.na(dataset$years_as_host)), "years_as_host"] <- median_years_as_host
  
  #
  # host_response_rate
  #
  
  predictions <- predict(model_response_rate,
                         dataset[which(is.na(dataset$host_response_rate)),
                                 c("host_location_country", "host_response_time",
                                   "host_nr_listings", "host_verified_amount",
                                   "years_as_host")])
  
  dataset[which(is.na(dataset$host_response_rate)), "host_response_rate"] <- pmin(predictions, 100)
  
  #
  # booking_min_nights, booking_max_night
  #
  
  invalid_min_max_nights <- subset(dataset, booking_min_nights > booking_max_nights,
                                   select=c("property_id", "booking_min_nights", "booking_max_nights"))
  for (i in 1:nrow(invalid_min_max_nights)) {
    dataset$booking_min_nights[dataset$property_id == invalid_min_max_nights[i, "property_id"]] <- invalid_min_max_nights[i, "booking_max_nights"]
    dataset$booking_max_nights[dataset$property_id == invalid_min_max_nights[i, "property_id"]] <- invalid_min_max_nights[i, "booking_min_nights"]
  }
  
  #
  # booking_availability_30/60/90/365
  #
  
  # Divide into categories
  dataset$category_30 <- cut(dataset$booking_availability_30, breaks = unique(c(-Inf, q30, Inf)), labels = c("low", "medium", "high"))
  dataset$category_60 <- cut(dataset$booking_availability_60, breaks = unique(c(-Inf, q60, Inf)), labels = c("low", "medium", "high"))
  dataset$category_90 <- cut(dataset$booking_availability_90, breaks = unique(c(-Inf, q90, Inf)), labels = c("low", "medium", "high"))
  dataset$category_365 <- cut(dataset$booking_availability_365, breaks = unique(c(-Inf, q365, Inf)), labels = c("low", "medium", "high"))
  
  # Create dummy columns for category_30
  dataset$category_30_low <- ifelse(dataset$category_30 == "low", 1, 0)
  dataset$category_30_medium <- ifelse(dataset$category_30 == "medium", 1, 0)
  dataset$category_30_high <- ifelse(dataset$category_30 == "high", 1, 0)
  
  # Create dummy columns for category_60
  dataset$category_60_low <- ifelse(dataset$category_60 == "low", 1, 0)
  dataset$category_60_medium <- ifelse(dataset$category_60 == "medium", 1, 0)
  dataset$category_60_high <- ifelse(dataset$category_60 == "high", 1, 0)
  
  # Create dummy columns for category_90
  dataset$category_90_low <- ifelse(dataset$category_90 == "low", 1, 0)
  dataset$category_90_medium <- ifelse(dataset$category_90 == "medium", 1, 0)
  dataset$category_90_high <- ifelse(dataset$category_90 == "high", 1, 0)
  
  # Create dummy columns for category_365
  dataset$category_365_low <- ifelse(dataset$category_365 == "low", 1, 0)
  dataset$category_365_medium <- ifelse(dataset$category_365 == "medium", 1, 0)
  dataset$category_365_high <- ifelse(dataset$category_365 == "high", 1, 0)
  
  #
  # reviews_num
  #
  
  dataset$reviews_num_cat <- rep(0, nrow(dataset))
  
  for (i in 1:length(dataset$reviews_num)){
    if (dataset$reviews_num[i] >= 1 & dataset$reviews_num[i]<=5){dataset$reviews_num_cat[i] = 1}
    else if (dataset$reviews_num[i] > 5 & dataset$reviews_num[i] <= 19){dataset$reviews_num_cat[i] = 2}
    else {dataset$reviews_num_cat[i] = 3}
  }
  
  #
  # reviews_first, reviews_last, reviews_per_month
  #
  
  dataset$review_period <- as.numeric(as.Date(dataset$reviews_last) - as.Date(dataset$reviews_first))
  
  # Contains a lot of missing values. Store which values were missing.
  dataset$review_period_was_missing <- as.numeric(is.na(dataset$review_period))
  
  #
  # reviews_acc, reviews_cleanliness, reviews_checkin, reviews_communication, reviews_location, reviews_value
  # reviews_rating
  #
  
  for (i in 1:length(dataset$reviews_acc)){
    if (is.na(dataset$reviews_acc[i]) & (is.na(dataset$reviews_per_month[i]) == FALSE)) {
      dataset$reviews_acc[i] = mean(train$reviews_acc,na.rm=TRUE)
    }
  }
  
  # do the same thing for the others
  
  for (i in 1:length(dataset$reviews_cleanliness)){
    if (is.na(dataset$reviews_cleanliness[i]) & (is.na(dataset$reviews_per_month[i]) == FALSE)) {
      dataset$reviews_cleanliness[i] = mean_reviews_cleanliness
    }
  }
  
  for (i in 1:length(dataset$reviews_checkin)){
    if (is.na(dataset$reviews_checkin[i]) & (is.na(dataset$reviews_per_month[i]) == FALSE)) {
      dataset$reviews_checkin[i] = mean_reviews_checkin
    }
  }
  
  for (i in 1:length(dataset$reviews_communication)){
    if (is.na(dataset$reviews_communication[i]) & (is.na(dataset$reviews_per_month[i]) == FALSE)) {
      dataset$reviews_communication[i] = mean_reviews_communication
    }
  }
  
  for (i in 1:length(dataset$reviews_location)){
    if (is.na(dataset$reviews_location[i]) & (is.na(dataset$reviews_per_month[i]) == FALSE)) {
      dataset$reviews_location[i] = mean_reviews_location
    }
  }
  
  for (i in 1:length(dataset$reviews_value)){
    if (is.na(dataset$reviews_value[i]) & (is.na(dataset$reviews_per_month[i]) == FALSE)) {
      dataset$reviews_value[i] = mean_reviews_value
    }
  }
  
  for (i in 1:length(dataset$reviews_rating)){
    if (is.na(dataset$reviews_rating[i]) & (is.na(dataset$reviews_per_month[i]) == FALSE)) {
      dataset$reviews_rating[i] = mean_reviews_rating
    }
  }
  
  
  #
  # extra
  #
  
  vector = numeric(nrow(dataset))
  for (i in 1:nrow(dataset)){
    if (grepl("Host Has Profile Pic",  dataset$extra[i], fixed = TRUE)){
      vector[i] = 1
    }
  }
  dataset$profile_pic <- factor(vector)
  
  vector = numeric(nrow(dataset))
  for (i in 1:nrow(dataset)){
    if (grepl("Is Location Exact",  dataset$extra[i], fixed = TRUE)){
      vector[i] = 1
    }
  }
  dataset$exact_location <- factor(vector)
  
  vector = numeric(nrow(dataset))
  for (i in 1:nrow(dataset)){
    if (grepl("Instant Bookable",  dataset$extra[i], fixed = TRUE)){
      vector[i] = 1
    }
  }
  dataset$instant_bookable <- factor(vector)
  
  vector = numeric(nrow(dataset))
  for (i in 1:nrow(dataset)){
    if (grepl("Host Is Superhost",  dataset$extra[i], fixed = TRUE)){
      vector[i] = 1
    }
  }
  dataset$superhost <- factor(vector)
  
  vector = numeric(nrow(dataset))
  for (i in 1:nrow(dataset)){
    if (grepl("Host Identity Verified",  dataset$extra[i], fixed = TRUE)){
      vector[i] = 1
    }
  }
  dataset$identified_host <- factor(vector)
  
  if (!("target" %in% colnames(dataset))) {
    keep <- keep[-c(1, 2)]
  }
  
  #
  # property_summary, property_space, property_desc, property_neighbourhood,
  # property_notes, property_transit, property_access, property_interaction,
  # property_rules
  #
  
  dataset$summary_score <- get_summary_score_3(dataset, important_words_high, important_words_low, summary_list)
  
  features <- c(features, "summary_score")
  
  # Also keep id column (necessary to identify predictions)
  keep <- c("property_id", keep)
  
  dataset_preprocessed <- dplyr::select(dataset, all_of(keep))
  
  colSums(is.na(dataset_preprocessed))
  
  # Write final data set to csv
  write.csv(dataset_preprocessed, paste0("data/", outfile, ".csv"),
            row.names = FALSE)
}

