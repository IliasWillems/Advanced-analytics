
# Clear work space
rm(list = ls())

# Load in data
train <- read.csv("data/train.csv", header=TRUE)

# Load packages
library(stopwords)

# Some useful functions

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
  
  return(score/log(length(words)))
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

################################################################################

# Maak de woordenschat

vocabulary <- data.frame("word" = character(), "value" = numeric(),
                         "occurrence" = numeric())

for (i in 1:nrow(train)) { # Deze for-loopt duurt wel een aantal minuutjes...
  
  if (i %% 1000 == 0) {
    message(round(i/nrow(train)*100, 2), "% completion of vocabulary")
  }
  
  summary <- train[i, "property_summary"]
  
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

# We definiëren de woordscore als "word_mean_value" x log(occurrence). Door
# vermenigvuldiging met log(occurrence) kunnen we ervoor zorgen dat de score
# van woorden die maar 1 keer voorkomen (outliers) gelijk is aan 0 en dat de
# score van een woord in steeds mindere maten toeneemt naarmate dat woord meer
# en meer voorkomt. Dit effect wordt afgekapt vanaf 100.
vocabulary$word_score <- vocabulary$word_mean_value*pmin(log(vocabulary$occurrence), log(100))
head(vocabulary[sort(vocabulary$word_score, index.return = TRUE, decreasing = TRUE)[[2]], ], 20)

# Een iets andere, meer directe aanpak: selecteer de woorden die in de
# beschrijvingen staan van eigendommen die hoog geprijsd zijn. Om deze aanpak
# toch enigzins robuust te maken vereisen we ook dat dit in meer dan 3 gevallen
# voorkomt.
IMPORTANCE_CUTOFF <- 150
vocabulary[which(vocabulary$word_mean_value > IMPORTANCE_CUTOFF & vocabulary$occurrence > 3), "word"]


################################################################################

# Het percentiel van de kleinste woordscore die we in rekening nemen bij het
# berekenen van de score van een summary.
score_threshold <- 0.80

train$summary_score <- rep(0, nrow(train))
for (i in 1:nrow(train)) {
  if (i %% 1000 == 0) {
    message(round(i/nrow(train)*100, 2), "% completion of score calculations")
  }
  train[i, "summary_score"] <- get_summary_score(train[i, "property_summary"],
                                                 vocabulary,
                                                 score_threshold)
}

# Lijkt absoluut niets van train$target te verklaren... :(
plot(train$summary_score, log(train$target))

################################################################################

IMPORTANCE_CUTOFF <- 90
high_price_indicative_words <- vocabulary[which(vocabulary$word_mean_value > IMPORTANCE_CUTOFF & vocabulary$occurrence > 10), "word"]
high_price_indicative_words

train$summary_score_2 <- rep(0, nrow(train))
for (i in 1:nrow(train)) {
  if (i %% 1000 == 0) {
    message(round(i/nrow(train)*100, 2), "% completion of score calculations")
  }
  
  train[i, "summary_score_2"] <- get_summary_score_2(train[i, "property_summary"],
                                                     high_price_indicative_words)
}

plot(train$summary_score_2, log(train$target))








