
# Clear work space
rm(list = ls())

# Load in data
train <- read.csv("data/train.csv", header=TRUE)
set.seed(123456)
train_size <- floor(0.80*nrow(train))
train_idx <- sample(1:nrow(train), train_size, replace = FALSE)
train <- train[train_idx, ]
validation <- train[-train_idx, ]

# Load packages
library(stopwords)
library(stringr)

# Some useful functions
source("Useful functions2.R")
 
################################################################################

# Maak de woordenschat

summary_vocabulary <- get_vocabulary("property_summary")
space_vocabulary <- get_vocabulary("property_space")
desc_vocabulary <- get_vocabulary("property_desc")
neighbourhood_vocabulary <- get_vocabulary("property_neighbourhood")
notes_vocabulary <- get_vocabulary("property_notes")
transit_vocabulary <- get_vocabulary("property_transit")
access_vocabulary <- get_vocabulary("property_access")
interaction_vocabulary <- get_vocabulary("property_interaction")
rules_vocabulary <- get_vocabulary("property_rules")

vocabulary_list <- list(summary_vocabulary,
                        space_vocabulary,
                        desc_vocabulary,
                        neighbourhood_vocabulary,
                        notes_vocabulary,
                        transit_vocabulary,
                        access_vocabulary,
                        interaction_vocabulary,
                        rules_vocabulary)

vocabulary_names <- c("summary_vocabulary", "space_vocabulary", "desc_vocabulary",
                      "neighbourhood_vocabulary", "notes_vocabulary", "transit_vocabulary",
                      "access_vocabulary", "interaction_vocabulary", "rules_vocabulary")


################################################################################

# Misschien is het geen goed idee om een hele tekst samen te vatten in een getal.
# Het kan een idee zijn om te kijken naar de 10 belangrijkste woorden om een
# hoge prijs te voorspellen en de 10 belangrijkste woorden om een lage prijs te
# voorspellen en deze op te slagen.

# Om meer robuuste resultaten te bekomen zullen we ook eisen dat de geselecteerde
# woorden minstens een bepaald aantal keer voorkomen in de training data.

get_important_words <- function(vocabulary, co_high, co_low, co_occ) {
  important_words_high <- vocabulary[which(vocabulary[, "word_mean_value"] > co_high & vocabulary[,"occurrence"] > co_occ), "word"]
  important_words_low <- vocabulary[which(vocabulary[, "word_mean_value"] < co_low & vocabulary[,"occurrence"] > co_occ), "word"]
  
  list("high" = important_words_high, "low" = important_words_low)
}

IMPORTANCE_CUTOFF_HIGH <- 70
IMPORTANCE_CUTOFF_LOW <- 35
OCCURENCE_CUTOFF <- 30

for (i in 1:length(vocabulary_names)) {
  voc <- vocabulary_list[[i]]
  out <- get_important_words(voc, IMPORTANCE_CUTOFF_HIGH, IMPORTANCE_CUTOFF_LOW, OCCURENCE_CUTOFF)
  message("\n", vocabulary_names[i])
  print( paste0("high: ", paste(out[[1]], sep = ", ")) )
  print( paste0("low: ", paste(out[[2]], sep = ", ")) )
}

# Dit geeft een hele lijst woorden. Veel ervan houden niet echt veel steek om
# gerelateerd te kunnen worden aan hoge of lage prijzen van een qualitatief
# standpunt en worden daarom ook genegeerd. Enkele schijnbaar nuttige woorden
# zijn:

# Voor hoge prijzen:
#   summary: gelegen, vlakbij, plenty, hip, deco (wellicht verwijzing naar "art
#            deco"), outside, whole, second (wellicht verwijzend naar een tweede
#            badkamer, terras, etc.),
#
#   space: hallway (indicatief voor een groot eigendom?), linens, en eigenlijk
#          ook gewoon al de rest van de opgelijste woorden.
#
#   desc: vlakbij, won (eventueel een of andere award gewonnen), alone, working,
#         forã (verwijzend naar 'foret' = bos?), hip, another (verwijzend naar
#         een tweede badkamer, terras, etc.), soleil, 3th (verwijzend naar ...),
#         terras, twin, groenplaats (hippe buurt in antwerpen?), ...
#
#   transit: stib (verwwijzend naar de metro in brussel), brussel, yser
#
#   interaction: local, bien

# Voor lage prijzen:
#   space: smartflats (kleine ruimte), upon (verwijzend naar iets dat de
#          bezoeker moet doen als die aankomt?)
#
#   desc: smartflats, emergencies, urgent
#
#   notes: All of them?
#
#   Access: contact ( --> upon arrival, the host needs to be contacted?)
#
#   Interaction: Emergencies, urgent
#
#   Rules: service, prepare, caution

################################################################################

# Is het mogelijk om property_sqfeet ad te leiden uit de tekstjes?

train$my_prop_sq_feet <- get_property_square_feet(train)

sqfeet_not_missing_idx <- which(!is.na(train$property_sqfeet))
train[sqfeet_not_missing_idx, c("my_prop_sq_feet", "property_sqfeet")]

# Dit lijkt niet te werken


# Deze stukjes code worden niet meer gebruikt

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
                                                 summary_vocabulary,
                                                 score_threshold)
}

# Lijkt absoluut niets van train$target te verklaren... :(
plot(train$summary_score, log(train$target))

################################################################################

IMPORTANCE_CUTOFF <- 90
high_price_indicative_words <- summary_vocabulary[which(summary_vocabulary$word_mean_value > IMPORTANCE_CUTOFF & summary_vocabulary$occurrence > 10), "word"]
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








