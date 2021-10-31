## INTRODUCTION


#Show the structure of the dataset
load("edx")
str(edx)

###########################################################################

## ANALYSIS


#load different libraries
library(tidyverse)
library(corrplot)
library(Hmisc)
library(lubridate)

set.seed(1, sample.kind = "Rounding") # set a seed so that the results are always the same
ind <- sample(1:9000055,50000) # take a sample of the edx data set
edx_smp <- edx[ind] # choose the sample from edx
edx_smp <- edx_smp %>% select(-title) # remove the title column
edx_smp$genres <- as.numeric(as.factor(edx_smp$genres)) # change the genre column to numeric factors
corrlt <- rcorr(as.matrix(edx_smp)) # create the correlation object
corrplot(corrlt$r, type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45, title = "Correlation of the Variables") # plot the correlations with their significance

edx_pc <- edx_smp %>% select(-userId) # deselect the userId
edx_pca <- prcomp(edx_pc, scale = TRUE) # produce principal components
screeplot(edx_pca, type = "line", main = "Scree plot", cex.main = 1.8) # plot screeplot

sil <- silhouette((cutree(clst.h,8)),clst.d) # create a silhouette object with 8 clusters
summary(sil)

###################################################################################

## METHOD

### MACHINE LEARNING MODELS


library(caret) # load caret library for machine learning
set.seed(33)
test_ind <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE) # make a data partition for a test set and train set
train <- edx[-test_ind,] # training set
test <- edx[test_ind,] # testing set
test <- test %>% 
  semi_join(train, by = "movieId") %>% semi_join(train, by = "userId") # get rid of movies or users that are in the test set but not the training set.
rmse <- function(ratings, predicted_ratings){
  sqrt(mean((ratings - predicted_ratings)^2))
} # a function that calculates the error that we have to minimize to get better models

#### FIRST MODEL

mu <- mean(train$rating) # average rating for all movies
first_rmse <- rmse(test$rating,mu) # the first prediction's error
first_rmse # print rmse

#### SECOND MODEL

avgs <- train %>% group_by(movieId) %>% summarise(a = mean(rating - mu)) # the average rating for each movie
predicted_ratings <- mu + test %>% 
  left_join(avgs, by='movieId') %>%
  .$a # our prediction of the rating of movies in the test set
second_rmse <- rmse(predicted_ratings,test$rating) # the second prediction's error
second_rmse

#### THIRD MODEL

avgr <- train %>% left_join(avgs, by = "movieId") %>% group_by(userId) %>% summarise(b = mean(rating - mu - a)) # the average rating given by each user
predicted_ratings <- test %>% left_join(avgs, by='movieId') %>% left_join(avgr, by='userId') %>% mutate(pred = mu + a + b) %>% .$pred # our prediction of the rating of movies in the test set
third_rmse <- rmse(predicted_ratings,test$rating) # the third prediction's error
third_rmse

#### FOURTH MODEL

avgt <- train %>% left_join(avgs, by = "movieId") %>% left_join(avgr, by = "userId") %>% group_by(genres) %>% summarise(c = mean(rating - mu - a - b)) # the average rating for each genre
predicted_ratings <- test %>% left_join(avgs, by='movieId') %>% left_join(avgr, by='userId') %>% left_join(avgt, by = "genres") %>% mutate(pred = mu + a + b + c) %>% .$pred # our prediction of the rating of movies in the test set
fourth_rmse <- rmse(predicted_ratings,test$rating) # the fourth prediction's error
fourth_rmse

#### FIFTH MODEL

train <- train %>% mutate(timestamp_t=format(as_datetime(train$timestamp), "%Y-%m-%d")) # add a new column to the training set with new date format
test <- test %>% mutate(timestamp_t=format(as_datetime(test$timestamp), "%Y-%m-%d")) # add a new column to the test set with new date format

avgu <- train %>% left_join(avgs, by = "movieId") %>% left_join(avgr, by = "userId") %>% left_join(avgt, by = "genres") %>% group_by(timestamp_t) %>% summarise(d = mean(rating - mu - a - b - c)) # the average rating for each timestamp
predicted_ratings <- test %>% left_join(avgs, by='movieId') %>% left_join(avgr, by='userId') %>% left_join(avgt, by = "genres") %>% left_join(avgu, by = "timestamp_t") %>% mutate(pred = mu + a + b + c + d) %>% .$pred # our prediction of the rating of movies in the test set
fifth_rmse <- rmse(predicted_ratings,test$rating) # the fifth prediction's error
fifth_rmse

#### SIXTH MODEL

pens <- seq(0, 10, 0.25) # the range of penalty constants to choose from
RMSES <- sapply(pens, function(p){
  a_r <- train %>%
    group_by(movieId) %>%
    summarise(a_r = sum(rating - mu)/(n()+p)) # regularization of the parameter
  b_r <- train %>% 
    left_join(a_r, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_r = sum(rating - a_r - mu)/(n()+p)) # regularization of the parameter
  c_r <- train %>% 
    left_join(a_r, by="movieId") %>%          
    left_join(b_r, by="userId") %>%
    group_by(genres) %>%
    summarise(c_r = sum(rating - a_r - b_r - mu)/(n()+p)) # regularization of the parameter 
  d_r <- train %>% 
    left_join(a_r, by="movieId") %>%
    left_join(b_r, by="userId") %>%
    left_join(c_r, by="genres") %>% 
    group_by(timestamp_t) %>%
    summarise(d_r = sum(rating - a_r - b_r - c_r - mu)/(n()+p)) # regularization of the parameter
  predicted_ratings <- 
    test %>% 
    left_join(a_r, by = "movieId") %>%
    left_join(b_r, by = "userId") %>%
    left_join(c_r, by = "genres") %>%
    left_join(d_r, by = "timestamp_t") %>%
    mutate(pred = mu + a_r + b_r + c_r + d_r) %>%
    .$pred # making the prediction of ratings
  return(rmse(predicted_ratings, test$rating))
}) # A function that returns rmses at different penalties

pen <- pens[which.min(RMSES)] # look for the penalty constant which gives the minimum error
pen # print the penalty constant

sixth_model <- function(p){
  a_r <- train %>%
    group_by(movieId) %>%
    summarise(a_r = sum(rating - mu)/(n()+p)) # regularization of the parameter
  b_r <- train %>% 
    left_join(a_r, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_r = sum(rating - a_r - mu)/(n()+p)) # regularization of the parameter
  c_r <- train %>% 
    left_join(a_r, by="movieId") %>%
    left_join(b_r, by="userId") %>%
    group_by(genres) %>%
    summarise(c_r = sum(rating - a_r - b_r - mu)/(n()+p)) # regularization of the parameter
  d_r <- train %>% 
    left_join(a_r, by="movieId") %>%
    left_join(b_r, by="userId") %>%
    left_join(c_r, by="genres") %>% 
    group_by(timestamp_t) %>%
    summarise(d_r = sum(rating - a_r - b_r - c_r - mu)/(n()+p)) # regularization of the parameter
  predicted_ratings <- 
    test %>% 
    left_join(a_r, by = "movieId") %>%
    left_join(b_r, by = "userId") %>%
    left_join(c_r, by = "genres") %>%
    left_join(d_r, by = "timestamp_t") %>%
    mutate(pred = mu + a_r + b_r + c_r + d_r) %>%
    .$pred # making the prediction of ratings
  return(rmse(predicted_ratings, test$rating))
} # A function that returns the lowest error
sixth_model(5.25)

################################################################################

## CONCLUSION WITH THE FINAL MODEL

load("validation")
validation_edit <- validation %>% 
  semi_join(train, by = "movieId") %>% semi_join(train, by = "userId") # get rid of movies or users that are in the validation set but not the training set.
validation_edit <- validation_edit %>% mutate(timestamp_t=format(as_datetime(validation_edit$timestamp), "%Y-%m-%d")) # add a new column to the validation set with new date format
sixth_model <- function(p){
  a_r <- train %>%
    group_by(movieId) %>%
    summarise(a_r = sum(rating - mu)/(n()+p))
  b_r <- train %>% 
    left_join(a_r, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_r = sum(rating - a_r - mu)/(n()+p))
  c_r <- train %>% 
    left_join(a_r, by="movieId") %>%
    left_join(b_r, by="userId") %>%
    group_by(genres) %>%
    summarise(c_r = sum(rating - a_r - b_r - mu)/(n()+p))
  d_r <- train %>% 
    left_join(a_r, by="movieId") %>%
    left_join(b_r, by="userId") %>%
    left_join(c_r, by="genres") %>% 
    group_by(timestamp_t) %>%
    summarise(d_r = sum(rating - a_r - b_r - c_r - mu)/(n()+p)) 
  predicted_ratings <- 
    validation_edit %>% 
    left_join(a_r, by = "movieId") %>%
    left_join(b_r, by = "userId") %>%
    left_join(c_r, by = "genres") %>%
    left_join(d_r, by = "timestamp_t") %>%
    mutate(pred = mu + a_r + b_r + c_r + d_r) %>%
    .$pred
  return(rmse(predicted_ratings, validation_edit$rating))
} # A function that returns the lowest error
sixth_model(5.25) # calling the function to calculate the error from the sixth model
