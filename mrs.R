#Show the structure of the dataset
load("edx")
str(edx)

library(tidyverse)
library(corrplot)
library(Hmisc)

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

library(caret)
set.seed(33)
test_ind <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE) # make a data partition for a test set and train set
train <- edx[-test_ind,] # training set
test <- edx[test_ind,] # testing set
test <- test %>% 
  semi_join(train, by = "movieId") %>% semi_join(train, by = "userId") # get rid of movies or users that are in the test set but not the training set.
rmse <- function(ratings, predicted_ratings){
  sqrt(mean((ratings - predicted_ratings)^2))
} # a function that calculates the error that we have to minimize to get better models

mu <- mean(train$rating) # average rating for all movies
first_rmse <- rmse(test$rating,mu) # the first prediction's error
first_rmse # print rmse

avgs <- train %>% group_by(movieId) %>% summarise(a = mean(rating - mu)) # the average rating for each movie
predicted_ratings <- mu + test %>% 
  left_join(avgs, by='movieId') %>%
  .$a # our prediction of the rating of movies in the test set
second_rmse <- rmse(predicted_ratings,test$rating) # the second prediction's error
second_rmse