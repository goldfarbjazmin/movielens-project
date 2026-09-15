# MovieLens Project - HarvardX Data Science Capstone
# Script: movielens_project.R
# Author: Jazmin Goldfarb

# 1. Enviroment Setup & Package Management
install.packages("tidyverse")
install.packages("caret")
install.packages("lubridate")
library("tidyverse")
library("caret")
library("lubridate")
# Loss function: RMSE
RMSE <- function(true_ratings, predicted_ratings) { sqrt(mean((true_ratings - predicted_ratings)^2, na.rm = TRUE))}
# 2. Dataset Generation
options(timeout = 240)
dl <- "ml-10M100K.zip"
if(!file.exists(dl)) {
  # Using curl with -k to prevent SSL handshake errors on GroupLens server
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", 
                dl, method = "curl", extra = "-k")
}
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl, method = "curl", extra = "-k")
ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file)) unzip(dl, ratings_file)
movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file)) unzip(dl, movies_file)
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE), stingsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>% mutate(userId = as.integer(userId), movieId = as.integer(movieId), rating = as.numeric(rating), timestamp = as.integer(timestamp))
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE), stingsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>% mutate(movieId = as.integer(movieId))
movielens <- left_join(ratings, movies, by = "movieId")
# Final test set will be 10% of MOvieLens data
set.seed(1, sample.kind = "Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index, ]
temp <- movielens[test_index, ]
final_holdout_test <- temp %>% semi_join(edx, by = "movieId") %>% semi_join(edx, by = "userId")
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)
# 3. Feature Engineering
edx <- edx %>% mutate(rating_year = year(as_datetime(timestamp)), release_year = as.numeric(str_extract(title, "(?<=\\()\\d{4}(?=\\))")), movie_age = rating_year - release_year)
final_holdout_test<- final_holdout_test %>% mutate( rating_year = year(as_datetime(timestamp)), release_year = as.numeric(str_extract(title, "(?<=\\()\\d{4}(?=\\))")), movie_age = rating_year - release_year)
# 4.Partitioning edx for Cross-Validation
set.seed(123, sample.kind = "Rounding")
val_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
train_set <- edx[-val_index, ]
val_temp <- edx[val_index, ]
test_set <- val_temp %>% semi_join(train_set, by = "movieId") %>% semi_join(train_set, by = "userId")
train_set <- rbind(train_set, anti_join(val_temp, test_set))
rm(val_index, val_temp)
# 5. Model development?
# Overall mean
mu_hat <- mean(train_set$rating)
baseline_rmse <- RMSE(test_set$rating, mu_hat)
cat(sprintf("Baseline (Mean) RMSE: %.5f\n", baseline_rmse))
# Movie effect (b_i)
movie_avgs <- train_set %>% group_by(movieId) %>% summarize(b_i = mean(rating - mu_hat))
pred_bi <- test_set %>% left_join(movie_avgs, by = "movieId") %>% mutate(pred = mu_hat + b_i) %>% pull(pred)
movie_rmse <- RMSE(test_set$rating, pred_bi)
cat(sprintf("Movie Effect RMSE: %.5f\n", movie_rmse))
# Movie + User Effect (b_i + b_u)
user_avgs <- train_set %>% 
  left_join(movie_avgs, by = "movieId") %>%
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu_hat - b_i))
pred_bi_bu <- test_set %>% 
  left_join(movie_avgs, by = "movieId") %>% 
  left_join(user_avgs, by = "userId") %>% 
  mutate(pred = mu_hat + b_i + b_u) %>% 
  pull(pred)
user_rmse <- RMSE(test_set$rating, pred_bi_bu)
cat(sprintf("Movie + User Effect RMSE: %.5f\n", user_rmse))
# Optimization: Regularization
lambdas <- seq(4.0, 5.5, 0.25)
rmses <- sapply(lambdas, function(l) { 
  mu <- mean (train_set$rating)
  
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu) / (n() + l))
  
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - mu - b_i) / (n() + l))
  
  b_g <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u) / (n() + l))
  
  b_t <- train_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>% 
    left_join(b_g, by = "genres") %>%
    group_by(movie_age) %>%
    summarize(b_t = sum(rating - mu - b_i - b_u - b_g) / (n() + l))
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_t, by = "movie_age") %>%
    mutate(
      b_i = replace_na(b_i, 0),
      b_u = replace_na(b_u, 0),
      b_g = replace_na(b_g, 0),
      b_t = replace_na(b_t, 0),
      pred = mu + b_i + b_u + b_g + b_t
    ) %>%
    pull(pred)
  
  return(RMSE(test_set$rating, predicted_ratings))
  })
optimal_lambda <- lambdas[which.min(rmses)]
cat(sprintf("Optimal Lambda determined by Cross-validation: %.2f\n", optimal_lambda))
# 6. Final Evaluation on final_holdout_test
# Model trained on full dataset using optimal_lambda
mu_full <- mean(edx$rating)
b_i_full <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu_full) / (n() + optimal_lambda))
b_u_full <- edx %>%
  left_join(b_i_full, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - mu_full - b_i) / (n() + optimal_lambda))
b_g_full <- edx %>%
  left_join(b_i_full, by = "movieId") %>%
  left_join(b_u_full, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu_full - b_i - b_u) / (n() + optimal_lambda))
b_t_full <- edx %>%
  left_join(b_i_full, by = "movieId") %>%
  left_join(b_u_full, by = "userId") %>%
  left_join(b_g_full, by = "genres") %>%
  group_by(movie_age) %>%
  summarize(b_t = sum(rating - mu_full - b_i - b_u - b_g) / (n() + optimal_lambda))
final_predictions <- final_holdout_test %>%
  left_join(b_i_full, by = "movieId") %>%
  left_join(b_u_full, by = "userId") %>%
  left_join(b_g_full, by = "genres") %>%
  left_join(b_t_full, by = "movie_age") %>%
  mutate(
    b_i = replace_na(b_i, 0),
    b_u = replace_na(b_u, 0),
    b_g = replace_na(b_g, 0),
    b_t = replace_na(b_t, 0),
    pred = mu_full + b_i + b_u + b_g + b_t
  ) %>%
  pull(pred)
final_rmse <- RMSE(final_holdout_test$rating, final_predictions)
# Console Output
cat("=====================================================\n")
cat(sprintf("FINAL EVALUATION ON HOLDOUT SET:\n"))
cat(sprintf("Achieved Final Holdout RMSE: %.5f\n", final_rmse))
cat("=====================================================\n")