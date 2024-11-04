# Machine Learning Model 1 : Original model

# Year : factor
# Space : Reef, Site, Transect


model_name = "original_model"

# 1. Load packages ----

library(tidyverse) # Core tidyverse packages
library(tidymodels) # Core tidymodels packages
library(sf)
library(DALEX)
library(DALEXtra)
library(caret)
library(xgboost)
library(vip)
library(future)
library(furrr)
library(mgcv)
library(INLA)

options(future.globals.maxSize = 100000*1024^2) # 100 Gb
plan(strategy = multisession, workers = 4)

start_time <- Sys.time()

# 2. Load data ----

dat_cov <- read.csv("Test_1/data/reef_data_NAs_with_cov_fixed.csv") %>% 
  filter(!is.na(COUNT)) %>% filter(fGROUP == "HCC") 

dat_cov <- dat_cov %>%
  mutate(REEF_SITE = paste(REEF_NAME, SITE_NO, sep = "_"),
         REEF_SITE_TRANSECT = paste(REEF_SITE, TRANSECT_NO, sep = "_")) %>% 
  dplyr::select(-c(P_CODE, fDEPTH, TRUE_COUNT, TOTAL, SITE_NO, TRANSECT_NO, tier, fGROUP
                   )) %>%
  mutate(across(c(REEF_NAME, REEF_SITE, REEF_SITE_TRANSECT, LONGITUDE, LATITUDE, fYEAR), as_factor)) %>%
  mutate(across(c(CYC, DHW, OT), ~ c(scale(.))))

# 1. Data preparation

## 1.1 Filter the category

data_split <- dat_cov 

## 1.2 Split into training and testing data

data_split <- initial_split(data_split, prop = 3/4)
data_train <- training(data_split)
data_test <- testing(data_split)


# 2. Hyperparameters tuning

## 2.1 Define the recipe

boosted_recipe <- recipe(COUNT ~ ., data = data_train) %>% 
  step_dummy(all_nominal_predictors())

## 2.2 Define the model

boosted_model <- boost_tree(learn_rate = tune(),
                            trees = tune(), 
                            min_n = tune(), 
                            tree_depth = tune()) %>% # Model type
  set_engine("xgboost") %>% # Model engine
  set_mode("regression") # Model mode

## 2.3 Define the workflow

boosted_workflow <- workflow() %>%
  add_recipe(boosted_recipe) %>% 
  add_model(boosted_model)

## 2.4 Create the grid - plus long quand size parametre est grand

tune_grid <- grid_space_filling(learn_rate(),
                                trees(),
                                tree_depth(),
                                min_n(),
                                size = 5) # to change original size = 30

## 2.5 Run the hyperparameters tuning - plus long en presence des covariates

tuned_results <- tune_grid(boosted_workflow,
                           resamples = vfold_cv(data_train, v = 2), # to change originql v = 5
                           grid = tune_grid)

## 2.6 Get best set of parameters

model_hyperparams <- select_best(tuned_results, metric = "rmse") %>% 
  select(-".config") %>% 
  as_tibble(.) %>%
  mutate(nb_training = nrow(data_train),
         nb_testing = nrow(data_test)) #%>% 
#  mutate(category = category_i)

# 3. Predicted vs observed

## 3.1 Redefine the model (with hyperparameters)

boosted_model <- boost_tree(learn_rate = model_hyperparams$learn_rate,
                            trees = model_hyperparams$trees, 
                            min_n = model_hyperparams$min_n, 
                            tree_depth = model_hyperparams$tree_depth) %>% # Model type
  set_engine("xgboost") %>% # Model engine
  set_mode("regression") # Model mode

## 3.2 Redefine the workflow

boosted_workflow <- workflow() %>%
  add_recipe(boosted_recipe) %>% 
  add_model(boosted_model)

## 3.3 Fit the final model

final_model <- boosted_workflow %>%
  last_fit(data_split)

final_fitted <- final_model$.workflow[[1]]

## 3.4 Model performance

model_performance <- collect_metrics(final_model) %>% 
  select(-".estimator", -".config") %>% 
  pivot_wider(names_from = ".metric", values_from = ".estimate") #%>% 
#  mutate(fGROUP = fGROUP_i)

## 3.4 Predicted vs Observed

result_pred_obs <- data_test %>% 
  mutate(yhat = predict(final_fitted, data_test)$.pred) %>% 
  rename(y = COUNT) %>% 
  select( y, yhat) #%>% 
#  mutate(fGROUP = fGROUP_i)

ggplot(result_pred_obs, aes(x = y, y = yhat)) + geom_point() + geom_abline(col = "red", linetype = "dashed") + theme_bw() + ggtitle("original model")   
# 4. Return the results

#  return(lst(model_hyperparams,
#             model_performance,
#             result_pred_obs))

end_time <- Sys.time()
computing_time <- end_time - start_time