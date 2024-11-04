# Machine Learning Model 3 : BSmodel

# Spacio temporal : Basis functions
# Add bs_model

model_name = "bs_model"

start_time <- Sys.time()

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

# 2. Load data ----

dat_cov <- read.csv("Test_1/data/reef_data_NAs_with_cov_fixed.csv") %>% 
  filter(!is.na(COUNT)) %>% filter(fGROUP == "HCC") 

#add basis functions 

## Picks number of spatial and temporal knots from mgcv models

pick_knots_mgcv <- function(data) {
  
  # Degree of freedom = number of unique location x number of years
  df <- data %>%
    group_by(LONGITUDE, LATITUDE) %>%
    dplyr::slice(1)
  
  max_kspat <- nrow(df)
  max_ktemp <- length(unique(dat$fYEAR))
  
  kspat <- seq(30, max_kspat, by = 20) # minium of 30 knots on the spatial dimension
  ktemp <- seq(8, max_ktemp, by = 2)  # minimum of 10 knots of the temporal dimension
  
  knbre<- expand.grid(kspat,ktemp)
  
  mod_list <- list()
  
  for ( i in 1 : nrow(knbre)){
    
    mod0 <- mgcv::gam(COUNT/TOTAL ~ te(LONGITUDE,LATITUDE, fYEAR, # inputs over which to smooth
                                       bs = c("tp", "cr"), # types of bases
                                       k=c(knbre[i,1],knbre[i,2]), # knot count in each dimension
                                       d=c(2,1)), # (s,t) basis dimension
                      data = dat,
                      control =  gam.control(scalePenalty = FALSE),
                      method = "GCV.Cp", family = binomial("logit"),
                      weights = TOTAL)
    
    mod_list[[i]] <- cbind(as.numeric(summary(mod0)$r.sq), as.numeric(summary(mod0)$s.table[[1]]), as.numeric(summary(mod0)$sp.criterion),
                           as.numeric(AIC(mod0)), knbre[i,1],knbre[i,2])
  }
  
  table_result <- do.call(rbind, mod_list)
  colnames(table_result) <- c("Rsquared", "edf", "GCV", "AIC", "kspat", "ktemp")
  
  ## Criteria
  # edf cannot be greater than degree of freedom
  ## lowest GCV
  ## highest r2
  ## lowest AIC
  
  table_result <- table_result %>%
    data.frame() %>%
    arrange(desc(Rsquared), GCV, desc(AIC))
  
  return(table_result)
}

knots <- pick_knots_mgcv(dat_cov)


## Add basis function

bs_model <- mgcv::gam(COUNT/TOTAL ~ te(LONGITUDE, LATITUDE, fYEAR,
                                       bs = c("tp","cr"),
                                       k= c(knots[1,5], knots[1,6]),
                                       d=c(2,1),
                                       fx = TRUE),
                      data = dat_cov,
                      control = gam.control(scalePenalty = FALSE),
                      method = "REML", family = binomial("logit")
)


# Modif data
dat_cov <- dat_cov %>%
  mutate(REEF_SITE = paste(REEF_NAME, SITE_NO, sep = "_"),
         REEF_SITE_TRANSECT = paste(REEF_SITE, TRANSECT_NO, sep = "_")) %>% 
  dplyr::select(-c(P_CODE, fDEPTH, TRUE_COUNT, TOTAL, SITE_NO, TRANSECT_NO, tier, fGROUP,
                   LONGITUDE, LATITUDE, fYEAR)) %>%
  mutate(across(c(REEF_NAME, REEF_SITE, REEF_SITE_TRANSECT), as_factor)) %>%
  mutate(across(c(CYC, DHW, OT), ~ c(scale(.)))) %>%
  mutate(model = predict(bs_model, newdata = dat_cov))



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

ggplot(result_pred_obs, aes(x = y, y = yhat)) +
  geom_point() + 
  geom_abline(col = "red", linetype = "dashed") + 
  ggtitle("BS_model") +
  theme_bw()

end_time <- Sys.time()
computing_time <- end_time - start_time

compt <- list(name = model_name, time = computing_time)
saveRDS(compt, file = paste(getwd(),model_name,".RDS"))
