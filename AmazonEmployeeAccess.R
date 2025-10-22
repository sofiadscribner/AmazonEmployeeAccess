# load packages

library(tidyverse)
library(tidymodels)
library(vroom)
library(ggmosaic)
library(embed)
library(kknn)

# read in data

train <- vroom('train.csv')
test <- vroom('test.csv')


train <- train %>%
  mutate(across(everything(), as.factor))

test <- test %>%
  mutate(across(everything(), as.factor))


# EXPLORATORY DATA ANALYSIS

# look at first few rows and check data types

head(train)

# bar plot of the two outcomes of the target variable "action"

ggplot(train, aes(x = as.factor(ACTION), fill = as.factor(ACTION))) +
  geom_bar(show.legend = FALSE) +
  labs(title = "Distribution of Action",
       x = "	ACTION (1 if the resource was approved, 0 if the resource was not)",
       y = "Count") +
  scale_fill_manual(values = c("darkgrey", "darkgreen")) +
  theme_minimal()

# it is clear that in our data it is much, much more common that a resources would be approved than not

train %>%
  summarise(across(everything(), n_distinct)) %>%
  pivot_longer(everything(), names_to = "feature", values_to = "n_unique") %>%
  filter(feature != "ACTION") %>%
  ggplot(aes(x = reorder(feature, n_unique), y = n_unique)) +
  geom_col(fill = "darkgreen") +
  coord_flip() +
  labs(title = "Number of Unique Categories per Feature",
       x = "Feature",
       y = "Distinct Values") +
  theme_minimal()

# I can see that there are literally thousands of unique categories in most of these variables. Role Family only has 67

# ENCODING RECIPE PRACTICE

practice_recipe <- recipe(ACTION ~., data = train) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors())

prep <- prep(practice_recipe)
baked <- bake(prep, new_data = train)
  
# LOGISTIC REGRESSION

# recipe

lencode_recipe <- recipe(ACTION ~., data = train) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

# model

logRegModel <- logistic_reg() %>%
  set_engine("glm")

# workflow

logReg_workflow <- workflow() %>%
  add_recipe(lencode_recipe) %>%
  add_model(logRegModel) %>%
  fit(data = train)

# make preds

logReg_preds <- predict(logReg_workflow,
                        new_data = test,
                        type = "prob")

# format for kaggle submission

logRegsub <- bind_cols(test %>% select(id), logReg_preds %>% select(.pred_1)) %>%
  rename(ACTION = .pred_1)

# save predictions locally

vroom_write(x=logRegsub, file="./logRegPreds.csv", delim=",")

# PENALIZED LOGISTIC REGRESSION

# recipe

pen_recipe <- recipe(ACTION ~., data = train) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors())

# model

pen_model <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

# workflow

pen_workflow <- workflow() %>%
  add_recipe(pen_recipe) %>%
  add_model(pen_model)

# Define the parameter space
pen_params <- parameters(penalty(range = c(-4, -1)), mixture())

# Build a regular grid across that space
tune_grid <- grid_regular(pen_params, levels = 5)

# split data for CV

folds <- vfold_cv(train, v = 4, repeats = 1)

# run CV

pen_cv_results <- pen_workflow %>%
  tune_grid(resamples = folds,
            grid = tune_grid,
            metrics = metric_set(roc_auc))

# find best parameters

best_params <- pen_cv_results %>%
  select_best(metric = "roc_auc")

# finalize workflow

final_pen_wf <-
  pen_workflow %>%
  finalize_workflow(best_params) %>%
  fit(data = train)


# make preds

pen_preds <- predict(final_pen_wf,
                        new_data = test,
                        type = "prob")

# format for kaggle submission

pen_log_reg_sub <- bind_cols(test %>% select(id), pen_preds %>% select(.pred_1)) %>%
  rename(ACTION = .pred_1)

# save predictions locally

vroom_write(x=pen_log_reg_sub, file="./PenLogRegPreds.csv", delim=",")


# RANDOM FOREST

# recipe

forest_recipe <- recipe(ACTION ~., data = train) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

# model

forest_model <- rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

# workflow

forest_workflow <- workflow() %>%
  add_recipe(forest_recipe) %>%
  add_model(forest_model)


# Build a regular grid across that space
tune_grid <- grid_regular(
  mtry(range = c(2, sqrt(ncol(train) - 1))),
  min_n(range = c(2, 10)),
  levels = 3
)

# split data for CV

folds <- vfold_cv(train, v = 4)

# run CV

forest_cv_results <- forest_workflow %>%
  tune_grid(resamples = folds,
            grid = tune_grid,
            metrics = metric_set(roc_auc))

# find best parameters

best_params <- forest_cv_results %>%
  select_best(metric = "roc_auc")

# finalize workflow

final_forest_wf <-
  forest_workflow %>%
  finalize_workflow(best_params) %>%
  fit(data = train)


# make preds

forest_preds <- predict(final_forest_wf,
                     new_data = test,
                     type = "prob")

# format for kaggle submission

forest_sub <- bind_cols(test %>% select(id), forest_preds %>% select(.pred_1)) %>%
  rename(ACTION = .pred_1)

# save predictions locally

vroom_write(x=forest_sub, file="./ForestPreds.csv", delim=",")

## K-NEAREST NEIGHBORS

# recipe

knn_recipe <- recipe(ACTION ~., data = train) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors())

# model

knn_model <- nearest_neighbor(neighbors = tune()) %>%
  set_engine("kknn") %>%
  set_mode("classification")

# workflow

knn_workflow <- workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(knn_model)


# Build a regular grid across that space
tune_grid <- grid_regular(
  neighbors(range = c(1, 51)), 
  levels = 6               
)

# split data for CV

folds <- vfold_cv(train, v = 4)

# run CV

knn_cv_results <- knn_workflow %>%
  tune_grid(resamples = folds,
            grid = tune_grid,
            metrics = metric_set(roc_auc))

# find best parameters

best_params <- knn_cv_results %>%
  select_best(metric = "roc_auc")

# finalize workflow

final_knn_wf <-
  knn_workflow %>%
  finalize_workflow(best_params) %>%
  fit(data = train)


# make preds

knn_preds <- predict(final_knn_wf,
                        new_data = test,
                        type = "prob")

# format for kaggle submission

knn_sub <- bind_cols(test %>% select(id), knn_preds %>% select(.pred_1)) %>%
  rename(ACTION = .pred_1)

# save predictions locally

vroom_write(x=knn_sub, file="./KNNPreds.csv", delim=",")