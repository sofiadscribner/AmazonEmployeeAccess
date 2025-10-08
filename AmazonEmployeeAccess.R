# load packages

library(tidyverse)
library(tidymodels)
library(vroom)
library(ggmosaic)
library(embed)

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