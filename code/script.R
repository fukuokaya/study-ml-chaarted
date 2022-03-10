d <- readxl::read_excel("~/OneDrive/research data/project/mlm-chaarted/source/220303_chaarted.xlsx") # import data from excel
df <- d %>% dplyr::filter(arm == "DTX and ADT") # select patients receiving ADT and DTX

# dataset for machine learning
df_ml <- df %>% 
  select(neutropenia, # survival parameter
         cab:local) %>% 
  mutate_if(is.character, factor) 

set.seed(123)
df_split <- initial_split(df_ml, strata = neutropenia, prop = 3/4)
df_train <- training(df_split)
df_test <- testing(df_split)

xgb_spec <- 
  boost_tree(trees = 1000, 
             tree_depth = tune(), 
             min_n = tune(), 
             loss_reduction = tune(),       
             sample_size = tune(), 
             mtry = tune(),  
             learn_rate = tune(),           
  ) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification") 
xgb_spec

xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), df_train),
  learn_rate(),
  size = 30
)
xgb_grid

xgb_wf <- workflow() %>%
  add_formula(neutropenia ~ .) %>%
  add_model(xgb_spec)

xgb_wf

set.seed(123)
df_folds <- vfold_cv(df_train, strata = neutropenia)

df_folds

doParallel::registerDoParallel()

set.seed(234)
xgb_res <- tune_grid(
  xgb_wf,
  resamples = df_folds,
  grid = xgb_grid,
  control = control_grid(save_pred = TRUE)
)

xgb_res

collect_metrics(xgb_res)

xgb_res %>%
  collect_metrics() %>%
  filter(.metric == "roc_auc") %>%
  select(mean, mtry:sample_size) %>%
  pivot_longer(mtry:sample_size,
               values_to = "value",
               names_to = "parameter"
  ) %>%
  ggplot(aes(value, mean, color = parameter)) +
  geom_point(alpha = 0.8, show.legend = FALSE) +
  facet_wrap(~ parameter, scales = "free_x") +
  labs(x = NULL, y = "AUC") 

show_best(xgb_res, "roc_auc")

best_auc <- select_best(xgb_res, "roc_auc")
best_auc

final_xgb <- finalize_workflow(
  xgb_wf,
  best_auc
)

final_xgb

library(vip)

final_xgb %>%
  fit(data = df_train) %>%
  pull_workflow_fit() %>%
  vip(geom = "point")

final_res <- last_fit(final_xgb, df_split)

collect_metrics(final_res)

final_res %>%
  collect_predictions() %>%
  roc_curve(neutropenia, .pred_no) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(size = 1.5, color = "midnightblue") +
  geom_abline(
    lty = 2, alpha = 0.5,
    color = "gray50",
    size = 1.2
  )