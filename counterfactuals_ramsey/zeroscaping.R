setwd("~/Austin Water/ramsey_welfare_result")
library(tidyverse)
library(readxl)
library(ggplot2)
library(ggpubr)
library(dplyr)
library(readr)
library(lubridate)
library(zoo)
library(RColorBrewer)
library("ggsci")
library(AER)
library(foreign)
library(mvtnorm)
library(reshape2)
library(data.table)
#library(lme4)
library(stargazer)
library(gganimate)
#library(gifski)
library(magick)
library(grid)
library(cowplot) # For get_legend
library(patchwork) # For plot_annotation

###### Prepare ##########
my_theme <- theme_minimal()+
  theme(title = element_text(hjust=0.5),legend.position='bottom')
theme_set(my_theme)
big_text <- theme(text = element_text(size=24))

small_text <- theme(text = element_text(size=17))

detail_0 = read_csv("cs_detail_results/detail_0.csv")
cs_0 = detail_0$cs_0
q_0 = detail_0$q_sum_hh0
r_0 = detail_0$r0
rm(detail_0)

#new_income = read_csv("../prem_key/prem_key_income.csv")

#new_income = new_income %>%
#  select(prem_id, e_income_descrete)

demand_2018_using_new_small = read_csv("../demand_2018_using_new_small.csv")
demand_2018_using_new_small = demand_2018_using_new_small[which(demand_2018_using_new_small$charge<=demand_2018_using_new_small$income),]
demand_key <- demand_2018_using_new_small %>% select(bill_ym, prem_id, income, CAP_HH)
#rm(demand_2018_using_new_small)

income = demand_key$income

#demand_key = merge(demand_key, new_income, by.x = c("prem_id"), by.y = c("prem_id"), all = T)

demand_key <- demand_key %>%
  mutate(income_strata = case_when(
    income < 6000 ~ "0~6k",
    income >= 6000 & income < 20000 ~ "6k~20k",
    income >= 20000 & income < 45000 ~ "20k~45k",
    income >= 45000 & income < 100000 ~ "45k~100k",
    income >= 100000 ~ ">100k"
  )) %>%
  mutate(income_strata = factor(income_strata, levels = c("0~6k", "6k~20k", "20k~45k", "45k~100k", ">100k")))

#demand_key <- demand_key %>%
#  mutate(income_strata = case_when(
#    e_income_descrete < 6000 ~ "0~6k",
#    e_income_descrete >= 6000 & e_income_descrete < 20000 ~ "6k~20k",
#    e_income_descrete >= 20000 & e_income_descrete < 45000 ~ "20k~45k",
#    e_income_descrete >= 45000 & e_income_descrete < 100000 ~ "45k~100k",
#    e_income_descrete >= 100000 ~ ">100k"
#  )) %>%
#  mutate(income_strata = factor(income_strata, levels = c("0~6k", "6k~20k", "20k~45k", "45k~100k", ">100k")))

demand_key_tally <- demand_key %>%
  count(income_strata) %>%
  mutate(proportion = n / sum(n))

demand_key$r_0 = r_0
demand_key$cs_0 = cs_0
demand_key$q_0 = q_0

demand_key = demand_key %>%
  group_by(bill_ym) %>%
  mutate(r_0_bill_ym = mean(r_0),
         cs_0_bill_ym = mean(cs_0),
         q_0_bill_ym = mean(q_0))

demand_key = demand_key %>%
  group_by(income_strata) %>%
  mutate(r_0_income_strata = mean(r_0),
         cs_0_income_strata = mean(cs_0),
         q_0_income_strata = mean(q_0))

demand_key = demand_key %>%
  group_by(income_strata, bill_ym) %>%
  mutate(r_0_income_strata_bill_ym = mean(r_0),
         cs_0_income_strata_bill_ym = mean(cs_0),
         q_0_income_strata_bill_ym = mean(q_0))

# A tibble: 5 × 3
#income_strata      n proportion
#<chr>          <int>      <dbl>
#1 1             217337     0.146 
#2 2             725664     0.489 
#3 3             320784     0.216 
#4 4             154788     0.104 
#5 5              65388     0.0441

demand_key$extra_consumption = case_when(
  demand_key$r_0>demand_key$income ~ 1,
  demand_key$r_0<=demand_key$income ~ 0
)

demand_key$quantity_strata = case_when(
  demand_key$q_0<=2 ~ "1",
  demand_key$q_0>2 & demand_key$q_0<=6  ~ "2",
  demand_key$q_0>6 &demand_key$q_0<=11  ~ "3",
  demand_key$q_0>11 & demand_key$q_0<=20  ~ "4",
  demand_key$q_0>20 ~ "5"
)

demand_key = demand_key %>%
  group_by(quantity_strata) %>%
  mutate(r_0_quantity_strata = mean(r_0),
         cs_0_quantity_strata = mean(cs_0),
         q_0_quantity_strata = mean(q_0))

ggplot(demand_key[which(demand_key$income < 250000 & demand_key$income > 1200), ], aes(x = income/1000, fill = as.factor(income_strata))) +
  geom_histogram(aes(y = ..count..), bins = 1000, alpha = 1) +
  scale_fill_manual(
    values = c("lightgreen", "mediumseagreen", "seagreen", "forestgreen", "darkgreen"),  # Distinct colors for each tier
    #values = c("lightblue", "lightgreen", "orange", "lightcoral", "darkviolet"),  # Distinct colors for each tier
    name = NULL
  ) +
  labs(title = "HH Monthly Income Histogram by Strata", x = "Monthly HH Income (k$)", y = "Count") +
  theme_minimal() + big_text + 
  theme(legend.position = "bottom")  # Move the legend to the bottom

income_id = demand_key %>% select(prem_id, income, CAP_HH, income_strata) %>% distinct()

income_id$quantity_strata = NULL

income_id = income_id %>% distinct()
edit_strata_df = function(x, str){
  x = data.frame(t(x))
  x<- x %>% slice(-1)
  
  x <- x %>%
    rownames_to_column(var = "step") %>% 
    rename_with(~ gsub("rate_", "", .x))  # Remove "rate_" from column names
  
  x$step = as.numeric(gsub("rate_", "",x$step))
  x <- x %>%
    arrange(step)
  
  x <- x %>%
    pivot_longer(cols = X1:X5, names_to = "income_strata", values_to = paste0(str,"_change_rate"))
  
  x <- x %>%
    mutate(income_strata = case_when(
      income_strata == "X1" ~ "0~6k",
      income_strata == "X2" ~ "6k~20k",
      income_strata == "X3" ~ "20k~45k",
      income_strata == "X4" ~ "45k~100k",
      income_strata == "X5" ~ ">100k"
    )) %>%
    mutate(income_strata = factor(income_strata, levels = c("0~6k", "6k~20k", "20k~45k", "45k~100k", ">100k")))
  return(x)
}

edit_strata_df_quantity = function(x, str){
  x = data.frame(t(x))
  x<- x %>% slice(-1)
  
  x <- x %>%
    rownames_to_column(var = "step") %>% 
    rename_with(~ gsub("rate_", "", .x))  # Remove "rate_" from column names
  
  x$step = as.numeric(gsub("rate_", "",x$step))
  x <- x %>%
    arrange(step)
  
  x <- x %>%
    pivot_longer(cols = X1:X5, names_to = "quantity_strata", values_to = paste0(str,"_change_rate"))
  
  x <- x %>%
    mutate(quantity_strata = case_when(
      quantity_strata == "X1" ~ "1",
      quantity_strata == "X2" ~ "2",
      quantity_strata == "X3" ~ "3",
      quantity_strata == "X4" ~ "4",
      quantity_strata == "X5" ~ "5"
    )) %>%
    mutate(quantity_strata = factor(quantity_strata, levels = c("1", "2", "3", "4", "5")))
  return(x)
}

custom_colors <- c("0~6k" = "#e22959", 
                   "6k~20k" = "#9f5553", 
                   "20k~45k" = "#4f8c9d", 
                   "45k~100k" = "#738c4e",
                   ">100k" = "#234043") 

custom_shapes <- c("0~6k" = 21, 
                   "6k~20k" = 22, 
                   "20k~45k" = 8, 
                   "45k~100k" = 18,
                   ">100k" = 3) 

custom_colors_q <- c("1" = "#e22959", 
                     "2" = "#9f5553", 
                     "3" = "#4f8c9d", 
                     "4" = "#738c4e",
                     "5" = "#234043") 

custom_shapes_q <- c("1" = 21, 
                     "2" = 22, 
                     "3" = 8, 
                     "4" = 18,
                     "5" = 3) 


academic_theme <- 
  theme_bw(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16),
    axis.title.x = element_text(face = "bold"),
    axis.title.y = element_text(face = "bold"),
    axis.text.x = element_text(angle = 0), # No rotation needed for simple gamma values
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 12),
    panel.grid.major = element_line(linewidth = 0.5, color = "grey90"),
    panel.grid.minor = element_line(linewidth = 0.25, color = "grey95")
  )



# --- Plot Definitions ---
my_plot_theme <- theme_minimal(base_size = 24) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    axis.title = element_text(face = "bold", size = 14),
    axis.text = element_text(size = 14),
    legend.position = "none", # IMPORTANT: Hide legend for individual plots
    panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5)
  )

sq_ndvi_avg_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/sq_ndvi_avg_bound_loss05_mean_ev_steps.csv")

colnames(sq_ndvi_avg_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(sq_ndvi_avg_bound_loss05_mean_ev_steps))*0.05-0.25)

sq_ndvi_avg_bound_loss05_mean_ev_steps_rate <-sq_ndvi_avg_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(sq_ndvi_avg_bound_loss05_mean_ev_steps)

sq_ndvi_avg_bound_loss05_mean_ev_steps_rate = cbind(sq_ndvi_avg_bound_loss05_mean_ev_steps_rate, demand_key)

mean_ev_rate_by_bill_ym <- sq_ndvi_avg_bound_loss05_mean_ev_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(income_strata,prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_by_bill_ym <- sq_ndvi_avg_bound_loss05_mean_ev_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(1:11, mean, na.rm = TRUE))

mean_ev_rate_by_strata_sq <- mean_ev_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_by_strata_sq <- mean_ev_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(2:12, mean, na.rm = TRUE))

mean_ev_rate_by_strata_sq = edit_strata_df(mean_ev_rate_by_strata_sq, "ev")
mean_ev_rate_by_strata_sq$ev_perct = as.numeric(mean_ev_rate_by_strata_sq$ev_change_rate)
mean_ev_rate_by_strata_sq$ev_change_rate = NULL

mean_ev_by_strata_sq= edit_strata_df(mean_ev_by_strata_sq, "ev")
mean_ev_by_strata_sq$ev = as.numeric(mean_ev_by_strata_sq$ev_change_rate)
mean_ev_by_strata_sq$ev_change_rate = NULL

sq_ndvi_avg_bound_loss05_var_ev_steps = read_csv("cs_detail_results/sq_ndvi_avg_bound_loss05_var_ev_steps.csv")

colnames(sq_ndvi_avg_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(sq_ndvi_avg_bound_loss05_var_ev_steps))*0.05-0.25)

sq_ndvi_avg_bound_loss05_var_ev_steps_rate <-sq_ndvi_avg_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(sq_ndvi_avg_bound_loss05_var_ev_steps)

sq_ndvi_avg_bound_loss05_var_ev_steps_rate = cbind(sq_ndvi_avg_bound_loss05_var_ev_steps_rate, demand_key)

var_ev_rate_by_bill_ym <- sq_ndvi_avg_bound_loss05_var_ev_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(income_strata,prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_by_bill_ym <- sq_ndvi_avg_bound_loss05_var_ev_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(1:11, mean, na.rm = TRUE))

var_ev_rate_by_strata_sq <- var_ev_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_by_strata_sq <- var_ev_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(2:12, mean, na.rm = TRUE))

var_ev_rate_by_strata_sq = edit_strata_df(var_ev_rate_by_strata_sq, "ev")
var_ev_rate_by_strata_sq$ev_perct = as.numeric(var_ev_rate_by_strata_sq$ev_change_rate)
var_ev_rate_by_strata_sq$ev_change_rate = NULL

var_ev_by_strata_sq= edit_strata_df(var_ev_by_strata_sq, "ev")
var_ev_by_strata_sq$ev = as.numeric(var_ev_by_strata_sq$ev_change_rate)
var_ev_by_strata_sq$ev_change_rate = NULL

process_ev_steps_data <- function(reduction_value, type, file_prefix = "low_avg_bound_loss05", path = "cs_detail_results/") {
  
  # 1. Construct the file path
  file_name <- sprintf("reduction_%.2f_%s_%s_ev_steps.csv", reduction_value, file_prefix, type)
  full_path <- file.path(path, file_name)
  
  if (!file.exists(full_path)) {
    warning(paste("File not found, skipping:", full_path))
    return(NULL)
  }
  
  # 2. Read and process the data
  message(paste("Processing:", full_path))
  ev_steps_data <- readr::read_csv(full_path, show_col_types = FALSE)
  
  # --- CORRECTED LOGIC: Use different formula based on type ---
  if (type == "mean") {
    # Formula for 'mean' files
    colnames(ev_steps_data) <- as.character(as.numeric(colnames(ev_steps_data)) * 0.05 - 0.25)
  } else {
    # Formula for 'var' files, as you specified
    colnames(ev_steps_data) <- as.character(as.numeric(colnames(ev_steps_data)) / 20 + 0.75)
  }
  
  # Store the new (correct) column names for summarization
  data_colnames <- colnames(ev_steps_data)
  
  # Calculate rate
  ev_steps_rate <- ev_steps_data %>%
    mutate(across(everything(), ~ (. / income) * 100, .names = "rate_{.col}"))
  
  # Bind key columns
  ev_steps_rate <- cbind(ev_steps_rate, demand_key)
  
  # 3. Perform summarizations using robust column selection
  mean_ev_rate_by_bill_ym <- ev_steps_rate %>%
    group_by(income_strata, bill_ym) %>%
    summarise(across(starts_with("rate_"), mean, na.rm = TRUE), .groups = 'drop')
  
  mean_ev_by_bill_ym <- ev_steps_rate %>%
    group_by(income_strata, bill_ym) %>%
    # Summarize only the original data columns by their correct names
    summarise(across(1:11, mean, na.rm = TRUE), .groups = 'drop')
  
  mean_ev_rate_by_strata <- mean_ev_rate_by_bill_ym %>%
    group_by(income_strata) %>%
    summarise(across(starts_with("rate_"), mean, na.rm = TRUE), .groups = 'drop')
  
  mean_ev_by_strata <- mean_ev_by_bill_ym %>%
    group_by(income_strata) %>%
    # Summarize the numeric data columns, excluding grouping variables
    summarise(across(2:12, mean, na.rm = TRUE), .groups = 'drop')
  
  # 4. Apply final transformations
  mean_ev_rate_by_strata <- edit_strata_df(mean_ev_rate_by_strata, "ev")
  mean_ev_rate_by_strata$ev_perct <- as.numeric(mean_ev_rate_by_strata$ev_change_rate)
  mean_ev_rate_by_strata$ev_change_rate <- NULL
  
  mean_ev_by_strata <- edit_strata_df(mean_ev_by_strata, "ev")
  mean_ev_by_strata$ev <- as.numeric(mean_ev_by_strata$ev_change_rate)
  mean_ev_by_strata$ev_change_rate <- NULL
  
  # 5. Return the final data frames
  return(list(rate = mean_ev_rate_by_strata, value = mean_ev_by_strata))
}
#processed_data <- process_ev_steps_data(reduction_value = 0.5, type = "var")
#rate_df <- processed_data$rate
#value_df <- processed_data$value

# Initialize empty dataframes to store the aggregated results
all_rates <- tibble()
all_values <- tibble()

# --- Step 1a: Add Baseline Data (Reduction = 0) ---
# Add the pre-calculated baseline data for reduction=0 for both 'mean' and 'var' types.
message("Adding baseline data for reduction = 0...")
if (exists("mean_ev_rate_by_strata_sq") && exists("mean_ev_by_strata_sq")) {
  # Filter for the specific 'mean' case at step -0.25
  rate_zero_reduction_mean <- mean_ev_rate_by_strata_sq %>%
    filter(income_strata == "0~6k", step == -0.25) %>%
    mutate(reduction = 1, type = "mean")
  
  value_zero_reduction_mean <- mean_ev_by_strata_sq %>%
    filter(income_strata == "0~6k", step == -0.25) %>%
    mutate(reduction = 1, type = "mean")
  
  # Append the baseline data for the 'mean' type
  all_rates <- bind_rows(all_rates, rate_zero_reduction_mean)
  all_values <- bind_rows(all_values, value_zero_reduction_mean)
  message("Successfully added baseline data for 'mean' type.")
} else {
  warning("Baseline dataframes for 'mean' type ('mean_ev_rate_by_strata_sq', 'mean_ev_by_strata_sq') not found. Skipping reduction = 0 for 'mean'.")
}

if (exists("var_ev_rate_by_strata_sq") && exists("var_ev_by_strata_sq")) {
  # Filter for the specific 'var' case at step 1.25 for the baseline data
  rate_zero_reduction_var <- var_ev_rate_by_strata_sq %>%
    filter(income_strata == "0~6k", step == 1.25) %>%
    mutate(reduction = 1, type = "var")
  
  value_zero_reduction_var <- var_ev_by_strata_sq %>%
    filter(income_strata == "0~6k", step == 1.25) %>%
    mutate(reduction = 1, type = "var")
  
  # Append the baseline data for the 'var' type
  all_rates <- bind_rows(all_rates, rate_zero_reduction_var)
  all_values <- bind_rows(all_values, value_zero_reduction_var)
  message("Successfully added baseline data for 'var' type.")
} else {
  warning("Baseline dataframes for 'var' type ('var_ev_rate_by_strata_sq', 'var_ev_by_strata_sq') not found. Skipping reduction = 0 for 'var'.")
}


# --- Step 1b: Loop through other reduction values ---
# Define the reduction values and types to loop through
reduction_values <- seq(0.1, 0.9, by = 0.1)
types <- c("mean", "var")

message("Starting data processing and filtering for reductions 0.1 to 0.9...")

# Loop through each reduction value and type
for (r_val in reduction_values) {
  for (type_val in types) {
    
    # --- I. PROCESS THE DATA ---
    # Call the main function to process the corresponding file.
    processed_data <- process_ev_steps_data(reduction_value = r_val, type = type_val)
    
    if (!is.null(processed_data)) {
      rate_df <- processed_data$rate
      value_df <- processed_data$value
      
      # --- II. FILTER THE RESULTS ---
      # NOTE: There is an inconsistency in the data. The baseline 'var' data uses
      # step = 1.25, but the looped CSV files do not contain that value.
      # We will use step = 0 for the looped 'var' files to get a result.
      step_to_filter <- if (type_val == "mean") -0.25 else 1.25
      
      filtered_rate <- rate_df %>%
        filter(income_strata == "0~6k", step == step_to_filter) %>%
        mutate(reduction = r_val, type = type_val)
      
      filtered_value <- value_df %>%
        filter(income_strata == "0~6k", step == step_to_filter) %>%
        mutate(reduction = r_val, type = type_val)
      
      # --- III. APPEND FOR PLOTTING ---
      all_rates <- bind_rows(all_rates, filtered_rate)
      all_values <- bind_rows(all_values, filtered_value)
      
      cat(sprintf("Successfully processed and filtered data for reduction %.1f (%s)\n", r_val, type_val))
    }
  }
}

message("Finished collecting data. Now generating plots...")

# --- Step 2: Plot the Results ---

all_rates <- all_rates %>% mutate(reduction_level = 1 - reduction)
all_values <- all_values %>% mutate(reduction_level = 1 - reduction)

if (nrow(all_rates) > 0 && nrow(all_values) > 0) {
  
  # Define shared legend labels
  legend_labels <- c(
    "mean" = expression(paste("mean, ", zeta[1], " = -0.25")),
    "var" = expression(paste("var, ", zeta[2], " = 1.25"))
  )
  
  # Plot 1: EV Rate Percentage vs. Reduction Value
  plot_rate <- ggplot(all_rates, aes(x = reduction_level, y = ev_perct, color = type, linetype = type)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") + # Add horizontal line at y=0
    geom_line(linewidth = 1) +
    geom_point(size = 3) +
    labs(
      title = NULL,
      subtitle = NULL,
      x = NULL,
      y = "Welfare Change (EV Rate %)"
    ) +
    scale_x_continuous(breaks = seq(0, 1, by = 0.1)) + # Set axis breaks from 0 to 1
    academic_theme + # Assuming you might have a custom theme called academic_theme
    theme(plot.title = element_text(face = "bold"), legend.position = "bottom") +
    scale_color_manual(
      name = "Specifications",
      values = c("mean" = "#0072B2", "var" = "#D55E00"),
      labels = legend_labels
    ) +
    scale_linetype_manual(
      name = "Specifications",
      values = c("mean" = "solid", "var" = "dashed"),
      labels = legend_labels
    )
  
  print(plot_rate)
  
  # Plot 2: Absolute EV Consumption vs. Reduction Value
  plot_value <- ggplot(all_values, aes(x = reduction_level, y = ev, color = type, linetype = type)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") + # Add horizontal line at y=0
    geom_line(linewidth = 1) +
    geom_point(size = 3) +
    labs(
      title = NULL,
      subtitle = NULL,
      x = NULL,
      y = "Welfare Change (Absolute EV $)"
    ) +
    scale_x_continuous(breaks = seq(0, 1, by = 0.1)) + # Set axis breaks from 0 to 1
    academic_theme + # Assuming you might have a custom theme called academic_theme
    theme(plot.title = element_text(face = "bold"), legend.position = "bottom") +
    scale_color_manual(
      name = "Specifications",
      values = c("mean" = "#0072B2", "var" = "#D55E00"),
      labels = legend_labels
    ) +
    scale_linetype_manual(
      name = "Specifications",
      values = c("mean" = "solid", "var" = "dashed"),
      labels = legend_labels
    )
  
  print(plot_value)
  
  # Calculate the difference from the baseline (reduction = 1)
  baseline_evs <- all_values %>%
    filter(reduction == 1) %>%
    select(type, baseline_ev = ev)
  
  all_values_diff <- all_values %>%
    left_join(baseline_evs, by = "type") %>%
    mutate(ev_diff = ev - baseline_ev)
  
  # Plot 3: Difference in EV vs. Reduction Value
  plot_diff <- ggplot(all_values_diff, aes(x = reduction_level, y = ev_diff, color = type, linetype = type)) +
    #geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_line(linewidth = 1) +
    geom_point(size = 3) +
    labs(
      title = NULL,
      subtitle = NULL,
      x =NULL,
      y = "Welfare Difference from Status Quo (Absolute EV $)"
    ) +
    scale_x_continuous(breaks = seq(0, 1, by = 0.1)) + # Set axis breaks from 0 to 1
    academic_theme +
    theme(plot.title = element_text(face = "bold"), legend.position = "bottom") +
    scale_color_manual(
      name = "Specifications",
      values = c("mean" = "#0072B2", "var" = "#D55E00"),
      labels = legend_labels
    ) +
    scale_linetype_manual(
      name = "Specifications",
      values = c("mean" = "solid", "var" = "dashed"),
      labels = legend_labels
    )
  
  print(plot_diff)
  
} else {
  message("No data found after filtering. Please check your data and file paths.")
}

# Arrange the EV Rate plot and the EV Difference plot
final_arranged_plot <- (plot_rate + plot_diff) +
  plot_layout(guides = "collect") + # Collect legends into one
  plot_annotation(
    subtitle = "For 0~6k Income Stratus",
    caption = "NDVI Reduction Level (0 = Status Quo)", # Use caption for a shared x-axis title
    theme = theme(plot.subtitle = element_text(hjust = 0.5, face = "bold",size = 16)) # Adjust subtitle size and center it
  ) &
  theme(
    legend.position = "right", # Move the legend to the right
    plot.caption = element_text(hjust = 0.5, size = 14, face = "bold") # Center the caption to act as the x-axis title
  )&
  theme(legend.position = "right") # Move the legend to the bottom

print(final_arranged_plot)

message("\n--- Starting Comparative Analysis ---")

  comparison_df <- all_values_diff %>%
    filter(reduction_level > 0) %>% # Exclude the baseline
    select(reduction_level, type, ev_diff) %>%
    pivot_wider(names_from = type, values_from = ev_diff)
  
  # Check if both 'mean' and 'var' columns exist after pivoting
    # Calculate the difference and the average difference
    comparison_df <- comparison_df %>%
      mutate(mean_minus_var = mean - var) # Calculate how much larger 'mean' is than 'var'
    
    average_difference <- mean(comparison_df$mean_minus_var, na.rm = TRUE)
    
    # Print the result in a clear message
    cat("\n--- Comparison of Welfare Difference (ev_diff) ---\n")
    cat("Excluding the status quo (reduction level = 0):\n")
    cat(sprintf(
      "On average, the welfare gain ('ev_diff') for the 'var' specification is $%.2f less than for the 'mean' specification.\n",
      average_difference
    ))
    cat("---------------------------------------------------\n")
