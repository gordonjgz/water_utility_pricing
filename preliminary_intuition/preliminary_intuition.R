setwd("~/Austin Water/preliminary_intuition")
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

detail_0 = read_csv("../ramsey_welfare_result/cs_detail_results/detail_0.csv")
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


##### status_quo ########

status_quo = read_csv("status_quo.csv")

#status_quo = cbind(status_quo, demand_key)

status_quo <- status_quo %>%
  mutate(across(starts_with("cs_"), 
                ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata) * 100, 
                .names = "rate_{.col}"),
         across(starts_with("cs_"), 
                ~ (. - demand_key$cs_0) / abs(demand_key$cs_0) * 100, 
                .names = "rate_ind_{.col}")
  )

status_quo <- status_quo %>%
  mutate(across(starts_with("q_"), 
                ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"),
         across(starts_with("q_"), 
                ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_ind_{.col}"),
         across(starts_with("q_"), 
                ~ (. - demand_key$q_0), .names = "diff_{.col}")
  )

status_quo <- status_quo %>%
  mutate(across(starts_with("r_"), 
                ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"),
         across(starts_with("r_"), 
                ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_ind_{.col}"),
         across(starts_with("r_"), 
                ~ (. - demand_key$r_0), .names = "diff_{.col}"))

status_quo <-status_quo %>%
  mutate(across(starts_with("ev_"), ~ (. ) / demand_key$income * 100, .names = "rate_{.col}"))

status_quo = status_quo %>%
  mutate(
    income = demand_key$income,
    income_strata = demand_key$income_strata,
    bill_ym= demand_key$bill_ym,
    prem_id= demand_key$prem_id,
    q_0  = demand_key$q_0
  )

status_quo_by_prem_id <- status_quo %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_1 = mean(rate_q_1),
            rate_r_1 = mean(rate_r_1),
            rate_ind_q_1 = mean(rate_ind_q_1),
            rate_ind_r_1 = mean(rate_ind_r_1),
            diff_q_1 = mean(diff_q_1),
            diff_r_1 = mean(diff_r_1),
            ev_1 = mean(ev_1),
            rate_ev_1 = mean(rate_ev_1),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()

# Get the names of your income strata levels in their factor order
all_strata <- levels(status_quo_by_prem_id$income_strata)

# Programmatically select the very first and very last level
strata_to_plot <- c(first(all_strata), last(all_strata))

plot_data_filtered_1 <- status_quo_by_prem_id %>%
  filter(income_strata %in% strata_to_plot)

status_quo_by_prem_id_dry <- status_quo %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_dry = mean(rate_q_dry),
            rate_r_dry = mean(rate_r_dry),
            rate_ind_q_dry = mean(rate_ind_q_dry),
            rate_ind_r_dry = mean(rate_ind_r_dry),
            diff_q_dry = mean(diff_q_dry),
            diff_r_dry = mean(diff_r_dry),
            ev_dry = mean(ev_dry),
            rate_ev_dry = mean(rate_ev_dry),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()

plot_data_filtered_dry <- status_quo_by_prem_id_dry %>%
  filter(income_strata %in% strata_to_plot)


status_quo_by_prem_id_wet <- status_quo %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_wet = mean(rate_q_wet),
            rate_r_wet = mean(rate_r_wet),
            rate_ind_q_wet = mean(rate_ind_q_wet),
            rate_ind_r_wet = mean(rate_ind_r_wet),
            diff_q_wet = mean(diff_q_wet),
            diff_r_wet = mean(diff_r_wet),
            ev_wet = mean(ev_wet),
            rate_ev_wet = mean(rate_ev_wet),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()

plot_data_filtered_wet <- status_quo_by_prem_id_wet %>%
  filter(income_strata %in% strata_to_plot)

status_quo_by_prem_id_low_var <- status_quo %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_low_var = mean(rate_q_low_var),
            rate_r_low_var = mean(rate_r_low_var),
            rate_ind_q_low_var = mean(rate_ind_q_low_var),
            rate_ind_r_low_var = mean(rate_ind_r_low_var),
            diff_q_low_var = mean(diff_q_low_var),
            diff_r_low_var = mean(diff_r_low_var),
            ev_low_var = mean(ev_low_var),
            rate_ev_low_var = mean(rate_ev_low_var),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()

plot_data_filtered_low_var <- status_quo_by_prem_id_low_var %>%
  filter(income_strata %in% strata_to_plot)

status_quo_by_prem_id_high_var <- status_quo %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_high_var = mean(rate_q_high_var),
            rate_r_high_var = mean(rate_r_high_var),
            rate_ind_q_high_var = mean(rate_ind_q_high_var),
            rate_ind_r_high_var = mean(rate_ind_r_high_var),
            diff_q_high_var = mean(diff_q_high_var),
            diff_r_high_var = mean(diff_r_high_var),
            ev_high_var = mean(ev_high_var),
            rate_ev_high_var = mean(rate_ev_high_var),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()

plot_data_filtered_high_var <- status_quo_by_prem_id_high_var %>%
  filter(income_strata %in% strata_to_plot)

# Prepare each dataframe individually by selecting and renaming columns
df_baseline <- plot_data_filtered_1 %>%
  mutate(condition = "Baseline") %>%
  select(
    income_strata,
    condition,
    q_change = rate_ind_q_1, # Rename to generic name
    r_change = rate_ind_r_1  # Rename to generic name
  )

df_dry <- plot_data_filtered_dry %>%
  mutate(condition = "Dry") %>%
  select(
    income_strata,
    condition,
    q_change = rate_ind_q_dry, # Rename to the SAME generic name
    r_change = rate_ind_r_dry  # Rename to the SAME generic name
  )

df_wet <- plot_data_filtered_wet %>%
  mutate(condition = "Rain") %>%
  select(
    income_strata,
    condition,
    q_change = rate_ind_q_wet, # Rename to the SAME generic name
    r_change = rate_ind_r_wet  # Rename to the SAME generic name
  )

df_low_var <- plot_data_filtered_low_var %>%
  mutate(condition = "Low Var") %>%
  select(
    income_strata,
    condition,
    q_change = rate_ind_q_low_var, # Rename to the SAME generic name
    r_change = rate_ind_r_low_var  # Rename to the SAME generic name
  )

df_high_var <- plot_data_filtered_high_var %>%
  mutate(condition = "High Var") %>%
  select(
    income_strata,
    condition,
    q_change = rate_ind_q_high_var, # Rename to the SAME generic name
    r_change = rate_ind_r_high_var  # Rename to the SAME generic name
  )

# Now, bind the clean dataframes. This will work perfectly without creating NAs.
all_data_clean <- bind_rows(df_dry, df_wet,df_low_var,df_high_var)

# Finally, pivot the clean, combined data into the long format for plotting
all_data_long <- all_data_clean %>%
  rename(
    `Quantity (Q %)` = q_change,
    `Payments (R %)` = r_change
  ) %>%
  pivot_longer(
    cols = c(`Quantity (Q %)`, `Payments (R %)`),
    names_to = "metric",
    values_to = "value"
  )

# Set the desired order for the conditions on the plot's x-axis
all_data_long$condition <- factor(all_data_long$condition, 
                                  levels = c("Dry", "Rain", "Low Var","High Var"))

all_data_long$metric <- factor(all_data_long$metric, 
                               levels = c("Quantity (Q %)", "Payments (R %)"))

custom_border_colors <- darken(custom_colors, amount = 0.3)

# Define the dodging width once to ensure it's identical in both layers
dodge <- position_dodge(width = 0.8)

y_limits <- range(all_data_long$value, na.rm = TRUE)
mean_only_data <- all_data_long %>%
  filter(condition == "Dry" | condition == "Rain")
ggplot(mean_only_data, aes(x = condition, y = value, fill = income_strata)) +
  geom_boxplot(position = dodge, outlier.shape = NA, color = "black") +
  stat_summary(fun = median, geom = "crossbar", width = 0.75, linewidth = 0.3, color = "white", position = dodge) +
  
  # This line forces the empty "Dry" and "Rain" categories to be displayed
  scale_x_discrete(drop = T) +
  # Set the y-axis limits to match the full dataset
  coord_cartesian(ylim = c(-25, 30)) +
  facet_wrap(~ metric) +
  scale_fill_manual(values = custom_colors) +
  labs(
    title = "",
    x = "Weather Condition",
    y = "Percentage Change (%)",
    fill = "Income Strata"
  ) +
  academic_theme+
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    strip.text = element_text(face = "bold")
  )

baseline_only_data <- all_data_long %>%
  filter(condition == "Baseline")
ggplot(baseline_only_data, aes(x = condition, y = value, fill = income_strata)) +
  geom_boxplot(position = dodge, outlier.shape = NA, color = "black") +
  stat_summary(fun = median, geom = "crossbar", width = 0.75, linewidth = 0.3, color = "white", position = dodge) +
  
  # This line forces the empty "Dry" and "Rain" categories to be displayed
  scale_x_discrete(drop = FALSE) +
  # Set the y-axis limits to match the full dataset
  coord_cartesian(ylim = y_limits) +
  facet_wrap(~ metric) +
  scale_fill_manual(values = custom_colors) +
  labs(
    title = "",
    x = "Weather Condition",
    y = "Percentage Change (%)",
    fill = "Income Strata"
  ) +
  academic_theme+
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    strip.text = element_text(face = "bold")
  )

ggplot(all_data_long, aes(x = condition, y = value, fill = income_strata)) +
  # Layer 1: The boxplot with BLACK outlines and whiskers
  geom_boxplot(
    position = dodge, 
    outlier.shape = NA, 
    color = "black" # Whiskers and border are black
  ) +
  # Layer 2: A WHITE median line drawn on top
  stat_summary(
    fun = median, 
    geom = "crossbar", 
    width = 0.75,     # Adjust width to match the box
    linewidth = 0.3,  # Make the line thick enough to see
    color = "white",  # Median line is white
    position = dodge  # CRITICAL: Must use the same dodge
  ) +
  # Set the y-axis limits to match the full dataset
  coord_cartesian(ylim = c(-25, 80)) +
  facet_wrap(~ metric) +
  scale_fill_manual(values = custom_colors) +
  labs(
    title = "",
    x = "Weather Condition",
    y = "Percentage Change (%)",
    fill = "Income Strata"
  ) +
  academic_theme+
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    strip.text = element_text(face = "bold")
  )

##### lowest_tier_increase_1 ########

lowest_tier_increase_1 = read_csv("lowest_tier_increase_1.csv")

#lowest_tier_increase_1 = cbind(lowest_tier_increase_1, demand_key)

lowest_tier_increase_1 <- lowest_tier_increase_1 %>%
  mutate(across(starts_with("cs_"), 
                ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata) * 100, 
                .names = "rate_{.col}"),
         across(starts_with("cs_"), 
                ~ (. - demand_key$cs_0) / abs(demand_key$cs_0) * 100, 
                .names = "rate_ind_{.col}")
         )

lowest_tier_increase_1 <- lowest_tier_increase_1 %>%
  mutate(across(starts_with("q_"), 
                ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"),
         across(starts_with("q_"), 
                ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_ind_{.col}"),
         across(starts_with("q_"), 
                ~ (. - demand_key$q_0), .names = "diff_{.col}")
         )

lowest_tier_increase_1 <- lowest_tier_increase_1 %>%
  mutate(across(starts_with("r_"), 
                ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"),
         across(starts_with("r_"), 
                ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_ind_{.col}"),
         across(starts_with("r_"), 
                ~ (. - demand_key$r_0), .names = "diff_{.col}"))

lowest_tier_increase_1 <-lowest_tier_increase_1 %>%
  mutate(across(starts_with("ev_"), ~ (. ) / demand_key$income * 100, .names = "rate_{.col}"))

lowest_tier_increase_1 = lowest_tier_increase_1 %>%
  mutate(
    income = demand_key$income,
    income_strata = demand_key$income_strata,
    bill_ym= demand_key$bill_ym,
    prem_id= demand_key$prem_id,
    q_0  = demand_key$q_0
  )

lowest_tier_increase_1_by_prem_id <- lowest_tier_increase_1 %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_1 = mean(rate_q_1),
            rate_r_1 = mean(rate_r_1),
            rate_ind_q_1 = mean(rate_ind_q_1),
            rate_ind_r_1 = mean(rate_ind_r_1),
            diff_q_1 = mean(diff_q_1),
            diff_r_1 = mean(diff_r_1),
            ev_1 = mean(ev_1),
            rate_ev_1 = mean(rate_ev_1),
            income_strata = unique(income_strata)
            ) %>%
  ungroup()

# Get the names of your income strata levels in their factor order
all_strata <- levels(lowest_tier_increase_1_by_prem_id$income_strata)

# Programmatically select the very first and very last level
strata_to_plot <- c(first(all_strata), last(all_strata))

plot_data_filtered_1 <- lowest_tier_increase_1_by_prem_id %>%
  filter(income_strata %in% strata_to_plot)

lowest_tier_increase_1_by_prem_id_dry <- lowest_tier_increase_1 %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_dry = mean(rate_q_dry),
            rate_r_dry = mean(rate_r_dry),
            rate_ind_q_dry = mean(rate_ind_q_dry),
            rate_ind_r_dry = mean(rate_ind_r_dry),
            diff_q_dry = mean(diff_q_dry),
            diff_r_dry = mean(diff_r_dry),
            ev_dry = mean(ev_dry),
            rate_ev_dry = mean(rate_ev_dry),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()

plot_data_filtered_dry <- lowest_tier_increase_1_by_prem_id_dry %>%
  filter(income_strata %in% strata_to_plot)


lowest_tier_increase_1_by_prem_id_wet <- lowest_tier_increase_1 %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_wet = mean(rate_q_wet),
            rate_r_wet = mean(rate_r_wet),
            rate_ind_q_wet = mean(rate_ind_q_wet),
            rate_ind_r_wet = mean(rate_ind_r_wet),
            diff_q_wet = mean(diff_q_wet),
            diff_r_wet = mean(diff_r_wet),
            ev_wet = mean(ev_wet),
            rate_ev_wet = mean(rate_ev_wet),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()

plot_data_filtered_wet <- lowest_tier_increase_1_by_prem_id_wet %>%
  filter(income_strata %in% strata_to_plot)

lowest_tier_increase_1_by_prem_id_low_var <- lowest_tier_increase_1 %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_low_var = mean(rate_q_low_var),
            rate_r_low_var = mean(rate_r_low_var),
            rate_ind_q_low_var = mean(rate_ind_q_low_var),
            rate_ind_r_low_var = mean(rate_ind_r_low_var),
            diff_q_low_var = mean(diff_q_low_var),
            diff_r_low_var = mean(diff_r_low_var),
            ev_low_var = mean(ev_low_var),
            rate_ev_low_var = mean(rate_ev_low_var),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()

plot_data_filtered_low_var <- lowest_tier_increase_1_by_prem_id_low_var %>%
  filter(income_strata %in% strata_to_plot)

lowest_tier_increase_1_by_prem_id_high_var <- lowest_tier_increase_1 %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_high_var = mean(rate_q_high_var),
            rate_r_high_var = mean(rate_r_high_var),
            rate_ind_q_high_var = mean(rate_ind_q_high_var),
            rate_ind_r_high_var = mean(rate_ind_r_high_var),
            diff_q_high_var = mean(diff_q_high_var),
            diff_r_high_var = mean(diff_r_high_var),
            ev_high_var = mean(ev_high_var),
            rate_ev_high_var = mean(rate_ev_high_var),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()

plot_data_filtered_high_var <- lowest_tier_increase_1_by_prem_id_high_var %>%
  filter(income_strata %in% strata_to_plot)

# Prepare each dataframe individually by selecting and renaming columns
df_baseline_p1 <- plot_data_filtered_1 %>%
  mutate(condition = "Baseline") %>%
  select(
    income_strata,
    condition,
    q_change = rate_ind_q_1, # Rename to generic name
    r_change = rate_ind_r_1  # Rename to generic name
  )

df_dry_p1 <- plot_data_filtered_dry %>%
  mutate(condition = "Dry") %>%
  select(
    income_strata,
    condition,
    q_change = rate_ind_q_dry, # Rename to the SAME generic name
    r_change = rate_ind_r_dry  # Rename to the SAME generic name
  )

df_wet_p1 <- plot_data_filtered_wet %>%
  mutate(condition = "Rain") %>%
  select(
    income_strata,
    condition,
    q_change = rate_ind_q_wet, # Rename to the SAME generic name
    r_change = rate_ind_r_wet  # Rename to the SAME generic name
  )

df_low_var_p1 <- plot_data_filtered_low_var %>%
  mutate(condition = "Low Var") %>%
  select(
    income_strata,
    condition,
    q_change = rate_ind_q_low_var, # Rename to the SAME generic name
    r_change = rate_ind_r_low_var  # Rename to the SAME generic name
  )

df_high_var_p1 <- plot_data_filtered_high_var %>%
  mutate(condition = "High Var") %>%
  select(
    income_strata,
    condition,
    q_change = rate_ind_q_high_var, # Rename to the SAME generic name
    r_change = rate_ind_r_high_var  # Rename to the SAME generic name
  )

# Now, bind the clean dataframes. This will work perfectly without creating NAs.
all_data_clean <- bind_rows(df_baseline, df_dry, df_wet)

# Finally, pivot the clean, combined data into the long format for plotting
all_data_long <- all_data_clean %>%
  rename(
    `Quantity (Q %)` = q_change,
    `Payments (R %)` = r_change
  ) %>%
  pivot_longer(
    cols = c(`Quantity (Q %)`, `Payments (R %)`),
    names_to = "metric",
    values_to = "value"
  )

# Set the desired order for the conditions on the plot's x-axis
all_data_long$condition <- factor(all_data_long$condition, 
                                  levels = c("Baseline", "Dry", "Rain"))

all_data_long$metric <- factor(all_data_long$metric, 
                               levels = c("Quantity (Q %)", "Payments (R %)"))

custom_border_colors <- darken(custom_colors, amount = 0.3)

# Define the dodging width once to ensure it's identical in both layers
dodge <- position_dodge(width = 0.8)

y_limits <- range(all_data_long$value, na.rm = TRUE)

baseline_only_data <- all_data_long %>%
  filter(condition == "Baseline")
ggplot(baseline_only_data, aes(x = condition, y = value, fill = income_strata)) +
  geom_boxplot(position = dodge, outlier.shape = NA, color = "black") +
  stat_summary(fun = median, geom = "crossbar", width = 0.75, linewidth = 0.3, color = "white", position = dodge) +
  
  # This line forces the empty "Dry" and "Rain" categories to be displayed
  scale_x_discrete(drop = FALSE) +
  # Set the y-axis limits to match the full dataset
  coord_cartesian(ylim = y_limits) +
  facet_wrap(~ metric) +
  scale_fill_manual(values = custom_colors) +
  labs(
    title = "",
    x = "Weather Condition",
    y = "Percentage Change (%)",
    fill = "Income Strata"
  ) +
  academic_theme+
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    strip.text = element_text(face = "bold")
  )

ggplot(all_data_long, aes(x = condition, y = value, fill = income_strata)) +
  # Layer 1: The boxplot with BLACK outlines and whiskers
  geom_boxplot(
    position = dodge, 
    outlier.shape = NA, 
    color = "black" # Whiskers and border are black
  ) +
  # Layer 2: A WHITE median line drawn on top
  stat_summary(
    fun = median, 
    geom = "crossbar", 
    width = 0.75,     # Adjust width to match the box
    linewidth = 0.3,  # Make the line thick enough to see
    color = "white",  # Median line is white
    position = dodge  # CRITICAL: Must use the same dodge
  ) +
  # Set the y-axis limits to match the full dataset
  coord_cartesian(ylim = y_limits) +
  facet_wrap(~ metric) +
  scale_fill_manual(values = custom_colors) +
  labs(
    title = "",
    x = "Weather Condition",
    y = "Percentage Change (%)",
    fill = "Income Strata"
  ) +
  academic_theme+
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    strip.text = element_text(face = "bold")
  )
  


##### highest_tier_increase_1 ########

highest_tier_increase_1 = read_csv("highest_tier_increase_1.csv")

#highest_tier_increase_1 = cbind(highest_tier_increase_1, demand_key)

highest_tier_increase_1 <- highest_tier_increase_1 %>%
  mutate(across(starts_with("cs_"), 
                ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata) * 100, 
                .names = "rate_{.col}"),
         across(starts_with("cs_"), 
                ~ (. - demand_key$cs_0) / abs(demand_key$cs_0) * 100, 
                .names = "rate_ind_{.col}")
  )

highest_tier_increase_1 <- highest_tier_increase_1 %>%
  mutate(across(starts_with("q_"), 
                ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"),
         across(starts_with("q_"), 
                ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_ind_{.col}"),
         across(starts_with("q_"), 
                ~ (. - demand_key$q_0), .names = "diff_{.col}")
  )

highest_tier_increase_1 <- highest_tier_increase_1 %>%
  mutate(across(starts_with("r_"), 
                ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"),
         across(starts_with("r_"), 
                ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_ind_{.col}"),
         across(starts_with("r_"), 
                ~ (. - demand_key$r_0), .names = "diff_{.col}"))

highest_tier_increase_1 <-highest_tier_increase_1 %>%
  mutate(across(starts_with("ev_"), ~ (. ) / demand_key$income * 100, .names = "rate_{.col}"))

highest_tier_increase_1 = highest_tier_increase_1 %>%
  mutate(
    income = demand_key$income,
    income_strata = demand_key$income_strata,
    bill_ym= demand_key$bill_ym,
    prem_id= demand_key$prem_id,
    q_0  = demand_key$q_0
  )

highest_tier_increase_1_by_prem_id <- highest_tier_increase_1 %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_1 = mean(rate_q_1),
            rate_r_1 = mean(rate_r_1),
            rate_ind_q_1 = mean(rate_ind_q_1),
            rate_ind_r_1 = mean(rate_ind_r_1),
            diff_q_1 = mean(diff_q_1),
            diff_r_1 = mean(diff_r_1),
            ev_1 = mean(ev_1),
            rate_ev_1 = mean(rate_ev_1),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()

# Get the names of your income strata levels in their factor order
all_strata <- levels(highest_tier_increase_1_by_prem_id$income_strata)

# Programmatically select the very first and very last level
strata_to_plot <- c(first(all_strata), last(all_strata))

plot_data_filtered_1 <- highest_tier_increase_1_by_prem_id %>%
  filter(income_strata %in% strata_to_plot)

highest_tier_increase_1_by_prem_id_dry <- highest_tier_increase_1 %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_dry = mean(rate_q_dry),
            rate_r_dry = mean(rate_r_dry),
            rate_ind_q_dry = mean(rate_ind_q_dry),
            rate_ind_r_dry = mean(rate_ind_r_dry),
            diff_q_dry = mean(diff_q_dry),
            diff_r_dry = mean(diff_r_dry),
            ev_dry = mean(ev_dry),
            rate_ev_dry = mean(rate_ev_dry),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()


plot_data_filtered_dry <- highest_tier_increase_1_by_prem_id_dry %>%
  filter(income_strata %in% strata_to_plot)


highest_tier_increase_1_by_prem_id_wet <- highest_tier_increase_1 %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_wet = mean(rate_q_wet),
            rate_r_wet = mean(rate_r_wet),
            rate_ind_q_wet = mean(rate_ind_q_wet),
            rate_ind_r_wet = mean(rate_ind_r_wet),
            diff_q_wet = mean(diff_q_wet),
            diff_r_wet = mean(diff_r_wet),
            ev_wet = mean(ev_wet),
            rate_ev_wet = mean(rate_ev_wet),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()


plot_data_filtered_wet <- highest_tier_increase_1_by_prem_id_wet %>%
  filter(income_strata %in% strata_to_plot)

highest_tier_increase_1_by_prem_id_low_var <- highest_tier_increase_1 %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_low_var = mean(rate_q_low_var),
            rate_r_low_var = mean(rate_r_low_var),
            rate_ind_q_low_var = mean(rate_ind_q_low_var),
            rate_ind_r_low_var = mean(rate_ind_r_low_var),
            diff_q_low_var = mean(diff_q_low_var),
            diff_r_low_var = mean(diff_r_low_var),
            ev_low_var = mean(ev_low_var),
            rate_ev_low_var = mean(rate_ev_low_var),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()


plot_data_filtered_low_var <- highest_tier_increase_1_by_prem_id_low_var %>%
  filter(income_strata %in% strata_to_plot)

highest_tier_increase_1_by_prem_id_high_var <- highest_tier_increase_1 %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_high_var = mean(rate_q_high_var),
            rate_r_high_var = mean(rate_r_high_var),
            rate_ind_q_high_var = mean(rate_ind_q_high_var),
            rate_ind_r_high_var = mean(rate_ind_r_high_var),
            diff_q_high_var = mean(diff_q_high_var),
            diff_r_high_var = mean(diff_r_high_var),
            ev_high_var = mean(ev_high_var),
            rate_ev_high_var = mean(rate_ev_high_var),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()


plot_data_filtered_high_var <- highest_tier_increase_1_by_prem_id_high_var %>%
  filter(income_strata %in% strata_to_plot)

# Prepare each dataframe individually by selecting and renaming columns
df_baseline_p5 <- plot_data_filtered_1 %>%
  mutate(condition = "Baseline") %>%
  select(
    income_strata,
    condition,
    q_change = rate_ind_q_1, # Rename to generic name
    r_change = rate_ind_r_1  # Rename to generic name
  )

df_dry_p5 <- plot_data_filtered_dry %>%
  mutate(condition = "Dry") %>%
  select(
    income_strata,
    condition,
    q_change = rate_ind_q_dry, # Rename to the SAME generic name
    r_change = rate_ind_r_dry  # Rename to the SAME generic name
  )

df_wet_p5 <- plot_data_filtered_wet %>%
  mutate(condition = "Rain") %>%
  select(
    income_strata,
    condition,
    q_change = rate_ind_q_wet, # Rename to the SAME generic name
    r_change = rate_ind_r_wet  # Rename to the SAME generic name
  )

df_low_var_p5 <- plot_data_filtered_low_var %>%
  mutate(condition = "Low Var") %>%
  select(
    income_strata,
    condition,
    q_change = rate_ind_q_low_var, # Rename to the SAME generic name
    r_change = rate_ind_r_low_var  # Rename to the SAME generic name
  )

df_high_var_p5 <- plot_data_filtered_high_var %>%
  mutate(condition = "High Var") %>%
  select(
    income_strata,
    condition,
    q_change = rate_ind_q_high_var, # Rename to the SAME generic name
    r_change = rate_ind_r_high_var  # Rename to the SAME generic name
  )

# Now, bind the clean dataframes. This will work perfectly without creating NAs.
all_data_clean <- bind_rows(df_baseline, df_dry, df_wet)

# Finally, pivot the clean, combined data into the long format for plotting
all_data_long <- all_data_clean %>%
  rename(
    `Quantity (Q %)` = q_change,
    `Payments (R %)` = r_change
  ) %>%
  pivot_longer(
    cols = c(`Quantity (Q %)`, `Payments (R %)`),
    names_to = "metric",
    values_to = "value"
  )

# Set the desired order for the conditions on the plot's x-axis
all_data_long$condition <- factor(all_data_long$condition, 
                                  levels = c("Baseline", "Dry", "Rain"))

all_data_long$metric <- factor(all_data_long$metric, 
                               levels = c("Quantity (Q %)", "Payments (R %)"))

custom_border_colors <- darken(custom_colors, amount = 0.3)

# Define the dodging width once to ensure it's identical in both layers
dodge <- position_dodge(width = 0.8)

y_limits <- range(all_data_long$value, na.rm = TRUE)

baseline_only_data <- all_data_long %>%
  filter(condition == "Baseline")
ggplot(baseline_only_data, aes(x = condition, y = value, fill = income_strata)) +
  geom_boxplot(position = dodge, outlier.shape = NA, color = "black") +
  stat_summary(fun = median, geom = "crossbar", width = 0.75, linewidth = 0.3, color = "white", position = dodge) +
  
  # This line forces the empty "Dry" and "Rain" categories to be displayed
  scale_x_discrete(drop = FALSE) +
  coord_cartesian(ylim = y_limits) +
  facet_wrap(~ metric) +
  scale_fill_manual(values = custom_colors) +
  labs(
    title = "",
    x = "Weather Condition",
    y = "Percentage Change (%)",
    fill = "Income Strata"
  ) +
  academic_theme+
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    strip.text = element_text(face = "bold")
  )

ggplot(all_data_long, aes(x = condition, y = value, fill = income_strata)) +
  # Layer 1: The boxplot with BLACK outlines and whiskers
  geom_boxplot(
    position = dodge, 
    outlier.shape = NA, 
    color = "black" # Whiskers and border are black
  ) +
  # Layer 2: A WHITE median line drawn on top
  stat_summary(
    fun = median, 
    geom = "crossbar", 
    width = 0.75,     # Adjust width to match the box
    linewidth = 0.3,  # Make the line thick enough to see
    color = "white",  # Median line is white
    position = dodge  # CRITICAL: Must use the same dodge
  ) +
  coord_cartesian(ylim = y_limits) +
  facet_wrap(~ metric) +
  scale_fill_manual(values = custom_colors) +
  labs(
    title = "",
    x = "Weather Condition",
    y = "Percentage Change (%)",
    fill = "Income Strata"
  ) +
  academic_theme+
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    strip.text = element_text(face = "bold")
  )





#### Combine Picture Dry ####

df_dry$scenerio = "status quo p"
df_dry_p1$scenerio = "p1 +1"
df_dry_p5$scenerio = "p5 +1"

# Now, bind the clean dataframes. This will work perfectly without creating NAs.
all_data_clean <- bind_rows(df_dry, df_dry_p1, df_dry_p5)

# Finally, pivot the clean, combined data into the long format for plotting
all_data_long <- all_data_clean %>%
  rename(
    `Quantity (Q %)` = q_change,
    `Payments (R %)` = r_change
  ) %>%
  pivot_longer(
    cols = c(`Quantity (Q %)`, `Payments (R %)`),
    names_to = "metric",
    values_to = "value"
  )

# Set the desired order for the conditions on the plot's x-axis
all_data_long$scenerio <- factor(all_data_long$scenerio, 
                                  levels = c("status quo p", "p1 +1", "p5 +1"))

all_data_long$metric <- factor(all_data_long$metric, 
                               levels = c("Quantity (Q %)", "Payments (R %)"))

custom_border_colors <- darken(custom_colors, amount = 0.3)

# Define the dodging width once to ensure it's identical in both layers
dodge <- position_dodge(width = 0.8)

y_limits <- range(all_data_long$value, na.rm = TRUE)

ggplot(all_data_long, aes(x = scenerio, y = value, fill = income_strata)) +
  geom_boxplot(position = dodge, outlier.shape = NA, color = "black") +
  stat_summary(fun = median, geom = "crossbar", width = 0.75, linewidth = 0.3, color = "white", position = dodge) +
  
  # This line forces the empty "Dry" and "Rain" categories to be displayed
  scale_x_discrete(drop = FALSE) +
  # Set the y-axis limits to match the full dataset
  coord_cartesian(ylim = y_limits) +
  facet_wrap(~ metric) +
  scale_fill_manual(values = custom_colors) +
  labs(
    title = "",
    x = "Scenerio",
    y = "Percentage Change (%)",
    fill = "Income Strata"
  ) +
  academic_theme+
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    strip.text = element_text(face = "bold")
  )

#### Combine Picture Wet ####

df_wet$scenerio = "status quo p"
df_wet_p1$scenerio = "p1 +1"
df_wet_p5$scenerio = "p5 +1"

# Now, bind the clean dataframes. This will work perfectly without creating NAs.
all_data_clean <- bind_rows(df_wet, df_wet_p1, df_wet_p5)

# Finally, pivot the clean, combined data into the long format for plotting
all_data_long <- all_data_clean %>%
  rename(
    `Quantity (Q %)` = q_change,
    `Payments (R %)` = r_change
  ) %>%
  pivot_longer(
    cols = c(`Quantity (Q %)`, `Payments (R %)`),
    names_to = "metric",
    values_to = "value"
  )

# Set the desired order for the conditions on the plot's x-axis
all_data_long$scenerio <- factor(all_data_long$scenerio, 
                                 levels = c("status quo p", "p1 +1", "p5 +1"))

all_data_long$metric <- factor(all_data_long$metric, 
                               levels = c("Quantity (Q %)", "Payments (R %)"))

custom_border_colors <- darken(custom_colors, amount = 0.3)

# Define the dodging width once to ensure it's identical in both layers
dodge <- position_dodge(width = 0.8)

y_limits <- range(all_data_long$value, na.rm = TRUE)

ggplot(all_data_long, aes(x = scenerio, y = value, fill = income_strata)) +
  geom_boxplot(position = dodge, outlier.shape = NA, color = "black") +
  stat_summary(fun = median, geom = "crossbar", width = 0.75, linewidth = 0.3, color = "white", position = dodge) +
  
  # This line forces the empty "Dry" and "Rain" categories to be displayed
  scale_x_discrete(drop = FALSE) +
  # Set the y-axis limits to match the full dataset
  coord_cartesian(ylim = y_limits) +
  facet_wrap(~ metric) +
  scale_fill_manual(values = custom_colors) +
  labs(
    title = "",
    x = "Scenerio",
    y = "Percentage Change (%)",
    fill = "Income Strata"
  ) +
  academic_theme+
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, face = "bold"),
    strip.text = element_text(face = "bold")
  )

##### temp ####

q1 = ggplot(plot_data_filtered, aes(x = income_strata, y = rate_ind_q_wet, fill = income_strata)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    x = "",
    y = "Percentage Change in Quantity (Q %)",
    #title = "Comparing Quantity Reduction Across Income Strata"
  ) +
  # --- CHANGE THIS LINE ---
  scale_fill_manual(values = custom_colors) + # Use scale_fill_manual to match the `fill` aesthetic
  # ----------------------
theme(legend.position = "none") + 
  academic_theme

r1 = ggplot(plot_data_filtered, aes(x = income_strata, y = rate_ind_r_wet, fill = income_strata)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    x = "",
    y = "Percentage Change in Payments (R %)",
    #title = "Comparing Payment Difference Across Income Strata"
  ) +
  # --- CHANGE THIS LINE ---
  scale_fill_manual(values = custom_colors) + # Use scale_fill_manual to match the `fill` aesthetic
  # ----------------------
theme(legend.position = "none") + 
  academic_theme

ev1 = ggplot(plot_data_filtered, aes(x = income_strata, y = ev_wet, fill = income_strata)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    x = "",
    y = "EV ($)",
    #title = "Comparing Payment Difference Across Income Strata"
  ) +
  # --- CHANGE THIS LINE ---
  scale_fill_manual(values = custom_colors) + # Use scale_fill_manual to match the `fill` aesthetic
  # ----------------------
theme(legend.position = "none") + 
  academic_theme

rate_ev1 = ggplot(plot_data_filtered, aes(x = income_strata, y = rate_ev_wet, fill = income_strata)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    x = "",
    y = "EV/I (%)",
    #title = "Comparing Payment Difference Across Income Strata"
  ) +
  # --- CHANGE THIS LINE ---
  scale_fill_manual(values = custom_colors) + # Use scale_fill_manual to match the `fill` aesthetic
  # ----------------------
theme(legend.position = "none") + 
  academic_theme

combined_plot_wet <- ggarrange(
  q1,
  r1,
  #ev1,
  #rate_ev1,
  ncol = 2, # Arrange in two columns
  common.legend = TRUE, # Use a single legend for both plots
  legend = "bottom" # Place the common legend on the right
)

lowest_tier_increase_1_by_prem_id_low_var <- lowest_tier_increase_1 %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_low_var = mean(rate_q_low_var),
            rate_r_low_var = mean(rate_r_low_var),
            rate_ind_q_low_var = mean(rate_ind_q_low_var),
            rate_ind_r_low_var = mean(rate_ind_r_low_var),
            diff_q_low_var = mean(diff_q_low_var),
            diff_r_low_var = mean(diff_r_low_var),
            ev_low_var = mean(ev_low_var),
            rate_ev_low_var = mean(rate_ev_low_var),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()


plot_data_filtered <- lowest_tier_increase_1_by_prem_id_low_var %>%
  filter(income_strata %in% strata_to_plot)


q1 = ggplot(plot_data_filtered, aes(x = income_strata, y = rate_ind_q_low_var, fill = income_strata)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    x = "",
    y = "Percentage Change in Quantity (Q %)",
    #title = "Comparing Quantity Reduction Across Income Strata"
  ) +
  # --- CHANGE THIS LINE ---
  scale_fill_manual(values = custom_colors) + # Use scale_fill_manual to match the `fill` aesthetic
  # ----------------------
theme(legend.position = "none") + 
  academic_theme

r1 = ggplot(plot_data_filtered, aes(x = income_strata, y = rate_ind_r_low_var, fill = income_strata)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    x = "",
    y = "Percentage Change in Payments (R %)",
    #title = "Comparing Payment Difference Across Income Strata"
  ) +
  # --- CHANGE THIS LINE ---
  scale_fill_manual(values = custom_colors) + # Use scale_fill_manual to match the `fill` aesthetic
  # ----------------------
theme(legend.position = "none") + 
  academic_theme

ev1 = ggplot(plot_data_filtered, aes(x = income_strata, y = ev_low_var, fill = income_strata)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    x = "",
    y = "EV ($)",
    #title = "Comparing Payment Difference Across Income Strata"
  ) +
  # --- CHANGE THIS LINE ---
  scale_fill_manual(values = custom_colors) + # Use scale_fill_manual to match the `fill` aesthetic
  # ----------------------
theme(legend.position = "none") + 
  academic_theme

rate_ev1 = ggplot(plot_data_filtered, aes(x = income_strata, y = rate_ev_low_var, fill = income_strata)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    x = "",
    y = "EV/I (%)",
    #title = "Comparing Payment Difference Across Income Strata"
  ) +
  # --- CHANGE THIS LINE ---
  scale_fill_manual(values = custom_colors) + # Use scale_fill_manual to match the `fill` aesthetic
  # ----------------------
theme(legend.position = "none") + 
  academic_theme

combined_plot <- ggarrange(
  q1,
  r1,
  ev1,
  rate_ev1,
  ncol = 4, # Arrange in two columns
  common.legend = TRUE, # Use a single legend for both plots
  legend = "bottom" # Place the common legend on the right
)

lowest_tier_increase_1_by_prem_id_high_var <- lowest_tier_increase_1 %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            rate_q_high_var = mean(rate_q_high_var),
            rate_r_high_var = mean(rate_r_high_var),
            rate_ind_q_high_var = mean(rate_ind_q_high_var),
            rate_ind_r_high_var = mean(rate_ind_r_high_var),
            diff_q_high_var = mean(diff_q_high_var),
            diff_r_high_var = mean(diff_r_high_var),
            ev_high_var = mean(ev_high_var),
            rate_ev_high_var = mean(rate_ev_high_var),
            income_strata = unique(income_strata)
  ) %>%
  ungroup()


plot_data_filtered <- lowest_tier_increase_1_by_prem_id_high_var %>%
  filter(income_strata %in% strata_to_plot)


q1 = ggplot(plot_data_filtered, aes(x = income_strata, y = rate_ind_q_high_var, fill = income_strata)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    x = "",
    y = "Percentage Change in Quantity (Q %)",
    #title = "Comparing Quantity Reduction Across Income Strata"
  ) +
  # --- CHANGE THIS LINE ---
  scale_fill_manual(values = custom_colors) + # Use scale_fill_manual to match the `fill` aesthetic
  # ----------------------
theme(legend.position = "none") + 
  academic_theme

r1 = ggplot(plot_data_filtered, aes(x = income_strata, y = rate_ind_r_high_var, fill = income_strata)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    x = "",
    y = "Percentage Change in Payments (R %)",
    #title = "Comparing Payment Difference Across Income Strata"
  ) +
  # --- CHANGE THIS LINE ---
  scale_fill_manual(values = custom_colors) + # Use scale_fill_manual to match the `fill` aesthetic
  # ----------------------
theme(legend.position = "none") + 
  academic_theme

ev1 = ggplot(plot_data_filtered, aes(x = income_strata, y = ev_high_var, fill = income_strata)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    x = "",
    y = "EV ($)",
    #title = "Comparing Payment Difference Across Income Strata"
  ) +
  # --- CHANGE THIS LINE ---
  scale_fill_manual(values = custom_colors) + # Use scale_fill_manual to match the `fill` aesthetic
  # ----------------------
theme(legend.position = "none") + 
  academic_theme

rate_ev1 = ggplot(plot_data_filtered, aes(x = income_strata, y = rate_ev_high_var, fill = income_strata)) +
  geom_boxplot(alpha = 0.7) +
  labs(
    x = "",
    y = "EV/I (%)",
    #title = "Comparing Payment Difference Across Income Strata"
  ) +
  # --- CHANGE THIS LINE ---
  scale_fill_manual(values = custom_colors) + # Use scale_fill_manual to match the `fill` aesthetic
  # ----------------------
theme(legend.position = "none") + 
  academic_theme

combined_plot <- ggarrange(
  q1,
  r1,
  ev1,
  rate_ev1,
  ncol = 4, # Arrange in two columns
  common.legend = TRUE, # Use a single legend for both plots
  legend = "bottom" # Place the common legend on the right
)





###### aggregated summary #####


lowest_tier_increase_1_by_bill_ym <- lowest_tier_increase_1 %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with(c("rate_", "ev_")), mean, na.rm = TRUE))

lowest_tier_increase_1_by_strata <- lowest_tier_increase_1_by_bill_ym  %>%
  group_by(income_strata) %>%
  summarise(across(starts_with(c("rate_", "ev_")), mean, na.rm = TRUE))


###### optimal price, no policy constraints #####
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
##### sq_weather_avg_bound_loss0_mean ########

sq_weather_avg_bound_loss0_mean_cs_steps = read_csv("cs_detail_results/sq_weather_avg_bound_loss0_mean_cs_steps.csv")

colnames(sq_weather_avg_bound_loss0_mean_cs_steps) = as.character(as.numeric(colnames(sq_weather_avg_bound_loss0_mean_cs_steps))*0.05-0.25)

sq_weather_avg_bound_loss0_mean_cs_steps_rate <-sq_weather_avg_bound_loss0_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(sq_weather_avg_bound_loss0_mean_cs_steps)

sq_weather_avg_bound_loss0_mean_q_steps = read_csv("cs_detail_results/sq_weather_avg_bound_loss0_mean_q_steps.csv")

colnames(sq_weather_avg_bound_loss0_mean_q_steps) = as.character(as.numeric(colnames(sq_weather_avg_bound_loss0_mean_q_steps))*0.05-0.25)

sq_weather_avg_bound_loss0_mean_q_steps_rate <-sq_weather_avg_bound_loss0_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(sq_weather_avg_bound_loss0_mean_q_steps)

sq_weather_avg_bound_loss0_mean_r_steps = read_csv("cs_detail_results/sq_weather_avg_bound_loss0_mean_r_steps.csv")

colnames(sq_weather_avg_bound_loss0_mean_r_steps) = as.character(as.numeric(colnames(sq_weather_avg_bound_loss0_mean_r_steps))*0.05-0.25)

sq_weather_avg_bound_loss0_mean_r_steps_rate <-sq_weather_avg_bound_loss0_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(sq_weather_avg_bound_loss0_mean_r_steps)

sq_weather_avg_bound_loss0_mean_ev_steps = read_csv("cs_detail_results/sq_weather_avg_bound_loss0_mean_ev_steps.csv")

colnames(sq_weather_avg_bound_loss0_mean_ev_steps) = as.character(as.numeric(colnames(sq_weather_avg_bound_loss0_mean_ev_steps))*0.05-0.25)

sq_weather_avg_bound_loss0_mean_ev_steps_rate <-sq_weather_avg_bound_loss0_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(sq_weather_avg_bound_loss0_mean_ev_steps)

sq_weather_avg_bound_loss0_mean_cs_steps_rate  = cbind(sq_weather_avg_bound_loss0_mean_cs_steps_rate , demand_key)
sq_weather_avg_bound_loss0_mean_q_steps_rate = cbind(sq_weather_avg_bound_loss0_mean_q_steps_rate, demand_key)
sq_weather_avg_bound_loss0_mean_r_steps_rate = cbind(sq_weather_avg_bound_loss0_mean_r_steps_rate, demand_key)
sq_weather_avg_bound_loss0_mean_ev_steps_rate = cbind(sq_weather_avg_bound_loss0_mean_ev_steps_rate, demand_key)

#sq_weather_avg_bound_loss0_mean_cs_steps_rate = sq_weather_avg_bound_loss0_mean_cs_steps_rate[which(sq_weather_avg_bound_loss0_mean_cs_steps_rate$q_0<100),]
#sq_weather_avg_bound_loss0_mean_ev_steps_rate = sq_weather_avg_bound_loss0_mean_ev_steps_rate[which(sq_weather_avg_bound_loss0_mean_ev_steps_rate$q_0<100),]

mean_cs_rate_by_bill_ym <- sq_weather_avg_bound_loss0_mean_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(income_strata,prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- sq_weather_avg_bound_loss0_mean_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(income_strata,prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- sq_weather_avg_bound_loss0_mean_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(income_strata,prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- sq_weather_avg_bound_loss0_mean_ev_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(income_strata,prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_by_bill_ym <- sq_weather_avg_bound_loss0_mean_ev_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(1:11, mean, na.rm = TRUE))

mean_cs_rate_by_strata_sq <- mean_cs_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_strata_sq <- mean_r_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_strata_sq <- mean_q_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_strata_sq <- mean_ev_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_by_strata_sq <- mean_ev_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(2:12, mean, na.rm = TRUE))

mean_cs_rate_by_strata_sq = edit_strata_df(mean_cs_rate_by_strata_sq, "cs")
mean_cs_rate_by_strata_sq$cs_change_rate = as.numeric(mean_cs_rate_by_strata_sq$cs_change_rate)

mean_r_rate_by_strata_sq = edit_strata_df(mean_r_rate_by_strata_sq, "r")
mean_r_rate_by_strata_sq$r_change_rate = as.numeric(mean_r_rate_by_strata_sq$r_change_rate)

mean_q_rate_by_strata_sq = edit_strata_df(mean_q_rate_by_strata_sq, "q")
mean_q_rate_by_strata_sq$q_change_rate = as.numeric(mean_q_rate_by_strata_sq$q_change_rate)

mean_ev_rate_by_strata_sq = edit_strata_df(mean_ev_rate_by_strata_sq, "ev")
mean_ev_rate_by_strata_sq$ev_perct = as.numeric(mean_ev_rate_by_strata_sq$ev_change_rate)
mean_ev_rate_by_strata_sq$ev_change_rate = NULL

mean_ev_by_strata_sq= edit_strata_df(mean_ev_by_strata_sq, "ev")
mean_ev_by_strata_sq$ev = as.numeric(mean_ev_by_strata_sq$ev_change_rate)
mean_ev_by_strata_sq$ev_change_rate = NULL

cs_avg = ggplot(mean_cs_rate_by_strata_sq , aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_cs_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "CS %",
    color = "Income Strata", 
    shape = "Income Strata") +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )


r_avg = ggplot(mean_r_rate_by_strata_sq , aes(x = step, y = r_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_r_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(0, 70))+
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "PS %",
    color = "Income Strata",
    shape = "Income Strata") +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

q_avg = ggplot(mean_q_rate_by_strata_sq , aes(x = step, y = q_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_q_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "Q %",
    color = "Income Strata",
    shape = "Income Strata") +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

ev_avg = ggplot(mean_ev_rate_by_strata_sq , aes(x = step, y = ev_perct, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_ev_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "EV/I (%)",
    color = NULL,
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )


mean_ev_rate_by_strata_sq$status = case_when(
  mean_ev_rate_by_strata_sq$step<0 ~ "low",
  mean_ev_rate_by_strata_sq$step>0 ~ "high"
)

mean_cs_rate_by_strata_sq$status = case_when(
  mean_cs_rate_by_strata_sq$step<0 ~ "low",
  mean_cs_rate_by_strata_sq$step>0 ~ "high"
)

sq_weather_loss0_mean_ev = mean_ev_rate_by_strata_sq %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

sq_weather_loss0_mean_cs = mean_cs_rate_by_strata_sq %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

shared_legend <- get_legend(r_avg)


# Arrange the three plots and the shared legend as the fourth item
sq_weather_loss0_mean_grid  <- ggarrange(
  r_avg + theme(legend.position = "none"),
  q_avg + theme(legend.position = "none"),
  ev_avg + theme(legend.position = "none"),
  shared_legend,
  ncol = 2,
  nrow = 2,
  labels = c("PS", "Q", "EV", ""), # Use an empty label for the legend's spot
  font.label = list(size = 15, face = "bold"),
  label.x = 0.02, label.y = 0.98,
  hjust = 0, vjust = 1
)

# Now, add a title to this grid using patchwork's plot_annotation
sq_weather_loss0_mean_grid_with_title <- sq_weather_loss0_mean_grid +
  plot_annotation(
    title = "Linear Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )

##### sq_weather_avg_bound_loss0_var ########

sq_weather_avg_bound_loss0_var_cs_steps = read_csv("cs_detail_results/sq_weather_avg_bound_loss0_var_cs_steps.csv")

colnames(sq_weather_avg_bound_loss0_var_cs_steps) = as.character(as.numeric(colnames(sq_weather_avg_bound_loss0_var_cs_steps))/20+0.75)

sq_weather_avg_bound_loss0_var_cs_steps_rate <-sq_weather_avg_bound_loss0_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(sq_weather_avg_bound_loss0_var_cs_steps)

sq_weather_avg_bound_loss0_var_q_steps = read_csv("cs_detail_results/sq_weather_avg_bound_loss0_var_q_steps.csv")

colnames(sq_weather_avg_bound_loss0_var_q_steps) = as.character(as.numeric(colnames(sq_weather_avg_bound_loss0_var_q_steps))/20+0.75)

sq_weather_avg_bound_loss0_var_q_steps_rate <-sq_weather_avg_bound_loss0_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(sq_weather_avg_bound_loss0_var_q_steps)

sq_weather_avg_bound_loss0_var_r_steps = read_csv("cs_detail_results/sq_weather_avg_bound_loss0_var_r_steps.csv")

colnames(sq_weather_avg_bound_loss0_var_r_steps) = as.character(as.numeric(colnames(sq_weather_avg_bound_loss0_var_r_steps))/20+0.75)

sq_weather_avg_bound_loss0_var_r_steps_rate <-sq_weather_avg_bound_loss0_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(sq_weather_avg_bound_loss0_var_r_steps)

sq_weather_avg_bound_loss0_var_ev_steps = read_csv("cs_detail_results/sq_weather_avg_bound_loss0_var_ev_steps.csv")

colnames(sq_weather_avg_bound_loss0_var_ev_steps) = as.character(as.numeric(colnames(sq_weather_avg_bound_loss0_var_ev_steps))/20+0.75)

sq_weather_avg_bound_loss0_var_ev_steps_rate <-sq_weather_avg_bound_loss0_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(sq_weather_avg_bound_loss0_var_ev_steps)

sq_weather_avg_bound_loss0_var_cs_steps_rate  = cbind(sq_weather_avg_bound_loss0_var_cs_steps_rate , demand_key)
sq_weather_avg_bound_loss0_var_q_steps_rate = cbind(sq_weather_avg_bound_loss0_var_q_steps_rate, demand_key)
sq_weather_avg_bound_loss0_var_r_steps_rate = cbind(sq_weather_avg_bound_loss0_var_r_steps_rate, demand_key)
sq_weather_avg_bound_loss0_var_ev_steps_rate = cbind(sq_weather_avg_bound_loss0_var_ev_steps_rate, demand_key)

#sq_weather_avg_bound_loss0_var_cs_steps_rate = sq_weather_avg_bound_loss0_var_cs_steps_rate[which(sq_weather_avg_bound_loss0_var_cs_steps_rate$q_0<100),]
#sq_weather_avg_bound_loss0_var_ev_steps_rate = sq_weather_avg_bound_loss0_var_ev_steps_rate[which(sq_weather_avg_bound_loss0_var_ev_steps_rate$q_0<100),]

var_cs_rate_by_bill_ym <- sq_weather_avg_bound_loss0_var_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- sq_weather_avg_bound_loss0_var_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- sq_weather_avg_bound_loss0_var_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- sq_weather_avg_bound_loss0_var_ev_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_by_bill_ym <- sq_weather_avg_bound_loss0_var_ev_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(1:11, mean, na.rm = TRUE))

var_cs_rate_by_strata_sq <- var_cs_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_strata_sq <- var_r_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_strata_sq <- var_q_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_strata_sq <- var_ev_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_by_strata_sq <- var_ev_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(2:12, mean, na.rm = TRUE))

var_cs_rate_by_strata_sq = edit_strata_df(var_cs_rate_by_strata_sq, "cs")
var_cs_rate_by_strata_sq$cs_change_rate = as.numeric(var_cs_rate_by_strata_sq$cs_change_rate)

var_r_rate_by_strata_sq = edit_strata_df(var_r_rate_by_strata_sq, "r")
var_r_rate_by_strata_sq$r_change_rate = as.numeric(var_r_rate_by_strata_sq$r_change_rate)

var_q_rate_by_strata_sq = edit_strata_df(var_q_rate_by_strata_sq, "q")
var_q_rate_by_strata_sq$q_change_rate = as.numeric(var_q_rate_by_strata_sq$q_change_rate)

var_ev_rate_by_strata_sq = edit_strata_df(var_ev_rate_by_strata_sq, "ev")
var_ev_rate_by_strata_sq$ev_perct = as.numeric(var_ev_rate_by_strata_sq$ev_change_rate)
var_ev_rate_by_strata_sq$ev_change_rate = NULL

var_ev_by_strata_sq = edit_strata_df(var_ev_by_strata_sq, "ev")
var_ev_by_strata_sq$ev = as.numeric(var_ev_by_strata_sq$ev_change_rate)
var_ev_by_strata_sq$ev_change_rate = NULL

cs_avg = ggplot(var_cs_rate_by_strata_sq , aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_cs_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "CS %",
    color = "Income Strata", 
    shape = "Income Strata") +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )


r_avg = ggplot(var_r_rate_by_strata_sq , aes(x = step, y = r_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_r_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(0, 70))+
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "PS %",
    color = "Income Strata",
    shape = "Income Strata") +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

q_avg = ggplot(var_q_rate_by_strata_sq , aes(x = step, y = q_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_q_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "Q %",
    color = "Income Strata",
    shape = "Income Strata") +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

ev_avg = ggplot(var_ev_rate_by_strata_sq , aes(x = step, y = ev_perct, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_ev_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "EV/I (%)",
    color = NULL,
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

var_ev_rate_by_strata_sq$status = case_when(
  var_ev_rate_by_strata_sq$step<1 ~ "low",
  var_ev_rate_by_strata_sq$step>1 ~ "high"
)

var_cs_rate_by_strata_sq$status = case_when(
  var_cs_rate_by_strata_sq$step<1 ~ "low",
  var_cs_rate_by_strata_sq$step>1 ~ "high"
)

sq_weather_loss0_var_ev = var_ev_rate_by_strata_sq %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

sq_weather_loss0_var_cs = var_cs_rate_by_strata_sq %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

shared_legend <- get_legend(r_avg)

# Arrange the three plots and the shared legend as the fourth item
sq_weather_loss0_var_grid <- ggarrange(
  r_avg + theme(legend.position = "none"),
  q_avg + theme(legend.position = "none"),
  ev_avg + theme(legend.position = "none"),
  shared_legend,
  ncol = 2,
  nrow = 2,
  labels = c("PS", "Q", "EV", ""), # Use an empty label for the legend's spot
  font.label = list(size = 15, face = "bold"),
  label.x = 0.02, label.y = 0.98,
  hjust = 0, vjust = 1
)

# Now, add a title to this grid using patchwork's plot_annotation
sq_weather_loss0_var_grid_with_title <- sq_weather_loss0_var_grid +
  plot_annotation(
    title = "Linear Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )







##### sq_p_mean ########

sq_p_mean_cs_steps = read_csv("cs_detail_results/sq_p_mean_cs_steps.csv")

colnames(sq_p_mean_cs_steps) = as.character(as.numeric(colnames(sq_p_mean_cs_steps))*0.05-0.25)

sq_p_mean_cs_steps_rate <-sq_p_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(sq_p_mean_cs_steps)

sq_p_mean_q_steps = read_csv("cs_detail_results/sq_p_mean_q_steps.csv")

colnames(sq_p_mean_q_steps) = as.character(as.numeric(colnames(sq_p_mean_q_steps))*0.05-0.25)

sq_p_mean_q_steps_rate <-sq_p_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(sq_p_mean_q_steps)

sq_p_mean_r_steps = read_csv("cs_detail_results/sq_p_mean_r_steps.csv")

colnames(sq_p_mean_r_steps) = as.character(as.numeric(colnames(sq_p_mean_r_steps))*0.05-0.25)

sq_p_mean_r_steps_rate <-sq_p_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(sq_p_mean_r_steps)

sq_p_mean_ev_steps = read_csv("cs_detail_results/sq_p_mean_ev_steps.csv")

colnames(sq_p_mean_ev_steps) = as.character(as.numeric(colnames(sq_p_mean_ev_steps))*0.05-0.25)

sq_p_mean_ev_steps_rate <-sq_p_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(sq_p_mean_ev_steps)

sq_p_mean_cs_steps_rate  = cbind(sq_p_mean_cs_steps_rate , demand_key)
sq_p_mean_q_steps_rate = cbind(sq_p_mean_q_steps_rate, demand_key)
sq_p_mean_r_steps_rate = cbind(sq_p_mean_r_steps_rate, demand_key)
sq_p_mean_ev_steps_rate = cbind(sq_p_mean_ev_steps_rate, demand_key)

#sq_p_mean_cs_steps_rate = sq_p_mean_cs_steps_rate[which(sq_p_mean_cs_steps_rate$q_0<100),]
#sq_p_mean_ev_steps_rate = sq_p_mean_ev_steps_rate[which(sq_p_mean_ev_steps_rate$q_0<100),]

mean_cs_rate_by_bill_ym <- sq_p_mean_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(income_strata,prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- sq_p_mean_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(income_strata,prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- sq_p_mean_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(income_strata,prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- sq_p_mean_ev_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(income_strata,prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_by_bill_ym <- sq_p_mean_ev_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(1:11, mean, na.rm = TRUE))

mean_cs_rate_by_strata_sq <- mean_cs_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_strata_sq <- mean_r_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_strata_sq <- mean_q_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_strata_sq <- mean_ev_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_by_strata_sq <- mean_ev_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(2:12, mean, na.rm = TRUE))

mean_cs_rate_by_strata_sq = edit_strata_df(mean_cs_rate_by_strata_sq, "cs")
mean_cs_rate_by_strata_sq$cs_change_rate = as.numeric(mean_cs_rate_by_strata_sq$cs_change_rate)

mean_r_rate_by_strata_sq = edit_strata_df(mean_r_rate_by_strata_sq, "r")
mean_r_rate_by_strata_sq$r_change_rate = as.numeric(mean_r_rate_by_strata_sq$r_change_rate)

mean_q_rate_by_strata_sq = edit_strata_df(mean_q_rate_by_strata_sq, "q")
mean_q_rate_by_strata_sq$q_change_rate = as.numeric(mean_q_rate_by_strata_sq$q_change_rate)

mean_ev_rate_by_strata_sq = edit_strata_df(mean_ev_rate_by_strata_sq, "ev")
mean_ev_rate_by_strata_sq$ev_perct = as.numeric(mean_ev_rate_by_strata_sq$ev_change_rate)
mean_ev_rate_by_strata_sq$ev_change_rate = NULL

mean_ev_by_strata_sq= edit_strata_df(mean_ev_by_strata_sq, "ev")
mean_ev_by_strata_sq$ev = as.numeric(mean_ev_by_strata_sq$ev_change_rate)
mean_ev_by_strata_sq$ev_change_rate = NULL

cs_avg = ggplot(mean_cs_rate_by_strata_sq , aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_cs_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "CS %",
    color = "Income Strata", 
    shape = "Income Strata") +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )


r_avg = ggplot(mean_r_rate_by_strata_sq , aes(x = step, y = r_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_r_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(0, 70))+
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "PS %",
    color = "Income Strata",
    shape = "Income Strata") +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

q_avg = ggplot(mean_q_rate_by_strata_sq , aes(x = step, y = q_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_q_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "Q %",
    color = "Income Strata",
    shape = "Income Strata") +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

ev_avg = ggplot(mean_ev_rate_by_strata_sq , aes(x = step, y = ev_perct, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_ev_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "EV/I (%)",
    color = NULL,
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )


mean_ev_rate_by_strata_sq$status = case_when(
  mean_ev_rate_by_strata_sq$step<0 ~ "low",
  mean_ev_rate_by_strata_sq$step>0 ~ "high"
)

mean_cs_rate_by_strata_sq$status = case_when(
  mean_cs_rate_by_strata_sq$step<0 ~ "low",
  mean_cs_rate_by_strata_sq$step>0 ~ "high"
)

sq_p_mean_ev = mean_ev_rate_by_strata_sq %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

sq_p_mean_cs = mean_cs_rate_by_strata_sq %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

shared_legend <- get_legend(r_avg)


# Arrange the three plots and the shared legend as the fourth item
sq_p_mean_grid  <- ggarrange(
  r_avg + theme(legend.position = "none"),
  q_avg + theme(legend.position = "none"),
  ev_avg + theme(legend.position = "none"),
  shared_legend,
  ncol = 2,
  nrow = 2,
  labels = c("PS", "Q", "EV", ""), # Use an empty label for the legend's spot
  font.label = list(size = 15, face = "bold"),
  label.x = 0.02, label.y = 0.98,
  hjust = 0, vjust = 1
)

# Now, add a title to this grid using patchwork's plot_annotation
sq_p_mean_grid_with_title <- sq_p_mean_grid +
  plot_annotation(
    title = "Linear Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )

##### sq_p_var ########

sq_p_var_cs_steps = read_csv("cs_detail_results/sq_p_var_cs_steps.csv")

colnames(sq_p_var_cs_steps) = as.character(as.numeric(colnames(sq_p_var_cs_steps))/20+0.75)

sq_p_var_cs_steps_rate <-sq_p_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(sq_p_var_cs_steps)

sq_p_var_q_steps = read_csv("cs_detail_results/sq_p_var_q_steps.csv")

colnames(sq_p_var_q_steps) = as.character(as.numeric(colnames(sq_p_var_q_steps))/20+0.75)

sq_p_var_q_steps_rate <-sq_p_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(sq_p_var_q_steps)

sq_p_var_r_steps = read_csv("cs_detail_results/sq_p_var_r_steps.csv")

colnames(sq_p_var_r_steps) = as.character(as.numeric(colnames(sq_p_var_r_steps))/20+0.75)

sq_p_var_r_steps_rate <-sq_p_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(sq_p_var_r_steps)

sq_p_var_ev_steps = read_csv("cs_detail_results/sq_p_var_ev_steps.csv")

colnames(sq_p_var_ev_steps) = as.character(as.numeric(colnames(sq_p_var_ev_steps))/20+0.75)

sq_p_var_ev_steps_rate <-sq_p_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(sq_p_var_ev_steps)

sq_p_var_cs_steps_rate  = cbind(sq_p_var_cs_steps_rate , demand_key)
sq_p_var_q_steps_rate = cbind(sq_p_var_q_steps_rate, demand_key)
sq_p_var_r_steps_rate = cbind(sq_p_var_r_steps_rate, demand_key)
sq_p_var_ev_steps_rate = cbind(sq_p_var_ev_steps_rate, demand_key)

#sq_p_var_cs_steps_rate = sq_p_var_cs_steps_rate[which(sq_p_var_cs_steps_rate$q_0<100),]
#sq_p_var_ev_steps_rate = sq_p_var_ev_steps_rate[which(sq_p_var_ev_steps_rate$q_0<100),]

var_cs_rate_by_bill_ym <- sq_p_var_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- sq_p_var_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- sq_p_var_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- sq_p_var_ev_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_by_bill_ym <- sq_p_var_ev_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(1:11, mean, na.rm = TRUE))

var_cs_rate_by_strata_sq <- var_cs_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_strata_sq <- var_r_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_strata_sq <- var_q_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_strata_sq <- var_ev_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_by_strata_sq <- var_ev_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(2:12, mean, na.rm = TRUE))

var_cs_rate_by_strata_sq = edit_strata_df(var_cs_rate_by_strata_sq, "cs")
var_cs_rate_by_strata_sq$cs_change_rate = as.numeric(var_cs_rate_by_strata_sq$cs_change_rate)

var_r_rate_by_strata_sq = edit_strata_df(var_r_rate_by_strata_sq, "r")
var_r_rate_by_strata_sq$r_change_rate = as.numeric(var_r_rate_by_strata_sq$r_change_rate)

var_q_rate_by_strata_sq = edit_strata_df(var_q_rate_by_strata_sq, "q")
var_q_rate_by_strata_sq$q_change_rate = as.numeric(var_q_rate_by_strata_sq$q_change_rate)

var_ev_rate_by_strata_sq = edit_strata_df(var_ev_rate_by_strata_sq, "ev")
var_ev_rate_by_strata_sq$ev_perct = as.numeric(var_ev_rate_by_strata_sq$ev_change_rate)
var_ev_rate_by_strata_sq$ev_change_rate = NULL

var_ev_by_strata_sq = edit_strata_df(var_ev_by_strata_sq, "ev")
var_ev_by_strata_sq$ev = as.numeric(var_ev_by_strata_sq$ev_change_rate)
var_ev_by_strata_sq$ev_change_rate = NULL

cs_avg = ggplot(var_cs_rate_by_strata_sq , aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_cs_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "CS %",
    color = "Income Strata", 
    shape = "Income Strata") +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )


r_avg = ggplot(var_r_rate_by_strata_sq , aes(x = step, y = r_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_r_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(0, 70))+
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "PS %",
    color = "Income Strata",
    shape = "Income Strata") +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

q_avg = ggplot(var_q_rate_by_strata_sq , aes(x = step, y = q_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_q_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "Q %",
    color = "Income Strata",
    shape = "Income Strata") +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

ev_avg = ggplot(var_ev_rate_by_strata_sq , aes(x = step, y = ev_perct, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_ev_rate_by_strata_sq, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "EV/I (%)",
    color = NULL,
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

var_ev_rate_by_strata_sq$status = case_when(
  var_ev_rate_by_strata_sq$step<1 ~ "low",
  var_ev_rate_by_strata_sq$step>1 ~ "high"
)

var_cs_rate_by_strata_sq$status = case_when(
  var_cs_rate_by_strata_sq$step<1 ~ "low",
  var_cs_rate_by_strata_sq$step>1 ~ "high"
)

sq_p_var_ev = var_ev_rate_by_strata_sq %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

sq_p_var_cs = var_cs_rate_by_strata_sq %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

shared_legend <- get_legend(r_avg)

# Arrange the three plots and the shared legend as the fourth item
sq_p_var_grid <- ggarrange(
  r_avg + theme(legend.position = "none"),
  q_avg + theme(legend.position = "none"),
  ev_avg + theme(legend.position = "none"),
  shared_legend,
  ncol = 2,
  nrow = 2,
  labels = c("PS", "Q", "EV", ""), # Use an empty label for the legend's spot
  font.label = list(size = 15, face = "bold"),
  label.x = 0.02, label.y = 0.98,
  hjust = 0, vjust = 1
)

# Now, add a title to this grid using patchwork's plot_annotation
sq_p_var_grid_with_title <- sq_p_var_grid +
  plot_annotation(
    title = "Linear Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )





