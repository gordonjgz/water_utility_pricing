setwd("C:/Users/Gordon Ji/Box Sync/Water_Data_Share/ramsey_welfare_result")
setwd("~/Library/CloudStorage/Box-Box/Water_Data_Share/ramsey_welfare_result")
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
library(lme4)
library(stargazer)
library(gganimate)
library(gifski)
library(magick)

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

demand_2018_using_new_small = read_csv("../demand_2018_using_new_small.csv")
demand_key <- demand_2018_using_new_small %>% select(bill_ym, prem_id, income, CAP_HH)
rm(demand_2018_using_new_small)

demand_key <- demand_key %>%
  mutate(income_strata = case_when(
    income < 6000 ~ "0~6k",
    income >= 6000 & income < 20000 ~ "6k~20k",
    income >= 20000 & income < 45000 ~ "20k~45k",
    income >= 45000 & income < 100000 ~ "45k~100k",
    income >= 100000 ~ ">100k"
  )) %>%
  mutate(income_strata = factor(income_strata, levels = c("0~6k", "6k~20k", "20k~45k", "45k~100k", ">100k")))

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


##### current_info_avg_bound_mean ########

current_info_avg_bound_mean_cs_steps = read_csv("cs_detail_results/current_info_avg_bound_mean_cs_steps.csv")
colnames(current_info_avg_bound_mean_cs_steps) = as.character(as.numeric(colnames(current_info_avg_bound_mean_cs_steps))/20-0.25)

current_info_avg_bound_mean_cs_steps_rate <-current_info_avg_bound_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_mean_cs_steps)

current_info_avg_bound_mean_q_steps = read_csv("cs_detail_results/current_info_avg_bound_mean_q_steps.csv")
colnames(current_info_avg_bound_mean_q_steps) = as.character(as.numeric(colnames(current_info_avg_bound_mean_q_steps))/20-0.25)

current_info_avg_bound_mean_q_steps_rate <-current_info_avg_bound_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_mean_q_steps)

current_info_avg_bound_mean_r_steps = read_csv("cs_detail_results/current_info_avg_bound_mean_r_steps.csv")
colnames(current_info_avg_bound_mean_r_steps) = as.character(as.numeric(colnames(current_info_avg_bound_mean_r_steps))/20-0.25)

current_info_avg_bound_mean_r_steps_rate <-current_info_avg_bound_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_mean_r_steps)

current_info_avg_bound_mean_cs_steps_rate  = cbind(current_info_avg_bound_mean_cs_steps_rate , demand_key)
current_info_avg_bound_mean_q_steps_rate = cbind(current_info_avg_bound_mean_q_steps_rate, demand_key)
current_info_avg_bound_mean_r_steps_rate = cbind(current_info_avg_bound_mean_r_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_avg_bound_mean_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_avg_bound_mean_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_avg_bound_mean_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_cs_rate_by_strata <- mean_cs_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_strata <- mean_r_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_strata <- mean_q_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_cs_rate_by_strata = edit_strata_df(mean_cs_rate_by_strata, "cs")
mean_cs_rate_by_strata$cs_change_rate = as.numeric(mean_cs_rate_by_strata$cs_change_rate)

mean_r_rate_by_strata = edit_strata_df(mean_r_rate_by_strata, "r")
mean_r_rate_by_strata$r_change_rate = as.numeric(mean_r_rate_by_strata$r_change_rate)

mean_q_rate_by_strata = edit_strata_df(mean_q_rate_by_strata, "q")
mean_q_rate_by_strata$q_change_rate = as.numeric(mean_q_rate_by_strata$q_change_rate)


cs = ggplot(mean_cs_rate_by_strata , aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_cs_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
       x = "Δ E[Prcp] (Inches)",
       y = "CS %",
       color = NULL, 
       shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

r = ggplot(mean_r_rate_by_strata , aes(x = step, y = r_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_r_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "Δ E[Prcp] (Inches)",
    y = "PS %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

q = ggplot(mean_q_rate_by_strata , aes(x = step, y = q_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_q_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "Δ E[Prcp] (Inches)",
    y = "Q %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

avg_bound_mean = ggarrange(cs, r,q, ncol = 2, nrow = 2, 
          labels = c("CS", "PS", "Q"),  # Adds labels to each plot
          font.label = list(size = 17, face = "bold"),  # Adjust label size
          #label.x = -0.01,  # Move label to the right
          #label.y = 1.01, # Move label slightly higher
          label.x = 0.9,  # Move label to the right
          label.y = 0.99,    # Keep label at the top
          hjust = 1, vjust = 1,  # Align to top-right corner
          common.legend = TRUE, 
          legend = "bottom")

##### current_info_avg_bound_var ########
cs_r_q_11 <- read_csv("cs_detail_results/cs_r_q_avg_11.csv", col_names = FALSE)
cs_11 = cs_r_q_11$X1
r_11 = cs_r_q_11$X2
q_11 = cs_r_q_11$X3

cs_r_q_115 <- read_csv("cs_detail_results/cs_r_q_avg_115.csv", col_names = FALSE)
cs_115 = cs_r_q_115$X1
r_115 = cs_r_q_115$X2
q_115 = cs_r_q_115$X3

cs_r_q_12 <- read_csv("cs_detail_results/cs_r_q_avg_12.csv", col_names = FALSE)
cs_12 = cs_r_q_12$X1
r_12 = cs_r_q_12$X2
q_12 = cs_r_q_12$X3

cs_r_q_125 <- read_csv("cs_detail_results/cs_r_q_avg_125.csv", col_names = FALSE)
cs_125 = cs_r_q_125$X1
r_125 = cs_r_q_125$X2
q_125 = cs_r_q_125$X3

current_info_avg_bound_var_cs_steps = read_csv("cs_detail_results/current_info_avg_bound_var_cs_steps.csv")
current_info_avg_bound_var_cs_steps$`7` = cs_11
current_info_avg_bound_var_cs_steps$`8` = cs_115
current_info_avg_bound_var_cs_steps$`9` = cs_12
current_info_avg_bound_var_cs_steps$`10` = cs_125

colnames(current_info_avg_bound_var_cs_steps) = as.character(as.numeric(colnames(current_info_avg_bound_var_cs_steps))/20+0.75)

current_info_avg_bound_var_cs_steps_rate <-current_info_avg_bound_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_var_cs_steps)

current_info_avg_bound_var_q_steps = read_csv("cs_detail_results/current_info_avg_bound_var_q_steps.csv")
current_info_avg_bound_var_q_steps$`7` = q_11
current_info_avg_bound_var_q_steps$`8` = q_115
current_info_avg_bound_var_q_steps$`9` = q_12
current_info_avg_bound_var_q_steps$`10` = q_125
colnames(current_info_avg_bound_var_q_steps) = as.character(as.numeric(colnames(current_info_avg_bound_var_q_steps))/20+0.75)

current_info_avg_bound_var_q_steps_rate <-current_info_avg_bound_var_q_steps %>%
  #mutate(across(everything(), ~ (. - demand_key$q_0_bill_ym) / demand_key$q_0_bill_ym*100, .names = "rate_{.col}"))
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
  
rm(current_info_avg_bound_var_q_steps)

current_info_avg_bound_var_r_steps = read_csv("cs_detail_results/current_info_avg_bound_var_r_steps.csv")
current_info_avg_bound_var_r_steps$`7` = r_11
current_info_avg_bound_var_r_steps$`8` = r_115
current_info_avg_bound_var_r_steps$`9` = r_12
current_info_avg_bound_var_r_steps$`10` = r_125
colnames(current_info_avg_bound_var_r_steps) = as.character(as.numeric(colnames(current_info_avg_bound_var_r_steps))/20+0.75)

current_info_avg_bound_var_r_steps_rate <-current_info_avg_bound_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_var_r_steps)

current_info_avg_bound_var_cs_steps_rate  = cbind(current_info_avg_bound_var_cs_steps_rate , demand_key)
current_info_avg_bound_var_q_steps_rate = cbind(current_info_avg_bound_var_q_steps_rate, demand_key)
current_info_avg_bound_var_r_steps_rate = cbind(current_info_avg_bound_var_r_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_avg_bound_var_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_avg_bound_var_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_avg_bound_var_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_cs_rate_by_strata <- var_cs_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_strata <- var_r_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_strata <- var_q_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_cs_rate_by_strata = edit_strata_df(var_cs_rate_by_strata, "cs")
var_cs_rate_by_strata$cs_change_rate = as.numeric(var_cs_rate_by_strata$cs_change_rate)

var_r_rate_by_strata = edit_strata_df(var_r_rate_by_strata, "r")
var_r_rate_by_strata$r_change_rate = as.numeric(var_r_rate_by_strata$r_change_rate)

var_q_rate_by_strata = edit_strata_df(var_q_rate_by_strata, "q")
var_q_rate_by_strata$q_change_rate = as.numeric(var_q_rate_by_strata$q_change_rate)


cs = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_cs_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_cartesian(ylim = c(-80, 50))+
  labs(
    x = "SD Ratio",
    y = "CS %",
    color = NULL, 
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

var_cs_rate_by_strata[var_cs_rate_by_strata$step<1,] %>% 
  group_by(income_strata) %>%
  summarise(cs_change_rate = mean(cs_change_rate))
# A tibble: 5 × 2
#income_strata cs_change_rate
#<fct>                  <dbl>
#  1 0~6k                    22.4
#2 6k~20k                  31.0
#3 20k~45k                 23.8
#4 45k~100k                26.6
#5 >100k                   15.5

# A tibble: 5 × 2
#income_strata cs_change_rate
#  <fct>                  <dbl>
#1 0~6k                   -31.5
#2 6k~20k                 -36.7
#3 20k~45k                -49.8
#4 45k~100k               -27.4
#5 >100k                  -25.2

r = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_r_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_cartesian(ylim = c(0, 60))+
  labs(
    x = "SD Ratio",
    y = "PS %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

q = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_q_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_cartesian(ylim = c(-7.5,10))+
  labs(
    x = "SD Ratio",
    y = "Q %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

avg_bound = ggarrange(cs, r,q, ncol = 2, nrow = 2, 
          labels = c("CS", "PS", "Q"),  # Adds labels to each plot
          font.label = list(size = 17, face = "bold"),  # Adjust label size
          #label.x = -0.01,  # Move label to the right
          #label.y = 1.01, # Move label slightly higher
          label.x = 0.9,  # Move label to the right
          label.y = 1,    # Keep label at the top
          hjust = 1, vjust = 1,  # Align to top-right corner
          common.legend = TRUE, 
          legend = "bottom")

ggplot(var_q_rate_by_bill_ym , aes(x =bill_ym, y = as.numeric(rate_0.85), color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes)+small_text

##### current_info_extreme_bound_mean ########

current_info_extreme_bound_mean_cs_steps = read_csv("cs_detail_results/current_info_extreme_bound_mean_cs_steps.csv")
colnames(current_info_extreme_bound_mean_cs_steps) = as.character(as.numeric(colnames(current_info_extreme_bound_mean_cs_steps))/20-0.25)

current_info_extreme_bound_mean_cs_steps_rate <-current_info_extreme_bound_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))

rm(current_info_extreme_bound_mean_cs_steps)

current_info_extreme_bound_mean_q_steps = read_csv("cs_detail_results/current_info_extreme_bound_mean_q_steps.csv")
colnames(current_info_extreme_bound_mean_q_steps) = as.character(as.numeric(colnames(current_info_extreme_bound_mean_q_steps))/20-0.25)

current_info_extreme_bound_mean_q_steps_rate <-current_info_extreme_bound_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))

rm(current_info_extreme_bound_mean_q_steps)

current_info_extreme_bound_mean_r_steps = read_csv("cs_detail_results/current_info_extreme_bound_mean_r_steps.csv")
colnames(current_info_extreme_bound_mean_r_steps) = as.character(as.numeric(colnames(current_info_extreme_bound_mean_r_steps))/20-0.25)

current_info_extreme_bound_mean_r_steps_rate <-current_info_extreme_bound_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))

rm(current_info_extreme_bound_mean_r_steps)

current_info_extreme_bound_mean_cs_steps_rate  = cbind(current_info_extreme_bound_mean_cs_steps_rate , demand_key)
current_info_extreme_bound_mean_q_steps_rate = cbind(current_info_extreme_bound_mean_q_steps_rate, demand_key)
current_info_extreme_bound_mean_r_steps_rate = cbind(current_info_extreme_bound_mean_r_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_extreme_bound_mean_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_extreme_bound_mean_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_extreme_bound_mean_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_cs_rate_by_strata <- mean_cs_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_strata <- mean_r_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_strata <- mean_q_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_cs_rate_by_strata = edit_strata_df(mean_cs_rate_by_strata, "cs")
mean_cs_rate_by_strata$cs_change_rate = as.numeric(mean_cs_rate_by_strata$cs_change_rate)

mean_r_rate_by_strata = edit_strata_df(mean_r_rate_by_strata, "r")
mean_r_rate_by_strata$r_change_rate = as.numeric(mean_r_rate_by_strata$r_change_rate)

mean_q_rate_by_strata = edit_strata_df(mean_q_rate_by_strata, "q")
mean_q_rate_by_strata$q_change_rate = as.numeric(mean_q_rate_by_strata$q_change_rate)


cs = ggplot(mean_cs_rate_by_strata , aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_cs_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "Δ E[Prcp] (Inches)",
    y = "CS %",
    color = NULL, 
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

r = ggplot(mean_r_rate_by_strata , aes(x = step, y = r_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_r_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "Δ E[Prcp] (Inches)",
    y = "PS %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

q = ggplot(mean_q_rate_by_strata , aes(x = step, y = q_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_q_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "Δ E[Prcp] (Inches)",
    y = "Q %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

ggarrange(cs, r,q, ncol = 2, nrow = 2, 
          labels = c("CS", "PS", "Q"),  # Adds labels to each plot
          font.label = list(size = 17, face = "bold"),  # Adjust label size
          #label.x = -0.01,  # Move label to the right
          #label.y = 1.01, # Move label slightly higher
          label.x = 0.9,  # Move label to the right
          label.y = 0.99,    # Keep label at the top
          hjust = 1, vjust = 1,  # Align to top-right corner
          common.legend = TRUE, 
          legend = "bottom")

##### current_info_extreme_bound_var ########
current_info_extreme_bound_var_cs_steps = read_csv("cs_detail_results/current_info_extreme_bound_var_cs_steps.csv")
colnames(current_info_extreme_bound_var_cs_steps) = as.character(as.numeric(colnames(current_info_extreme_bound_var_cs_steps))/20+0.75)

current_info_extreme_bound_var_cs_steps_rate <-current_info_extreme_bound_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))

rm(current_info_extreme_bound_var_cs_steps)

current_info_extreme_bound_var_q_steps = read_csv("cs_detail_results/current_info_extreme_bound_var_q_steps.csv")
colnames(current_info_extreme_bound_var_q_steps) = as.character(as.numeric(colnames(current_info_extreme_bound_var_q_steps))/20+0.75)

current_info_extreme_bound_var_q_steps_rate <-current_info_extreme_bound_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))

rm(current_info_extreme_bound_var_q_steps)

current_info_extreme_bound_var_r_steps = read_csv("cs_detail_results/current_info_extreme_bound_var_r_steps.csv")
colnames(current_info_extreme_bound_var_r_steps) = as.character(as.numeric(colnames(current_info_extreme_bound_var_r_steps))/20+0.75)

current_info_extreme_bound_var_r_steps_rate <-current_info_extreme_bound_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))

rm(current_info_extreme_bound_var_r_steps)

current_info_extreme_bound_var_cs_steps_rate  = cbind(current_info_extreme_bound_var_cs_steps_rate , demand_key)
current_info_extreme_bound_var_q_steps_rate = cbind(current_info_extreme_bound_var_q_steps_rate, demand_key)
current_info_extreme_bound_var_r_steps_rate = cbind(current_info_extreme_bound_var_r_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_extreme_bound_var_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_extreme_bound_var_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_extreme_bound_var_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_cs_rate_by_strata <- var_cs_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_strata <- var_r_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_strata <- var_q_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_cs_rate_by_strata = edit_strata_df(var_cs_rate_by_strata, "cs")
var_cs_rate_by_strata$cs_change_rate = as.numeric(var_cs_rate_by_strata$cs_change_rate)

var_r_rate_by_strata = edit_strata_df(var_r_rate_by_strata, "r")
var_r_rate_by_strata$r_change_rate = as.numeric(var_r_rate_by_strata$r_change_rate)

var_q_rate_by_strata = edit_strata_df(var_q_rate_by_strata, "q")
var_q_rate_by_strata$q_change_rate = as.numeric(var_q_rate_by_strata$q_change_rate)


cs = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_cs_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "SD Ratio",
    y = "CS %",
    color = NULL, 
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

r = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_r_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "SD Ratio",
    y = "PS %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

q = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_q_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "SD Ratio",
    y = "Q %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

extreme_bound = ggarrange(cs, r,q, ncol = 2, nrow = 2, 
          labels = c("CS", "PS", "Q"),  # Adds labels to each plot
          font.label = list(size = 17, face = "bold"),  # Adjust label size
          #label.x = -0.01,  # Move label to the right
          #label.y = 1.01, # Move label slightly higher
          label.x = 0.9,  # Move label to the right
          label.y = 1,    # Keep label at the top
          hjust = 1, vjust = 1,  # Align to top-right corner
          common.legend = TRUE, 
          legend = "bottom")

##### current_info_soft_bound_var ########
current_info_soft_bound_var_cs_steps = read_csv("cs_detail_results/current_info_soft_bound_var_cs_steps.csv")
colnames(current_info_soft_bound_var_cs_steps) = as.character(as.numeric(colnames(current_info_soft_bound_var_cs_steps))/20+0.75)

current_info_soft_bound_var_cs_steps_rate <-current_info_soft_bound_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))

rm(current_info_soft_bound_var_cs_steps)

current_info_soft_bound_var_q_steps = read_csv("cs_detail_results/current_info_soft_bound_var_q_steps.csv")
colnames(current_info_soft_bound_var_q_steps) = as.character(as.numeric(colnames(current_info_soft_bound_var_q_steps))/20+0.75)

current_info_soft_bound_var_q_steps_rate <-current_info_soft_bound_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))

rm(current_info_soft_bound_var_q_steps)

current_info_soft_bound_var_r_steps = read_csv("cs_detail_results/current_info_soft_bound_var_r_steps.csv")
colnames(current_info_soft_bound_var_r_steps) = as.character(as.numeric(colnames(current_info_soft_bound_var_r_steps))/20+0.75)

current_info_soft_bound_var_r_steps_rate <-current_info_soft_bound_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))

rm(current_info_soft_bound_var_r_steps)

current_info_soft_bound_var_cs_steps_rate  = cbind(current_info_soft_bound_var_cs_steps_rate , demand_key)
current_info_soft_bound_var_q_steps_rate = cbind(current_info_soft_bound_var_q_steps_rate, demand_key)
current_info_soft_bound_var_r_steps_rate = cbind(current_info_soft_bound_var_r_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_soft_bound_var_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_soft_bound_var_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_soft_bound_var_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_cs_rate_by_strata <- var_cs_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_strata <- var_r_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_strata <- var_q_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_cs_rate_by_strata = edit_strata_df(var_cs_rate_by_strata, "cs")
var_cs_rate_by_strata$cs_change_rate = as.numeric(var_cs_rate_by_strata$cs_change_rate)

var_r_rate_by_strata = edit_strata_df(var_r_rate_by_strata, "r")
var_r_rate_by_strata$r_change_rate = as.numeric(var_r_rate_by_strata$r_change_rate)

var_q_rate_by_strata = edit_strata_df(var_q_rate_by_strata, "q")
var_q_rate_by_strata$q_change_rate = as.numeric(var_q_rate_by_strata$q_change_rate)


cs = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_cs_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "SD Ratio",
    y = "CS %",
    color = NULL, 
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

r = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_r_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "SD Ratio",
    y = "PS %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

q = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_q_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "SD Ratio",
    y = "Q %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

r_alter = ggarrange(cs, r,q, ncol = 2, nrow = 2, 
          labels = c("CS", "PS", "Q"),  # Adds labels to each plot
          font.label = list(size = 17, face = "bold"),  # Adjust label size
          #label.x = -0.01,  # Move label to the right
          #label.y = 1.01, # Move label slightly higher
          label.x = 0.9,  # Move label to the right
          label.y = 1,    # Keep label at the top
          hjust = 1, vjust = 1,  # Align to top-right corner
          common.legend = TRUE, 
          legend = "bottom")

##### current_info_logr_bound_mean ########

current_info_logr_bound_mean_cs_steps = read_csv("cs_detail_results/current_info_logr_bound_mean_cs_steps.csv")
colnames(current_info_logr_bound_mean_cs_steps) = as.character(as.numeric(colnames(current_info_logr_bound_mean_cs_steps))/20-0.25)

current_info_logr_bound_mean_cs_steps_rate <-current_info_logr_bound_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))

rm(current_info_logr_bound_mean_cs_steps)

current_info_logr_bound_mean_q_steps = read_csv("cs_detail_results/current_info_logr_bound_mean_q_steps.csv")
colnames(current_info_logr_bound_mean_q_steps) = as.character(as.numeric(colnames(current_info_logr_bound_mean_q_steps))/20-0.25)

current_info_logr_bound_mean_q_steps_rate <-current_info_logr_bound_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))

rm(current_info_logr_bound_mean_q_steps)

current_info_logr_bound_mean_r_steps = read_csv("cs_detail_results/current_info_logr_bound_mean_r_steps.csv")
colnames(current_info_logr_bound_mean_r_steps) = as.character(as.numeric(colnames(current_info_logr_bound_mean_r_steps))/20-0.25)

current_info_logr_bound_mean_r_steps_rate <-current_info_logr_bound_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))

rm(current_info_logr_bound_mean_r_steps)

current_info_logr_bound_mean_cs_steps_rate  = cbind(current_info_logr_bound_mean_cs_steps_rate , demand_key)
current_info_logr_bound_mean_q_steps_rate = cbind(current_info_logr_bound_mean_q_steps_rate, demand_key)
current_info_logr_bound_mean_r_steps_rate = cbind(current_info_logr_bound_mean_r_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_logr_bound_mean_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_logr_bound_mean_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_logr_bound_mean_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_cs_rate_by_strata <- mean_cs_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_strata <- mean_r_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_strata <- mean_q_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_cs_rate_by_strata = edit_strata_df(mean_cs_rate_by_strata, "cs")
mean_cs_rate_by_strata$cs_change_rate = as.numeric(mean_cs_rate_by_strata$cs_change_rate)

mean_r_rate_by_strata = edit_strata_df(mean_r_rate_by_strata, "r")
mean_r_rate_by_strata$r_change_rate = as.numeric(mean_r_rate_by_strata$r_change_rate)

mean_q_rate_by_strata = edit_strata_df(mean_q_rate_by_strata, "q")
mean_q_rate_by_strata$q_change_rate = as.numeric(mean_q_rate_by_strata$q_change_rate)


cs = ggplot(mean_cs_rate_by_strata , aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_cs_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "Δ E[Prcp] (Inches)",
    y = "CS %",
    color = NULL, 
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

r = ggplot(mean_r_rate_by_strata , aes(x = step, y = r_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_r_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "Δ E[Prcp] (Inches)",
    y = "PS %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

q = ggplot(mean_q_rate_by_strata , aes(x = step, y = q_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_q_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "Δ E[Prcp] (Inches)",
    y = "Q %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

logr_bound_mean = ggarrange(cs, r,q, ncol = 2, nrow = 2, 
          labels = c("CS", "PS", "Q"),  # Adds labels to each plot
          font.label = list(size = 17, face = "bold"),  # Adjust label size
          #label.x = -0.01,  # Move label to the right
          #label.y = 1.01, # Move label slightly higher
          label.x = 0.9,  # Move label to the right
          label.y = 0.99,    # Keep label at the top
          hjust = 1, vjust = 1,  # Align to top-right corner
          common.legend = TRUE, 
          legend = "bottom")

##### current_info_logr_bound_var ########
cs_r_q_1 <- read_csv("cs_detail_results/cs_r_q_logr_1.csv", col_names = FALSE)
cs_1 = cs_r_q_1$X1
r_1 = cs_r_q_1$X2
q_1 = cs_r_q_1$X3

cs_r_q_105 <- read_csv("cs_detail_results/cs_r_q_logr_105.csv", col_names = FALSE)
cs_105 = cs_r_q_105$X1
r_105 = cs_r_q_105$X2
q_105 = cs_r_q_105$X3

cs_r_q_11 <- read_csv("cs_detail_results/cs_r_q_logr_11.csv", col_names = FALSE)
cs_11 = cs_r_q_11$X1
r_11 = cs_r_q_11$X2
q_11 = cs_r_q_11$X3

current_info_logr_bound_var_cs_steps = read_csv("cs_detail_results/current_info_logr_bound_var_cs_steps.csv")
current_info_logr_bound_var_cs_steps$`5` = cs_1
current_info_logr_bound_var_cs_steps$`6` = cs_105
current_info_logr_bound_var_cs_steps$`7` = cs_11
colnames(current_info_logr_bound_var_cs_steps) = as.character(as.numeric(colnames(current_info_logr_bound_var_cs_steps))/20+0.75)

current_info_logr_bound_var_cs_steps_rate <-current_info_logr_bound_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))

rm(current_info_logr_bound_var_cs_steps)

current_info_logr_bound_var_q_steps = read_csv("cs_detail_results/current_info_logr_bound_var_q_steps.csv")
current_info_logr_bound_var_q_steps$`5` = q_1
current_info_logr_bound_var_q_steps$`6` = q_105
current_info_logr_bound_var_q_steps$`7` = q_11
colnames(current_info_logr_bound_var_q_steps) = as.character(as.numeric(colnames(current_info_logr_bound_var_q_steps))/20+0.75)

current_info_logr_bound_var_q_steps_rate <-current_info_logr_bound_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))

rm(current_info_logr_bound_var_q_steps)

current_info_logr_bound_var_r_steps = read_csv("cs_detail_results/current_info_logr_bound_var_r_steps.csv")
current_info_logr_bound_var_r_steps$`5` = r_1
current_info_logr_bound_var_r_steps$`6` = r_105
current_info_logr_bound_var_r_steps$`7` = r_11
colnames(current_info_logr_bound_var_r_steps) = as.character(as.numeric(colnames(current_info_logr_bound_var_r_steps))/20+0.75)

current_info_logr_bound_var_r_steps_rate <-current_info_logr_bound_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))

rm(current_info_logr_bound_var_r_steps)

current_info_logr_bound_var_cs_steps_rate  = cbind(current_info_logr_bound_var_cs_steps_rate , demand_key)
current_info_logr_bound_var_q_steps_rate = cbind(current_info_logr_bound_var_q_steps_rate, demand_key)
current_info_logr_bound_var_r_steps_rate = cbind(current_info_logr_bound_var_r_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_logr_bound_var_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_logr_bound_var_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_logr_bound_var_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_cs_rate_by_strata <- var_cs_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_strata <- var_r_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_strata <- var_q_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_cs_rate_by_strata = edit_strata_df(var_cs_rate_by_strata, "cs")
var_cs_rate_by_strata$cs_change_rate = as.numeric(var_cs_rate_by_strata$cs_change_rate)

var_r_rate_by_strata = edit_strata_df(var_r_rate_by_strata, "r")
var_r_rate_by_strata$r_change_rate = as.numeric(var_r_rate_by_strata$r_change_rate)

var_q_rate_by_strata = edit_strata_df(var_q_rate_by_strata, "q")
var_q_rate_by_strata$q_change_rate = as.numeric(var_q_rate_by_strata$q_change_rate)


cs = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_cs_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "SD Ratio",
    y = "CS %",
    color = NULL, 
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  coord_cartesian(ylim = c(-80, 50))+
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

var_cs_rate_by_strata[var_cs_rate_by_strata$step<1,] %>% 
  group_by(income_strata) %>%
  summarise(cs_change_rate = mean(cs_change_rate))
#income_strata cs_change_rate
#<fct>                  <dbl>
#  1 0~6k                   20.9 
#2 6k~20k                 14.9 
#3 20k~45k                 9.56
#4 45k~100k               26.5 
#5 >100k                  21.4 

#income_strata cs_change_rate
#<fct>                  <dbl>
#1 0~6k                   -31.3
#2 6k~20k                 -20.3
#3 20k~45k                -28.7
#4 45k~100k               -28.8
#5 >100k                  -47.8

r = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_r_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_cartesian(ylim = c(0, 60))+
  labs(
    x = "SD Ratio",
    y = "PS %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

q = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_q_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_cartesian(ylim = c(-10,10))+
  labs(
    x = "SD Ratio",
    y = "Q %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

logr_bound = ggarrange(cs, r,q, ncol = 2, nrow = 2, 
                          labels = c("CS", "PS", "Q"),  # Adds labels to each plot
                          font.label = list(size = 17, face = "bold"),  # Adjust label size
                          #label.x = -0.01,  # Move label to the right
                          #label.y = 1.01, # Move label slightly higher
                          label.x = 0.9,  # Move label to the right
                          label.y = 1,    # Keep label at the top
                          hjust = 1, vjust = 1,  # Align to top-right corner
                          common.legend = TRUE, 
                          legend = "bottom")


##### current_info_gamma05_bound_var ########
cs_r_q <- read_csv("cs_detail_results/cs_r_q_gamma05_115.csv", col_names = FALSE)
cs_115 = cs_r_q$X1
r_115 = cs_r_q$X2
q_115 = cs_r_q$X3
current_info_gamma05_bound_var_cs_steps = read_csv("cs_detail_results/current_info_gamma05_bound_var_cs_steps.csv")
current_info_gamma05_bound_var_cs_steps$`8` = cs_115

colnames(current_info_gamma05_bound_var_cs_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_var_cs_steps))/20+0.75)

current_info_gamma05_bound_var_cs_steps_rate <-current_info_gamma05_bound_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_var_cs_steps)

current_info_gamma05_bound_var_q_steps = read_csv("cs_detail_results/current_info_gamma05_bound_var_q_steps.csv")
current_info_gamma05_bound_var_q_steps$`8` = q_115

colnames(current_info_gamma05_bound_var_q_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_var_q_steps))/20+0.75)

current_info_gamma05_bound_var_q_steps_rate <-current_info_gamma05_bound_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_var_q_steps)

current_info_gamma05_bound_var_r_steps = read_csv("cs_detail_results/current_info_gamma05_bound_var_r_steps.csv")
current_info_gamma05_bound_var_r_steps$`8` = r_115

colnames(current_info_gamma05_bound_var_r_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_var_r_steps))/20+0.75)

current_info_gamma05_bound_var_r_steps_rate <-current_info_gamma05_bound_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_var_r_steps)

current_info_gamma05_bound_var_cs_steps_rate  = cbind(current_info_gamma05_bound_var_cs_steps_rate , demand_key)
current_info_gamma05_bound_var_q_steps_rate = cbind(current_info_gamma05_bound_var_q_steps_rate, demand_key)
current_info_gamma05_bound_var_r_steps_rate = cbind(current_info_gamma05_bound_var_r_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_gamma05_bound_var_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_gamma05_bound_var_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_gamma05_bound_var_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_cs_rate_by_strata <- var_cs_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_strata <- var_r_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_strata <- var_q_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_cs_rate_by_strata = edit_strata_df(var_cs_rate_by_strata, "cs")
var_cs_rate_by_strata$cs_change_rate = as.numeric(var_cs_rate_by_strata$cs_change_rate)

var_r_rate_by_strata = edit_strata_df(var_r_rate_by_strata, "r")
var_r_rate_by_strata$r_change_rate = as.numeric(var_r_rate_by_strata$r_change_rate)

var_q_rate_by_strata = edit_strata_df(var_q_rate_by_strata, "q")
var_q_rate_by_strata$q_change_rate = as.numeric(var_q_rate_by_strata$q_change_rate)


cs = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_cs_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "SD Ratio",
    y = "CS %",
    color = NULL, 
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  coord_cartesian(ylim = c(-80, 50))+
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

var_cs_rate_by_strata[var_cs_rate_by_strata$step>1,] %>% 
  group_by(income_strata) %>%
  summarise(cs_change_rate = mean(cs_change_rate))
# A tibble: 5 × 2
#income_strata cs_change_rate
#<fct>                  <dbl>
#  1 0~6k                    22.2
#2 6k~20k                  30.3
#3 20k~45k                 23.3
#4 45k~100k                26.0
#5 >100k                   20.2

#income_strata cs_change_rate
#<fct>                  <dbl>
#1 0~6k                   -31.6
#2 6k~20k                 -27.6
#3 20k~45k                -35.3
#4 45k~100k               -28.7
#5 >100k                  -38.4

r = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_r_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_cartesian(ylim = c(0, 70))+
  labs(
    x = "SD Ratio",
    y = "PS %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

q = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_q_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = "SD Ratio",
    y = "Q %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

gamma05_bound = ggarrange(cs, r,q, ncol = 2, nrow = 2, 
                       labels = c("CS", "PS", "Q"),  # Adds labels to each plot
                       font.label = list(size = 17, face = "bold"),  # Adjust label size
                       #label.x = -0.01,  # Move label to the right
                       #label.y = 1.01, # Move label slightly higher
                       label.x = 0.9,  # Move label to the right
                       label.y = 1,    # Keep label at the top
                       hjust = 1, vjust = 1,  # Align to top-right corner
                       common.legend = TRUE, 
                       legend = "bottom")
