setwd("C:/Users/Gordon Ji/Box Sync/Water_Data_Share/ramsey_welfare_result")
setwd("~/Library/CloudStorage/Box-Box/Water_Data_Share/ramsey_welfare_result")
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
rm(demand_2018_using_new_small)

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

# A tibble: 5 Ã 3
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


academic_theme <- theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 10), # Added legend title styling
    legend.text = element_text(size = 9),
    axis.title.x = element_text(size = 11, face = "bold", margin = margin(t = 10)), # Bold, slightly larger, margin
    axis.title.y = element_text(size = 11, face = "bold", margin = margin(r = 10)), # Bold, slightly larger, margin
    axis.text = element_text(size = 9),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
    panel.grid.minor = element_blank(), # Typically remove minor grids
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5) # Add subtle border
  )


##### current_info_avg_bound_loss1_mean ########

current_info_avg_bound_loss1_mean_cs_steps = read_csv("cs_detail_results/current_info_avg_bound_loss1_mean_cs_steps.csv")

colnames(current_info_avg_bound_loss1_mean_cs_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss1_mean_cs_steps))*0.05-0.25)

current_info_avg_bound_loss1_mean_cs_steps_rate <-current_info_avg_bound_loss1_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss1_mean_cs_steps)

current_info_avg_bound_loss1_mean_q_steps = read_csv("cs_detail_results/current_info_avg_bound_loss1_mean_q_steps.csv")

colnames(current_info_avg_bound_loss1_mean_q_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss1_mean_q_steps))*0.05-0.25)

current_info_avg_bound_loss1_mean_q_steps_rate <-current_info_avg_bound_loss1_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss1_mean_q_steps)

current_info_avg_bound_loss1_mean_r_steps = read_csv("cs_detail_results/current_info_avg_bound_loss1_mean_r_steps.csv")

colnames(current_info_avg_bound_loss1_mean_r_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss1_mean_r_steps))*0.05-0.25)

current_info_avg_bound_loss1_mean_r_steps_rate <-current_info_avg_bound_loss1_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss1_mean_r_steps)

current_info_avg_bound_loss1_mean_ev_steps = read_csv("cs_detail_results/current_info_avg_bound_loss1_mean_ev_steps.csv")

colnames(current_info_avg_bound_loss1_mean_ev_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss1_mean_ev_steps))*0.05-0.25)

current_info_avg_bound_loss1_mean_ev_steps_rate <-current_info_avg_bound_loss1_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss1_mean_ev_steps)

current_info_avg_bound_loss1_mean_cs_steps_rate  = cbind(current_info_avg_bound_loss1_mean_cs_steps_rate , demand_key)
current_info_avg_bound_loss1_mean_q_steps_rate = cbind(current_info_avg_bound_loss1_mean_q_steps_rate, demand_key)
current_info_avg_bound_loss1_mean_r_steps_rate = cbind(current_info_avg_bound_loss1_mean_r_steps_rate, demand_key)
current_info_avg_bound_loss1_mean_ev_steps_rate = cbind(current_info_avg_bound_loss1_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_avg_bound_loss1_mean_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_avg_bound_loss1_mean_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_avg_bound_loss1_mean_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- current_info_avg_bound_loss1_mean_ev_steps_rate %>%
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

mean_ev_rate_by_strata <- mean_ev_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_cs_rate_by_strata = edit_strata_df(mean_cs_rate_by_strata, "cs")
mean_cs_rate_by_strata$cs_change_rate = as.numeric(mean_cs_rate_by_strata$cs_change_rate)

mean_r_rate_by_strata = edit_strata_df(mean_r_rate_by_strata, "r")
mean_r_rate_by_strata$r_change_rate = as.numeric(mean_r_rate_by_strata$r_change_rate)

mean_q_rate_by_strata = edit_strata_df(mean_q_rate_by_strata, "q")
mean_q_rate_by_strata$q_change_rate = as.numeric(mean_q_rate_by_strata$q_change_rate)

mean_ev_rate_by_strata = edit_strata_df(mean_ev_rate_by_strata, "ev")
mean_ev_rate_by_strata$ev_perct = as.numeric(mean_ev_rate_by_strata$ev_change_rate)
mean_ev_rate_by_strata$ev_change_rate = NULL

cs_avg = ggplot(mean_cs_rate_by_strata , aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_cs_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "CS %",
    color = NULL, 
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )


r_avg = ggplot(mean_r_rate_by_strata , aes(x = step, y = r_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_r_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(0, 70))+
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "R %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

q_avg = ggplot(mean_q_rate_by_strata , aes(x = step, y = q_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_q_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "q %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

ev_avg = ggplot(mean_ev_rate_by_strata , aes(x = step, y = ev_perct, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_ev_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  #geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "EV/I (%)",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

avg_bound_loss1_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct))

avg_bound_loss1_mean_grid <- ggarrange(cs_avg,r_avg,q_avg,ev_avg, ncol = 2, nrow = 2, 
                            labels = c("CS", 
                                       "PS", 
                                       "Q",
                                       "EV"),  # Adds labels to each plot
                            font.label = list(size = 15, face = "bold"),
                            label.x = 0.02, label.y = 0.98,
                            hjust = 0, vjust = 1,
                            common.legend = FALSE, # IMPORTANT: No legend here
                            legend = "none")

# Now, add a title to this grid using patchwork's plot_annotation
avg_bound_loss1_mean_grid_with_title <- avg_bound_loss1_mean_grid +
  plot_annotation(
    title = "Linear Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )

##### current_info_gamma1_bound_loss1_mean ########

current_info_gamma1_bound_loss1_mean_cs_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss1_mean_cs_steps.csv")

colnames(current_info_gamma1_bound_loss1_mean_cs_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss1_mean_cs_steps))*0.05-0.25)

current_info_gamma1_bound_loss1_mean_cs_steps_rate <-current_info_gamma1_bound_loss1_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss1_mean_cs_steps)

current_info_gamma1_bound_loss1_mean_q_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss1_mean_q_steps.csv")

colnames(current_info_gamma1_bound_loss1_mean_q_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss1_mean_q_steps))*0.05-0.25)

current_info_gamma1_bound_loss1_mean_q_steps_rate <-current_info_gamma1_bound_loss1_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss1_mean_q_steps)

current_info_gamma1_bound_loss1_mean_r_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss1_mean_r_steps.csv")

colnames(current_info_gamma1_bound_loss1_mean_r_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss1_mean_r_steps))*0.05-0.25)

current_info_gamma1_bound_loss1_mean_r_steps_rate <-current_info_gamma1_bound_loss1_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss1_mean_r_steps)

current_info_gamma1_bound_loss1_mean_ev_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss1_mean_ev_steps.csv")

colnames(current_info_gamma1_bound_loss1_mean_ev_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss1_mean_ev_steps))*0.05-0.25)

current_info_gamma1_bound_loss1_mean_ev_steps_rate <-current_info_gamma1_bound_loss1_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss1_mean_ev_steps)

current_info_gamma1_bound_loss1_mean_cs_steps_rate  = cbind(current_info_gamma1_bound_loss1_mean_cs_steps_rate , demand_key)
current_info_gamma1_bound_loss1_mean_q_steps_rate = cbind(current_info_gamma1_bound_loss1_mean_q_steps_rate, demand_key)
current_info_gamma1_bound_loss1_mean_r_steps_rate = cbind(current_info_gamma1_bound_loss1_mean_r_steps_rate, demand_key)
current_info_gamma1_bound_loss1_mean_ev_steps_rate = cbind(current_info_gamma1_bound_loss1_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_gamma1_bound_loss1_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_gamma1_bound_loss1_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_gamma1_bound_loss1_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- current_info_gamma1_bound_loss1_mean_ev_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_cs_rate_by_strata <- mean_cs_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_strata <- mean_r_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_strata <- mean_q_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_strata <- mean_ev_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))


#mean_cs_rate_by_strata = edit_strata_df_quantity(mean_cs_rate_by_strata, "cs")
mean_cs_rate_by_strata = edit_strata_df(mean_cs_rate_by_strata, "cs")
mean_cs_rate_by_strata$cs_change_rate = as.numeric(mean_cs_rate_by_strata$cs_change_rate)

#mean_r_rate_by_strata = edit_strata_df_quantity(mean_r_rate_by_strata, "r")
mean_r_rate_by_strata = edit_strata_df(mean_r_rate_by_strata, "r")
mean_r_rate_by_strata$r_change_rate = as.numeric(mean_r_rate_by_strata$r_change_rate)

#mean_q_rate_by_strata = edit_strata_df_quantity(mean_q_rate_by_strata, "q")
mean_q_rate_by_strata = edit_strata_df(mean_q_rate_by_strata, "q")
mean_q_rate_by_strata$q_change_rate = as.numeric(mean_q_rate_by_strata$q_change_rate)

#mean_ev_rate_by_strata = edit_strata_df_quantity(mean_ev_rate_by_strata, "ev")
mean_ev_rate_by_strata = edit_strata_df(mean_ev_rate_by_strata, "ev")
mean_ev_rate_by_strata$ev_perct = as.numeric(mean_ev_rate_by_strata$ev_change_rate)
mean_ev_rate_by_strata$ev_change_rate = NULL

cs_gamma1 = ggplot(mean_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
                                        color = income_strata, shape = income_strata
                                        #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  #geom_point(data = subset(mean_cs_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_point(data = subset(mean_cs_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "CS %",
    color = NULL, 
    shape = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  #coord_cartesian(ylim = c(-80, 50))+
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )


r_gamma1 = ggplot(mean_r_rate_by_strata , aes(x = step, y = r_change_rate, 
                                      color = income_strata, shape = income_strata
                                      #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_r_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_point(data = subset(mean_r_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(0, 70))+
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "PS %",
    color = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

q_gamma1 = ggplot(mean_q_rate_by_strata , aes(x = step, y = q_change_rate, 
                                      color = income_strata, shape = income_strata
                                      #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_q_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_point(data = subset(mean_q_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "Q %",
    color = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

ev_gamma1 = ggplot(mean_ev_rate_by_strata , aes(x = step, y = ev_perct, 
                                        color = income_strata, shape = income_strata
                                        #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_ev_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_point(data = subset(mean_ev_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  #geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "EV/I (%)",
    color = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

gamma1_bound_loss1_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct))

gamma1_bound_loss1_mean_grid = ggarrange(cs_gamma1,r_gamma1,q_gamma1,ev_gamma1, ncol = 2, nrow = 2, 
                               labels = c("CS", 
                                          "PS", 
                                          "Q",
                                          "EV"),  # Adds labels to each plot
                               font.label = list(size = 15, face = "bold"),
                               label.x = 0.02, label.y = 0.98,
                               hjust = 0, vjust = 1,
                               common.legend = FALSE, # IMPORTANT: No legend here
                               legend = "none")

# Now, add a title to this grid using patchwork's plot_annotation
gamma1_bound_loss1_mean_grid_with_title <- gamma1_bound_loss1_mean_grid +
  plot_annotation(
    title = "Concave Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )

mean_cs_rate_by_strata[mean_cs_rate_by_strata$step>1,] %>% 
  group_by(income_strata) %>%
  summarise(cs_change_rate = mean(cs_change_rate))
# A tibble: 5 x 2
#income_strata cs_change_rate
#<fct>                  <dbl>
#1 0~6k                    21.4
#2 6k~20k                  23.1
#3 20k~45k                 17.5
#4 45k~100k                24.8
#5 >100k                   22.8

# A tibble: 5 x 2
#income_strata cs_change_rate
#<fct>                  <dbl>
#  1 0~6k                   -31.0
#2 6k~20k                 -35.1
#3 20k~45k                -43.4
#4 45k~100k               -26.5
#5 >100k                  -31.1


# Step 1: Create a temporary plot with visible content
legend_plot <- ggplot(mean_cs_rate_by_strata, aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, stroke = 1) +
  labs(color = "Income Strata", shape = "Income Strata") +
  scale_color_manual(values = custom_colors) +
  scale_shape_manual(values = custom_shapes) +
  guides(
    color = guide_legend(override.aes = list(linewidth = 1, size = 3, shape = custom_shapes)),
    shape = "none"  # combine shape into color legend
  ) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 13),
    legend.text = element_text(size = 12)
  )

plot_grob <- ggplotGrob(legend_plot)

legend_index <- which(sapply(plot_grob$grobs, function(x) x$name) == "guide-box")

common_legend <- plot_grob$grobs[[legend_index]]

# --- Combine the two grids with the common legend ---

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss1_mean_grid_with_title, gamma1_bound_loss1_mean_grid_with_title,
                                              ncol = 2, nrow = 1,
                                              # Add labels for the overall columns
                                              labels = c("Scenario 1: Mean Loss Impact", "Scenario 2: Gamma 1 Loss Impact"),
                                              font.label = list(size = 18, face = "bold", color = "black"),
                                              label.x = 0, # Position label at the start of the column (relative to grid)
                                              label.y = 1.05, # Position label slightly above the plot area
                                              hjust = 0, vjust = 0, # Align to top-left of the label's "box"
                                              common.legend = T, # No common legend at this stage
                                              legend = "none")

# Now, combine the combined grids with the extracted common legend
avg_gamma1_mean <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                      ncol = 1, nrow = 2,
                                      heights = c(1, 0.2)) # Adjust heights as needed
grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))





##### current_info_avg_bound_loss1_var ########

current_info_avg_bound_loss1_var_cs_steps = read_csv("cs_detail_results/current_info_avg_bound_loss1_var_cs_steps.csv")

colnames(current_info_avg_bound_loss1_var_cs_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss1_var_cs_steps))/20+0.75)

current_info_avg_bound_loss1_var_cs_steps_rate <-current_info_avg_bound_loss1_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss1_var_cs_steps)

current_info_avg_bound_loss1_var_q_steps = read_csv("cs_detail_results/current_info_avg_bound_loss1_var_q_steps.csv")

colnames(current_info_avg_bound_loss1_var_q_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss1_var_q_steps))/20+0.75)

current_info_avg_bound_loss1_var_q_steps_rate <-current_info_avg_bound_loss1_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss1_var_q_steps)

current_info_avg_bound_loss1_var_r_steps = read_csv("cs_detail_results/current_info_avg_bound_loss1_var_r_steps.csv")

colnames(current_info_avg_bound_loss1_var_r_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss1_var_r_steps))/20+0.75)

current_info_avg_bound_loss1_var_r_steps_rate <-current_info_avg_bound_loss1_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss1_var_r_steps)

current_info_avg_bound_loss1_var_ev_steps = read_csv("cs_detail_results/current_info_avg_bound_loss1_var_ev_steps.csv")

colnames(current_info_avg_bound_loss1_var_ev_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss1_var_ev_steps))/20+0.75)

current_info_avg_bound_loss1_var_ev_steps_rate <-current_info_avg_bound_loss1_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss1_var_ev_steps)

current_info_avg_bound_loss1_var_cs_steps_rate  = cbind(current_info_avg_bound_loss1_var_cs_steps_rate , demand_key)
current_info_avg_bound_loss1_var_q_steps_rate = cbind(current_info_avg_bound_loss1_var_q_steps_rate, demand_key)
current_info_avg_bound_loss1_var_r_steps_rate = cbind(current_info_avg_bound_loss1_var_r_steps_rate, demand_key)
current_info_avg_bound_loss1_var_ev_steps_rate = cbind(current_info_avg_bound_loss1_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_avg_bound_loss1_var_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_avg_bound_loss1_var_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_avg_bound_loss1_var_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_avg_bound_loss1_var_ev_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
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

var_ev_rate_by_strata <- var_ev_rate_by_bill_ym %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_cs_rate_by_strata = edit_strata_df(var_cs_rate_by_strata, "cs")
var_cs_rate_by_strata$cs_change_rate = as.numeric(var_cs_rate_by_strata$cs_change_rate)

var_r_rate_by_strata = edit_strata_df(var_r_rate_by_strata, "r")
var_r_rate_by_strata$r_change_rate = as.numeric(var_r_rate_by_strata$r_change_rate)

var_q_rate_by_strata = edit_strata_df(var_q_rate_by_strata, "q")
var_q_rate_by_strata$q_change_rate = as.numeric(var_q_rate_by_strata$q_change_rate)

var_ev_rate_by_strata = edit_strata_df(var_ev_rate_by_strata, "ev")
var_ev_rate_by_strata$ev_perct = as.numeric(var_ev_rate_by_strata$ev_change_rate)
var_ev_rate_by_strata$ev_change_rate = NULL

cs_avg = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_cs_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "CS %",
    color = NULL, 
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )


r_avg = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_r_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(0, 70))+
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "PS %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

q_avg = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_q_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "Q %",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

ev_avg = ggplot(var_ev_rate_by_strata , aes(x = step, y = ev_perct, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_ev_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  #geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "EV/I (%)",
    color = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

avg_bound_loss1_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct))

avg_bound_loss1_var_grid <- ggarrange(cs_avg,r_avg,q_avg,ev_avg, ncol = 2, nrow = 2, 
                                       labels = c("CS", 
                                                  "PS", 
                                                  "Q",
                                                  "EV"),  # Adds labels to each plot
                                       font.label = list(size = 15, face = "bold"),
                                       label.x = 0.02, label.y = 0.98,
                                       hjust = 0, vjust = 1,
                                       common.legend = FALSE, # IMPORTANT: No legend here
                                       legend = "none")

# Now, add a title to this grid using patchwork's plot_annotation
avg_bound_loss1_var_grid_with_title <- avg_bound_loss1_var_grid +
  plot_annotation(
    title = "Linear Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )

##### current_info_gamma1_bound_loss1_var ########

current_info_gamma1_bound_loss1_var_cs_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss1_var_cs_steps.csv")

colnames(current_info_gamma1_bound_loss1_var_cs_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss1_var_cs_steps))/20+0.75)

current_info_gamma1_bound_loss1_var_cs_steps_rate <-current_info_gamma1_bound_loss1_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss1_var_cs_steps)

current_info_gamma1_bound_loss1_var_q_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss1_var_q_steps.csv")

colnames(current_info_gamma1_bound_loss1_var_q_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss1_var_q_steps))/20+0.75)

current_info_gamma1_bound_loss1_var_q_steps_rate <-current_info_gamma1_bound_loss1_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss1_var_q_steps)

current_info_gamma1_bound_loss1_var_r_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss1_var_r_steps.csv")

colnames(current_info_gamma1_bound_loss1_var_r_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss1_var_r_steps))/20+0.75)

current_info_gamma1_bound_loss1_var_r_steps_rate <-current_info_gamma1_bound_loss1_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss1_var_r_steps)

current_info_gamma1_bound_loss1_var_ev_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss1_var_ev_steps.csv")

colnames(current_info_gamma1_bound_loss1_var_ev_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss1_var_ev_steps))/20+0.75)

current_info_gamma1_bound_loss1_var_ev_steps_rate <-current_info_gamma1_bound_loss1_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss1_var_ev_steps)

current_info_gamma1_bound_loss1_var_cs_steps_rate  = cbind(current_info_gamma1_bound_loss1_var_cs_steps_rate , demand_key)
current_info_gamma1_bound_loss1_var_q_steps_rate = cbind(current_info_gamma1_bound_loss1_var_q_steps_rate, demand_key)
current_info_gamma1_bound_loss1_var_r_steps_rate = cbind(current_info_gamma1_bound_loss1_var_r_steps_rate, demand_key)
current_info_gamma1_bound_loss1_var_ev_steps_rate = cbind(current_info_gamma1_bound_loss1_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_gamma1_bound_loss1_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_gamma1_bound_loss1_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_gamma1_bound_loss1_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_gamma1_bound_loss1_var_ev_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_cs_rate_by_strata <- var_cs_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_strata <- var_r_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_strata <- var_q_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_strata <- var_ev_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

#var_cs_rate_by_strata = edit_strata_df_quantity(var_cs_rate_by_strata, "cs")
var_cs_rate_by_strata = edit_strata_df(var_cs_rate_by_strata, "cs")
var_cs_rate_by_strata$cs_change_rate = as.numeric(var_cs_rate_by_strata$cs_change_rate)

#var_r_rate_by_strata = edit_strata_df_quantity(var_r_rate_by_strata, "r")
var_r_rate_by_strata = edit_strata_df(var_r_rate_by_strata, "r")
var_r_rate_by_strata$r_change_rate = as.numeric(var_r_rate_by_strata$r_change_rate)

#var_q_rate_by_strata = edit_strata_df_quantity(var_q_rate_by_strata, "q")
var_q_rate_by_strata = edit_strata_df(var_q_rate_by_strata, "q")
var_q_rate_by_strata$q_change_rate = as.numeric(var_q_rate_by_strata$q_change_rate)

#var_ev_rate_by_strata = edit_strata_df_quantity(var_ev_rate_by_strata, "ev")
var_ev_rate_by_strata = edit_strata_df(var_ev_rate_by_strata, "ev")
var_ev_rate_by_strata$ev_perct = as.numeric(var_ev_rate_by_strata$ev_change_rate)
var_ev_rate_by_strata$ev_change_rate = NULL

cs_gamma1 = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
                                                color = income_strata, shape = income_strata
                                                #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  #geom_point(data = subset(var_cs_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_point(data = subset(var_cs_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "CS %",
    color = NULL, 
    shape = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  #coord_cartesian(ylim = c(-80, 50))+
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )


r_gamma1 = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, 
                                              color = income_strata, shape = income_strata
                                              #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_r_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_point(data = subset(var_r_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(0, 70))+
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "PS %",
    color = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

q_gamma1 = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, 
                                              color = income_strata, shape = income_strata
                                              #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_q_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_point(data = subset(var_q_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "Q %",
    color = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

ev_gamma1 = ggplot(var_ev_rate_by_strata , aes(x = step, y = ev_perct, 
                                                color = income_strata, shape = income_strata
                                                #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_ev_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_point(data = subset(var_ev_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  #geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "EV/I (%)",
    color = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

gamma1_bound_loss1_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct))

gamma1_bound_loss1_var_grid = ggarrange(cs_gamma1,r_gamma1,q_gamma1,ev_gamma1, ncol = 2, nrow = 2, 
                                         labels = c("CS", 
                                                    "PS", 
                                                    "Q",
                                                    "EV"),  # Adds labels to each plot
                                         font.label = list(size = 15, face = "bold"),
                                         label.x = 0.02, label.y = 0.98,
                                         hjust = 0, vjust = 1,
                                         common.legend = FALSE, # IMPORTANT: No legend here
                                         legend = "none")

# Now, add a title to this grid using patchwork's plot_annotation
gamma1_bound_loss1_var_grid_with_title <- gamma1_bound_loss1_var_grid +
  plot_annotation(
    title = "Concave Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )

var_cs_rate_by_strata[var_cs_rate_by_strata$step>1,] %>% 
  group_by(income_strata) %>%
  summarise(cs_change_rate = var(cs_change_rate))
# A tibble: 5 x 2
#income_strata cs_change_rate
#<fct>                  <dbl>
#1 0~6k                    21.4
#2 6k~20k                  23.1
#3 20k~45k                 17.5
#4 45k~100k                24.8
#5 >100k                   22.8

# A tibble: 5 x 2
#income_strata cs_change_rate
#<fct>                  <dbl>
#  1 0~6k                   -31.0
#2 6k~20k                 -35.1
#3 20k~45k                -43.4
#4 45k~100k               -26.5
#5 >100k                  -31.1


# --- Combine the two grids with the common legend ---

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss1_var_grid_with_title, gamma1_bound_loss1_var_grid_with_title,
                                              ncol = 2, nrow = 1,
                                              # Add labels for the overall columns
                                              labels = c("Scenario 1: var Loss Impact", "Scenario 2: Gamma 1 Loss Impact"),
                                              font.label = list(size = 18, face = "bold", color = "black"),
                                              label.x = 0, # Position label at the start of the column (relative to grid)
                                              label.y = 1.05, # Position label slightly above the plot area
                                              hjust = 0, vjust = 0, # Align to top-left of the label's "box"
                                              common.legend = T, # No common legend at this stage
                                              legend = "none")

# Now, combine the combined grids with the extracted common legend
avg_gamma1_var <- ggarrange(combined_grids_with_outer_labels, common_legend,
                             ncol = 1, nrow = 2,
                             heights = c(1, 0.2)) # Adjust heights as needed
grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### avg_bound_loss1_q #####

var_cs_rate_by_bill_ym <- current_info_avg_bound_loss1_var_cs_steps_rate %>%
  group_by(quantity_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_avg_bound_loss1_var_q_steps_rate %>%
  group_by(quantity_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_avg_bound_loss1_var_r_steps_rate %>%
  group_by(quantity_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_avg_bound_loss1_var_ev_steps_rate %>%
  group_by(quantity_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_cs_rate_by_strata <- var_cs_rate_by_bill_ym %>%
  group_by(quantity_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_strata <- var_r_rate_by_bill_ym %>%
  group_by(quantity_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_strata <- var_q_rate_by_bill_ym %>%
  group_by(quantity_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_strata <- var_ev_rate_by_bill_ym %>%
  group_by(quantity_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_cs_rate_by_strata = edit_strata_df_quantity(var_cs_rate_by_strata, "cs")
var_cs_rate_by_strata$cs_change_rate = as.numeric(var_cs_rate_by_strata$cs_change_rate)

var_r_rate_by_strata = edit_strata_df_quantity(var_r_rate_by_strata, "r")
var_r_rate_by_strata$r_change_rate = as.numeric(var_r_rate_by_strata$r_change_rate)

var_q_rate_by_strata = edit_strata_df_quantity(var_q_rate_by_strata, "q")
var_q_rate_by_strata$q_change_rate = as.numeric(var_q_rate_by_strata$q_change_rate)

var_ev_rate_by_strata = edit_strata_df_quantity(var_ev_rate_by_strata, "ev")
var_ev_rate_by_strata$ev_perct = as.numeric(var_ev_rate_by_strata$ev_change_rate)
var_ev_rate_by_strata$ev_change_rate = NULL

cs = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, color = quantity_strata, shape = quantity_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_cs_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = "SD Ratio",
    y = "CS %",
    color = NULL, 
    shape = NULL) +
  scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes_q)+ # Apply shapes
  #coord_cartesian(ylim = c(-80, 50))+
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text


r = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, color = quantity_strata, shape = quantity_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_r_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(0, 70))+
  labs(
    x = "SD Ratio",
    y = "PS %",
    color = NULL) +
  scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

q = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, color = quantity_strata, shape = quantity_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_q_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = "SD Ratio",
    y = "Q %",
    color = NULL) +
  scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

ev = ggplot(var_ev_rate_by_strata , aes(x = step, y = ev_perct, color = quantity_strata, shape = quantity_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_ev_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  #geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = "SD Ratio",
    y = "EV/I (%)",
    color = NULL) +
  scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  theme_minimal() + 
  theme(
    legend.position = "bottom",  # Move legend to the bottom
    axis.title.x = element_text(size = 12),  # Change x-axis label size
    axis.title.y = element_text(size = 15)   # Ensure only one y-axis label size setting
  ) + small_text

avg_bound_loss1_q = ggarrange(cs,r,q,ev, ncol = 2, nrow = 2, 
                              labels = c("CS", 
                                         "PS", 
                                         "Q",
                                         "EV"),  # Adds labels to each plot
                              font.label = list(size = 17, face = "bold"),  # Adjust label size
                              #label.x = -0.01,  # Move label to the right
                              #label.y = 1.01, # Move label slightly higher
                              label.x = 0.9,  # Move label to the right
                              label.y = 1,    # Keep label at the top
                              hjust = 1, vjust = 1,  # Align to top-right corner
                              common.legend = TRUE, 
                              legend = "bottom")


###### Comparison between avg bound and gamma1 bound ######

current_info_avg_bound_loss1_var_ev_steps_rate_125 = current_info_avg_bound_loss1_var_ev_steps_rate %>%
  select(income_strata,quantity_strata, rate_1.25, q_0, r_0, cs_0)

current_info_avg_bound_loss1_var_r_steps_rate_125 = current_info_avg_bound_loss1_var_r_steps_rate %>%
  select(income_strata,quantity_strata, `1.25`,rate_1.25, q_0, r_0, cs_0)

current_info_avg_bound_loss1_var_q_steps_rate_125 = current_info_avg_bound_loss1_var_q_steps_rate %>%
  select(income_strata,quantity_strata, `1.25`,rate_1.25, q_0, r_0, cs_0)

current_info_avg_bound_loss1_var_cs_steps_rate_125 = current_info_avg_bound_loss1_var_cs_steps_rate %>%
  select(income_strata,quantity_strata, `1.25`,rate_1.25, q_0, r_0, cs_0)

#> sd(current_info_avg_bound_loss1_var_r_steps_rate_125$`1.25`)
#[1] 172.6831

#> sd(current_info_avg_bound_loss1_var_r_steps_rate_125$r_0)
#[1] 122.2203

current_info_gamma1_bound_loss1_var_ev_steps_rate_125 = current_info_gamma1_bound_loss1_var_ev_steps_rate %>%
  select(income_strata,quantity_strata, rate_1.25, q_0, r_0, cs_0)

current_info_gamma1_bound_loss1_var_r_steps_rate_125 = current_info_gamma1_bound_loss1_var_r_steps_rate %>%
  select(income_strata,quantity_strata, `1.25`,rate_1.25, q_0, r_0, cs_0)


current_info_gamma1_bound_loss1_var_q_steps_rate_125 = current_info_gamma1_bound_loss1_var_q_steps_rate %>%
  select(income_strata,quantity_strata, `1.25`,rate_1.25, q_0, r_0, cs_0)

current_info_gamma1_bound_loss1_var_cs_steps_rate_125 = current_info_gamma1_bound_loss1_var_cs_steps_rate %>%
  select(income_strata,quantity_strata, `1.25`,rate_1.25, q_0, r_0, cs_0)


#> sd(current_info_gamma1_bound_loss1_var_r_steps_rate_125$`1.25`)
#[1] 165.6033

#> sd(current_info_gamma1_bound_loss1_var_r_steps_rate_125$r_0)
#[1] 122.2203

current_info_loss1_var_steps_rate_125_compare = current_info_gamma1_bound_loss1_var_r_steps_rate_125
current_info_loss1_var_steps_rate_125_compare$gamma1_r_1.25= current_info_loss1_var_steps_rate_125_compare$`1.25`
current_info_loss1_var_steps_rate_125_compare$gamma1_r_rate_1.25= current_info_loss1_var_steps_rate_125_compare$rate_1.25
current_info_loss1_var_steps_rate_125_compare$avg_r_1.25 = current_info_avg_bound_loss1_var_r_steps_rate_125$`1.25`
current_info_loss1_var_steps_rate_125_compare$avg_r_rate_1.25 = current_info_avg_bound_loss1_var_r_steps_rate_125$rate_1.25
rm(current_info_avg_bound_loss1_var_r_steps_rate_125, current_info_gamma1_bound_loss1_var_r_steps_rate_125)
current_info_loss1_var_steps_rate_125_compare$`1.25` = NULL
current_info_loss1_var_steps_rate_125_compare$rate_1.25 = NULL

current_info_loss1_var_steps_rate_125_compare$gamma1_cs_1.25= current_info_gamma1_bound_loss1_var_cs_steps_rate_125$`1.25`
current_info_loss1_var_steps_rate_125_compare$gamma1_q_1.25= current_info_gamma1_bound_loss1_var_q_steps_rate_125$`1.25`
current_info_loss1_var_steps_rate_125_compare$gamma1_ev_1.25= current_info_gamma1_bound_loss1_var_ev_steps_rate_125$rate_1.25
current_info_loss1_var_steps_rate_125_compare$avg_cs_1.25= current_info_avg_bound_loss1_var_cs_steps_rate_125$`1.25`
current_info_loss1_var_steps_rate_125_compare$avg_q_1.25= current_info_avg_bound_loss1_var_q_steps_rate_125$`1.25`
current_info_loss1_var_steps_rate_125_compare$avg_ev_1.25= current_info_avg_bound_loss1_var_ev_steps_rate_125$rate_1.25
current_info_loss1_var_steps_rate_125_compare$avg_cs_rate_1.25= current_info_avg_bound_loss1_var_cs_steps_rate_125$rate_1.25
current_info_loss1_var_steps_rate_125_compare$gamma1_cs_rate_1.25= current_info_gamma1_bound_loss1_var_cs_steps_rate_125$rate_1.25
current_info_loss1_var_steps_rate_125_compare$avg_q_rate_1.25= current_info_avg_bound_loss1_var_q_steps_rate_125$rate_1.25
current_info_loss1_var_steps_rate_125_compare$gamma1_q_rate_1.25= current_info_gamma1_bound_loss1_var_q_steps_rate_125$rate_1.25

rm(current_info_avg_bound_loss1_var_cs_steps_rate_125, current_info_gamma1_bound_loss1_var_cs_steps_rate_125)
rm(current_info_avg_bound_loss1_var_q_steps_rate_125, current_info_gamma1_bound_loss1_var_q_steps_rate_125)
rm(current_info_avg_bound_loss1_var_ev_steps_rate_125, current_info_gamma1_bound_loss1_var_ev_steps_rate_125)

current_info_loss1_var_steps_rate_125_compare$bill_ym = demand_key$bill_ym

current_info_loss1_var_steps_rate_125_compare$income = demand_key$income

current_info_loss1_var_steps_rate_125_compare_bill_ym_income = current_info_loss1_var_steps_rate_125_compare %>%
  group_by(income_strata, bill_ym) %>%
  summarise(r_0 = sum(r_0),
            cs_0 = sum(cs_0),
            q_0 = sum(q_0),
            avg_r_125 = sum(avg_r_1.25),
            gamma1_r_125 = sum(gamma1_r_1.25),
            avg_q_125 = sum(avg_q_1.25),
            gamma1_q_125 = sum(gamma1_q_1.25),
            avg_cs_125 = sum(avg_cs_1.25),
            gamma1_cs_125 = sum(gamma1_cs_1.25),
            avg_ev_125 = sum(avg_ev_1.25),
            gamma1_ev_125 = sum(gamma1_ev_1.25))

# Reshape the data to long format for plotting multiple y-variables
var_r_rate_long_income <- current_info_loss1_var_steps_rate_125_compare_bill_ym_income %>%
  pivot_longer(
    cols = c(r_0, avg_r_125, gamma1_r_125), # Specify the columns to pivot
    names_to = "metric", # Name of the new column
    values_to = "value" # Name of the new column for values
  )

# Get the unique income strata levels
strata_levels <- unique(var_r_rate_long_income$income_strata)

# Create a list to store the plots
plot_list <- list()

# Define specific linetypes for each metric
custom_linetypes <- c("r_0" = "solid",
                      "avg_r_125" = "dashed",
                      "gamma1_r_125" = "dotted")

# Loop through each strata level to create a plot
for (current_strata in strata_levels) {
  
  # Create the base plot with all data faded
  p <- ggplot(var_r_rate_long_income, aes(x = bill_ym, y = value, linetype = metric)) +
    # Plot all data with low alpha (faded) and a muted color (e.g., grey)
    geom_line(aes(color = income_strata), size = 1.5, alpha = 0.2) +
    geom_point(aes(color = income_strata, shape = income_strata), size = 4, stroke = 1.2, alpha = 0.2) +
    # Add vertical and horizontal lines
    # Apply custom colors for the faded data (optional, could use a single grey scale)
    scale_color_manual(values = custom_colors) +
    # Apply custom shapes for the faded data
    scale_shape_manual(values = custom_shapes) +
    # Customize linetypes for metrics
    scale_linetype_manual(values = custom_linetypes) + # Use the custom linetypes
    # Add informative labels and title
    labs(
      title = paste("R Metrics over Bill Month: Highlighting", current_strata),
      x = "Bill Month",
      y = "Value",
      color = "Income Strata",
      shape = "Income Strata",
      linetype = "Metric" # Legend title for linetype
    ) +
    # Apply a cleaner theme
    theme_minimal() +
    theme(
      legend.position = "bottom",
      axis.title.x = element_text(size = 12),
      axis.title.y = element_text(size = 15)
    )
  
  # Overlay the data for the current strata with full opacity and custom colors/shapes
  p <- p +
    geom_line(data = subset(var_r_rate_long_income, income_strata == current_strata),
              aes(color = income_strata), size = 1.5, alpha = 1) +
    geom_point(data = subset(var_r_rate_long_income, income_strata == current_strata),
               aes(color = income_strata, shape = income_strata), size = 4, stroke = 1.2, alpha = 1) +
    # Re-apply scales to ensure the correct colors/shapes are used for the highlighted data
    scale_color_manual(values = custom_colors) +
    scale_shape_manual(values = custom_shapes) +
    # Re-apply linetype scale for the highlighted data
    scale_linetype_manual(values = custom_linetypes)
  
  
  # Store the plot in the list
  plot_list[[current_strata]] <- p
}

current_info_loss1_var_steps_rate_125_compare_bill_ym_quantity = current_info_loss1_var_steps_rate_125_compare %>%
  group_by(quantity_strata, bill_ym) %>%
  summarise(r_0 = sum(r_0),
            cs_0 = sum(cs_0),
            q_0 = sum(q_0),
            avg_r_125 = sum(avg_r_1.25),
            gamma1_r_125 = sum(gamma1_r_1.25),
            avg_q_125 = sum(avg_q_1.25),
            gamma1_q_125 = sum(gamma1_q_1.25),
            avg_cs_125 = sum(avg_cs_1.25),
            gamma1_cs_125 = sum(gamma1_cs_1.25),
            avg_ev_125 = sum(avg_ev_1.25),
            gamma1_ev_125 = sum(gamma1_ev_1.25))

# Function to plot specified metrics over Bill Month, highlighting each strata
plot_metrics_by_strata <- function(data, metrics_to_plot, custom_colors, custom_shapes, custom_linetypes, plot_title_prefix) {
  
  # Validate that all metrics_to_plot exist in the data
  if (!all(metrics_to_plot %in% colnames(data))) {
    stop("One or more of the specified metrics are not found in the data.")
  }
  
  # Select relevant columns and pivot to long format for the specified metrics
  # This creates a temporary long dataset just for plotting within this function
  data_long <- data %>%
    select(quantity_strata, bill_ym, all_of(metrics_to_plot)) %>%
    pivot_longer(
      cols = all_of(metrics_to_plot), # Pivot only the specified metrics
      names_to = "metric",
      values_to = "value"
    )
  
  # Get the unique strata levels
  strata_levels <- unique(data_long$quantity_strata)
  
  # Create a list to store the plots
  plot_list <- list()
  
  # Loop through each strata level to create a plot
  for (current_strata in strata_levels) {
    
    # Create the base plot with all data faded
    p <- ggplot(data_long, aes(x = bill_ym, y = value, linetype = metric)) +
      # Plot all data with low alpha (faded)
      geom_line(aes(color = quantity_strata), size = 1.5, alpha = 0.2) +
      geom_point(aes(color = quantity_strata, shape = quantity_strata), size = 4, stroke = 1.2, alpha = 0.2) +
      # Apply custom scales for color and shape (mapping to quantity_strata)
      scale_color_manual(values = custom_colors) +
      scale_shape_manual(values = custom_shapes) +
      # Apply custom linetypes for metrics
      scale_linetype_manual(values = custom_linetypes) +
      # Add informative labels and title
      labs(
        title = paste(plot_title_prefix, "over Bill Month: Highlighting", current_strata),
        x = "Bill Month",
        y = "Value",
        color = "Quantity Strata", # Updated from Income Strata based on variable name
        shape = "Quantity Strata", # Updated from Income Strata
        linetype = "Metric"
      ) +
      # Apply a cleaner theme
      theme_minimal() +
      theme(
        legend.position = "bottom",
        axis.title.x = element_text(size = 12),
        axis.title.y = element_text(size = 15)
      )
    
    # Overlay the data for the current strata with full opacity
    # Subset the long data for the current strata for highlighting
    data_current_strata <- subset(data_long, quantity_strata == current_strata)
    
    p <- p +
      geom_line(data = data_current_strata,
                aes(color = quantity_strata), size = 1.5, alpha = 1) +
      geom_point(data = data_current_strata,
                 aes(color = quantity_strata, shape = quantity_strata), size = 4, stroke = 1.2, alpha = 1) +
      # Re-apply scales to ensure consistency for the highlighted data
      scale_color_manual(values = custom_colors) +
      scale_shape_manual(values = custom_shapes) +
      scale_linetype_manual(values = custom_linetypes)
    
    
    # Store the plot in the list, named by strata level
    plot_list[[current_strata]] <- p
  }
  
  return(plot_list)
}

r_metrics_to_plot <- c("r_0", "avg_r_125", "gamma1_r_125") # Use exact column names from your data
custom_linetypes_r <- c("r_0" = "solid", "avg_r_125" = "dashed", "gamma1_r_125" = "dotted") # Define linetypes for R metrics

r_plots <- plot_metrics_by_strata(
  data = current_info_loss1_var_steps_rate_125_compare_bill_ym_quantity,
  metrics_to_plot = r_metrics_to_plot,
  custom_colors = custom_colors_q, # Use your defined custom colors for strata
  custom_shapes = custom_shapes_q, # Use your defined custom shapes for strata
  custom_linetypes = custom_linetypes_r,
  plot_title_prefix = "R Metrics"
)

q_metrics_to_plot <- c("q_0", "avg_q_125", "gamma1_q_125") # Use exact column names from your data
custom_linetypes_q <- c("q_0" = "solid", "avg_q_125" = "dashed", "gamma1_q_125" = "dotted") # Define linetypes for Q metrics

q_plots <- plot_metrics_by_strata(
  data = current_info_loss1_var_steps_rate_125_compare_bill_ym_quantity,
  metrics_to_plot = q_metrics_to_plot,
  custom_colors = custom_colors_q, # Reuse strata colors/shapes
  custom_shapes = custom_shapes_q,
  custom_linetypes = custom_linetypes_q,
  plot_title_prefix = "Q Metrics"
)


ev_metrics_to_plot <- c("avg_ev_125", "gamma1_ev_125") # Use exact column names from your data
custom_linetypes_ev <- c("avg_ev_125" = "dashed", "gamma1_ev_125" = "dotted") # Define linetypes for Q metrics

ev_plots <- plot_metrics_by_strata(
  data = current_info_loss1_var_steps_rate_125_compare_bill_ym_quantity,
  metrics_to_plot = ev_metrics_to_plot,
  custom_colors = custom_colors_q, # Reuse strata colors/shapes
  custom_shapes = custom_shapes_q,
  custom_linetypes = custom_linetypes_ev,
  plot_title_prefix = "EV Metrics"
)

current_info_loss1_var_steps_rate_125_compare_small = current_info_loss1_var_steps_rate_125_compare[which(current_info_loss1_var_steps_rate_125_compare$income<=250000 & 
                                                                                                            current_info_loss1_var_steps_rate_125_compare$q_0<=100),]


##### current_info_gamma05_bound_loss1_mean ########

current_info_gamma05_bound_loss1_mean_cs_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss1_mean_cs_steps.csv")

colnames(current_info_gamma05_bound_loss1_mean_cs_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss1_mean_cs_steps))*0.05-0.25)

current_info_gamma05_bound_loss1_mean_cs_steps_rate <-current_info_gamma05_bound_loss1_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss1_mean_cs_steps)

current_info_gamma05_bound_loss1_mean_q_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss1_mean_q_steps.csv")

colnames(current_info_gamma05_bound_loss1_mean_q_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss1_mean_q_steps))*0.05-0.25)

current_info_gamma05_bound_loss1_mean_q_steps_rate <-current_info_gamma05_bound_loss1_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss1_mean_q_steps)

current_info_gamma05_bound_loss1_mean_r_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss1_mean_r_steps.csv")

colnames(current_info_gamma05_bound_loss1_mean_r_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss1_mean_r_steps))*0.05-0.25)

current_info_gamma05_bound_loss1_mean_r_steps_rate <-current_info_gamma05_bound_loss1_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss1_mean_r_steps)

current_info_gamma05_bound_loss1_mean_ev_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss1_mean_ev_steps.csv")

colnames(current_info_gamma05_bound_loss1_mean_ev_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss1_mean_ev_steps))*0.05-0.25)

current_info_gamma05_bound_loss1_mean_ev_steps_rate <-current_info_gamma05_bound_loss1_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss1_mean_ev_steps)

current_info_gamma05_bound_loss1_mean_cs_steps_rate  = cbind(current_info_gamma05_bound_loss1_mean_cs_steps_rate , demand_key)
current_info_gamma05_bound_loss1_mean_q_steps_rate = cbind(current_info_gamma05_bound_loss1_mean_q_steps_rate, demand_key)
current_info_gamma05_bound_loss1_mean_r_steps_rate = cbind(current_info_gamma05_bound_loss1_mean_r_steps_rate, demand_key)
current_info_gamma05_bound_loss1_mean_ev_steps_rate = cbind(current_info_gamma05_bound_loss1_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_gamma05_bound_loss1_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_gamma05_bound_loss1_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_gamma05_bound_loss1_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- current_info_gamma05_bound_loss1_mean_ev_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_cs_rate_by_strata <- mean_cs_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_strata <- mean_r_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_strata <- mean_q_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_strata <- mean_ev_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

#mean_cs_rate_by_strata = edit_strata_df_quantity(mean_cs_rate_by_strata, "cs")
mean_cs_rate_by_strata = edit_strata_df(mean_cs_rate_by_strata, "cs")
mean_cs_rate_by_strata$cs_change_rate = as.numeric(mean_cs_rate_by_strata$cs_change_rate)

#mean_r_rate_by_strata = edit_strata_df_quantity(mean_r_rate_by_strata, "r")
mean_r_rate_by_strata = edit_strata_df(mean_r_rate_by_strata, "r")
mean_r_rate_by_strata$r_change_rate = as.numeric(mean_r_rate_by_strata$r_change_rate)

#mean_q_rate_by_strata = edit_strata_df_quantity(mean_q_rate_by_strata, "q")
mean_q_rate_by_strata = edit_strata_df(mean_q_rate_by_strata, "q")
mean_q_rate_by_strata$q_change_rate = as.numeric(mean_q_rate_by_strata$q_change_rate)

#mean_ev_rate_by_strata = edit_strata_df_quantity(mean_ev_rate_by_strata, "ev")
mean_ev_rate_by_strata = edit_strata_df(mean_ev_rate_by_strata, "ev")
mean_ev_rate_by_strata$ev_perct = as.numeric(mean_ev_rate_by_strata$ev_change_rate)
mean_ev_rate_by_strata$ev_change_rate = NULL

cs_gamma05 = ggplot(mean_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
                                                color = income_strata, shape = income_strata
                                                #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  #geom_point(data = subset(mean_cs_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_point(data = subset(mean_cs_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "CS %",
    color = NULL, 
    shape = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  #coord_cartesian(ylim = c(-80, 50))+
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )


r_gamma05 = ggplot(mean_r_rate_by_strata , aes(x = step, y = r_change_rate, 
                                              color = income_strata, shape = income_strata
                                              #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_r_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_point(data = subset(mean_r_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(0, 70))+
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "PS %",
    color = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

q_gamma05 = ggplot(mean_q_rate_by_strata , aes(x = step, y = q_change_rate, 
                                              color = income_strata, shape = income_strata
                                              #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_q_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_point(data = subset(mean_q_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "Q %",
    color = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

ev_gamma05 = ggplot(mean_ev_rate_by_strata , aes(x = step, y = ev_perct, 
                                                color = income_strata, shape = income_strata
                                                #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_ev_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_point(data = subset(mean_ev_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  #geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"),
    y = "EV/I (%)",
    color = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

gamma05_bound_loss1_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct))

gamma05_bound_loss1_mean_grid = ggarrange(cs_gamma05,r_gamma05,q_gamma05,ev_gamma05, ncol = 2, nrow = 2, 
                                         labels = c("CS", 
                                                    "PS", 
                                                    "Q",
                                                    "EV"),  # Adds labels to each plot
                                         font.label = list(size = 15, face = "bold"),
                                         label.x = 0.02, label.y = 0.98,
                                         hjust = 0, vjust = 1,
                                         common.legend = FALSE, # IMPORTANT: No legend here
                                         legend = "none")

# Now, add a title to this grid using patchwork's plot_annotation
gamma05_bound_loss1_mean_grid_with_title <- gamma05_bound_loss1_mean_grid +
  plot_annotation(
    title = "Concave Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )

mean_cs_rate_by_strata[mean_cs_rate_by_strata$step>1,] %>% 
  group_by(income_strata) %>%
  summarise(cs_change_rate = mean(cs_change_rate))
# A tibble: 5 x 2
#income_strata cs_change_rate
#<fct>                  <dbl>
#1 0~6k                    21.4
#2 6k~20k                  23.1
#3 20k~45k                 17.5
#4 45k~100k                24.8
#5 >100k                   22.8

# A tibble: 5 x 2
#income_strata cs_change_rate
#<fct>                  <dbl>
#  1 0~6k                   -31.0
#2 6k~20k                 -35.1
#3 20k~45k                -43.4
#4 45k~100k               -26.5
#5 >100k                  -31.1


# Step 1: Create a temporary plot with visible content
legend_plot <- ggplot(mean_cs_rate_by_strata, aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
  geom_line(linewidth = 1) +
  geom_point(size = 3, stroke = 1) +
  labs(color = "Income Strata", shape = "Income Strata") +
  scale_color_manual(values = custom_colors) +
  scale_shape_manual(values = custom_shapes) +
  guides(
    color = guide_legend(override.aes = list(linewidth = 1, size = 3, shape = custom_shapes)),
    shape = "none"  # combine shape into color legend
  ) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 13),
    legend.text = element_text(size = 12)
  )

plot_grob <- ggplotGrob(legend_plot)

legend_index <- which(sapply(plot_grob$grobs, function(x) x$name) == "guide-box")

common_legend <- plot_grob$grobs[[legend_index]]

# --- Combine the two grids with the common legend ---

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss1_mean_grid_with_title, gamma05_bound_loss1_mean_grid_with_title,
                                              ncol = 2, nrow = 1,
                                              # Add labels for the overall columns
                                              labels = c("Scenario 1: Mean Loss Impact", "Scenario 2: Gamma 1 Loss Impact"),
                                              font.label = list(size = 18, face = "bold", color = "black"),
                                              label.x = 0, # Position label at the start of the column (relative to grid)
                                              label.y = 1.05, # Position label slightly above the plot area
                                              hjust = 0, vjust = 0, # Align to top-left of the label's "box"
                                              common.legend = T, # No common legend at this stage
                                              legend = "none")

# Now, combine the combined grids with the extracted common legend
avg_gamma05_mean <- ggarrange(combined_grids_with_outer_labels, common_legend,
                             ncol = 1, nrow = 2,
                             heights = c(1, 0.2)) # Adjust heights as needed
grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))


##### current_info_gamma05_bound_loss1_var ########

current_info_gamma05_bound_loss1_var_cs_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss1_var_cs_steps.csv")

colnames(current_info_gamma05_bound_loss1_var_cs_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss1_var_cs_steps))/20+0.75)

current_info_gamma05_bound_loss1_var_cs_steps_rate <-current_info_gamma05_bound_loss1_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss1_var_cs_steps)

current_info_gamma05_bound_loss1_var_q_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss1_var_q_steps.csv")

colnames(current_info_gamma05_bound_loss1_var_q_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss1_var_q_steps))/20+0.75)

current_info_gamma05_bound_loss1_var_q_steps_rate <-current_info_gamma05_bound_loss1_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss1_var_q_steps)

current_info_gamma05_bound_loss1_var_r_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss1_var_r_steps.csv")

colnames(current_info_gamma05_bound_loss1_var_r_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss1_var_r_steps))/20+0.75)

current_info_gamma05_bound_loss1_var_r_steps_rate <-current_info_gamma05_bound_loss1_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss1_var_r_steps)

current_info_gamma05_bound_loss1_var_ev_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss1_var_ev_steps.csv")

colnames(current_info_gamma05_bound_loss1_var_ev_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss1_var_ev_steps))/20+0.75)

current_info_gamma05_bound_loss1_var_ev_steps_rate <-current_info_gamma05_bound_loss1_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss1_var_ev_steps)

current_info_gamma05_bound_loss1_var_cs_steps_rate  = cbind(current_info_gamma05_bound_loss1_var_cs_steps_rate , demand_key)
current_info_gamma05_bound_loss1_var_q_steps_rate = cbind(current_info_gamma05_bound_loss1_var_q_steps_rate, demand_key)
current_info_gamma05_bound_loss1_var_r_steps_rate = cbind(current_info_gamma05_bound_loss1_var_r_steps_rate, demand_key)
current_info_gamma05_bound_loss1_var_ev_steps_rate = cbind(current_info_gamma05_bound_loss1_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_gamma05_bound_loss1_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_gamma05_bound_loss1_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_gamma05_bound_loss1_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_gamma05_bound_loss1_var_ev_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_cs_rate_by_strata <- var_cs_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_strata <- var_r_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_strata <- var_q_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_strata <- var_ev_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

#var_cs_rate_by_strata = edit_strata_df_quantity(var_cs_rate_by_strata, "cs")
var_cs_rate_by_strata = edit_strata_df(var_cs_rate_by_strata, "cs")
var_cs_rate_by_strata$cs_change_rate = as.numeric(var_cs_rate_by_strata$cs_change_rate)

#var_r_rate_by_strata = edit_strata_df_quantity(var_r_rate_by_strata, "r")
var_r_rate_by_strata = edit_strata_df(var_r_rate_by_strata, "r")
var_r_rate_by_strata$r_change_rate = as.numeric(var_r_rate_by_strata$r_change_rate)

#var_q_rate_by_strata = edit_strata_df_quantity(var_q_rate_by_strata, "q")
var_q_rate_by_strata = edit_strata_df(var_q_rate_by_strata, "q")
var_q_rate_by_strata$q_change_rate = as.numeric(var_q_rate_by_strata$q_change_rate)

#var_ev_rate_by_strata = edit_strata_df_quantity(var_ev_rate_by_strata, "ev")
var_ev_rate_by_strata = edit_strata_df(var_ev_rate_by_strata, "ev")
var_ev_rate_by_strata$ev_perct = as.numeric(var_ev_rate_by_strata$ev_change_rate)
var_ev_rate_by_strata$ev_change_rate = NULL

cs_gamma05 = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
                                               color = income_strata, shape = income_strata
                                               #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  #geom_point(data = subset(var_cs_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_point(data = subset(var_cs_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "CS %",
    color = NULL, 
    shape = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  #coord_cartesian(ylim = c(-80, 50))+
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )


r_gamma05 = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, 
                                             color = income_strata, shape = income_strata
                                             #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_r_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_point(data = subset(var_r_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(0, 70))+
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "PS %",
    color = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

q_gamma05 = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, 
                                             color = income_strata, shape = income_strata
                                             #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_q_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_point(data = subset(var_q_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "Q %",
    color = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

ev_gamma05 = ggplot(var_ev_rate_by_strata , aes(x = step, y = ev_perct, 
                                               color = income_strata, shape = income_strata
                                               #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_ev_rate_by_strata, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_point(data = subset(var_ev_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
  #geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  #coord_cartesian(ylim = c(-7.5, 10))+
  labs(
    x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"),
    y = "EV/I (%)",
    color = NULL) +
  #scale_color_manual(values = custom_colors_q) +  # Apply gradient-like colors
  #scale_shape_manual(values = custom_shapes_q) + # Apply shapes
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

gamma05_bound_loss1_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct))


gamma05_bound_loss1_var_grid = ggarrange(cs_gamma05,r_gamma05,q_gamma05,ev_gamma05, ncol = 2, nrow = 2, 
                                        labels = c("CS", 
                                                   "PS", 
                                                   "Q",
                                                   "EV"),  # Adds labels to each plot
                                        font.label = list(size = 15, face = "bold"),
                                        label.x = 0.02, label.y = 0.98,
                                        hjust = 0, vjust = 1,
                                        common.legend = FALSE, # IMPORTANT: No legend here
                                        legend = "none")

# Now, add a title to this grid using patchwork's plot_annotation
gamma05_bound_loss1_var_grid_with_title <- gamma05_bound_loss1_var_grid +
  plot_annotation(
    title = "Concave Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )

var_cs_rate_by_strata[var_cs_rate_by_strata$step>1,] %>% 
  group_by(income_strata) %>%
  summarise(cs_change_rate = var(cs_change_rate))
# A tibble: 5 x 2
#income_strata cs_change_rate
#<fct>                  <dbl>
#1 0~6k                    21.4
#2 6k~20k                  23.1
#3 20k~45k                 17.5
#4 45k~100k                24.8
#5 >100k                   22.8

# A tibble: 5 x 2
#income_strata cs_change_rate
#<fct>                  <dbl>
#  1 0~6k                   -31.0
#2 6k~20k                 -35.1
#3 20k~45k                -43.4
#4 45k~100k               -26.5
#5 >100k                  -31.1


# --- Combine the two grids with the common legend ---

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss1_var_grid_with_title, gamma05_bound_loss1_var_grid_with_title,
                                              ncol = 2, nrow = 1,
                                              # Add labels for the overall columns
                                              labels = c("Scenario 1: var Loss Impact", "Scenario 2: Gamma 1 Loss Impact"),
                                              font.label = list(size = 18, face = "bold", color = "black"),
                                              label.x = 0, # Position label at the start of the column (relative to grid)
                                              label.y = 1.05, # Position label slightly above the plot area
                                              hjust = 0, vjust = 0, # Align to top-left of the label's "box"
                                              common.legend = T, # No common legend at this stage
                                              legend = "none")

# Now, combine the combined grids with the extracted common legend
avg_gamma05_var <- ggarrange(combined_grids_with_outer_labels, common_legend,
                            ncol = 1, nrow = 2,
                            heights = c(1, 0.2)) # Adjust heights as needed
grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))








##### Trade Off Study

loss1_mean_ev = data.frame(avg_bound_loss1_mean_ev$income_strata)
colnames(loss1_mean_ev) = c("income_strata")
loss1_mean_ev$mean_ev_perct_0 = avg_bound_loss1_mean_ev$mean_ev_perct
loss1_mean_ev$sd_ev_perct_0 = avg_bound_loss1_mean_ev$sd_ev_perct
loss1_mean_ev$mean_ev_perct_05 = gamma05_bound_loss1_mean_ev$mean_ev_perct
loss1_mean_ev$sd_ev_perct_05 = gamma05_bound_loss1_mean_ev$sd_ev_perct
loss1_mean_ev$mean_ev_perct_1 = gamma1_bound_loss1_mean_ev$mean_ev_perct
loss1_mean_ev$sd_ev_perct_1 = gamma1_bound_loss1_mean_ev$sd_ev_perct

loss1_var_ev = data.frame(avg_bound_loss1_var_ev$income_strata)
colnames(loss1_var_ev) = c("income_strata")
loss1_var_ev$mean_ev_perct_0 = avg_bound_loss1_var_ev$mean_ev_perct
loss1_var_ev$sd_ev_perct_0 = avg_bound_loss1_var_ev$sd_ev_perct
loss1_var_ev$mean_ev_perct_05 = gamma05_bound_loss1_var_ev$mean_ev_perct
loss1_var_ev$sd_ev_perct_05 = gamma05_bound_loss1_var_ev$sd_ev_perct
loss1_var_ev$mean_ev_perct_1 = gamma1_bound_loss1_var_ev$mean_ev_perct
loss1_var_ev$sd_ev_perct_1 = gamma1_bound_loss1_var_ev$sd_ev_perct


write.csv(loss1_mean_ev, "trade_off_study/loss1_mean_ev.csv", row.names = FALSE)
write.csv(loss1_var_ev, "trade_off_study/loss1_var_ev.csv", row.names = FALSE)
#################################################################
### Graph to showcase mismatch between income and tiers #####
#################################################################

q0_income = ggplot(current_info_loss1_var_steps_rate_125_compare,
       aes(x = log(income), y = log(q_0), fill = income_strata)) +
  stat_bin_2d(aes(alpha = after_stat(count)),
              binwidth = c(0.1, 0.1), # Adjust binwidth as needed for clarity and density
              position = "identity") +
  # Add horizontal lines at specified log(q_0) values
  geom_hline(yintercept = c(log(2), log(6), log(11), log(20)),
             linetype = "dashed", # Use dashed lines
             color = "black",    # Use black color
             linewidth = 0.8) +   # Adjust line thickness if needed
  # Add labels for the horizontal lines
  annotate("text", x = max(log(current_info_loss1_var_steps_rate_125_compare$income), na.rm = TRUE) * 1.02, y = log(2), label = "q₀ = 2", hjust = 0, vjust = -0.5, size = 4, fontface = "bold") +
  annotate("text", x = max(log(current_info_loss1_var_steps_rate_125_compare$income), na.rm = TRUE) * 1.02, y = log(6), label = "q₀ = 6", hjust = 0, vjust = -0.5, size = 4, fontface = "bold") +
  annotate("text", x = max(log(current_info_loss1_var_steps_rate_125_compare$income), na.rm = TRUE) * 1.02, y = log(11), label = "q₀ = 11", hjust = 0, vjust = -0.5, size = 4, fontface = "bold") +
  annotate("text", x = max(log(current_info_loss1_var_steps_rate_125_compare$income), na.rm = TRUE) * 1.02, y = log(20), label = "q₀ = 20", hjust = 0, vjust = -0.5, size = 4, fontface = "bold") +
  scale_fill_manual(values = custom_colors,
                    name = "Income Strata") +
  scale_alpha_continuous(range = c(0.5, 1), # Adjust range (min_alpha, max_alpha) to control transparency
                         name = "Number of Points") +
  labs(
    title = "Binned Plot of log(q0) vs log(Income) by Income Strata",
    x = "log(Income)",  # Corrected x-axis label
    y = "log(q_0)"     # Corrected y-axis label
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    panel.grid.major = element_line(color = "gray80"),
    panel.grid.minor = element_line(color = "gray90"),
    plot.margin = unit(c(5.5, 10, 5.5, 5.5), "points") # Add some space on the right for labels
  )

q1_avg_income = ggplot(current_info_loss1_var_steps_rate_125_compare,
       aes(x = log(income), y = log(avg_q_1.25), fill = income_strata)) +
  stat_bin_2d(aes(alpha = after_stat(count)),
              binwidth = c(0.1, 0.1), # Adjust binwidth as needed for clarity and density
              position = "identity") +
  # Add horizontal lines at specified log(q_0) values
  #geom_hline(yintercept = c(log(2), log(6), log(11), log(20)),
   #          linetype = "dashed", # Use dashed lines
   #          color = "black",    # Use black color
  #           linewidth = 0.8) +   # Adjust line thickness if needed
  # Add labels for the horizontal lines
  #annotate("text", x = max(log(current_info_loss1_var_steps_rate_125_compare$income), na.rm = TRUE) * 1.02, y = log(2), label = "q₀ = 2", hjust = 0, vjust = -0.5, size = 4, fontface = "bold") +
  #annotate("text", x = max(log(current_info_loss1_var_steps_rate_125_compare$income), na.rm = TRUE) * 1.02, y = log(6), label = "q₀ = 6", hjust = 0, vjust = -0.5, size = 4, fontface = "bold") +
  #annotate("text", x = max(log(current_info_loss1_var_steps_rate_125_compare$income), na.rm = TRUE) * 1.02, y = log(11), label = "q₀ = 11", hjust = 0, vjust = -0.5, size = 4, fontface = "bold") +
  #annotate("text", x = max(log(current_info_loss1_var_steps_rate_125_compare$income), na.rm = TRUE) * 1.02, y = log(20), label = "q₀ = 20", hjust = 0, vjust = -0.5, size = 4, fontface = "bold") +
  scale_fill_manual(values = custom_colors,
                    name = "Income Strata") +
  scale_alpha_continuous(range = c(0.5, 1), # Adjust range (min_alpha, max_alpha) to control transparency
                         name = "Number of Points") +
  labs(
    title = "Binned Plot of log(q')_linear vs log(Income) by Income Strata",
    x = "log(Income)",  # Corrected x-axis label
    y = "log(q')_linear"     # Corrected y-axis label
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    panel.grid.major = element_line(color = "gray80"),
    panel.grid.minor = element_line(color = "gray90"),
    plot.margin = unit(c(5.5, 10, 5.5, 5.5), "points") # Add some space on the right for labels
  )

q1_gamma1_income = ggplot(current_info_loss1_var_steps_rate_125_compare,
                       aes(x = log(income), y = log(gamma1_q_1.25), fill = income_strata)) +
  stat_bin_2d(aes(alpha = after_stat(count)),
              binwidth = c(0.1, 0.1), # Adjust binwidth as needed for clarity and density
              position = "identity") +
  # Add horizontal lines at specified log(q_0) values
  #geom_hline(yintercept = c(log(2), log(6), log(11), log(20)),
  #          linetype = "dashed", # Use dashed lines
  #          color = "black",    # Use black color
  #           linewidth = 0.8) +   # Adjust line thickness if needed
  # Add labels for the horizontal lines
  #annotate("text", x = max(log(current_info_loss1_var_steps_rate_125_compare$income), na.rm = TRUE) * 1.02, y = log(2), label = "q₀ = 2", hjust = 0, vjust = -0.5, size = 4, fontface = "bold") +
  #annotate("text", x = max(log(current_info_loss1_var_steps_rate_125_compare$income), na.rm = TRUE) * 1.02, y = log(6), label = "q₀ = 6", hjust = 0, vjust = -0.5, size = 4, fontface = "bold") +
  #annotate("text", x = max(log(current_info_loss1_var_steps_rate_125_compare$income), na.rm = TRUE) * 1.02, y = log(11), label = "q₀ = 11", hjust = 0, vjust = -0.5, size = 4, fontface = "bold") +
  #annotate("text", x = max(log(current_info_loss1_var_steps_rate_125_compare$income), na.rm = TRUE) * 1.02, y = log(20), label = "q₀ = 20", hjust = 0, vjust = -0.5, size = 4, fontface = "bold") +
  scale_fill_manual(values = custom_colors,
                    name = "Income Strata") +
  scale_alpha_continuous(range = c(0.5, 1), # Adjust range (min_alpha, max_alpha) to control transparency
                         name = "Number of Points") +
  labs(
    title = "Binned Plot of log(q')_gamma1 vs log(Income) by Income Strata",
    x = "log(Income)",  # Corrected x-axis label
    y = "log(q')_gamma1"     # Corrected y-axis label
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.title = element_text(size = 16),
    legend.text = element_text(size = 14),
    axis.title = element_text(size = 16),
    axis.text = element_text(size = 14),
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
    panel.grid.major = element_line(color = "gray80"),
    panel.grid.minor = element_line(color = "gray90"),
    plot.margin = unit(c(5.5, 10, 5.5, 5.5), "points") # Add some space on the right for labels
  )

avg_ev_by_q0 = ggplot(current_info_loss1_var_steps_rate_125_compare[current_info_loss1_var_steps_rate_125_compare$avg_ev_1.25<200,], 
       aes(x = q_0, y = avg_ev_1.25, fill = income_strata)) +
  stat_bin_2d(binwidth = c(10, 2), alpha = 0.8, position = "identity") +  # Adjust binwidth to (5, 1)
  scale_fill_manual(values = custom_colors) +
  labs(
    title = "Binned Plot of EV/I at SD Ratio = 1.25 vs q_0 by Income Strata",
    x = "q_0 (kGallon)",
    y = "EV/I (%)",
    fill = "Income Strata"
  ) +
  theme_minimal(base_size = 14) +  # Larger base font size for academic readability
  theme(
    legend.position = "bottom",         # Move legend to bottom
    legend.title = element_text(size = 16), # Larger legend title
    legend.text = element_text(size = 14),  # Adjust legend text size
    axis.title = element_text(size = 16),   # Larger axis titles
    axis.text = element_text(size = 14),    # Larger axis labels
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),  # Title centered and bold
    panel.grid.major = element_line(color = "gray80"),   # Lighter grid lines
    panel.grid.minor = element_line(color = "gray90")    # Lighter minor grid lines
  )
gamma1_ev_by_q0 = ggplot(current_info_loss1_var_steps_rate_125_compare[current_info_loss1_var_steps_rate_125_compare$gamma1_ev_1.25<200,], 
                      aes(x = q_0, y = gamma1_ev_1.25, fill = income_strata)) +
  stat_bin_2d(binwidth = c(10, 2), alpha = 0.8, position = "identity") +  # Adjust binwidth to (5, 1)
  scale_fill_manual(values = custom_colors) +
  labs(
    title = "Binned Plot of EV/I at SD Ratio = 1.25 vs q_0 by Income Strata",
    x = "q_0 (kGallon)",
    y = "EV/I (%)",
    fill = "Income Strata"
  ) +
  theme_minimal(base_size = 14) +  # Larger base font size for academic readability
  theme(
    legend.position = "bottom",         # Move legend to bottom
    legend.title = element_text(size = 16), # Larger legend title
    legend.text = element_text(size = 14),  # Adjust legend text size
    axis.title = element_text(size = 16),   # Larger axis titles
    axis.text = element_text(size = 14),    # Larger axis labels
    plot.title = element_text(size = 18, face = "bold", hjust = 0.5),  # Title centered and bold
    panel.grid.major = element_line(color = "gray80"),   # Lighter grid lines
    panel.grid.minor = element_line(color = "gray90")    # Lighter minor grid lines
  )

#################################################################################
#### Why concave bound makes price tier more disperse <=> supposed better equity
#################################################################################

current_info_loss1_var_steps_rate_125_compare_bill_ym = current_info_loss1_var_steps_rate_125_compare %>%
  group_by(bill_ym) %>%
  summarise(r_0 = sum(r_0),
            cs_0 = sum(cs_0),
            q_0 = sum(q_0),
            avg_r_125 = sum(avg_r_1.25),
            gamma1_r_125 = sum(gamma1_r_1.25),
            avg_q_125 = sum(avg_q_1.25),
            gamma1_q_125 = sum(gamma1_q_1.25),
            avg_cs_125 = sum(avg_cs_1.25),
            gamma1_cs_125 = sum(gamma1_cs_1.25),
            avg_ev_125 = sum(avg_ev_1.25),
            gamma1_ev_125 = sum(gamma1_ev_1.25))



sd(current_info_loss1_var_steps_rate_125_compare_bill_ym$r_0)
#[1] 2866018

sd(current_info_loss1_var_steps_rate_125_compare_bill_ym$avg_r_125)
#[1] 5549694

sd(current_info_loss1_var_steps_rate_125_compare_bill_ym$gamma1_r_125)
#[1] 5571152

### The sd actually didn't decrease

current_info_loss1_var_steps_rate_125_compare_bill_ym$avg_r_0 = mean(current_info_loss1_var_steps_rate_125_compare_bill_ym$r_0)
current_info_loss1_var_steps_rate_125_compare_bill_ym$avg_avg_r_125 = mean(current_info_loss1_var_steps_rate_125_compare_bill_ym$avg_r_125)
current_info_loss1_var_steps_rate_125_compare_bill_ym$avg_gamma1_r_125 = mean(current_info_loss1_var_steps_rate_125_compare_bill_ym$gamma1_r_125)

current_info_loss1_var_steps_rate_125_compare_bill_ym$log_gamma1_r_125 = log(current_info_loss1_var_steps_rate_125_compare_bill_ym$gamma1_r_125)
#### for concave bound, need to raise r for all months such that this more strict revenue constraint needs to be satisfied. 

#### Why specifically raise price for higher quantity users and lower price for lower quantity users?


current_info_loss1_var_steps_rate_125_compare_bill_ym_quantity$r_diff = current_info_loss1_var_steps_rate_125_compare_bill_ym_quantity$gamma1_r_125 - current_info_loss1_var_steps_rate_125_compare_bill_ym_quantity$avg_r_125

current_info_loss1_var_steps_rate_125_compare_bill_ym_quantity$q_diff = current_info_loss1_var_steps_rate_125_compare_bill_ym_quantity$gamma1_q_125 - current_info_loss1_var_steps_rate_125_compare_bill_ym_quantity$avg_q_125



ggplot(data = current_info_loss1_var_steps_rate_125_compare_bill_ym_quantity,
       aes(x = avg_r_125, y = gamma1_r_125)) +
  geom_point(aes(color = quantity_strata, shape = quantity_strata), size = 4, stroke = 1.2) + # Add the scatter plot layer
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) + # Add the y=x line
  scale_color_manual(values = custom_colors_q) +
  scale_shape_manual(values = custom_shapes_q) +
  labs(
    title = "Scatter Plot of gamma1_r_125 vs avg_r_125",
    x = "Average R 125",
    y = "Gamma1 R 125"
  ) +
  theme_minimal()


ggplot(data = current_info_loss1_var_steps_rate_125_compare_bill_ym_quantity,
       aes(x = avg_q_125, y = gamma1_q_125)) +
  geom_point(aes(color = quantity_strata, shape = quantity_strata), size = 4, stroke = 1.2) + # Add the scatter plot layer
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) + # Add the y=x line
  scale_color_manual(values = custom_colors_q) +
  scale_shape_manual(values = custom_shapes_q) +
  labs(
    title = "Scatter Plot of gamma1_r_125 vs avg_r_125",
    x = "Average Q 125",
    y = "Gamma1 Q 125"
  ) +
  theme_minimal()

unique_avg_r_0_value <- unique(current_info_loss1_var_steps_rate_125_compare_bill_ym$avg_r_0)

# Create a small data frame for this single point
additional_point_data <- data.frame(
  x_coord = c(unique(current_info_loss1_var_steps_rate_125_compare_bill_ym$avg_avg_r_125)
              #, unique(current_info_loss1_var_steps_rate_125_compare_bill_ym$avg_avg_r_125)
              ),
  y_coord = c(exp(mean(current_info_loss1_var_steps_rate_125_compare_bill_ym$log_gamma1_r_125) )
              #, unique(current_info_loss1_var_steps_rate_125_compare_bill_ym$avg_gamma1_r_125) )
              ))

# Create the plot
p = ggplot(data = current_info_loss1_var_steps_rate_125_compare_bill_ym,
       aes(x = avg_r_125, y = gamma1_r_125)) +
  geom_point(size = 4, stroke = 1.2) + # Add the scatter plot layer
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed", size = 1) + # Add the y=x line
  geom_vline(xintercept = unique_avg_r_0_value[1], color = "blue", linetype = "dotted", size = 1) + # Add the vertical line at unique(avg_r_0)
  geom_hline(yintercept = unique_avg_r_0_value[1], color = "blue", linetype = "dotted", size = 1) + # Add the vertical line at unique(avg_r_0)
  labs(
    title = "Scatter Plot of gamma1_r_125 vs avg_r_125 with reference lines",
    x = "Average R 125",
    y = "Gamma1 R 125"
  ) +
  theme_minimal()


  p <- p + geom_point(data = additional_point_data, aes(x = x_coord, y = y_coord),
                      color = "darkgreen", # Choose a distinct color
                      shape = 8,         # Choose a distinct shape (e.g., a star)
                      size = 6,          # Make the point larger
                      stroke = 1.5)      # Increase stroke for better visibility

current_info_loss1_var_steps_rate_125_compare_bill_ym$r_diff = current_info_loss1_var_steps_rate_125_compare_bill_ym$gamma1_r_125 - current_info_loss1_var_steps_rate_125_compare_bill_ym$avg_r_125
current_info_loss1_var_steps_rate_125_compare_bill_ym$q_diff = current_info_loss1_var_steps_rate_125_compare_bill_ym$gamma1_q_125 - current_info_loss1_var_steps_rate_125_compare_bill_ym$avg_q_125





