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


##### current_info_avg_bound_loss05_mean ########

current_info_avg_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/current_info_avg_bound_loss05_mean_cs_steps.csv")

colnames(current_info_avg_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss05_mean_cs_steps))*0.05-0.25)

current_info_avg_bound_loss05_mean_cs_steps_rate <-current_info_avg_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss05_mean_cs_steps)

current_info_avg_bound_loss05_mean_q_steps = read_csv("cs_detail_results/current_info_avg_bound_loss05_mean_q_steps.csv")

colnames(current_info_avg_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss05_mean_q_steps))*0.05-0.25)

current_info_avg_bound_loss05_mean_q_steps_rate <-current_info_avg_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss05_mean_q_steps)

current_info_avg_bound_loss05_mean_r_steps = read_csv("cs_detail_results/current_info_avg_bound_loss05_mean_r_steps.csv")

colnames(current_info_avg_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss05_mean_r_steps))*0.05-0.25)

current_info_avg_bound_loss05_mean_r_steps_rate <-current_info_avg_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss05_mean_r_steps)

current_info_avg_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/current_info_avg_bound_loss05_mean_ev_steps.csv")

colnames(current_info_avg_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss05_mean_ev_steps))*0.05-0.25)

current_info_avg_bound_loss05_mean_ev_steps_rate <-current_info_avg_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss05_mean_ev_steps)

current_info_avg_bound_loss05_mean_cs_steps_rate  = cbind(current_info_avg_bound_loss05_mean_cs_steps_rate , demand_key)
current_info_avg_bound_loss05_mean_q_steps_rate = cbind(current_info_avg_bound_loss05_mean_q_steps_rate, demand_key)
current_info_avg_bound_loss05_mean_r_steps_rate = cbind(current_info_avg_bound_loss05_mean_r_steps_rate, demand_key)
current_info_avg_bound_loss05_mean_ev_steps_rate = cbind(current_info_avg_bound_loss05_mean_ev_steps_rate, demand_key)

#current_info_avg_bound_loss05_mean_cs_steps_rate = current_info_avg_bound_loss05_mean_cs_steps_rate[which(current_info_avg_bound_loss05_mean_cs_steps_rate$q_0<100),]
#current_info_avg_bound_loss05_mean_ev_steps_rate = current_info_avg_bound_loss05_mean_ev_steps_rate[which(current_info_avg_bound_loss05_mean_ev_steps_rate$q_0<100),]

mean_cs_rate_by_bill_ym <- current_info_avg_bound_loss05_mean_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_avg_bound_loss05_mean_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_avg_bound_loss05_mean_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- current_info_avg_bound_loss05_mean_ev_steps_rate %>%
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

mean_ev_rate_by_strata$status = case_when(
  mean_ev_rate_by_strata$step<0 ~ "low",
  mean_ev_rate_by_strata$step>0 ~ "high"
)

mean_cs_rate_by_strata$status = case_when(
  mean_cs_rate_by_strata$step<0 ~ "low",
  mean_cs_rate_by_strata$step>0 ~ "high"
)

avg_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

avg_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


avg_bound_loss05_mean_grid <- ggarrange(cs_avg,r_avg,q_avg,ev_avg, ncol = 2, nrow = 2, 
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
avg_bound_loss05_mean_grid_with_title <- avg_bound_loss05_mean_grid +
  plot_annotation(
    title = "Linear Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )

##### current_info_avg_bound_loss05_var ########

current_info_avg_bound_loss05_var_cs_steps = read_csv("cs_detail_results/current_info_avg_bound_loss05_var_cs_steps.csv")

colnames(current_info_avg_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss05_var_cs_steps))/20+0.75)

current_info_avg_bound_loss05_var_cs_steps_rate <-current_info_avg_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss05_var_cs_steps)

current_info_avg_bound_loss05_var_q_steps = read_csv("cs_detail_results/current_info_avg_bound_loss05_var_q_steps.csv")

colnames(current_info_avg_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss05_var_q_steps))/20+0.75)

current_info_avg_bound_loss05_var_q_steps_rate <-current_info_avg_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss05_var_q_steps)

current_info_avg_bound_loss05_var_r_steps = read_csv("cs_detail_results/current_info_avg_bound_loss05_var_r_steps.csv")

colnames(current_info_avg_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss05_var_r_steps))/20+0.75)

current_info_avg_bound_loss05_var_r_steps_rate <-current_info_avg_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss05_var_r_steps)

current_info_avg_bound_loss05_var_ev_steps = read_csv("cs_detail_results/current_info_avg_bound_loss05_var_ev_steps.csv")

colnames(current_info_avg_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(current_info_avg_bound_loss05_var_ev_steps))/20+0.75)

current_info_avg_bound_loss05_var_ev_steps_rate <-current_info_avg_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_avg_bound_loss05_var_ev_steps)

current_info_avg_bound_loss05_var_cs_steps_rate  = cbind(current_info_avg_bound_loss05_var_cs_steps_rate , demand_key)
current_info_avg_bound_loss05_var_q_steps_rate = cbind(current_info_avg_bound_loss05_var_q_steps_rate, demand_key)
current_info_avg_bound_loss05_var_r_steps_rate = cbind(current_info_avg_bound_loss05_var_r_steps_rate, demand_key)
current_info_avg_bound_loss05_var_ev_steps_rate = cbind(current_info_avg_bound_loss05_var_ev_steps_rate, demand_key)

#current_info_avg_bound_loss05_var_cs_steps_rate = current_info_avg_bound_loss05_var_cs_steps_rate[which(current_info_avg_bound_loss05_var_cs_steps_rate$q_0<100),]
#current_info_avg_bound_loss05_var_ev_steps_rate = current_info_avg_bound_loss05_var_ev_steps_rate[which(current_info_avg_bound_loss05_var_ev_steps_rate$q_0<100),]

var_cs_rate_by_bill_ym <- current_info_avg_bound_loss05_var_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_avg_bound_loss05_var_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_avg_bound_loss05_var_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_avg_bound_loss05_var_ev_steps_rate %>%
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

var_ev_rate_by_strata$status = case_when(
  var_ev_rate_by_strata$step<1 ~ "low",
  var_ev_rate_by_strata$step>1 ~ "high"
)

var_cs_rate_by_strata$status = case_when(
  var_cs_rate_by_strata$step<1 ~ "low",
  var_cs_rate_by_strata$step>1 ~ "high"
)

avg_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

avg_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

avg_bound_loss05_var_grid <- ggarrange(cs_avg,r_avg,q_avg,ev_avg, ncol = 2, nrow = 2, 
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
avg_bound_loss05_var_grid_with_title <- avg_bound_loss05_var_grid +
  plot_annotation(
    title = "Linear Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )

##### current_info_gamma05_bound_loss05_mean ########

current_info_gamma05_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss05_mean_cs_steps.csv")

colnames(current_info_gamma05_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss05_mean_cs_steps))*0.05-0.25)

current_info_gamma05_bound_loss05_mean_cs_steps_rate <-current_info_gamma05_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss05_mean_cs_steps)

current_info_gamma05_bound_loss05_mean_q_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss05_mean_q_steps.csv")

colnames(current_info_gamma05_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss05_mean_q_steps))*0.05-0.25)

current_info_gamma05_bound_loss05_mean_q_steps_rate <-current_info_gamma05_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss05_mean_q_steps)

current_info_gamma05_bound_loss05_mean_r_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss05_mean_r_steps.csv")

colnames(current_info_gamma05_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss05_mean_r_steps))*0.05-0.25)

current_info_gamma05_bound_loss05_mean_r_steps_rate <-current_info_gamma05_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss05_mean_r_steps)

current_info_gamma05_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss05_mean_ev_steps.csv")

colnames(current_info_gamma05_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss05_mean_ev_steps))*0.05-0.25)

current_info_gamma05_bound_loss05_mean_ev_steps_rate <-current_info_gamma05_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss05_mean_ev_steps)

current_info_gamma05_bound_loss05_mean_cs_steps_rate  = cbind(current_info_gamma05_bound_loss05_mean_cs_steps_rate , demand_key)
current_info_gamma05_bound_loss05_mean_q_steps_rate = cbind(current_info_gamma05_bound_loss05_mean_q_steps_rate, demand_key)
current_info_gamma05_bound_loss05_mean_r_steps_rate = cbind(current_info_gamma05_bound_loss05_mean_r_steps_rate, demand_key)
current_info_gamma05_bound_loss05_mean_ev_steps_rate = cbind(current_info_gamma05_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_gamma05_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_gamma05_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_gamma05_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- current_info_gamma05_bound_loss05_mean_ev_steps_rate %>%
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

mean_ev_rate_by_strata$status = case_when(
  mean_ev_rate_by_strata$step<0 ~ "low",
  mean_ev_rate_by_strata$step>0 ~ "high"
)

mean_cs_rate_by_strata$status = case_when(
  mean_cs_rate_by_strata$step<0 ~ "low",
  mean_cs_rate_by_strata$step>0 ~ "high"
)

gamma05_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma05_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

gamma05_bound_loss05_mean_grid = ggarrange(cs_gamma05,r_gamma05,q_gamma05,ev_gamma05, ncol = 2, nrow = 2, 
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
gamma05_bound_loss05_mean_grid_with_title <- gamma05_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_mean_grid_with_title, gamma05_bound_loss05_mean_grid_with_title,
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
avg_gamma05_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                    ncol = 1, nrow = 2,
                                    heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### current_info_gamma05_bound_loss05_var ########

current_info_gamma05_bound_loss05_var_cs_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss05_var_cs_steps.csv")

colnames(current_info_gamma05_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss05_var_cs_steps))/20+0.75)

current_info_gamma05_bound_loss05_var_cs_steps_rate <-current_info_gamma05_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss05_var_cs_steps)

current_info_gamma05_bound_loss05_var_q_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss05_var_q_steps.csv")

colnames(current_info_gamma05_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss05_var_q_steps))/20+0.75)

current_info_gamma05_bound_loss05_var_q_steps_rate <-current_info_gamma05_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss05_var_q_steps)

current_info_gamma05_bound_loss05_var_r_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss05_var_r_steps.csv")

colnames(current_info_gamma05_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss05_var_r_steps))/20+0.75)

current_info_gamma05_bound_loss05_var_r_steps_rate <-current_info_gamma05_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss05_var_r_steps)

current_info_gamma05_bound_loss05_var_ev_steps = read_csv("cs_detail_results/current_info_gamma05_bound_loss05_var_ev_steps.csv")

colnames(current_info_gamma05_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(current_info_gamma05_bound_loss05_var_ev_steps))/20+0.75)

current_info_gamma05_bound_loss05_var_ev_steps_rate <-current_info_gamma05_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma05_bound_loss05_var_ev_steps)

current_info_gamma05_bound_loss05_var_cs_steps_rate  = cbind(current_info_gamma05_bound_loss05_var_cs_steps_rate , demand_key)
current_info_gamma05_bound_loss05_var_q_steps_rate = cbind(current_info_gamma05_bound_loss05_var_q_steps_rate, demand_key)
current_info_gamma05_bound_loss05_var_r_steps_rate = cbind(current_info_gamma05_bound_loss05_var_r_steps_rate, demand_key)
current_info_gamma05_bound_loss05_var_ev_steps_rate = cbind(current_info_gamma05_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_gamma05_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_gamma05_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_gamma05_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_gamma05_bound_loss05_var_ev_steps_rate %>%
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

var_ev_rate_by_strata$status = case_when(
  var_ev_rate_by_strata$step<1 ~ "low",
  var_ev_rate_by_strata$step>1 ~ "high"
)

var_cs_rate_by_strata$status = case_when(
  var_cs_rate_by_strata$step<1 ~ "low",
  var_cs_rate_by_strata$step>1 ~ "high"
)

gamma05_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma05_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


gamma05_bound_loss05_var_grid = ggarrange(cs_gamma05,r_gamma05,q_gamma05,ev_gamma05, ncol = 2, nrow = 2, 
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
gamma05_bound_loss05_var_grid_with_title <- gamma05_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_var_grid_with_title, gamma05_bound_loss05_var_grid_with_title,
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
avg_gamma05_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                   ncol = 1, nrow = 2,
                                   heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))





##### current_info_gamma04_bound_loss05_mean ########

current_info_gamma04_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/current_info_gamma04_bound_loss05_mean_cs_steps.csv")

colnames(current_info_gamma04_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(current_info_gamma04_bound_loss05_mean_cs_steps))*0.05-0.25)

current_info_gamma04_bound_loss05_mean_cs_steps_rate <-current_info_gamma04_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma04_bound_loss05_mean_cs_steps)

current_info_gamma04_bound_loss05_mean_q_steps = read_csv("cs_detail_results/current_info_gamma04_bound_loss05_mean_q_steps.csv")

colnames(current_info_gamma04_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(current_info_gamma04_bound_loss05_mean_q_steps))*0.05-0.25)

current_info_gamma04_bound_loss05_mean_q_steps_rate <-current_info_gamma04_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma04_bound_loss05_mean_q_steps)

current_info_gamma04_bound_loss05_mean_r_steps = read_csv("cs_detail_results/current_info_gamma04_bound_loss05_mean_r_steps.csv")

colnames(current_info_gamma04_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(current_info_gamma04_bound_loss05_mean_r_steps))*0.05-0.25)

current_info_gamma04_bound_loss05_mean_r_steps_rate <-current_info_gamma04_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma04_bound_loss05_mean_r_steps)

current_info_gamma04_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/current_info_gamma04_bound_loss05_mean_ev_steps.csv")

colnames(current_info_gamma04_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(current_info_gamma04_bound_loss05_mean_ev_steps))*0.05-0.25)

current_info_gamma04_bound_loss05_mean_ev_steps_rate <-current_info_gamma04_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma04_bound_loss05_mean_ev_steps)

current_info_gamma04_bound_loss05_mean_cs_steps_rate  = cbind(current_info_gamma04_bound_loss05_mean_cs_steps_rate , demand_key)
current_info_gamma04_bound_loss05_mean_q_steps_rate = cbind(current_info_gamma04_bound_loss05_mean_q_steps_rate, demand_key)
current_info_gamma04_bound_loss05_mean_r_steps_rate = cbind(current_info_gamma04_bound_loss05_mean_r_steps_rate, demand_key)
current_info_gamma04_bound_loss05_mean_ev_steps_rate = cbind(current_info_gamma04_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_gamma04_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_gamma04_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_gamma04_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- current_info_gamma04_bound_loss05_mean_ev_steps_rate %>%
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

cs_gamma04 = ggplot(mean_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma04 = ggplot(mean_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma04 = ggplot(mean_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma04 = ggplot(mean_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

mean_ev_rate_by_strata$status = case_when(
  mean_ev_rate_by_strata$step<0 ~ "low",
  mean_ev_rate_by_strata$step>0 ~ "high"
)

mean_cs_rate_by_strata$status = case_when(
  mean_cs_rate_by_strata$step<0 ~ "low",
  mean_cs_rate_by_strata$step>0 ~ "high"
)

gamma04_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma04_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


gamma04_bound_loss05_mean_grid = ggarrange(cs_gamma04,r_gamma04,q_gamma04,ev_gamma04, ncol = 2, nrow = 2, 
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
gamma04_bound_loss05_mean_grid_with_title <- gamma04_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_mean_grid_with_title, gamma04_bound_loss05_mean_grid_with_title,
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
avg_gamma04_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                      ncol = 1, nrow = 2,
                                      heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### current_info_gamma04_bound_loss05_var ########

current_info_gamma04_bound_loss05_var_cs_steps = read_csv("cs_detail_results/current_info_gamma04_bound_loss05_var_cs_steps.csv")

colnames(current_info_gamma04_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(current_info_gamma04_bound_loss05_var_cs_steps))/20+0.75)

current_info_gamma04_bound_loss05_var_cs_steps_rate <-current_info_gamma04_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma04_bound_loss05_var_cs_steps)

current_info_gamma04_bound_loss05_var_q_steps = read_csv("cs_detail_results/current_info_gamma04_bound_loss05_var_q_steps.csv")

colnames(current_info_gamma04_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(current_info_gamma04_bound_loss05_var_q_steps))/20+0.75)

current_info_gamma04_bound_loss05_var_q_steps_rate <-current_info_gamma04_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma04_bound_loss05_var_q_steps)

current_info_gamma04_bound_loss05_var_r_steps = read_csv("cs_detail_results/current_info_gamma04_bound_loss05_var_r_steps.csv")

colnames(current_info_gamma04_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(current_info_gamma04_bound_loss05_var_r_steps))/20+0.75)

current_info_gamma04_bound_loss05_var_r_steps_rate <-current_info_gamma04_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma04_bound_loss05_var_r_steps)

current_info_gamma04_bound_loss05_var_ev_steps = read_csv("cs_detail_results/current_info_gamma04_bound_loss05_var_ev_steps.csv")

colnames(current_info_gamma04_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(current_info_gamma04_bound_loss05_var_ev_steps))/20+0.75)

current_info_gamma04_bound_loss05_var_ev_steps_rate <-current_info_gamma04_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma04_bound_loss05_var_ev_steps)

current_info_gamma04_bound_loss05_var_cs_steps_rate  = cbind(current_info_gamma04_bound_loss05_var_cs_steps_rate , demand_key)
current_info_gamma04_bound_loss05_var_q_steps_rate = cbind(current_info_gamma04_bound_loss05_var_q_steps_rate, demand_key)
current_info_gamma04_bound_loss05_var_r_steps_rate = cbind(current_info_gamma04_bound_loss05_var_r_steps_rate, demand_key)
current_info_gamma04_bound_loss05_var_ev_steps_rate = cbind(current_info_gamma04_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_gamma04_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_gamma04_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_gamma04_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_gamma04_bound_loss05_var_ev_steps_rate %>%
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

cs_gamma04 = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma04 = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma04 = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma04 = ggplot(var_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

var_ev_rate_by_strata$status = case_when(
  var_ev_rate_by_strata$step<1 ~ "low",
  var_ev_rate_by_strata$step>1 ~ "high"
)

var_cs_rate_by_strata$status = case_when(
  var_cs_rate_by_strata$step<1 ~ "low",
  var_cs_rate_by_strata$step>1 ~ "high"
)


gamma04_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma04_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


gamma04_bound_loss05_var_grid = ggarrange(cs_gamma04,r_gamma04,q_gamma04,ev_gamma04, ncol = 2, nrow = 2, 
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
gamma04_bound_loss05_var_grid_with_title <- gamma04_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_var_grid_with_title, gamma04_bound_loss05_var_grid_with_title,
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
avg_gamma04_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                     ncol = 1, nrow = 2,
                                     heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))



##### current_info_gamma03_bound_loss05_mean ########

current_info_gamma03_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/current_info_gamma03_bound_loss05_mean_cs_steps.csv")

colnames(current_info_gamma03_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(current_info_gamma03_bound_loss05_mean_cs_steps))*0.05-0.25)

current_info_gamma03_bound_loss05_mean_cs_steps_rate <-current_info_gamma03_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma03_bound_loss05_mean_cs_steps)

current_info_gamma03_bound_loss05_mean_q_steps = read_csv("cs_detail_results/current_info_gamma03_bound_loss05_mean_q_steps.csv")

colnames(current_info_gamma03_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(current_info_gamma03_bound_loss05_mean_q_steps))*0.05-0.25)

current_info_gamma03_bound_loss05_mean_q_steps_rate <-current_info_gamma03_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma03_bound_loss05_mean_q_steps)

current_info_gamma03_bound_loss05_mean_r_steps = read_csv("cs_detail_results/current_info_gamma03_bound_loss05_mean_r_steps.csv")

colnames(current_info_gamma03_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(current_info_gamma03_bound_loss05_mean_r_steps))*0.05-0.25)

current_info_gamma03_bound_loss05_mean_r_steps_rate <-current_info_gamma03_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma03_bound_loss05_mean_r_steps)

current_info_gamma03_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/current_info_gamma03_bound_loss05_mean_ev_steps.csv")

colnames(current_info_gamma03_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(current_info_gamma03_bound_loss05_mean_ev_steps))*0.05-0.25)

current_info_gamma03_bound_loss05_mean_ev_steps_rate <-current_info_gamma03_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma03_bound_loss05_mean_ev_steps)

current_info_gamma03_bound_loss05_mean_cs_steps_rate  = cbind(current_info_gamma03_bound_loss05_mean_cs_steps_rate , demand_key)
current_info_gamma03_bound_loss05_mean_q_steps_rate = cbind(current_info_gamma03_bound_loss05_mean_q_steps_rate, demand_key)
current_info_gamma03_bound_loss05_mean_r_steps_rate = cbind(current_info_gamma03_bound_loss05_mean_r_steps_rate, demand_key)
current_info_gamma03_bound_loss05_mean_ev_steps_rate = cbind(current_info_gamma03_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_gamma03_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_gamma03_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_gamma03_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- current_info_gamma03_bound_loss05_mean_ev_steps_rate %>%
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

cs_gamma03 = ggplot(mean_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma03 = ggplot(mean_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma03 = ggplot(mean_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma03 = ggplot(mean_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

mean_ev_rate_by_strata$status = case_when(
  mean_ev_rate_by_strata$step<0 ~ "low",
  mean_ev_rate_by_strata$step>0 ~ "high"
)

mean_cs_rate_by_strata$status = case_when(
  mean_cs_rate_by_strata$step<0 ~ "low",
  mean_cs_rate_by_strata$step>0 ~ "high"
)

gamma03_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma03_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


gamma03_bound_loss05_mean_grid = ggarrange(cs_gamma03,r_gamma03,q_gamma03,ev_gamma03, ncol = 2, nrow = 2, 
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
gamma03_bound_loss05_mean_grid_with_title <- gamma03_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_mean_grid_with_title, gamma03_bound_loss05_mean_grid_with_title,
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
avg_gamma03_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                      ncol = 1, nrow = 2,
                                      heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### current_info_gamma03_bound_loss05_var ########

current_info_gamma03_bound_loss05_var_cs_steps = read_csv("cs_detail_results/current_info_gamma03_bound_loss05_var_cs_steps.csv")

colnames(current_info_gamma03_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(current_info_gamma03_bound_loss05_var_cs_steps))/20+0.75)

current_info_gamma03_bound_loss05_var_cs_steps_rate <-current_info_gamma03_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma03_bound_loss05_var_cs_steps)

current_info_gamma03_bound_loss05_var_q_steps = read_csv("cs_detail_results/current_info_gamma03_bound_loss05_var_q_steps.csv")

colnames(current_info_gamma03_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(current_info_gamma03_bound_loss05_var_q_steps))/20+0.75)

current_info_gamma03_bound_loss05_var_q_steps_rate <-current_info_gamma03_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma03_bound_loss05_var_q_steps)

current_info_gamma03_bound_loss05_var_r_steps = read_csv("cs_detail_results/current_info_gamma03_bound_loss05_var_r_steps.csv")

colnames(current_info_gamma03_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(current_info_gamma03_bound_loss05_var_r_steps))/20+0.75)

current_info_gamma03_bound_loss05_var_r_steps_rate <-current_info_gamma03_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma03_bound_loss05_var_r_steps)

current_info_gamma03_bound_loss05_var_ev_steps = read_csv("cs_detail_results/current_info_gamma03_bound_loss05_var_ev_steps.csv")

colnames(current_info_gamma03_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(current_info_gamma03_bound_loss05_var_ev_steps))/20+0.75)

current_info_gamma03_bound_loss05_var_ev_steps_rate <-current_info_gamma03_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma03_bound_loss05_var_ev_steps)

current_info_gamma03_bound_loss05_var_cs_steps_rate  = cbind(current_info_gamma03_bound_loss05_var_cs_steps_rate , demand_key)
current_info_gamma03_bound_loss05_var_q_steps_rate = cbind(current_info_gamma03_bound_loss05_var_q_steps_rate, demand_key)
current_info_gamma03_bound_loss05_var_r_steps_rate = cbind(current_info_gamma03_bound_loss05_var_r_steps_rate, demand_key)
current_info_gamma03_bound_loss05_var_ev_steps_rate = cbind(current_info_gamma03_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_gamma03_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_gamma03_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_gamma03_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_gamma03_bound_loss05_var_ev_steps_rate %>%
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

cs_gamma03 = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma03 = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma03 = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma03 = ggplot(var_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

var_ev_rate_by_strata$status = case_when(
  var_ev_rate_by_strata$step<1 ~ "low",
  var_ev_rate_by_strata$step>1 ~ "high"
)

var_cs_rate_by_strata$status = case_when(
  var_cs_rate_by_strata$step<1 ~ "low",
  var_cs_rate_by_strata$step>1 ~ "high"
)


gamma03_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma03_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


gamma03_bound_loss05_var_grid = ggarrange(cs_gamma03,r_gamma03,q_gamma03,ev_gamma03, ncol = 2, nrow = 2, 
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
gamma03_bound_loss05_var_grid_with_title <- gamma03_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_var_grid_with_title, gamma03_bound_loss05_var_grid_with_title,
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
avg_gamma03_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                     ncol = 1, nrow = 2,
                                     heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))



##### current_info_gamma025_bound_loss05_mean ########

current_info_gamma025_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/current_info_gamma025_bound_loss05_mean_cs_steps.csv")

colnames(current_info_gamma025_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(current_info_gamma025_bound_loss05_mean_cs_steps))*0.05-0.25)

current_info_gamma025_bound_loss05_mean_cs_steps_rate <-current_info_gamma025_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma025_bound_loss05_mean_cs_steps)

current_info_gamma025_bound_loss05_mean_q_steps = read_csv("cs_detail_results/current_info_gamma025_bound_loss05_mean_q_steps.csv")

colnames(current_info_gamma025_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(current_info_gamma025_bound_loss05_mean_q_steps))*0.05-0.25)

current_info_gamma025_bound_loss05_mean_q_steps_rate <-current_info_gamma025_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma025_bound_loss05_mean_q_steps)

current_info_gamma025_bound_loss05_mean_r_steps = read_csv("cs_detail_results/current_info_gamma025_bound_loss05_mean_r_steps.csv")

colnames(current_info_gamma025_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(current_info_gamma025_bound_loss05_mean_r_steps))*0.05-0.25)

current_info_gamma025_bound_loss05_mean_r_steps_rate <-current_info_gamma025_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma025_bound_loss05_mean_r_steps)

current_info_gamma025_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/current_info_gamma025_bound_loss05_mean_ev_steps.csv")

colnames(current_info_gamma025_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(current_info_gamma025_bound_loss05_mean_ev_steps))*0.05-0.25)

current_info_gamma025_bound_loss05_mean_ev_steps_rate <-current_info_gamma025_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma025_bound_loss05_mean_ev_steps)

current_info_gamma025_bound_loss05_mean_cs_steps_rate  = cbind(current_info_gamma025_bound_loss05_mean_cs_steps_rate , demand_key)
current_info_gamma025_bound_loss05_mean_q_steps_rate = cbind(current_info_gamma025_bound_loss05_mean_q_steps_rate, demand_key)
current_info_gamma025_bound_loss05_mean_r_steps_rate = cbind(current_info_gamma025_bound_loss05_mean_r_steps_rate, demand_key)
current_info_gamma025_bound_loss05_mean_ev_steps_rate = cbind(current_info_gamma025_bound_loss05_mean_ev_steps_rate, demand_key)

#current_info_gamma025_bound_loss05_mean_cs_steps_rate = current_info_gamma025_bound_loss05_mean_cs_steps_rate[which(current_info_gamma025_bound_loss05_mean_cs_steps_rate$q_0<100),]
#current_info_gamma025_bound_loss05_mean_ev_steps_rate = current_info_gamma025_bound_loss05_mean_ev_steps_rate[which(current_info_gamma025_bound_loss05_mean_ev_steps_rate$q_0<100),]

mean_cs_rate_by_bill_ym <- current_info_gamma025_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_gamma025_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_gamma025_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- current_info_gamma025_bound_loss05_mean_ev_steps_rate %>%
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

cs_gamma025 = ggplot(mean_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma025 = ggplot(mean_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma025 = ggplot(mean_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma025 = ggplot(mean_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

mean_ev_rate_by_strata$status = case_when(
  mean_ev_rate_by_strata$step<0 ~ "low",
  mean_ev_rate_by_strata$step>0 ~ "high"
)

mean_cs_rate_by_strata$status = case_when(
  mean_cs_rate_by_strata$step<0 ~ "low",
  mean_cs_rate_by_strata$step>0 ~ "high"
)

gamma025_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma025_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


gamma025_bound_loss05_mean_grid = ggarrange(cs_gamma025,r_gamma025,q_gamma025,ev_gamma025, ncol = 2, nrow = 2, 
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
gamma025_bound_loss05_mean_grid_with_title <- gamma025_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_mean_grid_with_title, gamma025_bound_loss05_mean_grid_with_title,
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
avg_gamma025_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                     ncol = 1, nrow = 2,
                                     heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### current_info_gamma025_bound_loss05_var ########

current_info_gamma025_bound_loss05_var_cs_steps = read_csv("cs_detail_results/current_info_gamma025_bound_loss05_var_cs_steps.csv")

colnames(current_info_gamma025_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(current_info_gamma025_bound_loss05_var_cs_steps))/20+0.75)

current_info_gamma025_bound_loss05_var_cs_steps_rate <-current_info_gamma025_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma025_bound_loss05_var_cs_steps)

current_info_gamma025_bound_loss05_var_q_steps = read_csv("cs_detail_results/current_info_gamma025_bound_loss05_var_q_steps.csv")

colnames(current_info_gamma025_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(current_info_gamma025_bound_loss05_var_q_steps))/20+0.75)

current_info_gamma025_bound_loss05_var_q_steps_rate <-current_info_gamma025_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma025_bound_loss05_var_q_steps)

current_info_gamma025_bound_loss05_var_r_steps = read_csv("cs_detail_results/current_info_gamma025_bound_loss05_var_r_steps.csv")

colnames(current_info_gamma025_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(current_info_gamma025_bound_loss05_var_r_steps))/20+0.75)

current_info_gamma025_bound_loss05_var_r_steps_rate <-current_info_gamma025_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma025_bound_loss05_var_r_steps)

current_info_gamma025_bound_loss05_var_ev_steps = read_csv("cs_detail_results/current_info_gamma025_bound_loss05_var_ev_steps.csv")

colnames(current_info_gamma025_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(current_info_gamma025_bound_loss05_var_ev_steps))/20+0.75)

current_info_gamma025_bound_loss05_var_ev_steps_rate <-current_info_gamma025_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma025_bound_loss05_var_ev_steps)

current_info_gamma025_bound_loss05_var_cs_steps_rate  = cbind(current_info_gamma025_bound_loss05_var_cs_steps_rate , demand_key)
current_info_gamma025_bound_loss05_var_q_steps_rate = cbind(current_info_gamma025_bound_loss05_var_q_steps_rate, demand_key)
current_info_gamma025_bound_loss05_var_r_steps_rate = cbind(current_info_gamma025_bound_loss05_var_r_steps_rate, demand_key)
current_info_gamma025_bound_loss05_var_ev_steps_rate = cbind(current_info_gamma025_bound_loss05_var_ev_steps_rate, demand_key)

#current_info_gamma025_bound_loss05_var_cs_steps_rate = current_info_gamma025_bound_loss05_var_cs_steps_rate[which(current_info_gamma025_bound_loss05_var_cs_steps_rate$q_0<100),]
#current_info_gamma025_bound_loss05_var_ev_steps_rate = current_info_gamma025_bound_loss05_var_ev_steps_rate[which(current_info_gamma025_bound_loss05_var_ev_steps_rate$q_0<100),]


var_cs_rate_by_bill_ym <- current_info_gamma025_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_gamma025_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_gamma025_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_gamma025_bound_loss05_var_ev_steps_rate %>%
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

cs_gamma025 = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma025 = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma025 = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma025 = ggplot(var_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

var_ev_rate_by_strata$status = case_when(
  var_ev_rate_by_strata$step<1 ~ "low",
  var_ev_rate_by_strata$step>1 ~ "high"
)

var_cs_rate_by_strata$status = case_when(
  var_cs_rate_by_strata$step<1 ~ "low",
  var_cs_rate_by_strata$step>1 ~ "high"
)


gamma025_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma025_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


gamma025_bound_loss05_var_grid = ggarrange(cs_gamma025,r_gamma025,q_gamma025,ev_gamma025, ncol = 2, nrow = 2, 
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
gamma025_bound_loss05_var_grid_with_title <- gamma025_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_var_grid_with_title, gamma025_bound_loss05_var_grid_with_title,
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
avg_gamma025_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                    ncol = 1, nrow = 2,
                                    heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))



##### current_info_gamma02_bound_loss05_mean ########

current_info_gamma02_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/current_info_gamma02_bound_loss05_mean_cs_steps.csv")

colnames(current_info_gamma02_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(current_info_gamma02_bound_loss05_mean_cs_steps))*0.05-0.25)

current_info_gamma02_bound_loss05_mean_cs_steps_rate <-current_info_gamma02_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma02_bound_loss05_mean_cs_steps)

current_info_gamma02_bound_loss05_mean_q_steps = read_csv("cs_detail_results/current_info_gamma02_bound_loss05_mean_q_steps.csv")

colnames(current_info_gamma02_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(current_info_gamma02_bound_loss05_mean_q_steps))*0.05-0.25)

current_info_gamma02_bound_loss05_mean_q_steps_rate <-current_info_gamma02_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma02_bound_loss05_mean_q_steps)

current_info_gamma02_bound_loss05_mean_r_steps = read_csv("cs_detail_results/current_info_gamma02_bound_loss05_mean_r_steps.csv")

colnames(current_info_gamma02_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(current_info_gamma02_bound_loss05_mean_r_steps))*0.05-0.25)

current_info_gamma02_bound_loss05_mean_r_steps_rate <-current_info_gamma02_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma02_bound_loss05_mean_r_steps)

current_info_gamma02_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/current_info_gamma02_bound_loss05_mean_ev_steps.csv")

colnames(current_info_gamma02_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(current_info_gamma02_bound_loss05_mean_ev_steps))*0.05-0.25)

current_info_gamma02_bound_loss05_mean_ev_steps_rate <-current_info_gamma02_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma02_bound_loss05_mean_ev_steps)

current_info_gamma02_bound_loss05_mean_cs_steps_rate  = cbind(current_info_gamma02_bound_loss05_mean_cs_steps_rate , demand_key)
current_info_gamma02_bound_loss05_mean_q_steps_rate = cbind(current_info_gamma02_bound_loss05_mean_q_steps_rate, demand_key)
current_info_gamma02_bound_loss05_mean_r_steps_rate = cbind(current_info_gamma02_bound_loss05_mean_r_steps_rate, demand_key)
current_info_gamma02_bound_loss05_mean_ev_steps_rate = cbind(current_info_gamma02_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_gamma02_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_gamma02_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_gamma02_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- current_info_gamma02_bound_loss05_mean_ev_steps_rate %>%
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

cs_gamma02 = ggplot(mean_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma02 = ggplot(mean_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma02 = ggplot(mean_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma02 = ggplot(mean_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

mean_ev_rate_by_strata$status = case_when(
  mean_ev_rate_by_strata$step<0 ~ "low",
  mean_ev_rate_by_strata$step>0 ~ "high"
)

mean_cs_rate_by_strata$status = case_when(
  mean_cs_rate_by_strata$step<0 ~ "low",
  mean_cs_rate_by_strata$step>0 ~ "high"
)

gamma02_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma02_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


gamma02_bound_loss05_mean_grid = ggarrange(cs_gamma02,r_gamma02,q_gamma02,ev_gamma02, ncol = 2, nrow = 2, 
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
gamma02_bound_loss05_mean_grid_with_title <- gamma02_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_mean_grid_with_title, gamma02_bound_loss05_mean_grid_with_title,
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
avg_gamma02_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                      ncol = 1, nrow = 2,
                                      heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### current_info_gamma02_bound_loss05_var ########

current_info_gamma02_bound_loss05_var_cs_steps = read_csv("cs_detail_results/current_info_gamma02_bound_loss05_var_cs_steps.csv")

colnames(current_info_gamma02_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(current_info_gamma02_bound_loss05_var_cs_steps))/20+0.75)

current_info_gamma02_bound_loss05_var_cs_steps_rate <-current_info_gamma02_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma02_bound_loss05_var_cs_steps)

current_info_gamma02_bound_loss05_var_q_steps = read_csv("cs_detail_results/current_info_gamma02_bound_loss05_var_q_steps.csv")

colnames(current_info_gamma02_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(current_info_gamma02_bound_loss05_var_q_steps))/20+0.75)

current_info_gamma02_bound_loss05_var_q_steps_rate <-current_info_gamma02_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma02_bound_loss05_var_q_steps)

current_info_gamma02_bound_loss05_var_r_steps = read_csv("cs_detail_results/current_info_gamma02_bound_loss05_var_r_steps.csv")

colnames(current_info_gamma02_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(current_info_gamma02_bound_loss05_var_r_steps))/20+0.75)

current_info_gamma02_bound_loss05_var_r_steps_rate <-current_info_gamma02_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma02_bound_loss05_var_r_steps)

current_info_gamma02_bound_loss05_var_ev_steps = read_csv("cs_detail_results/current_info_gamma02_bound_loss05_var_ev_steps.csv")

colnames(current_info_gamma02_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(current_info_gamma02_bound_loss05_var_ev_steps))/20+0.75)

current_info_gamma02_bound_loss05_var_ev_steps_rate <-current_info_gamma02_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma02_bound_loss05_var_ev_steps)

current_info_gamma02_bound_loss05_var_cs_steps_rate  = cbind(current_info_gamma02_bound_loss05_var_cs_steps_rate , demand_key)
current_info_gamma02_bound_loss05_var_q_steps_rate = cbind(current_info_gamma02_bound_loss05_var_q_steps_rate, demand_key)
current_info_gamma02_bound_loss05_var_r_steps_rate = cbind(current_info_gamma02_bound_loss05_var_r_steps_rate, demand_key)
current_info_gamma02_bound_loss05_var_ev_steps_rate = cbind(current_info_gamma02_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_gamma02_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_gamma02_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_gamma02_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_gamma02_bound_loss05_var_ev_steps_rate %>%
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

cs_gamma02 = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma02 = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma02 = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma02 = ggplot(var_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

var_ev_rate_by_strata$status = case_when(
  var_ev_rate_by_strata$step<1 ~ "low",
  var_ev_rate_by_strata$step>1 ~ "high"
)

var_cs_rate_by_strata$status = case_when(
  var_cs_rate_by_strata$step<1 ~ "low",
  var_cs_rate_by_strata$step>1 ~ "high"
)


gamma02_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma02_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


gamma02_bound_loss05_var_grid = ggarrange(cs_gamma02,r_gamma02,q_gamma02,ev_gamma02, ncol = 2, nrow = 2, 
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
gamma02_bound_loss05_var_grid_with_title <- gamma02_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_var_grid_with_title, gamma02_bound_loss05_var_grid_with_title,
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
avg_gamma02_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                     ncol = 1, nrow = 2,
                                     heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))



##### current_info_gamma01_bound_loss05_mean ########

current_info_gamma01_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/current_info_gamma01_bound_loss05_mean_cs_steps.csv")

colnames(current_info_gamma01_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(current_info_gamma01_bound_loss05_mean_cs_steps))*0.05-0.25)

current_info_gamma01_bound_loss05_mean_cs_steps_rate <-current_info_gamma01_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma01_bound_loss05_mean_cs_steps)

current_info_gamma01_bound_loss05_mean_q_steps = read_csv("cs_detail_results/current_info_gamma01_bound_loss05_mean_q_steps.csv")

colnames(current_info_gamma01_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(current_info_gamma01_bound_loss05_mean_q_steps))*0.05-0.25)

current_info_gamma01_bound_loss05_mean_q_steps_rate <-current_info_gamma01_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma01_bound_loss05_mean_q_steps)

current_info_gamma01_bound_loss05_mean_r_steps = read_csv("cs_detail_results/current_info_gamma01_bound_loss05_mean_r_steps.csv")

colnames(current_info_gamma01_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(current_info_gamma01_bound_loss05_mean_r_steps))*0.05-0.25)

current_info_gamma01_bound_loss05_mean_r_steps_rate <-current_info_gamma01_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma01_bound_loss05_mean_r_steps)

current_info_gamma01_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/current_info_gamma01_bound_loss05_mean_ev_steps.csv")

colnames(current_info_gamma01_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(current_info_gamma01_bound_loss05_mean_ev_steps))*0.05-0.25)

current_info_gamma01_bound_loss05_mean_ev_steps_rate <-current_info_gamma01_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma01_bound_loss05_mean_ev_steps)

current_info_gamma01_bound_loss05_mean_cs_steps_rate  = cbind(current_info_gamma01_bound_loss05_mean_cs_steps_rate , demand_key)
current_info_gamma01_bound_loss05_mean_q_steps_rate = cbind(current_info_gamma01_bound_loss05_mean_q_steps_rate, demand_key)
current_info_gamma01_bound_loss05_mean_r_steps_rate = cbind(current_info_gamma01_bound_loss05_mean_r_steps_rate, demand_key)
current_info_gamma01_bound_loss05_mean_ev_steps_rate = cbind(current_info_gamma01_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_gamma01_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_gamma01_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_gamma01_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- current_info_gamma01_bound_loss05_mean_ev_steps_rate %>%
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


cs_gamma01 = ggplot(mean_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma01 = ggplot(mean_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma01 = ggplot(mean_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma01 = ggplot(mean_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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


mean_ev_rate_by_strata$status = case_when(
  mean_ev_rate_by_strata$step<0 ~ "low",
  mean_ev_rate_by_strata$step>0 ~ "high"
)

mean_cs_rate_by_strata$status = case_when(
  mean_cs_rate_by_strata$step<0 ~ "low",
  mean_cs_rate_by_strata$step>0 ~ "high"
)


gamma01_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma01_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


gamma01_bound_loss05_mean_grid = ggarrange(cs_gamma01,r_gamma01,q_gamma01,ev_gamma01, ncol = 2, nrow = 2, 
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
gamma01_bound_loss05_mean_grid_with_title <- gamma01_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_mean_grid_with_title, gamma01_bound_loss05_mean_grid_with_title,
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
avg_gamma01_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                      ncol = 1, nrow = 2,
                                      heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### current_info_gamma01_bound_loss05_var ########

current_info_gamma01_bound_loss05_var_cs_steps = read_csv("cs_detail_results/current_info_gamma01_bound_loss05_var_cs_steps.csv")

colnames(current_info_gamma01_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(current_info_gamma01_bound_loss05_var_cs_steps))/20+0.75)

current_info_gamma01_bound_loss05_var_cs_steps_rate <-current_info_gamma01_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma01_bound_loss05_var_cs_steps)

current_info_gamma01_bound_loss05_var_q_steps = read_csv("cs_detail_results/current_info_gamma01_bound_loss05_var_q_steps.csv")

colnames(current_info_gamma01_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(current_info_gamma01_bound_loss05_var_q_steps))/20+0.75)

current_info_gamma01_bound_loss05_var_q_steps_rate <-current_info_gamma01_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma01_bound_loss05_var_q_steps)

current_info_gamma01_bound_loss05_var_r_steps = read_csv("cs_detail_results/current_info_gamma01_bound_loss05_var_r_steps.csv")

colnames(current_info_gamma01_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(current_info_gamma01_bound_loss05_var_r_steps))/20+0.75)

current_info_gamma01_bound_loss05_var_r_steps_rate <-current_info_gamma01_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma01_bound_loss05_var_r_steps)

current_info_gamma01_bound_loss05_var_ev_steps = read_csv("cs_detail_results/current_info_gamma01_bound_loss05_var_ev_steps.csv")

colnames(current_info_gamma01_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(current_info_gamma01_bound_loss05_var_ev_steps))/20+0.75)

current_info_gamma01_bound_loss05_var_ev_steps_rate <-current_info_gamma01_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma01_bound_loss05_var_ev_steps)

current_info_gamma01_bound_loss05_var_cs_steps_rate  = cbind(current_info_gamma01_bound_loss05_var_cs_steps_rate , demand_key)
current_info_gamma01_bound_loss05_var_q_steps_rate = cbind(current_info_gamma01_bound_loss05_var_q_steps_rate, demand_key)
current_info_gamma01_bound_loss05_var_r_steps_rate = cbind(current_info_gamma01_bound_loss05_var_r_steps_rate, demand_key)
current_info_gamma01_bound_loss05_var_ev_steps_rate = cbind(current_info_gamma01_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_gamma01_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_gamma01_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_gamma01_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_gamma01_bound_loss05_var_ev_steps_rate %>%
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

cs_gamma01 = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma01 = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma01 = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma01 = ggplot(var_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

var_ev_rate_by_strata$status = case_when(
  var_ev_rate_by_strata$step<1 ~ "low",
  var_ev_rate_by_strata$step>1 ~ "high"
)

var_cs_rate_by_strata$status = case_when(
  var_cs_rate_by_strata$step<1 ~ "low",
  var_cs_rate_by_strata$step>1 ~ "high"
)


gamma01_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma01_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


gamma01_bound_loss05_var_grid = ggarrange(cs_gamma01,r_gamma01,q_gamma01,ev_gamma01, ncol = 2, nrow = 2, 
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
gamma01_bound_loss05_var_grid_with_title <- gamma01_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_var_grid_with_title, gamma01_bound_loss05_var_grid_with_title,
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
avg_gamma01_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                     ncol = 1, nrow = 2,
                                     heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))











##### Individual Level ####

avg_mean_high_stratus = current_info_avg_bound_loss05_mean_ev_steps_rate[which(current_info_avg_bound_loss05_mean_ev_steps_rate$income_strata==">100k"),]
avg_mean_low_stratus = current_info_avg_bound_loss05_mean_ev_steps_rate[which(current_info_avg_bound_loss05_mean_ev_steps_rate$income_strata=="0~6k"),]

gamma025_mean_high_stratus = current_info_gamma025_bound_loss05_mean_ev_steps_rate[which(current_info_gamma025_bound_loss05_mean_ev_steps_rate$income_strata==">100k"),]
gamma025_mean_low_stratus = current_info_gamma025_bound_loss05_mean_ev_steps_rate[which(current_info_gamma025_bound_loss05_mean_ev_steps_rate$income_strata=="0~6k"),]

avg_low = ggplot(avg_mean_low_stratus, aes(x = q_0, y = rate_0.25)) +
  geom_hex(bins = 75, color = "white", size = 0.1) +
  scale_fill_viridis_c(trans = "log10",
                       option = "D",
                       direction = -1,
                       bquote(log[10](Count~of~Observations))) +
  coord_cartesian(#xlim = c(0, 500),
                  ylim = c(-90, 10)
                  ) +

  labs(title = "Linear Constraint - 0~6k", # Updated title to reflect rate_0.25
    x = "Initial Quantity (q_0) (kGal)",
    y =expression("EV/I when "~zeta[1] ~ "=0.25"), # Updated y-axis label to reflect rate_0.25
    fill = "Log10(Count of Observations)") +
  
  academic_theme +
  theme(plot.title = element_text(hjust = 0.5))

gamma025_low = ggplot(gamma025_mean_low_stratus, aes(x = q_0, y = rate_0.25)) +
  geom_hex(bins = 75, color = "white", size = 0.1) +
  scale_fill_viridis_c(trans = "log10",
                       option = "D",
                       direction = -1,
                       bquote(log[10](Count~of~Observations))) +
  coord_cartesian(#xlim = c(0, 500),
                  ylim = c(-90, 10)
                  ) +
  
  labs(title = "Concave Constraint - 0~6k",
    x = "Initial Quantity (q_0) (kGal)",
    y =expression("EV/I when "~zeta[1] ~ "=0.25"), # Updated y-axis label to reflect rate_0.25
    fill = "Log10(Count of Observations)") +
  
  academic_theme +
  theme(plot.title = element_text(hjust = 0.5))

avg_gamma025_low <- ggarrange(
  avg_low,
  gamma025_low,
  ncol = 2, # Arrange in two columns
  common.legend = TRUE, # Use a single legend for both plots
  legend = "bottom" # Place the common legend on the right
)


avg_high = ggplot(avg_mean_high_stratus, aes(x = q_0, y = rate_0.25)) +
  geom_hex(bins = 75, color = "white", size = 0.1) +
  scale_fill_viridis_c(trans = "log10",
                       option = "D",
                       direction = -1,
                       bquote(log[10](Count~of~Observations))) +
  coord_cartesian(#xlim = c(0, 500),
                  ylim = c(-9, 1)
                  ) +
  
  labs(title = "Linear Constraint - >100k", # Updated title to reflect rate_0.25
    x = "Initial Quantity (q_0) (kGal)",
    y =expression("EV/I when "~zeta[1] ~ "=0.25"), # Updated y-axis label to reflect rate_0.25
    fill = "Log10(Count of Observations)") +
  
  academic_theme +
  theme(plot.title = element_text(hjust = 0.5))

gamma025_high = ggplot(gamma025_mean_high_stratus, aes(x = q_0, y = rate_0.25)) +
  geom_hex(bins = 75, color = "white", size = 0.1) +
  scale_fill_viridis_c(trans = "log10",
                       option = "D",
                       direction = -1,
                       bquote(log[10](Count~of~Observations))) +
  coord_cartesian(#xlim = c(0, 500),
                  ylim = c(-9, 1)
                  ) +
  
  labs(title = "Concave Constraint- >100k",
    x = "Initial Quantity (q_0) (kGal)",
    y =expression("EV/I when "~zeta[1] ~ "=0.25"), # Updated y-axis label to reflect rate_0.25
    fill = "Log10(Count of Observations)") +
  
  academic_theme +
  theme(plot.title = element_text(hjust = 0.5))

avg_gamma025_high <- ggarrange(
  avg_high,
  gamma025_high,
  ncol = 2, # Arrange in two columns
  common.legend = TRUE, # Use a single legend for both plots
  legend = "bottom" # Place the common legend on the right
)




#### Rm Sd decrease ####
avg_mean_bill_ym  = current_info_avg_bound_loss05_mean_r_steps_rate %>%
  group_by(bill_ym) %>%
  summarise(across(1:11, sum, na.rm = TRUE))

avg_mean_bill_ym = avg_mean_bill_ym - 28058192.92994599

colSums(avg_mean_bill_ym)

gamma025_mean_bill_ym  = current_info_gamma025_bound_loss05_mean_r_steps_rate %>%
  group_by(bill_ym) %>%
  summarise(across(1:11, sum, na.rm = TRUE))

gamma025_mean_bill_ym = gamma025_mean_bill_ym - 28058192.92994599

colSums(gamma025_mean_bill_ym) - colSums(avg_mean_bill_ym)

avg_var_bill_ym  = current_info_avg_bound_loss05_var_r_steps_rate %>%
  group_by(bill_ym) %>%
  summarise(across(1:11, sum, na.rm = TRUE))

avg_var_bill_ym = avg_var_bill_ym - 28058192.92994599

colSums(avg_var_bill_ym)

gamma025_var_bill_ym  = current_info_gamma025_bound_loss05_var_r_steps_rate %>%
  group_by(bill_ym) %>%
  summarise(across(1:11, sum, na.rm = TRUE))

gamma025_var_bill_ym = gamma025_var_bill_ym - 28058192.92994599

colSums(gamma025_var_bill_ym) - colSums(avg_var_bill_ym)



##### Trade Off Study #####

avg_bound_loss05_mean_ev$gamma = 0
gamma01_bound_loss05_mean_ev$gamma = 0.1
gamma02_bound_loss05_mean_ev$gamma = 0.2
gamma025_bound_loss05_mean_ev$gamma = 0.25
gamma03_bound_loss05_mean_ev$gamma = 0.3
gamma04_bound_loss05_mean_ev$gamma = 0.4
gamma05_bound_loss05_mean_ev$gamma = 0.5
loss05_mean_ev = rbind(avg_bound_loss05_mean_ev,gamma01_bound_loss05_mean_ev,
                       gamma02_bound_loss05_mean_ev,
                       gamma025_bound_loss05_mean_ev, 
                       gamma03_bound_loss05_mean_ev,gamma04_bound_loss05_mean_ev,
                       gamma05_bound_loss05_mean_ev)

avg_bound_loss05_var_ev$gamma = 0
gamma01_bound_loss05_var_ev$gamma = 0.1
gamma02_bound_loss05_var_ev$gamma = 0.2
gamma025_bound_loss05_var_ev$gamma = 0.25
gamma03_bound_loss05_var_ev$gamma = 0.3
gamma04_bound_loss05_var_ev$gamma = 0.4
gamma05_bound_loss05_var_ev$gamma = 0.5
loss05_var_ev = rbind(avg_bound_loss05_var_ev,gamma01_bound_loss05_var_ev,
                       gamma02_bound_loss05_var_ev,
                       gamma025_bound_loss05_var_ev, 
                       gamma03_bound_loss05_var_ev,gamma04_bound_loss05_var_ev,
                       gamma05_bound_loss05_var_ev)

avg_bound_loss05_mean_cs$gamma = 0
gamma01_bound_loss05_mean_cs$gamma = 0.1
gamma02_bound_loss05_mean_cs$gamma = 0.2
gamma025_bound_loss05_mean_cs$gamma = 0.25
gamma03_bound_loss05_mean_cs$gamma = 0.3
gamma04_bound_loss05_mean_cs$gamma = 0.4
gamma05_bound_loss05_mean_cs$gamma = 0.5
loss05_mean_cs = rbind(avg_bound_loss05_mean_cs,gamma01_bound_loss05_mean_cs,
                       gamma02_bound_loss05_mean_cs,
                       gamma025_bound_loss05_mean_cs, 
                       gamma03_bound_loss05_mean_cs,gamma04_bound_loss05_mean_cs,
                       gamma05_bound_loss05_mean_cs)

avg_bound_loss05_var_cs$gamma = 0
gamma01_bound_loss05_var_cs$gamma = 0.1
gamma02_bound_loss05_var_cs$gamma = 0.2
gamma025_bound_loss05_var_cs$gamma = 0.25
gamma03_bound_loss05_var_cs$gamma = 0.3
gamma04_bound_loss05_var_cs$gamma = 0.4
gamma05_bound_loss05_var_cs$gamma = 0.5
loss05_var_cs = rbind(avg_bound_loss05_var_cs,gamma01_bound_loss05_var_cs,
                      gamma02_bound_loss05_var_cs,
                      gamma025_bound_loss05_var_cs, 
                      gamma03_bound_loss05_var_cs,gamma04_bound_loss05_var_cs,
                      gamma05_bound_loss05_var_cs)

write.csv(loss05_mean_ev, "trade_off_study/loss05_mean_ev.csv", row.names = FALSE)
write.csv(loss05_var_ev, "trade_off_study/loss05_var_ev.csv", row.names = FALSE)

write.csv(loss05_mean_cs, "trade_off_study/loss05_mean_cs.csv", row.names = FALSE)
write.csv(loss05_var_cs, "trade_off_study/loss05_var_cs.csv", row.names = FALSE)

##### Trade Off Study Graph All Strata #####

data_filter = loss05_mean_ev %>%
  filter(status == "low")

low_mean = ggplot(data_filter , aes(x = gamma, y = mean_ev_perct, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(data_filter, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  #geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_cartesian(ylim = c(-10, -3.5))+
  labs(
    title = expression(zeta[1] < 0),
    x = expression(gamma),
    y = expression("EV %"),
    color = NULL, 
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

data_filter = loss05_mean_ev %>%
  filter(status == "high")

high_mean = ggplot(data_filter , aes(x = gamma, y = mean_ev_perct, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(data_filter, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  #geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_cartesian(ylim = c(-10, -3.5))+
  labs(
    title = expression(zeta[1] > 0),
    x = expression(gamma),
    y = expression("EV %"),
    color = NULL, 
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

combined_plot <- ggarrange(
  low_mean,
  high_mean,
  ncol = 2, # Arrange in two columns
  common.legend = TRUE, # Use a single legend for both plots
  legend = "bottom" # Place the common legend on the right
)

data_filter = loss05_var_ev %>%
  filter(status == "low")

low_mean = ggplot(data_filter , aes(x = gamma, y = mean_ev_perct, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(data_filter, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  #geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_cartesian(ylim = c(-10, -3.5))+
  labs(
    title = expression(zeta[2] < 1),
    x = expression(gamma),
    y = expression("EV %"),
    color = NULL, 
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

data_filter = loss05_var_ev %>%
  filter(status == "high")

high_mean = ggplot(data_filter , aes(x = gamma, y = mean_ev_perct, color = income_strata, shape = income_strata)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(data_filter, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  #geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  #geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  coord_cartesian(ylim = c(-10, -3.5))+
  labs(
    title = expression(zeta[2] > 1),
    x = expression(gamma),
    y = expression("EV %"),
    color = NULL, 
    shape = NULL) +
  scale_color_manual(values = custom_colors) +  # Apply gradient-like colors
  scale_shape_manual(values = custom_shapes) + # Apply shapes
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

combined_plot <- ggarrange(
  low_mean,
  high_mean,
  ncol = 2, # Arrange in two columns
  common.legend = TRUE, # Use a single legend for both plots
  legend = "bottom" # Place the common legend on the right
)

##### Trade Off Study Graph lowest Strata zeta_1#####

metric_colors <- c("Mean" = "#1B9E77",
                   "SD" = "#D95F02",
                   "5th Percentile" = "#7570B3")

# New shape scheme for the three metrics
metric_shapes <- c("Mean" = 19,  # Solid circle
                   "SD" = 17,   # Solid triangle
                   "5th Percentile" = 15) # Solid square

data_filter_lowest_strata <- loss05_mean_ev %>%
  filter(status == "low", income_strata == "0~6k") %>% # Filter for specific strata and status
  pivot_longer(
    cols = c(mean_ev_perct, sd_ev_perct, ev_five_quantile),
    names_to = "metric_type",
    values_to = "value"
  ) %>%
  mutate(
    # Create readable labels for the metric type
    metric_type = case_when(
      metric_type == "mean_ev_perct" ~ "Mean",
      metric_type == "sd_ev_perct" ~ "SD",
      metric_type == "ev_five_quantile" ~ "5th Percentile",
      TRUE ~ metric_type
    ),
    # Ensure factor levels for correct legend order and mapping
    metric_type = factor(metric_type, levels = names(metric_colors))
  )

# --- Create the ggplot visualization ---
low_mean_lowest_strata_plot <- ggplot(data_filter_lowest_strata,
                                      aes(x = gamma, y = value,
                                          color = metric_type, # Color by metric type
                                          shape = metric_type, # Shape by metric type
                                          group = metric_type)) + # Group by metric type for lines
  geom_line(linewidth = 1) + # Slightly thicker lines for clarity
  geom_point(size = 4, stroke = 1.2) + # Larger points with a border
  #coord_cartesian(ylim = c(-10, -3.5)) + # Apply the y-axis limits
  labs(
    title = expression(paste(zeta[1]< 0)), # More descriptive title
    x = expression(gamma), # X-axis label with gamma symbol
    y = expression("EV/I %"), # Y-axis label with mathematical expression
    color = "Metric Type", # Legend title for color
    shape = "Metric Type"  # Legend title for shape
  ) +
  scale_color_manual(values = metric_colors) + # Apply new colors for metrics
  scale_shape_manual(values = metric_shapes) + # Apply new shapes for metrics
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

low_mean_faceted_metrics_plot <- ggplot(data_filter_lowest_strata,
                                        aes(x = gamma, y = value, group = 1)) + # Group by 1 for single line per facet
  geom_line(color = "steelblue", linewidth = 1) + # Single line color
  geom_point(color = "steelblue", shape = 19, size = 4, stroke = 1.2) + # Single point color and shape
  #coord_cartesian(ylim = c(-10, 1)) + # Adjusted Y-lim to better accommodate SD EV which is higher
  labs(
    title = expression(zeta[1] < 0),
    x = expression(gamma),
    y = expression("EV/I %")
  ) +
  facet_wrap(~ metric_type, scales = "free_y", ncol = 1) + # Facet by metric_type with free Y-scales
  academic_theme # Apply the custom theme


data_filter_lowest_strata <- loss05_mean_ev %>%
  filter(status == "high", income_strata == "0~6k") %>% # Filter for specific strata and status
  pivot_longer(
    cols = c(mean_ev_perct, sd_ev_perct, ev_five_quantile),
    names_to = "metric_type",
    values_to = "value"
  ) %>%
  mutate(
    # Create readable labels for the metric type
    metric_type = case_when(
      metric_type == "mean_ev_perct" ~ "Mean",
      metric_type == "sd_ev_perct" ~ "SD",
      metric_type == "ev_five_quantile" ~ "5th Percentile",
      TRUE ~ metric_type
    ),
    # Ensure factor levels for correct legend order and mapping
    metric_type = factor(metric_type, levels = names(metric_colors))
  )

# --- Create the ggplot visualization ---
high_mean_lowest_strata_plot <- ggplot(data_filter_lowest_strata,
                                       aes(x = gamma, y = value,
                                           color = metric_type, # Color by metric type
                                           shape = metric_type, # Shape by metric type
                                           group = metric_type)) + # Group by metric type for lines
  geom_line(linewidth = 1) + # Slightly thicker lines for clarity
  geom_point(size = 4, stroke = 1.2) + # Larger points with a border
  #coord_cartesian(ylim = c(-10, -3.5)) + # Apply the y-axis limits
  labs(
    title = expression(paste(zeta[1] > 0)), # More descriptive title
    x = expression(gamma), # X-axis label with gamma symbol
    y = expression("EV/I %"), # Y-axis label with mathematical expression
    color = "Metric Type", # Legend title for color
    shape = "Metric Type"  # Legend title for shape
  ) +
  scale_color_manual(values = metric_colors) + # Apply new colors for metrics
  scale_shape_manual(values = metric_shapes) + # Apply new shapes for metrics
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

high_mean_faceted_metrics_plot <- ggplot(data_filter_lowest_strata,
                                         aes(x = gamma, y = value, group = 1)) + # Group by 1 for single line per facet
  geom_line(color = "steelblue", linewidth = 1) + # Single line color
  geom_point(color = "steelblue", shape = 19, size = 4, stroke = 1.2) + # Single point color and shape
  #coord_cartesian(ylim = c(-10, 1)) + # Adjusted Y-lim to better accommodate SD EV which is higher
  labs(
    title = expression(zeta[1] > 0),
    x = expression(gamma),
    y = expression("EV/I %")
  ) +
  facet_wrap(~ metric_type, scales = "free_y", ncol = 1) + # Facet by metric_type with free Y-scales
  academic_theme # Apply the custom theme


combined_lowest_strata_plot_zeta1 <- ggarrange(
  low_mean_lowest_strata_plot,
  high_mean_lowest_strata_plot,
  ncol = 2, # Arrange in two columns
  common.legend = TRUE, # Use a single legend for both plots
  legend = "bottom" # Place the common legend on the right
)

combined_lowest_faceted_strata_plot_zeta1 <- ggarrange(
  low_mean_faceted_metrics_plot,
  high_mean_faceted_metrics_plot,
  ncol = 2, # Arrange in two columns
  common.legend = TRUE, # Use a single legend for both plots
  legend = "bottom" # Place the common legend on the right
)

##### Trade Off Study Graph lowest Strata zeta_2#####


data_filter_lowest_strata <- loss05_var_ev %>%
  filter(status == "low", income_strata == "0~6k") %>% # Filter for specific strata and status
  pivot_longer(
    cols = c(mean_ev_perct, sd_ev_perct, ev_five_quantile),
    names_to = "metric_type",
    values_to = "value"
  ) %>%
  mutate(
    # Create readable labels for the metric type
    metric_type = case_when(
      metric_type == "mean_ev_perct" ~ "Mean",
      metric_type == "sd_ev_perct" ~ "SD",
      metric_type == "ev_five_quantile" ~ "5th Percentile",
      TRUE ~ metric_type
    ),
    # Ensure factor levels for correct legend order and mapping
    metric_type = factor(metric_type, levels = names(metric_colors))
  )

# --- Create the ggplot visualization ---
low_mean_lowest_strata_plot <- ggplot(data_filter_lowest_strata,
                                      aes(x = gamma, y = value,
                                          color = metric_type, # Color by metric type
                                          shape = metric_type, # Shape by metric type
                                          group = metric_type)) + # Group by metric type for lines
  geom_line(linewidth = 1) + # Slightly thicker lines for clarity
  geom_point(size = 4, stroke = 1.2) + # Larger points with a border
  #coord_cartesian(ylim = c(-10, -3.5)) + # Apply the y-axis limits
  labs(
    title = expression(paste(zeta[2]< 1)), # More descriptive title
    x = expression(gamma), # X-axis label with gamma symbol
    y = expression("EV/I %"), # Y-axis label with mathematical expression
    color = "Metric Type", # Legend title for color
    shape = "Metric Type"  # Legend title for shape
  ) +
  scale_color_manual(values = metric_colors) + # Apply new colors for metrics
  scale_shape_manual(values = metric_shapes) + # Apply new shapes for metrics
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

low_mean_faceted_metrics_plot <- ggplot(data_filter_lowest_strata,
                                         aes(x = gamma, y = value, group = 1)) + # Group by 1 for single line per facet
  geom_line(color = "steelblue", linewidth = 1) + # Single line color
  geom_point(color = "steelblue", shape = 19, size = 4, stroke = 1.2) + # Single point color and shape
  #coord_cartesian(ylim = c(-10, 1)) + # Adjusted Y-lim to better accommodate SD EV which is higher
  labs(
    title = expression(zeta[2] < 1),
    x = expression(gamma),
    y = expression("EV/I %")
  ) +
  facet_wrap(~ metric_type, scales = "free_y", ncol = 1) + # Facet by metric_type with free Y-scales
  academic_theme # Apply the custom theme


data_filter_lowest_strata <- loss05_var_ev %>%
  filter(status == "high", income_strata == "0~6k") %>% # Filter for specific strata and status
  pivot_longer(
    cols = c(mean_ev_perct, sd_ev_perct, ev_five_quantile),
    names_to = "metric_type",
    values_to = "value"
  ) %>%
  mutate(
    # Create readable labels for the metric type
    metric_type = case_when(
      metric_type == "mean_ev_perct" ~ "Mean",
      metric_type == "sd_ev_perct" ~ "SD",
      metric_type == "ev_five_quantile" ~ "5th Percentile",
      TRUE ~ metric_type
    ),
    # Ensure factor levels for correct legend order and mapping
    metric_type = factor(metric_type, levels = names(metric_colors))
  )

# --- Create the ggplot visualization ---
high_mean_lowest_strata_plot <- ggplot(data_filter_lowest_strata,
                                       aes(x = gamma, y = value,
                                           color = metric_type, # Color by metric type
                                           shape = metric_type, # Shape by metric type
                                           group = metric_type)) + # Group by metric type for lines
  geom_line(linewidth = 1) + # Slightly thicker lines for clarity
  geom_point(size = 4, stroke = 1.2) + # Larger points with a border
  #coord_cartesian(ylim = c(-10, -3.5)) + # Apply the y-axis limits
  labs(
    title = expression(paste(zeta[2] > 1)), # More descriptive title
    x = expression(gamma), # X-axis label with gamma symbol
    y = expression("EV/I %"), # Y-axis label with mathematical expression
    color = "Metric Type", # Legend title for color
    shape = "Metric Type"  # Legend title for shape
  ) +
  scale_color_manual(values = metric_colors) + # Apply new colors for metrics
  scale_shape_manual(values = metric_shapes) + # Apply new shapes for metrics
  academic_theme + # Apply the custom theme
  guides( # Adjust legend key appearance
    color = guide_legend(override.aes = list(linewidth = 1, size = 3)),
    shape = guide_legend(override.aes = list(linewidth = 1, size = 3))
  )

high_mean_faceted_metrics_plot <- ggplot(data_filter_lowest_strata,
                                         aes(x = gamma, y = value, group = 1)) + # Group by 1 for single line per facet
  geom_line(color = "steelblue", linewidth = 1) + # Single line color
  geom_point(color = "steelblue", shape = 19, size = 4, stroke = 1.2) + # Single point color and shape
  #coord_cartesian(ylim = c(-10, 1)) + # Adjusted Y-lim to better accommodate SD EV which is higher
  labs(
    title = expression(zeta[2] > 1),
    x = expression(gamma),
    y = expression("EV/I %")
  ) +
  facet_wrap(~ metric_type, scales = "free_y", ncol = 1) + # Facet by metric_type with free Y-scales
  academic_theme # Apply the custom theme


combined_lowest_strata_plot_zeta2 <- ggarrange(
  low_mean_lowest_strata_plot,
  high_mean_lowest_strata_plot,
  ncol = 2, # Arrange in two columns
  common.legend = TRUE, # Use a single legend for both plots
  legend = "bottom" # Place the common legend on the right
)

combined_lowest_faceted_strata_plot_zeta2 <- ggarrange(
  low_mean_faceted_metrics_plot,
  high_mean_faceted_metrics_plot,
  ncol = 2, # Arrange in two columns
  common.legend = TRUE, # Use a single legend for both plots
  legend = "bottom" # Place the common legend on the right
)


#### Clean Up #####

# Get a list of all objects in the current environment
all_objects <- ls()

# Identify dataframes that start with "current"
dataframes_to_delete <- c()
for (obj_name in all_objects) {
  if (startsWith(obj_name, "current") && is.data.frame(get(obj_name))) {
    dataframes_to_delete <- c(dataframes_to_delete, obj_name)
  }
}

# Delete the identified dataframes
if (length(dataframes_to_delete) > 0) {
  rm(list = dataframes_to_delete)
  cat("Deleted dataframes starting with 'current':", paste(dataframes_to_delete, collapse = ", "), "\n")
} else {
  cat("No dataframes starting with 'current' found.\n")
}









#### Prem Id Level ####

mean_cs_rate_by_prem_id = current_info_avg_bound_loss05_var_cs_steps_rate %>%
  group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_prem_id = current_info_avg_bound_loss05_var_ev_steps_rate %>%
  group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

avg_bound_loss05_var_cs_prem_id = data.frame(mean_cs_rate_by_prem_id$prem_id)
colnames(avg_bound_loss05_var_cs_prem_id) = c("prem_id")

avg_bound_loss05_var_cs_prem_id$cs_mean_low = apply(mean_cs_rate_by_prem_id[, 2:7], 1, mean)
avg_bound_loss05_var_cs_prem_id$cs_mean_high = apply(mean_cs_rate_by_prem_id[, 7:12], 1, mean)
avg_bound_loss05_var_cs_prem_id$cs_floor_low = apply(mean_cs_rate_by_prem_id[, 2:7], 1, function (x) quantile(x, 0.05))
avg_bound_loss05_var_cs_prem_id$cs_floor_high = apply(mean_cs_rate_by_prem_id[, 7:12], 1, function (x) quantile(x, 0.05))
avg_bound_loss05_var_cs_prem_id$cs_sd_low = apply(mean_cs_rate_by_prem_id[, 2:7], 1, sd)
avg_bound_loss05_var_cs_prem_id$cs_sd_high = apply(mean_cs_rate_by_prem_id[, 7:12], 1, sd)

avg_bound_loss05_var_ev_prem_id = data.frame(mean_ev_rate_by_prem_id$prem_id)
colnames(avg_bound_loss05_var_ev_prem_id) = c("prem_id")

avg_bound_loss05_var_ev_prem_id$ev_mean_low = apply(mean_ev_rate_by_prem_id[, 2:7], 1, mean)
avg_bound_loss05_var_ev_prem_id$ev_mean_high = apply(mean_ev_rate_by_prem_id[, 7:12], 1, mean)
avg_bound_loss05_var_ev_prem_id$ev_floor_low = apply(mean_ev_rate_by_prem_id[, 2:7], 1, function (x) quantile(x, 0.05))
avg_bound_loss05_var_ev_prem_id$ev_floor_high = apply(mean_ev_rate_by_prem_id[, 7:12], 1, function (x) quantile(x, 0.05))
avg_bound_loss05_var_ev_prem_id$ev_sd_low = apply(mean_ev_rate_by_prem_id[, 2:7], 1, sd)
avg_bound_loss05_var_ev_prem_id$ev_sd_high = apply(mean_ev_rate_by_prem_id[, 7:12], 1, sd)


loss05_mean_cs_prem_id = avg_bound_loss05_mean_cs_prem_id

colnames(loss05_mean_cs_prem_id) = c("prem_id", "cs_mean_low_0","cs_mean_high_0","cs_sd_low_0","cs_sd_high_0","cs_floor_low_0","cs_floor_high_0")
loss05_mean_cs_prem_id$cs_mean_low_025 = gamma025_bound_loss05_mean_cs_prem_id$cs_mean_low
loss05_mean_cs_prem_id$cs_mean_high_025 = gamma025_bound_loss05_mean_cs_prem_id$cs_mean_high
loss05_mean_cs_prem_id$cs_sd_low_025 = gamma025_bound_loss05_mean_cs_prem_id$cs_sd_low
loss05_mean_cs_prem_id$cs_sd_high_025 = gamma025_bound_loss05_mean_cs_prem_id$cs_sd_high
loss05_mean_cs_prem_id$cs_floor_low_025 = gamma025_bound_loss05_mean_cs_prem_id$cs_floor_low
loss05_mean_cs_prem_id$cs_floor_high_025 = gamma025_bound_loss05_mean_cs_prem_id$cs_floor_high

loss05_mean_cs_prem_id$cs_mean_low_01 = gamma01_bound_loss05_mean_cs_prem_id$cs_mean_low
loss05_mean_cs_prem_id$cs_mean_high_01 = gamma01_bound_loss05_mean_cs_prem_id$cs_mean_high
loss05_mean_cs_prem_id$cs_sd_low_01 = gamma01_bound_loss05_mean_cs_prem_id$cs_sd_low
loss05_mean_cs_prem_id$cs_sd_high_01 = gamma01_bound_loss05_mean_cs_prem_id$cs_sd_high
loss05_mean_cs_prem_id$cs_floor_low_01 = gamma01_bound_loss05_mean_cs_prem_id$cs_floor_low
loss05_mean_cs_prem_id$cs_floor_high_01 = gamma01_bound_loss05_mean_cs_prem_id$cs_floor_high

loss05_mean_cs_prem_id$cs_mean_low_05 = gamma05_bound_loss05_mean_cs_prem_id$cs_mean_low
loss05_mean_cs_prem_id$cs_mean_high_05 = gamma05_bound_loss05_mean_cs_prem_id$cs_mean_high
loss05_mean_cs_prem_id$cs_sd_low_05 = gamma05_bound_loss05_mean_cs_prem_id$cs_sd_low
loss05_mean_cs_prem_id$cs_sd_high_05 = gamma05_bound_loss05_mean_cs_prem_id$cs_sd_high
loss05_mean_cs_prem_id$cs_floor_low_05 = gamma05_bound_loss05_mean_cs_prem_id$cs_floor_low
loss05_mean_cs_prem_id$cs_floor_high_05 = gamma05_bound_loss05_mean_cs_prem_id$cs_floor_high


loss05_mean_cs_prem_id = merge(loss05_mean_cs_prem_id, income_id, by.x = c("prem_id"), by.y = c("prem_id"), all.x = T)

loss05_mean_cs_strata = loss05_mean_cs_prem_id %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("cs_"), median, na.rm = TRUE))

df_long_all <- loss05_mean_cs_strata %>%
  pivot_longer(
    cols = starts_with("cs_"), # Selects all columns that start with 'cs_'
    names_to = c("variable_type", "side", "gamma_str"), # New column names to create
    names_pattern = "cs_([a-z]+)_([a-z]+)_(\\d+\\.?\\d*)$", # Regex to extract parts
    values_to = "value" # Column to store the numerical values
  ) %>%
  mutate(
    # Convert gamma_str to a proper numeric gamma value
    gamma = as.numeric(gamma_str),
    gamma = case_when(
      !grepl("\\.", gamma_str) & nchar(gamma_str) == 2 ~ gamma / 10,  # e.g., "01" -> 0.01, "05" -> 0.05
      !grepl("\\.", gamma_str) & nchar(gamma_str) == 3 ~ gamma / 100, # e.g., "025" -> 0.025
      TRUE ~ gamma # For "0", "0.25", "0.5", etc., already correct
    ),
    gamma = as.factor(gamma), # Convert gamma to a factor for plotting
    variable_type = as.factor(variable_type), # Convert variable type to factor
    side = as.factor(side) # Convert side to factor
  ) %>%
  # Select and reorder columns for clarity
  select(income_strata, gamma, variable_type, side, value)

# 2. Filter the data for 'mean' variable_type and 'low' side
df_filtered <- df_long_all %>%
  filter(variable_type == "mean", side == "low")

# 3. Create the ggplot visualization for academic paper quality
ggplot(df_filtered, aes(x = gamma, y = value,
                                 color = income_strata,
                                 shape = income_strata,
                                 group = income_strata)) +
  geom_line(linewidth = 0.8) + # Increased line thickness
  geom_point(size = 3.5, stroke = 0.8) + # Increased point size and added a border
  scale_color_manual(values = custom_colors) + # Apply custom colors
  scale_shape_manual(values = custom_shapes) + # Apply custom shapes
  labs(
    #title = "Mean Low by Gamma Value Across Income Strata",
    x = expression(paste("Gamma (", gamma, ")")), # Use LaTeX-like gamma symbol
    y = "CS %",
    color = "Income Strata",
    shape = "Income Strata"
  ) +
  theme_bw(base_size = 14) + # Use black and white theme with a larger base font size
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 16), # Center and bold title
    axis.title = element_text(face = "bold"), # Bold axis titles
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1), # Rotate and adjust x-axis labels
    legend.position = "right", # Place legend on the right
    legend.title = element_text(face = "bold"), # Bold legend title
    legend.text = element_text(size = 12), # Adjust legend text size
    panel.grid.major = element_line(linewidth = 0.5, color = "grey90"), # Fine-tune grid lines
    panel.grid.minor = element_line(linewidth = 0.25, color = "grey95")
  )



##### current_info_gamma1_bound_loss05_mean ########

current_info_gamma1_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss05_mean_cs_steps.csv")

colnames(current_info_gamma1_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss05_mean_cs_steps))*0.05-0.25)

current_info_gamma1_bound_loss05_mean_cs_steps_rate <-current_info_gamma1_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss05_mean_cs_steps)

current_info_gamma1_bound_loss05_mean_q_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss05_mean_q_steps.csv")

colnames(current_info_gamma1_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss05_mean_q_steps))*0.05-0.25)

current_info_gamma1_bound_loss05_mean_q_steps_rate <-current_info_gamma1_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss05_mean_q_steps)

current_info_gamma1_bound_loss05_mean_r_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss05_mean_r_steps.csv")

colnames(current_info_gamma1_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss05_mean_r_steps))*0.05-0.25)

current_info_gamma1_bound_loss05_mean_r_steps_rate <-current_info_gamma1_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss05_mean_r_steps)

current_info_gamma1_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss05_mean_ev_steps.csv")

colnames(current_info_gamma1_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss05_mean_ev_steps))*0.05-0.25)

current_info_gamma1_bound_loss05_mean_ev_steps_rate <-current_info_gamma1_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss05_mean_ev_steps)

current_info_gamma1_bound_loss05_mean_cs_steps_rate  = cbind(current_info_gamma1_bound_loss05_mean_cs_steps_rate , demand_key)
current_info_gamma1_bound_loss05_mean_q_steps_rate = cbind(current_info_gamma1_bound_loss05_mean_q_steps_rate, demand_key)
current_info_gamma1_bound_loss05_mean_r_steps_rate = cbind(current_info_gamma1_bound_loss05_mean_r_steps_rate, demand_key)
current_info_gamma1_bound_loss05_mean_ev_steps_rate = cbind(current_info_gamma1_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_gamma1_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_gamma1_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_gamma1_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- current_info_gamma1_bound_loss05_mean_ev_steps_rate %>%
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

gamma1_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma1_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

gamma1_bound_loss05_mean_grid = ggarrange(cs_gamma1,r_gamma1,q_gamma1,ev_gamma1, ncol = 2, nrow = 2, 
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
gamma1_bound_loss05_mean_grid_with_title <- gamma1_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_mean_grid_with_title, gamma1_bound_loss1_mean_grid_with_title,
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
avg_gamma1_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                    ncol = 1, nrow = 2,
                                    heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))





##### current_info_gamma1_bound_loss05_var ########

current_info_gamma1_bound_loss05_var_cs_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss05_var_cs_steps.csv")

colnames(current_info_gamma1_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss05_var_cs_steps))/20+0.75)

current_info_gamma1_bound_loss05_var_cs_steps_rate <-current_info_gamma1_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss05_var_cs_steps)

current_info_gamma1_bound_loss05_var_q_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss05_var_q_steps.csv")

colnames(current_info_gamma1_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss05_var_q_steps))/20+0.75)

current_info_gamma1_bound_loss05_var_q_steps_rate <-current_info_gamma1_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss05_var_q_steps)

current_info_gamma1_bound_loss05_var_r_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss05_var_r_steps.csv")

colnames(current_info_gamma1_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss05_var_r_steps))/20+0.75)

current_info_gamma1_bound_loss05_var_r_steps_rate <-current_info_gamma1_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss05_var_r_steps)

current_info_gamma1_bound_loss05_var_ev_steps = read_csv("cs_detail_results/current_info_gamma1_bound_loss05_var_ev_steps.csv")

colnames(current_info_gamma1_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(current_info_gamma1_bound_loss05_var_ev_steps))/20+0.75)

current_info_gamma1_bound_loss05_var_ev_steps_rate <-current_info_gamma1_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma1_bound_loss05_var_ev_steps)

current_info_gamma1_bound_loss05_var_cs_steps_rate  = cbind(current_info_gamma1_bound_loss05_var_cs_steps_rate , demand_key)
current_info_gamma1_bound_loss05_var_q_steps_rate = cbind(current_info_gamma1_bound_loss05_var_q_steps_rate, demand_key)
current_info_gamma1_bound_loss05_var_r_steps_rate = cbind(current_info_gamma1_bound_loss05_var_r_steps_rate, demand_key)
current_info_gamma1_bound_loss05_var_ev_steps_rate = cbind(current_info_gamma1_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_gamma1_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_gamma1_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_gamma1_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_gamma1_bound_loss05_var_ev_steps_rate %>%
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

gamma1_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma1_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

gamma1_bound_loss05_var_grid = ggarrange(cs_gamma1,r_gamma1,q_gamma1,ev_gamma1, ncol = 2, nrow = 2, 
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
gamma1_bound_loss05_var_grid_with_title <- gamma1_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_var_grid_with_title, gamma1_bound_loss05_var_grid_with_title,
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
avg_gamma1_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                   ncol = 1, nrow = 2,
                                   heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))





