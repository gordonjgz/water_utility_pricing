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



##### montecarlo_weather_avg_bound_loss05_mean ########

montecarlo_weather_avg_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/montecarlo_weather_avg_bound_loss05_mean_cs_steps.csv")

colnames(montecarlo_weather_avg_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(montecarlo_weather_avg_bound_loss05_mean_cs_steps))*0.05-0.25)

montecarlo_weather_avg_bound_loss05_mean_cs_steps_rate <-montecarlo_weather_avg_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(montecarlo_weather_avg_bound_loss05_mean_cs_steps)

montecarlo_weather_avg_bound_loss05_mean_q_steps = read_csv("cs_detail_results/montecarlo_weather_avg_bound_loss05_mean_q_steps.csv")

colnames(montecarlo_weather_avg_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(montecarlo_weather_avg_bound_loss05_mean_q_steps))*0.05-0.25)

montecarlo_weather_avg_bound_loss05_mean_q_steps_rate <-montecarlo_weather_avg_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_avg_bound_loss05_mean_q_steps)

montecarlo_weather_avg_bound_loss05_mean_r_steps = read_csv("cs_detail_results/montecarlo_weather_avg_bound_loss05_mean_r_steps.csv")

colnames(montecarlo_weather_avg_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(montecarlo_weather_avg_bound_loss05_mean_r_steps))*0.05-0.25)

montecarlo_weather_avg_bound_loss05_mean_r_steps_rate <-montecarlo_weather_avg_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_avg_bound_loss05_mean_r_steps)

montecarlo_weather_avg_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/montecarlo_weather_avg_bound_loss05_mean_ev_steps.csv")

colnames(montecarlo_weather_avg_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(montecarlo_weather_avg_bound_loss05_mean_ev_steps))*0.05-0.25)

montecarlo_weather_avg_bound_loss05_mean_ev_steps_rate <-montecarlo_weather_avg_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(montecarlo_weather_avg_bound_loss05_mean_ev_steps)

montecarlo_weather_avg_bound_loss05_mean_cs_steps_rate  = cbind(montecarlo_weather_avg_bound_loss05_mean_cs_steps_rate , demand_key)
montecarlo_weather_avg_bound_loss05_mean_q_steps_rate = cbind(montecarlo_weather_avg_bound_loss05_mean_q_steps_rate, demand_key)
montecarlo_weather_avg_bound_loss05_mean_r_steps_rate = cbind(montecarlo_weather_avg_bound_loss05_mean_r_steps_rate, demand_key)
montecarlo_weather_avg_bound_loss05_mean_ev_steps_rate = cbind(montecarlo_weather_avg_bound_loss05_mean_ev_steps_rate, demand_key)

#montecarlo_weather_avg_bound_loss05_mean_cs_steps_rate = montecarlo_weather_avg_bound_loss05_mean_cs_steps_rate[which(montecarlo_weather_avg_bound_loss05_mean_cs_steps_rate$q_0<100),]
#montecarlo_weather_avg_bound_loss05_mean_ev_steps_rate = montecarlo_weather_avg_bound_loss05_mean_ev_steps_rate[which(montecarlo_weather_avg_bound_loss05_mean_ev_steps_rate$q_0<100),]

mean_cs_rate_by_bill_ym <- montecarlo_weather_avg_bound_loss05_mean_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- montecarlo_weather_avg_bound_loss05_mean_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- montecarlo_weather_avg_bound_loss05_mean_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- montecarlo_weather_avg_bound_loss05_mean_ev_steps_rate %>%
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

mc_avg_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

mc_avg_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


mc_avg_bound_loss05_mean_grid <- ggarrange(cs_avg,r_avg,q_avg,ev_avg, ncol = 2, nrow = 2, 
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
mc_avg_bound_loss05_mean_grid_with_title <- mc_avg_bound_loss05_mean_grid +
  plot_annotation(
    title = "Linear Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )

##### montecarlo_weather_avg_bound_loss05_var ########

montecarlo_weather_avg_bound_loss05_var_cs_steps = read_csv("cs_detail_results/montecarlo_weather_avg_bound_loss05_var_cs_steps.csv")

colnames(montecarlo_weather_avg_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(montecarlo_weather_avg_bound_loss05_var_cs_steps))/20+0.75)

montecarlo_weather_avg_bound_loss05_var_cs_steps_rate <-montecarlo_weather_avg_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(montecarlo_weather_avg_bound_loss05_var_cs_steps)

montecarlo_weather_avg_bound_loss05_var_q_steps = read_csv("cs_detail_results/montecarlo_weather_avg_bound_loss05_var_q_steps.csv")

colnames(montecarlo_weather_avg_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(montecarlo_weather_avg_bound_loss05_var_q_steps))/20+0.75)

montecarlo_weather_avg_bound_loss05_var_q_steps_rate <-montecarlo_weather_avg_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_avg_bound_loss05_var_q_steps)

montecarlo_weather_avg_bound_loss05_var_r_steps = read_csv("cs_detail_results/montecarlo_weather_avg_bound_loss05_var_r_steps.csv")

colnames(montecarlo_weather_avg_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(montecarlo_weather_avg_bound_loss05_var_r_steps))/20+0.75)

montecarlo_weather_avg_bound_loss05_var_r_steps_rate <-montecarlo_weather_avg_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_avg_bound_loss05_var_r_steps)

montecarlo_weather_avg_bound_loss05_var_ev_steps = read_csv("cs_detail_results/montecarlo_weather_avg_bound_loss05_var_ev_steps.csv")

colnames(montecarlo_weather_avg_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(montecarlo_weather_avg_bound_loss05_var_ev_steps))/20+0.75)

montecarlo_weather_avg_bound_loss05_var_ev_steps_rate <-montecarlo_weather_avg_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(montecarlo_weather_avg_bound_loss05_var_ev_steps)

montecarlo_weather_avg_bound_loss05_var_cs_steps_rate  = cbind(montecarlo_weather_avg_bound_loss05_var_cs_steps_rate , demand_key)
montecarlo_weather_avg_bound_loss05_var_q_steps_rate = cbind(montecarlo_weather_avg_bound_loss05_var_q_steps_rate, demand_key)
montecarlo_weather_avg_bound_loss05_var_r_steps_rate = cbind(montecarlo_weather_avg_bound_loss05_var_r_steps_rate, demand_key)
montecarlo_weather_avg_bound_loss05_var_ev_steps_rate = cbind(montecarlo_weather_avg_bound_loss05_var_ev_steps_rate, demand_key)

#montecarlo_weather_avg_bound_loss05_var_cs_steps_rate = montecarlo_weather_avg_bound_loss05_var_cs_steps_rate[which(montecarlo_weather_avg_bound_loss05_var_cs_steps_rate$q_0<100),]
#montecarlo_weather_avg_bound_loss05_var_ev_steps_rate = montecarlo_weather_avg_bound_loss05_var_ev_steps_rate[which(montecarlo_weather_avg_bound_loss05_var_ev_steps_rate$q_0<100),]

var_cs_rate_by_bill_ym <- montecarlo_weather_avg_bound_loss05_var_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- montecarlo_weather_avg_bound_loss05_var_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- montecarlo_weather_avg_bound_loss05_var_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- montecarlo_weather_avg_bound_loss05_var_ev_steps_rate %>%
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

mc_avg_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

mc_avg_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

mc_avg_bound_loss05_var_grid <- ggarrange(cs_avg,r_avg,q_avg,ev_avg, ncol = 2, nrow = 2, 
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
mc_avg_bound_loss05_var_grid_with_title <- mc_avg_bound_loss05_var_grid +
  plot_annotation(
    title = "Linear Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )


##### montecarlo_weather_gamma01_bound_loss05_mean ########

montecarlo_weather_gamma01_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/montecarlo_weather_gamma01_bound_loss05_mean_cs_steps.csv")

colnames(montecarlo_weather_gamma01_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma01_bound_loss05_mean_cs_steps))*0.05-0.25)

montecarlo_weather_gamma01_bound_loss05_mean_cs_steps_rate <-montecarlo_weather_gamma01_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma01_bound_loss05_mean_cs_steps)

montecarlo_weather_gamma01_bound_loss05_mean_q_steps = read_csv("cs_detail_results/montecarlo_weather_gamma01_bound_loss05_mean_q_steps.csv")

colnames(montecarlo_weather_gamma01_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma01_bound_loss05_mean_q_steps))*0.05-0.25)

montecarlo_weather_gamma01_bound_loss05_mean_q_steps_rate <-montecarlo_weather_gamma01_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma01_bound_loss05_mean_q_steps)

montecarlo_weather_gamma01_bound_loss05_mean_r_steps = read_csv("cs_detail_results/montecarlo_weather_gamma01_bound_loss05_mean_r_steps.csv")

colnames(montecarlo_weather_gamma01_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma01_bound_loss05_mean_r_steps))*0.05-0.25)

montecarlo_weather_gamma01_bound_loss05_mean_r_steps_rate <-montecarlo_weather_gamma01_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma01_bound_loss05_mean_r_steps)

montecarlo_weather_gamma01_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/montecarlo_weather_gamma01_bound_loss05_mean_ev_steps.csv")

colnames(montecarlo_weather_gamma01_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma01_bound_loss05_mean_ev_steps))*0.05-0.25)

montecarlo_weather_gamma01_bound_loss05_mean_ev_steps_rate <-montecarlo_weather_gamma01_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma01_bound_loss05_mean_ev_steps)

montecarlo_weather_gamma01_bound_loss05_mean_cs_steps_rate  = cbind(montecarlo_weather_gamma01_bound_loss05_mean_cs_steps_rate , demand_key)
montecarlo_weather_gamma01_bound_loss05_mean_q_steps_rate = cbind(montecarlo_weather_gamma01_bound_loss05_mean_q_steps_rate, demand_key)
montecarlo_weather_gamma01_bound_loss05_mean_r_steps_rate = cbind(montecarlo_weather_gamma01_bound_loss05_mean_r_steps_rate, demand_key)
montecarlo_weather_gamma01_bound_loss05_mean_ev_steps_rate = cbind(montecarlo_weather_gamma01_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- montecarlo_weather_gamma01_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- montecarlo_weather_gamma01_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- montecarlo_weather_gamma01_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- montecarlo_weather_gamma01_bound_loss05_mean_ev_steps_rate %>%
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

mc_gamma01_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

mc_gamma01_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

mc_gamma01_bound_loss05_mean_grid = ggarrange(cs_gamma01,r_gamma01,q_gamma01,ev_gamma01, ncol = 2, nrow = 2, 
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
mc_gamma01_bound_loss05_mean_grid_with_title <- mc_gamma01_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(mc_avg_bound_loss05_mean_grid_with_title, mc_gamma01_bound_loss05_mean_grid_with_title,
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
mc_avg_gamma01_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                        ncol = 1, nrow = 2,
                                        heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### montecarlo_weather_gamma01_bound_loss05_var ########

montecarlo_weather_gamma01_bound_loss05_var_cs_steps = read_csv("cs_detail_results/montecarlo_weather_gamma01_bound_loss05_var_cs_steps.csv")

colnames(montecarlo_weather_gamma01_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma01_bound_loss05_var_cs_steps))/20+0.75)

montecarlo_weather_gamma01_bound_loss05_var_cs_steps_rate <-montecarlo_weather_gamma01_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma01_bound_loss05_var_cs_steps)

montecarlo_weather_gamma01_bound_loss05_var_q_steps = read_csv("cs_detail_results/montecarlo_weather_gamma01_bound_loss05_var_q_steps.csv")

colnames(montecarlo_weather_gamma01_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma01_bound_loss05_var_q_steps))/20+0.75)

montecarlo_weather_gamma01_bound_loss05_var_q_steps_rate <-montecarlo_weather_gamma01_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma01_bound_loss05_var_q_steps)

montecarlo_weather_gamma01_bound_loss05_var_r_steps = read_csv("cs_detail_results/montecarlo_weather_gamma01_bound_loss05_var_r_steps.csv")

colnames(montecarlo_weather_gamma01_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma01_bound_loss05_var_r_steps))/20+0.75)

montecarlo_weather_gamma01_bound_loss05_var_r_steps_rate <-montecarlo_weather_gamma01_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma01_bound_loss05_var_r_steps)

montecarlo_weather_gamma01_bound_loss05_var_ev_steps = read_csv("cs_detail_results/montecarlo_weather_gamma01_bound_loss05_var_ev_steps.csv")

colnames(montecarlo_weather_gamma01_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma01_bound_loss05_var_ev_steps))/20+0.75)

montecarlo_weather_gamma01_bound_loss05_var_ev_steps_rate <-montecarlo_weather_gamma01_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma01_bound_loss05_var_ev_steps)

montecarlo_weather_gamma01_bound_loss05_var_cs_steps_rate  = cbind(montecarlo_weather_gamma01_bound_loss05_var_cs_steps_rate , demand_key)
montecarlo_weather_gamma01_bound_loss05_var_q_steps_rate = cbind(montecarlo_weather_gamma01_bound_loss05_var_q_steps_rate, demand_key)
montecarlo_weather_gamma01_bound_loss05_var_r_steps_rate = cbind(montecarlo_weather_gamma01_bound_loss05_var_r_steps_rate, demand_key)
montecarlo_weather_gamma01_bound_loss05_var_ev_steps_rate = cbind(montecarlo_weather_gamma01_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- montecarlo_weather_gamma01_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- montecarlo_weather_gamma01_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- montecarlo_weather_gamma01_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- montecarlo_weather_gamma01_bound_loss05_var_ev_steps_rate %>%
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

mc_gamma01_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

mc_gamma01_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


mc_gamma01_bound_loss05_var_grid = ggarrange(cs_gamma01,r_gamma01,q_gamma01,ev_gamma01, ncol = 2, nrow = 2, 
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
mc_gamma01_bound_loss05_var_grid_with_title <- mc_gamma01_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(mc_avg_bound_loss05_var_grid_with_title, mc_gamma01_bound_loss05_var_grid_with_title,
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
mc_avg_gamma01_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                       ncol = 1, nrow = 2,
                                       heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))



##### montecarlo_weather_gamma02_bound_loss05_mean ########

montecarlo_weather_gamma02_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/montecarlo_weather_gamma02_bound_loss05_mean_cs_steps.csv")

colnames(montecarlo_weather_gamma02_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma02_bound_loss05_mean_cs_steps))*0.05-0.25)

montecarlo_weather_gamma02_bound_loss05_mean_cs_steps_rate <-montecarlo_weather_gamma02_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma02_bound_loss05_mean_cs_steps)

montecarlo_weather_gamma02_bound_loss05_mean_q_steps = read_csv("cs_detail_results/montecarlo_weather_gamma02_bound_loss05_mean_q_steps.csv")

colnames(montecarlo_weather_gamma02_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma02_bound_loss05_mean_q_steps))*0.05-0.25)

montecarlo_weather_gamma02_bound_loss05_mean_q_steps_rate <-montecarlo_weather_gamma02_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma02_bound_loss05_mean_q_steps)

montecarlo_weather_gamma02_bound_loss05_mean_r_steps = read_csv("cs_detail_results/montecarlo_weather_gamma02_bound_loss05_mean_r_steps.csv")

colnames(montecarlo_weather_gamma02_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma02_bound_loss05_mean_r_steps))*0.05-0.25)

montecarlo_weather_gamma02_bound_loss05_mean_r_steps_rate <-montecarlo_weather_gamma02_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma02_bound_loss05_mean_r_steps)

montecarlo_weather_gamma02_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/montecarlo_weather_gamma02_bound_loss05_mean_ev_steps.csv")

colnames(montecarlo_weather_gamma02_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma02_bound_loss05_mean_ev_steps))*0.05-0.25)

montecarlo_weather_gamma02_bound_loss05_mean_ev_steps_rate <-montecarlo_weather_gamma02_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma02_bound_loss05_mean_ev_steps)

montecarlo_weather_gamma02_bound_loss05_mean_cs_steps_rate  = cbind(montecarlo_weather_gamma02_bound_loss05_mean_cs_steps_rate , demand_key)
montecarlo_weather_gamma02_bound_loss05_mean_q_steps_rate = cbind(montecarlo_weather_gamma02_bound_loss05_mean_q_steps_rate, demand_key)
montecarlo_weather_gamma02_bound_loss05_mean_r_steps_rate = cbind(montecarlo_weather_gamma02_bound_loss05_mean_r_steps_rate, demand_key)
montecarlo_weather_gamma02_bound_loss05_mean_ev_steps_rate = cbind(montecarlo_weather_gamma02_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- montecarlo_weather_gamma02_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- montecarlo_weather_gamma02_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- montecarlo_weather_gamma02_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- montecarlo_weather_gamma02_bound_loss05_mean_ev_steps_rate %>%
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

mc_gamma02_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

mc_gamma02_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

mc_gamma02_bound_loss05_mean_grid = ggarrange(cs_gamma02,r_gamma02,q_gamma02,ev_gamma02, ncol = 2, nrow = 2, 
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
mc_gamma02_bound_loss05_mean_grid_with_title <- mc_gamma02_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(mc_avg_bound_loss05_mean_grid_with_title, mc_gamma02_bound_loss05_mean_grid_with_title,
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
mc_avg_gamma02_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                        ncol = 1, nrow = 2,
                                        heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### montecarlo_weather_gamma02_bound_loss05_var ########

montecarlo_weather_gamma02_bound_loss05_var_cs_steps = read_csv("cs_detail_results/montecarlo_weather_gamma02_bound_loss05_var_cs_steps.csv")

colnames(montecarlo_weather_gamma02_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma02_bound_loss05_var_cs_steps))/20+0.75)

montecarlo_weather_gamma02_bound_loss05_var_cs_steps_rate <-montecarlo_weather_gamma02_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma02_bound_loss05_var_cs_steps)

montecarlo_weather_gamma02_bound_loss05_var_q_steps = read_csv("cs_detail_results/montecarlo_weather_gamma02_bound_loss05_var_q_steps.csv")

colnames(montecarlo_weather_gamma02_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma02_bound_loss05_var_q_steps))/20+0.75)

montecarlo_weather_gamma02_bound_loss05_var_q_steps_rate <-montecarlo_weather_gamma02_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma02_bound_loss05_var_q_steps)

montecarlo_weather_gamma02_bound_loss05_var_r_steps = read_csv("cs_detail_results/montecarlo_weather_gamma02_bound_loss05_var_r_steps.csv")

colnames(montecarlo_weather_gamma02_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma02_bound_loss05_var_r_steps))/20+0.75)

montecarlo_weather_gamma02_bound_loss05_var_r_steps_rate <-montecarlo_weather_gamma02_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma02_bound_loss05_var_r_steps)

montecarlo_weather_gamma02_bound_loss05_var_ev_steps = read_csv("cs_detail_results/montecarlo_weather_gamma02_bound_loss05_var_ev_steps.csv")

colnames(montecarlo_weather_gamma02_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma02_bound_loss05_var_ev_steps))/20+0.75)

montecarlo_weather_gamma02_bound_loss05_var_ev_steps_rate <-montecarlo_weather_gamma02_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma02_bound_loss05_var_ev_steps)

montecarlo_weather_gamma02_bound_loss05_var_cs_steps_rate  = cbind(montecarlo_weather_gamma02_bound_loss05_var_cs_steps_rate , demand_key)
montecarlo_weather_gamma02_bound_loss05_var_q_steps_rate = cbind(montecarlo_weather_gamma02_bound_loss05_var_q_steps_rate, demand_key)
montecarlo_weather_gamma02_bound_loss05_var_r_steps_rate = cbind(montecarlo_weather_gamma02_bound_loss05_var_r_steps_rate, demand_key)
montecarlo_weather_gamma02_bound_loss05_var_ev_steps_rate = cbind(montecarlo_weather_gamma02_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- montecarlo_weather_gamma02_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- montecarlo_weather_gamma02_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- montecarlo_weather_gamma02_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- montecarlo_weather_gamma02_bound_loss05_var_ev_steps_rate %>%
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

mc_gamma02_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

mc_gamma02_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


mc_gamma02_bound_loss05_var_grid = ggarrange(cs_gamma02,r_gamma02,q_gamma02,ev_gamma02, ncol = 2, nrow = 2, 
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
mc_gamma02_bound_loss05_var_grid_with_title <- mc_gamma02_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(mc_avg_bound_loss05_var_grid_with_title, mc_gamma02_bound_loss05_var_grid_with_title,
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
mc_avg_gamma02_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                       ncol = 1, nrow = 2,
                                       heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))

##### montecarlo_weather_gamma03_bound_loss05_mean ########

montecarlo_weather_gamma03_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/montecarlo_weather_gamma03_bound_loss05_mean_cs_steps.csv")

colnames(montecarlo_weather_gamma03_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma03_bound_loss05_mean_cs_steps))*0.05-0.25)

montecarlo_weather_gamma03_bound_loss05_mean_cs_steps_rate <-montecarlo_weather_gamma03_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma03_bound_loss05_mean_cs_steps)

montecarlo_weather_gamma03_bound_loss05_mean_q_steps = read_csv("cs_detail_results/montecarlo_weather_gamma03_bound_loss05_mean_q_steps.csv")

colnames(montecarlo_weather_gamma03_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma03_bound_loss05_mean_q_steps))*0.05-0.25)

montecarlo_weather_gamma03_bound_loss05_mean_q_steps_rate <-montecarlo_weather_gamma03_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma03_bound_loss05_mean_q_steps)

montecarlo_weather_gamma03_bound_loss05_mean_r_steps = read_csv("cs_detail_results/montecarlo_weather_gamma03_bound_loss05_mean_r_steps.csv")

colnames(montecarlo_weather_gamma03_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma03_bound_loss05_mean_r_steps))*0.05-0.25)

montecarlo_weather_gamma03_bound_loss05_mean_r_steps_rate <-montecarlo_weather_gamma03_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma03_bound_loss05_mean_r_steps)

montecarlo_weather_gamma03_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/montecarlo_weather_gamma03_bound_loss05_mean_ev_steps.csv")

colnames(montecarlo_weather_gamma03_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma03_bound_loss05_mean_ev_steps))*0.05-0.25)

montecarlo_weather_gamma03_bound_loss05_mean_ev_steps_rate <-montecarlo_weather_gamma03_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma03_bound_loss05_mean_ev_steps)

montecarlo_weather_gamma03_bound_loss05_mean_cs_steps_rate  = cbind(montecarlo_weather_gamma03_bound_loss05_mean_cs_steps_rate , demand_key)
montecarlo_weather_gamma03_bound_loss05_mean_q_steps_rate = cbind(montecarlo_weather_gamma03_bound_loss05_mean_q_steps_rate, demand_key)
montecarlo_weather_gamma03_bound_loss05_mean_r_steps_rate = cbind(montecarlo_weather_gamma03_bound_loss05_mean_r_steps_rate, demand_key)
montecarlo_weather_gamma03_bound_loss05_mean_ev_steps_rate = cbind(montecarlo_weather_gamma03_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- montecarlo_weather_gamma03_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- montecarlo_weather_gamma03_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- montecarlo_weather_gamma03_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- montecarlo_weather_gamma03_bound_loss05_mean_ev_steps_rate %>%
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

mc_gamma03_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

mc_gamma03_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

mc_gamma03_bound_loss05_mean_grid = ggarrange(cs_gamma03,r_gamma03,q_gamma03,ev_gamma03, ncol = 2, nrow = 2, 
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
mc_gamma03_bound_loss05_mean_grid_with_title <- mc_gamma03_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(mc_avg_bound_loss05_mean_grid_with_title, mc_gamma03_bound_loss05_mean_grid_with_title,
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
mc_avg_gamma03_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                     ncol = 1, nrow = 2,
                                     heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### montecarlo_weather_gamma03_bound_loss05_var ########

montecarlo_weather_gamma03_bound_loss05_var_cs_steps = read_csv("cs_detail_results/montecarlo_weather_gamma03_bound_loss05_var_cs_steps.csv")

colnames(montecarlo_weather_gamma03_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma03_bound_loss05_var_cs_steps))/20+0.75)

montecarlo_weather_gamma03_bound_loss05_var_cs_steps_rate <-montecarlo_weather_gamma03_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma03_bound_loss05_var_cs_steps)

montecarlo_weather_gamma03_bound_loss05_var_q_steps = read_csv("cs_detail_results/montecarlo_weather_gamma03_bound_loss05_var_q_steps.csv")

colnames(montecarlo_weather_gamma03_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma03_bound_loss05_var_q_steps))/20+0.75)

montecarlo_weather_gamma03_bound_loss05_var_q_steps_rate <-montecarlo_weather_gamma03_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma03_bound_loss05_var_q_steps)

montecarlo_weather_gamma03_bound_loss05_var_r_steps = read_csv("cs_detail_results/montecarlo_weather_gamma03_bound_loss05_var_r_steps.csv")

colnames(montecarlo_weather_gamma03_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma03_bound_loss05_var_r_steps))/20+0.75)

montecarlo_weather_gamma03_bound_loss05_var_r_steps_rate <-montecarlo_weather_gamma03_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma03_bound_loss05_var_r_steps)

montecarlo_weather_gamma03_bound_loss05_var_ev_steps = read_csv("cs_detail_results/montecarlo_weather_gamma03_bound_loss05_var_ev_steps.csv")

colnames(montecarlo_weather_gamma03_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma03_bound_loss05_var_ev_steps))/20+0.75)

montecarlo_weather_gamma03_bound_loss05_var_ev_steps_rate <-montecarlo_weather_gamma03_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma03_bound_loss05_var_ev_steps)

montecarlo_weather_gamma03_bound_loss05_var_cs_steps_rate  = cbind(montecarlo_weather_gamma03_bound_loss05_var_cs_steps_rate , demand_key)
montecarlo_weather_gamma03_bound_loss05_var_q_steps_rate = cbind(montecarlo_weather_gamma03_bound_loss05_var_q_steps_rate, demand_key)
montecarlo_weather_gamma03_bound_loss05_var_r_steps_rate = cbind(montecarlo_weather_gamma03_bound_loss05_var_r_steps_rate, demand_key)
montecarlo_weather_gamma03_bound_loss05_var_ev_steps_rate = cbind(montecarlo_weather_gamma03_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- montecarlo_weather_gamma03_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- montecarlo_weather_gamma03_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- montecarlo_weather_gamma03_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- montecarlo_weather_gamma03_bound_loss05_var_ev_steps_rate %>%
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

mc_gamma03_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

mc_gamma03_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


mc_gamma03_bound_loss05_var_grid = ggarrange(cs_gamma03,r_gamma03,q_gamma03,ev_gamma03, ncol = 2, nrow = 2, 
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
mc_gamma03_bound_loss05_var_grid_with_title <- mc_gamma03_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(mc_avg_bound_loss05_var_grid_with_title, mc_gamma03_bound_loss05_var_grid_with_title,
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
mc_avg_gamma03_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                    ncol = 1, nrow = 2,
                                    heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))

##### montecarlo_weather_gamma04_bound_loss05_mean ########

montecarlo_weather_gamma04_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/montecarlo_weather_gamma04_bound_loss05_mean_cs_steps.csv")

colnames(montecarlo_weather_gamma04_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma04_bound_loss05_mean_cs_steps))*0.05-0.25)

montecarlo_weather_gamma04_bound_loss05_mean_cs_steps_rate <-montecarlo_weather_gamma04_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma04_bound_loss05_mean_cs_steps)

montecarlo_weather_gamma04_bound_loss05_mean_q_steps = read_csv("cs_detail_results/montecarlo_weather_gamma04_bound_loss05_mean_q_steps.csv")

colnames(montecarlo_weather_gamma04_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma04_bound_loss05_mean_q_steps))*0.05-0.25)

montecarlo_weather_gamma04_bound_loss05_mean_q_steps_rate <-montecarlo_weather_gamma04_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma04_bound_loss05_mean_q_steps)

montecarlo_weather_gamma04_bound_loss05_mean_r_steps = read_csv("cs_detail_results/montecarlo_weather_gamma04_bound_loss05_mean_r_steps.csv")

colnames(montecarlo_weather_gamma04_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma04_bound_loss05_mean_r_steps))*0.05-0.25)

montecarlo_weather_gamma04_bound_loss05_mean_r_steps_rate <-montecarlo_weather_gamma04_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma04_bound_loss05_mean_r_steps)

montecarlo_weather_gamma04_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/montecarlo_weather_gamma04_bound_loss05_mean_ev_steps.csv")

colnames(montecarlo_weather_gamma04_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma04_bound_loss05_mean_ev_steps))*0.05-0.25)

montecarlo_weather_gamma04_bound_loss05_mean_ev_steps_rate <-montecarlo_weather_gamma04_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma04_bound_loss05_mean_ev_steps)

montecarlo_weather_gamma04_bound_loss05_mean_cs_steps_rate  = cbind(montecarlo_weather_gamma04_bound_loss05_mean_cs_steps_rate , demand_key)
montecarlo_weather_gamma04_bound_loss05_mean_q_steps_rate = cbind(montecarlo_weather_gamma04_bound_loss05_mean_q_steps_rate, demand_key)
montecarlo_weather_gamma04_bound_loss05_mean_r_steps_rate = cbind(montecarlo_weather_gamma04_bound_loss05_mean_r_steps_rate, demand_key)
montecarlo_weather_gamma04_bound_loss05_mean_ev_steps_rate = cbind(montecarlo_weather_gamma04_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- montecarlo_weather_gamma04_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- montecarlo_weather_gamma04_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- montecarlo_weather_gamma04_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- montecarlo_weather_gamma04_bound_loss05_mean_ev_steps_rate %>%
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

mc_gamma04_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

mc_gamma04_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

mc_gamma04_bound_loss05_mean_grid = ggarrange(cs_gamma04,r_gamma04,q_gamma04,ev_gamma04, ncol = 2, nrow = 2, 
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
mc_gamma04_bound_loss05_mean_grid_with_title <- mc_gamma04_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(mc_avg_bound_loss05_mean_grid_with_title, mc_gamma04_bound_loss05_mean_grid_with_title,
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
mc_avg_gamma04_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                        ncol = 1, nrow = 2,
                                        heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### montecarlo_weather_gamma04_bound_loss05_var ########

montecarlo_weather_gamma04_bound_loss05_var_cs_steps = read_csv("cs_detail_results/montecarlo_weather_gamma04_bound_loss05_var_cs_steps.csv")

colnames(montecarlo_weather_gamma04_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma04_bound_loss05_var_cs_steps))/20+0.75)

montecarlo_weather_gamma04_bound_loss05_var_cs_steps_rate <-montecarlo_weather_gamma04_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma04_bound_loss05_var_cs_steps)

montecarlo_weather_gamma04_bound_loss05_var_q_steps = read_csv("cs_detail_results/montecarlo_weather_gamma04_bound_loss05_var_q_steps.csv")

colnames(montecarlo_weather_gamma04_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma04_bound_loss05_var_q_steps))/20+0.75)

montecarlo_weather_gamma04_bound_loss05_var_q_steps_rate <-montecarlo_weather_gamma04_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma04_bound_loss05_var_q_steps)

montecarlo_weather_gamma04_bound_loss05_var_r_steps = read_csv("cs_detail_results/montecarlo_weather_gamma04_bound_loss05_var_r_steps.csv")

colnames(montecarlo_weather_gamma04_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma04_bound_loss05_var_r_steps))/20+0.75)

montecarlo_weather_gamma04_bound_loss05_var_r_steps_rate <-montecarlo_weather_gamma04_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma04_bound_loss05_var_r_steps)

montecarlo_weather_gamma04_bound_loss05_var_ev_steps = read_csv("cs_detail_results/montecarlo_weather_gamma04_bound_loss05_var_ev_steps.csv")

colnames(montecarlo_weather_gamma04_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma04_bound_loss05_var_ev_steps))/20+0.75)

montecarlo_weather_gamma04_bound_loss05_var_ev_steps_rate <-montecarlo_weather_gamma04_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma04_bound_loss05_var_ev_steps)

montecarlo_weather_gamma04_bound_loss05_var_cs_steps_rate  = cbind(montecarlo_weather_gamma04_bound_loss05_var_cs_steps_rate , demand_key)
montecarlo_weather_gamma04_bound_loss05_var_q_steps_rate = cbind(montecarlo_weather_gamma04_bound_loss05_var_q_steps_rate, demand_key)
montecarlo_weather_gamma04_bound_loss05_var_r_steps_rate = cbind(montecarlo_weather_gamma04_bound_loss05_var_r_steps_rate, demand_key)
montecarlo_weather_gamma04_bound_loss05_var_ev_steps_rate = cbind(montecarlo_weather_gamma04_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- montecarlo_weather_gamma04_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- montecarlo_weather_gamma04_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- montecarlo_weather_gamma04_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- montecarlo_weather_gamma04_bound_loss05_var_ev_steps_rate %>%
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

mc_gamma04_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

mc_gamma04_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


mc_gamma04_bound_loss05_var_grid = ggarrange(cs_gamma04,r_gamma04,q_gamma04,ev_gamma04, ncol = 2, nrow = 2, 
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
mc_gamma04_bound_loss05_var_grid_with_title <- mc_gamma04_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(mc_avg_bound_loss05_var_grid_with_title, mc_gamma04_bound_loss05_var_grid_with_title,
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
mc_avg_gamma04_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                       ncol = 1, nrow = 2,
                                       heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))

##### montecarlo_weather_gamma05_bound_loss05_mean ########

montecarlo_weather_gamma05_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/montecarlo_weather_gamma05_bound_loss05_mean_cs_steps.csv")

colnames(montecarlo_weather_gamma05_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma05_bound_loss05_mean_cs_steps))*0.05-0.25)

montecarlo_weather_gamma05_bound_loss05_mean_cs_steps_rate <-montecarlo_weather_gamma05_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma05_bound_loss05_mean_cs_steps)

montecarlo_weather_gamma05_bound_loss05_mean_q_steps = read_csv("cs_detail_results/montecarlo_weather_gamma05_bound_loss05_mean_q_steps.csv")

colnames(montecarlo_weather_gamma05_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma05_bound_loss05_mean_q_steps))*0.05-0.25)

montecarlo_weather_gamma05_bound_loss05_mean_q_steps_rate <-montecarlo_weather_gamma05_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma05_bound_loss05_mean_q_steps)

montecarlo_weather_gamma05_bound_loss05_mean_r_steps = read_csv("cs_detail_results/montecarlo_weather_gamma05_bound_loss05_mean_r_steps.csv")

colnames(montecarlo_weather_gamma05_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma05_bound_loss05_mean_r_steps))*0.05-0.25)

montecarlo_weather_gamma05_bound_loss05_mean_r_steps_rate <-montecarlo_weather_gamma05_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma05_bound_loss05_mean_r_steps)

montecarlo_weather_gamma05_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/montecarlo_weather_gamma05_bound_loss05_mean_ev_steps.csv")

colnames(montecarlo_weather_gamma05_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma05_bound_loss05_mean_ev_steps))*0.05-0.25)

montecarlo_weather_gamma05_bound_loss05_mean_ev_steps_rate <-montecarlo_weather_gamma05_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma05_bound_loss05_mean_ev_steps)

montecarlo_weather_gamma05_bound_loss05_mean_cs_steps_rate  = cbind(montecarlo_weather_gamma05_bound_loss05_mean_cs_steps_rate , demand_key)
montecarlo_weather_gamma05_bound_loss05_mean_q_steps_rate = cbind(montecarlo_weather_gamma05_bound_loss05_mean_q_steps_rate, demand_key)
montecarlo_weather_gamma05_bound_loss05_mean_r_steps_rate = cbind(montecarlo_weather_gamma05_bound_loss05_mean_r_steps_rate, demand_key)
montecarlo_weather_gamma05_bound_loss05_mean_ev_steps_rate = cbind(montecarlo_weather_gamma05_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- montecarlo_weather_gamma05_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- montecarlo_weather_gamma05_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- montecarlo_weather_gamma05_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- montecarlo_weather_gamma05_bound_loss05_mean_ev_steps_rate %>%
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

mc_gamma05_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

mc_gamma05_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

mc_gamma05_bound_loss05_mean_grid = ggarrange(cs_gamma05,r_gamma05,q_gamma05,ev_gamma05, ncol = 2, nrow = 2, 
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
mc_gamma05_bound_loss05_mean_grid_with_title <- mc_gamma05_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(mc_avg_bound_loss05_mean_grid_with_title, mc_gamma05_bound_loss05_mean_grid_with_title,
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
mc_avg_gamma05_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                        ncol = 1, nrow = 2,
                                        heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### montecarlo_weather_gamma05_bound_loss05_var ########

montecarlo_weather_gamma05_bound_loss05_var_cs_steps = read_csv("cs_detail_results/montecarlo_weather_gamma05_bound_loss05_var_cs_steps.csv")

colnames(montecarlo_weather_gamma05_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma05_bound_loss05_var_cs_steps))/20+0.75)

montecarlo_weather_gamma05_bound_loss05_var_cs_steps_rate <-montecarlo_weather_gamma05_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma05_bound_loss05_var_cs_steps)

montecarlo_weather_gamma05_bound_loss05_var_q_steps = read_csv("cs_detail_results/montecarlo_weather_gamma05_bound_loss05_var_q_steps.csv")

colnames(montecarlo_weather_gamma05_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma05_bound_loss05_var_q_steps))/20+0.75)

montecarlo_weather_gamma05_bound_loss05_var_q_steps_rate <-montecarlo_weather_gamma05_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma05_bound_loss05_var_q_steps)

montecarlo_weather_gamma05_bound_loss05_var_r_steps = read_csv("cs_detail_results/montecarlo_weather_gamma05_bound_loss05_var_r_steps.csv")

colnames(montecarlo_weather_gamma05_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma05_bound_loss05_var_r_steps))/20+0.75)

montecarlo_weather_gamma05_bound_loss05_var_r_steps_rate <-montecarlo_weather_gamma05_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma05_bound_loss05_var_r_steps)

montecarlo_weather_gamma05_bound_loss05_var_ev_steps = read_csv("cs_detail_results/montecarlo_weather_gamma05_bound_loss05_var_ev_steps.csv")

colnames(montecarlo_weather_gamma05_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(montecarlo_weather_gamma05_bound_loss05_var_ev_steps))/20+0.75)

montecarlo_weather_gamma05_bound_loss05_var_ev_steps_rate <-montecarlo_weather_gamma05_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(montecarlo_weather_gamma05_bound_loss05_var_ev_steps)

montecarlo_weather_gamma05_bound_loss05_var_cs_steps_rate  = cbind(montecarlo_weather_gamma05_bound_loss05_var_cs_steps_rate , demand_key)
montecarlo_weather_gamma05_bound_loss05_var_q_steps_rate = cbind(montecarlo_weather_gamma05_bound_loss05_var_q_steps_rate, demand_key)
montecarlo_weather_gamma05_bound_loss05_var_r_steps_rate = cbind(montecarlo_weather_gamma05_bound_loss05_var_r_steps_rate, demand_key)
montecarlo_weather_gamma05_bound_loss05_var_ev_steps_rate = cbind(montecarlo_weather_gamma05_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- montecarlo_weather_gamma05_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- montecarlo_weather_gamma05_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- montecarlo_weather_gamma05_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- montecarlo_weather_gamma05_bound_loss05_var_ev_steps_rate %>%
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

mc_gamma05_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

mc_gamma05_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


mc_gamma05_bound_loss05_var_grid = ggarrange(cs_gamma05,r_gamma05,q_gamma05,ev_gamma05, ncol = 2, nrow = 2, 
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
mc_gamma05_bound_loss05_var_grid_with_title <- mc_gamma05_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(mc_avg_bound_loss05_var_grid_with_title, mc_gamma05_bound_loss05_var_grid_with_title,
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
mc_avg_gamma05_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                       ncol = 1, nrow = 2,
                                       heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))






##### Clean up #####
# Get a list of all objects in the current environment
all_objects <- ls()

# Identify dataframes that start with "current"
dataframes_to_delete <- c()
for (obj_name in all_objects) {
  if (startsWith(obj_name, "montecarlo") && is.data.frame(get(obj_name))) {
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

##### Individual Level #####

demand_char = demand_2018_using_new_small %>%
  select(charge, NDVI, deflator, total_housevalue, bathroom, bedroom, spa, spa_area,
         fountain_wtr, fountain_wtr_area, heavy_water_app, heavy_water_app_area,hvac_residential,
         house_area, lawn_area, heavy_water_spa, heavy_water_spa_area,prev_NDVI, total_area, lawn_percentage,mean_e_diff,
         mean_TMAX_1, IQR_TMAX_1, total_PRCP, IQR_PRCP
         )
#rm(demand_2018_using_new_small)

demand_char = demand_char %>%
  #select(mean_TMAX_1, IQR_TMAX_1, total_PRCP, IQR_PRCP) %>%
  mutate(total_PRCP_dry = pmax(total_PRCP - 0.25, 1e-16) ,
         total_PRCP_wet = pmax(total_PRCP + 0.25, 1e-16)
         )

montecarlo_weather_avg_bound_loss05_mean_ev_steps_rate = cbind(montecarlo_weather_avg_bound_loss05_mean_ev_steps_rate, demand_char)
#montecarlo_weather_gamma03_bound_loss05_mean_ev_steps_rate = cbind(montecarlo_weather_gamma03_bound_loss05_mean_ev_steps_rate, demand_char)

avg_mean_high_stratus = montecarlo_weather_avg_bound_loss05_mean_ev_steps_rate[which(montecarlo_weather_avg_bound_loss05_mean_ev_steps_rate$income_strata==">100k"),]
avg_mean_low_stratus = montecarlo_weather_avg_bound_loss05_mean_ev_steps_rate[which(montecarlo_weather_avg_bound_loss05_mean_ev_steps_rate$income_strata=="0~6k"),]
avg_q_mean_low_stratus = montecarlo_weather_avg_bound_loss05_mean_q_steps_rate[which(montecarlo_weather_avg_bound_loss05_mean_q_steps_rate$income_strata=="0~6k"),]

#gamma03_mean_high_stratus = montecarlo_weather_gamma03_bound_loss05_mean_ev_steps_rate[which(montecarlo_weather_gamma03_bound_loss05_mean_ev_steps_rate$income_strata==">100k"),]
#gamma03_mean_low_stratus = montecarlo_weather_gamma03_bound_loss05_mean_ev_steps_rate[which(montecarlo_weather_gamma03_bound_loss05_mean_ev_steps_rate$income_strata=="0~6k"),]

avg_mean_low_stratus = avg_mean_low_stratus %>%
  group_by(prem_id) %>%
  mutate(mean_ev_rate_by_bill_ym_dry = mean(`rate_-0.25`),
         mean_ev_rate_by_bill_ym_wet = mean(`rate_0.25`),
         ) %>%
  ungroup()

avg_q_mean_low_stratus = avg_q_mean_low_stratus %>%
  select(`-0.25`, `0.25`) %>%
  rename(quantity_dry = `-0.25`,
         quantity_wet = `0.25`)

avg_mean_low_stratus = cbind(avg_mean_low_stratus, avg_q_mean_low_stratus)

avg_mean_low_stratus = avg_mean_low_stratus %>%
  rename(ev_rate_dry = `rate_-0.25`,
         ev_rate_wet = `rate_0.25`,
         ev_dry = `-0.25`,
         ev_wet = `0.25`)

avg_mean_low_stratus$NDVI_diff = avg_mean_low_stratus$NDVI - avg_mean_low_stratus$prev_NDVI

beta_prcp = 0.008

avg_mean_low_stratus = avg_mean_low_stratus %>%
  mutate(cf_NDVI_dry = NDVI + beta_prcp * -0.25,
         cf_NDVI_wet = NDVI + beta_prcp * 0.25,
         cf_prev_NDVI_dry = prev_NDVI + beta_prcp * -0.25,
         cf_prev_NDVI_wet = NDVI + beta_prcp * 0.25,
         cf_NDVI_diff_dry = cf_NDVI_dry - cf_prev_NDVI_dry ,
         cf_NDVI_diff_wet = cf_NDVI_wet - cf_prev_NDVI_wet ,
         )

avg_mean_low_stratus = avg_mean_low_stratus %>%
  group_by(prem_id) %>%
  mutate(
    prev_prev_NDVI = lag(prev_NDVI, 1),
    cf_prev_prev_NDVI_dry = lag(cf_prev_NDVI_dry, 1),
    cf_prev_prev_NDVI_wet = lag(cf_prev_NDVI_wet, 1),
    cf_prev_NDVI_diff_dry = cf_prev_NDVI_dry - cf_prev_prev_NDVI_dry ,
    cf_prev_NDVI_diff_wet = cf_prev_NDVI_wet - cf_prev_prev_NDVI_wet ,
    prev_NDVI_diff = prev_NDVI - prev_prev_NDVI,
    mean_TMAX_1_lag = lag(mean_TMAX_1, 1),
    IQR_TMAX_1_lag = lag(IQR_TMAX_1, 1),
    total_PRCP_lag = lag(total_PRCP, 1),
    IQR_PRCP_lag = lag(IQR_PRCP, 1),
    avg_NDVI = mean(NDVI)
  )

avg_mean_low_stratus$NDVI_category = case_when(
  avg_mean_low_stratus$NDVI <0.2 ~ "1",
  avg_mean_low_stratus$NDVI >=0.2 &  avg_mean_low_stratus$NDVI <0.35~ "2",
  avg_mean_low_stratus$NDVI >=0.35 &  avg_mean_low_stratus$NDVI <0.5~ "3",
  avg_mean_low_stratus$NDVI >=0.5 ~ "4",
)

model_dry = lm(quantity_dry~bedroom + bathroom +cf_prev_NDVI_diff_dry+
                 mean_TMAX_1 + IQR_TMAX_1 + total_PRCP + IQR_PRCP +
                 mean_TMAX_1_lag + IQR_TMAX_1_lag + total_PRCP_lag + IQR_PRCP_lag + NDVI_category+
             heavy_water_spa_area + lawn_percentage + mean_e_diff, data = avg_mean_low_stratus)
summary(model_dry)

model_wet = lm(quantity_wet~bedroom + bathroom +cf_prev_NDVI_diff_wet+
                 mean_TMAX_1 + IQR_TMAX_1 + total_PRCP + IQR_PRCP +
                 mean_TMAX_1_lag + IQR_TMAX_1_lag + total_PRCP_lag + IQR_PRCP_lag + NDVI_category+
                 heavy_water_spa_area + lawn_percentage + mean_e_diff, data = avg_mean_low_stratus)
summary(model_wet)

# Reshape the data from wide to long format to plot multiple x-variables
data_long <- pivot_longer(
  avg_mean_low_stratus,
  cols = c(cf_prev_NDVI_wet, cf_prev_NDVI_dry, prev_NDVI),
  names_to = "NDVI_Type",
  values_to = "NDVI_Value"
)

# Create the ggplot using the new long-format data
ggplot(data_long, aes(x = NDVI_Value, y = quantity_dry)) +
  # geom_hex creates hexagonal bins. Darker color = more points.
  geom_hex() +
  # You can still overlay the trend line
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  # To see the different NDVI types, you can create separate plots with faceting
  facet_wrap(~ NDVI_Type, scales = "free_x") +
  labs(title = "Density of Points for Each NDVI Metric") +
  theme_minimal()

# Create the ggplot using the new long-format data
ggplot(avg_mean_low_stratus, aes(x = NDVI_category, y = quantity_dry)) +
  # geom_hex creates hexagonal bins. Darker color = more points.
  geom_hex() +
  # You can still overlay the trend line
  geom_smooth(method = "lm", se = FALSE, color = "red") +

  labs(title = "Density of Points for Each NDVI Metric") +
  theme_minimal()


###### on prem_id_level #####



avg_mean_low_stratus_sum = avg_mean_low_stratus %>%
  group_by(prem_id) %>%
  summarise(total_housevalue = mean(total_housevalue),
            bedroom = mean(bedroom),
            bathroom = mean(bathroom),
            NDVI = mean(NDVI),
            income = mean(income),
            spa = mean(spa),
            spa_area = mean(spa_area),
            fountain_wtr = mean(fountain_wtr),
            fountain_wtr_area = mean(fountain_wtr_area),
            heavy_water_app = mean(heavy_water_app),
            heavy_water_app_area = mean(heavy_water_app_area),
            hvac_residential = mean(hvac_residential),
            house_area = mean(house_area),
            lawn_area = mean(lawn_area),
            heavy_water_spa = mean(heavy_water_app),
            heavy_water_spa_area = mean(heavy_water_spa_area),
            prev_NDVI = mean(prev_NDVI),
            NDVI_diff = mean(NDVI_diff),
            total_area = mean(total_area),
            lawn_percentage = mean(lawn_percentage),
            mean_e_diff = mean(mean_e_diff),
            mean_ev_rate_dry = mean(ev_rate_dry),
            mean_ev_rate_wet = mean(ev_rate_wet),
            mean_quantity_dry = mean(quantity_dry),
            mean_quantity_wet = mean(quantity_wet)
            )

model = lm(mean_ev_rate_by_bill_ym~bedroom + bathroom + NDVI + prev_NDVI + income + 
    heavy_water_spa_area + lawn_percentage + mean_e_diff, data = avg_mean_low_stratus_sum)
summary(model)

avg_mean_low_stratus_low = avg_mean_low_stratus[avg_mean_low_stratus$mean_ev_rate_by_bill_ym < -20,]



