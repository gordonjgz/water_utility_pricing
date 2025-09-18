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
##### sq_ndvi_avg_bound_loss05_mean ########

sq_ndvi_avg_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/sq_ndvi_avg_bound_loss05_mean_cs_steps.csv")

colnames(sq_ndvi_avg_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(sq_ndvi_avg_bound_loss05_mean_cs_steps))*0.05-0.25)

sq_ndvi_avg_bound_loss05_mean_cs_steps_rate <-sq_ndvi_avg_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
  #mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(sq_ndvi_avg_bound_loss05_mean_cs_steps)

sq_ndvi_avg_bound_loss05_mean_q_steps = read_csv("cs_detail_results/sq_ndvi_avg_bound_loss05_mean_q_steps.csv")

colnames(sq_ndvi_avg_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(sq_ndvi_avg_bound_loss05_mean_q_steps))*0.05-0.25)

sq_ndvi_avg_bound_loss05_mean_q_steps_rate <-sq_ndvi_avg_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
  #mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(sq_ndvi_avg_bound_loss05_mean_q_steps)

sq_ndvi_avg_bound_loss05_mean_r_steps = read_csv("cs_detail_results/sq_ndvi_avg_bound_loss05_mean_r_steps.csv")

colnames(sq_ndvi_avg_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(sq_ndvi_avg_bound_loss05_mean_r_steps))*0.05-0.25)

sq_ndvi_avg_bound_loss05_mean_r_steps_rate <-sq_ndvi_avg_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
  #mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(sq_ndvi_avg_bound_loss05_mean_r_steps)

sq_ndvi_avg_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/sq_ndvi_avg_bound_loss05_mean_ev_steps.csv")

colnames(sq_ndvi_avg_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(sq_ndvi_avg_bound_loss05_mean_ev_steps))*0.05-0.25)

sq_ndvi_avg_bound_loss05_mean_ev_steps_rate <-sq_ndvi_avg_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(sq_ndvi_avg_bound_loss05_mean_ev_steps)

sq_ndvi_avg_bound_loss05_mean_cs_steps_rate  = cbind(sq_ndvi_avg_bound_loss05_mean_cs_steps_rate , demand_key)
sq_ndvi_avg_bound_loss05_mean_q_steps_rate = cbind(sq_ndvi_avg_bound_loss05_mean_q_steps_rate, demand_key)
sq_ndvi_avg_bound_loss05_mean_r_steps_rate = cbind(sq_ndvi_avg_bound_loss05_mean_r_steps_rate, demand_key)
sq_ndvi_avg_bound_loss05_mean_ev_steps_rate = cbind(sq_ndvi_avg_bound_loss05_mean_ev_steps_rate, demand_key)

#sq_ndvi_avg_bound_loss05_mean_cs_steps_rate = sq_ndvi_avg_bound_loss05_mean_cs_steps_rate[which(sq_ndvi_avg_bound_loss05_mean_cs_steps_rate$q_0<100),]
#sq_ndvi_avg_bound_loss05_mean_ev_steps_rate = sq_ndvi_avg_bound_loss05_mean_ev_steps_rate[which(sq_ndvi_avg_bound_loss05_mean_ev_steps_rate$q_0<100),]

mean_cs_rate_by_bill_ym <- sq_ndvi_avg_bound_loss05_mean_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(income_strata,prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- sq_ndvi_avg_bound_loss05_mean_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(income_strata,prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- sq_ndvi_avg_bound_loss05_mean_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(income_strata,prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- sq_ndvi_avg_bound_loss05_mean_ev_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(income_strata,prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_by_bill_ym <- sq_ndvi_avg_bound_loss05_mean_ev_steps_rate %>%
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

sq_avg_bound_loss05_mean_ev = mean_ev_rate_by_strata_sq %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

sq_avg_bound_loss05_mean_cs = mean_cs_rate_by_strata_sq %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

shared_legend <- get_legend(r_avg)


# Arrange the three plots and the shared legend as the fourth item
sq_avg_bound_loss05_mean_grid  <- ggarrange(
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
sq_avg_bound_loss05_mean_grid_with_title <- sq_avg_bound_loss05_mean_grid +
  plot_annotation(
    title = "Linear Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )

##### sq_ndvi_avg_bound_loss05_var ########

sq_ndvi_avg_bound_loss05_var_cs_steps = read_csv("cs_detail_results/sq_ndvi_avg_bound_loss05_var_cs_steps.csv")

colnames(sq_ndvi_avg_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(sq_ndvi_avg_bound_loss05_var_cs_steps))/20+0.75)

sq_ndvi_avg_bound_loss05_var_cs_steps_rate <-sq_ndvi_avg_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(sq_ndvi_avg_bound_loss05_var_cs_steps)

sq_ndvi_avg_bound_loss05_var_q_steps = read_csv("cs_detail_results/sq_ndvi_avg_bound_loss05_var_q_steps.csv")

colnames(sq_ndvi_avg_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(sq_ndvi_avg_bound_loss05_var_q_steps))/20+0.75)

sq_ndvi_avg_bound_loss05_var_q_steps_rate <-sq_ndvi_avg_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(sq_ndvi_avg_bound_loss05_var_q_steps)

sq_ndvi_avg_bound_loss05_var_r_steps = read_csv("cs_detail_results/sq_ndvi_avg_bound_loss05_var_r_steps.csv")

colnames(sq_ndvi_avg_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(sq_ndvi_avg_bound_loss05_var_r_steps))/20+0.75)

sq_ndvi_avg_bound_loss05_var_r_steps_rate <-sq_ndvi_avg_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(sq_ndvi_avg_bound_loss05_var_r_steps)

sq_ndvi_avg_bound_loss05_var_ev_steps = read_csv("cs_detail_results/sq_ndvi_avg_bound_loss05_var_ev_steps.csv")

colnames(sq_ndvi_avg_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(sq_ndvi_avg_bound_loss05_var_ev_steps))/20+0.75)

sq_ndvi_avg_bound_loss05_var_ev_steps_rate <-sq_ndvi_avg_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(sq_ndvi_avg_bound_loss05_var_ev_steps)

sq_ndvi_avg_bound_loss05_var_cs_steps_rate  = cbind(sq_ndvi_avg_bound_loss05_var_cs_steps_rate , demand_key)
sq_ndvi_avg_bound_loss05_var_q_steps_rate = cbind(sq_ndvi_avg_bound_loss05_var_q_steps_rate, demand_key)
sq_ndvi_avg_bound_loss05_var_r_steps_rate = cbind(sq_ndvi_avg_bound_loss05_var_r_steps_rate, demand_key)
sq_ndvi_avg_bound_loss05_var_ev_steps_rate = cbind(sq_ndvi_avg_bound_loss05_var_ev_steps_rate, demand_key)

#sq_ndvi_avg_bound_loss05_var_cs_steps_rate = sq_ndvi_avg_bound_loss05_var_cs_steps_rate[which(sq_ndvi_avg_bound_loss05_var_cs_steps_rate$q_0<100),]
#sq_ndvi_avg_bound_loss05_var_ev_steps_rate = sq_ndvi_avg_bound_loss05_var_ev_steps_rate[which(sq_ndvi_avg_bound_loss05_var_ev_steps_rate$q_0<100),]

var_cs_rate_by_bill_ym <- sq_ndvi_avg_bound_loss05_var_cs_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- sq_ndvi_avg_bound_loss05_var_q_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- sq_ndvi_avg_bound_loss05_var_r_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- sq_ndvi_avg_bound_loss05_var_ev_steps_rate %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_by_bill_ym <- sq_ndvi_avg_bound_loss05_var_ev_steps_rate %>%
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

sq_avg_bound_loss05_var_ev = var_ev_rate_by_strata_sq %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

sq_avg_bound_loss05_var_cs = var_cs_rate_by_strata_sq %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

shared_legend <- get_legend(r_avg)

# Arrange the three plots and the shared legend as the fourth item
sq_avg_bound_loss05_var_grid <- ggarrange(
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
sq_avg_bound_loss05_var_grid_with_title <- sq_avg_bound_loss05_var_grid +
  plot_annotation(
    title = "Linear Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )




##### half_ndvi_low_avg_bound_loss05_mean ########

half_ndvi_low_avg_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/reduction_0.50_low_avg_bound_loss05_mean_cs_steps.csv")

colnames(half_ndvi_low_avg_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(half_ndvi_low_avg_bound_loss05_mean_cs_steps))*0.05-0.25)

half_ndvi_low_avg_bound_loss05_mean_cs_steps_rate <-half_ndvi_low_avg_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(half_ndvi_low_avg_bound_loss05_mean_cs_steps)

half_ndvi_low_avg_bound_loss05_mean_q_steps = read_csv("cs_detail_results/reduction_0.50_low_avg_bound_loss05_mean_q_steps.csv")

colnames(half_ndvi_low_avg_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(half_ndvi_low_avg_bound_loss05_mean_q_steps))*0.05-0.25)

half_ndvi_low_avg_bound_loss05_mean_q_steps_rate <-half_ndvi_low_avg_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(half_ndvi_low_avg_bound_loss05_mean_q_steps)

half_ndvi_low_avg_bound_loss05_mean_r_steps = read_csv("cs_detail_results/reduction_0.50_low_avg_bound_loss05_mean_r_steps.csv")

colnames(half_ndvi_low_avg_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(half_ndvi_low_avg_bound_loss05_mean_r_steps))*0.05-0.25)

half_ndvi_low_avg_bound_loss05_mean_r_steps_rate <-half_ndvi_low_avg_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(half_ndvi_low_avg_bound_loss05_mean_r_steps)

half_ndvi_low_avg_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/reduction_0.50_low_avg_bound_loss05_mean_ev_steps.csv")

colnames(half_ndvi_low_avg_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(half_ndvi_low_avg_bound_loss05_mean_ev_steps))*0.05-0.25)

half_ndvi_low_avg_bound_loss05_mean_ev_steps_rate <-half_ndvi_low_avg_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(half_ndvi_low_avg_bound_loss05_mean_ev_steps)

half_ndvi_low_avg_bound_loss05_mean_cs_steps_rate  = cbind(half_ndvi_low_avg_bound_loss05_mean_cs_steps_rate , demand_key)
half_ndvi_low_avg_bound_loss05_mean_q_steps_rate = cbind(half_ndvi_low_avg_bound_loss05_mean_q_steps_rate, demand_key)
half_ndvi_low_avg_bound_loss05_mean_r_steps_rate = cbind(half_ndvi_low_avg_bound_loss05_mean_r_steps_rate, demand_key)
half_ndvi_low_avg_bound_loss05_mean_ev_steps_rate = cbind(half_ndvi_low_avg_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- half_ndvi_low_avg_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- half_ndvi_low_avg_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- half_ndvi_low_avg_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- half_ndvi_low_avg_bound_loss05_mean_ev_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_by_bill_ym <- half_ndvi_low_avg_bound_loss05_mean_ev_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(1:11, mean, na.rm = TRUE))

mean_cs_rate_by_strata_low_ndvi <- mean_cs_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_strata_low_ndvi <- mean_r_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_strata_low_ndvi <- mean_q_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_strata_low_ndvi <- mean_ev_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_by_strata_low_ndvi <- mean_ev_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(2:12, mean, na.rm = TRUE))

#mean_cs_rate_by_strata = edit_strata_df_quantity(mean_cs_rate_by_strata, "cs")
mean_cs_rate_by_strata_low_ndvi = edit_strata_df(mean_cs_rate_by_strata_low_ndvi, "cs")
mean_cs_rate_by_strata_low_ndvi$cs_change_rate = as.numeric(mean_cs_rate_by_strata_low_ndvi$cs_change_rate)

#mean_r_rate_by_strata = edit_strata_df_quantity(mean_r_rate_by_strata, "r")
mean_r_rate_by_strata_low_ndvi = edit_strata_df(mean_r_rate_by_strata_low_ndvi, "r")
mean_r_rate_by_strata_low_ndvi$r_change_rate = as.numeric(mean_r_rate_by_strata_low_ndvi$r_change_rate)

#mean_q_rate_by_strata = edit_strata_df_quantity(mean_q_rate_by_strata, "q")
mean_q_rate_by_strata_low_ndvi = edit_strata_df(mean_q_rate_by_strata_low_ndvi, "q")
mean_q_rate_by_strata_low_ndvi$q_change_rate = as.numeric(mean_q_rate_by_strata_low_ndvi$q_change_rate)

#mean_ev_rate_by_strata = edit_strata_df_quantity(mean_ev_rate_by_strata, "ev")
mean_ev_rate_by_strata_low_ndvi = edit_strata_df(mean_ev_rate_by_strata_low_ndvi, "ev")
mean_ev_rate_by_strata_low_ndvi$ev_perct = as.numeric(mean_ev_rate_by_strata_low_ndvi$ev_change_rate)
mean_ev_rate_by_strata_low_ndvi$ev_change_rate = NULL

mean_ev_by_strata_low_ndvi = edit_strata_df(mean_ev_by_strata_low_ndvi, "ev")
mean_ev_by_strata_low_ndvi$ev = as.numeric(mean_ev_by_strata_low_ndvi$ev_change_rate)
mean_ev_by_strata_low_ndvi$ev_change_rate = NULL

cs_half_ndvi = ggplot(mean_cs_rate_by_strata_low_ndvi , aes(x = step, y = cs_change_rate, 
                                                 color = income_strata, shape = income_strata
                                                 #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  #geom_point(data = subset(mean_cs_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_point(data = subset(mean_cs_rate_by_strata_low_ndvi, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
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


r_half_ndvi = ggplot(mean_r_rate_by_strata_low_ndvi , aes(x = step, y = r_change_rate, 
                                               color = income_strata, shape = income_strata
                                               #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_r_rate_by_strata_low_ndvi, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
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

q_half_ndvi = ggplot(mean_q_rate_by_strata_low_ndvi , aes(x = step, y = q_change_rate, 
                                               color = income_strata, shape = income_strata
                                               #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_q_rate_by_strata_low_ndvi, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
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

ev_half_ndvi = ggplot(mean_ev_rate_by_strata_low_ndvi , aes(x = step, y = ev_perct, 
                                                 color = income_strata, shape = income_strata
                                                 #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(mean_ev_rate_by_strata_low_ndvi, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
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

mean_ev_rate_by_strata_low_ndvi$status = case_when(
  mean_ev_rate_by_strata_low_ndvi$step<0 ~ "low",
  mean_ev_rate_by_strata_low_ndvi$step>0 ~ "high"
)

mean_cs_rate_by_strata_low_ndvi$status = case_when(
  mean_cs_rate_by_strata_low_ndvi$step<0 ~ "low",
  mean_cs_rate_by_strata_low_ndvi$step>0 ~ "high"
)

half_ndvi_bound_loss05_mean_ev = mean_ev_rate_by_strata_low_ndvi %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

half_ndvi_bound_loss05_mean_cs = mean_cs_rate_by_strata_low_ndvi %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

half_ndvi_bound_loss05_mean_grid = ggarrange(cs_half_ndvi,r_half_ndvi,q_half_ndvi,ev_half_ndvi, ncol = 2, nrow = 2, 
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
half_ndvi_bound_loss05_mean_grid_with_title <- half_ndvi_bound_loss05_mean_grid +
  plot_annotation(
    title = "Concave Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )

mean_cs_rate_by_strata_low_ndvi[mean_cs_rate_by_strata_low_ndvi$step>1,] %>% 
  group_by(income_strata) %>%
  summarise(cs_change_rate = mean(cs_change_rate))

# Step 1: Create a temporary plot with visible content
legend_plot <- ggplot(mean_cs_rate_by_strata_low_ndvi, aes(x = step, y = cs_change_rate, color = income_strata, shape = income_strata)) +
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

combined_grids_with_outer_labels <- ggarrange(sq_avg_bound_loss05_mean_grid_with_title, half_ndvi_bound_loss05_mean_grid_with_title,
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
half_ndvi_bound_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                        ncol = 1, nrow = 2,
                                        heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### half_ndvi_low_avg_bound_loss05_var ########

half_ndvi_low_avg_bound_loss05_var_cs_steps = read_csv("cs_detail_results/reduction_0.50_low_avg_bound_loss05_var_cs_steps.csv")

colnames(half_ndvi_low_avg_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(half_ndvi_low_avg_bound_loss05_var_cs_steps))/20+0.75)

half_ndvi_low_avg_bound_loss05_var_cs_steps_rate <-half_ndvi_low_avg_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(half_ndvi_low_avg_bound_loss05_var_cs_steps)

half_ndvi_low_avg_bound_loss05_var_q_steps = read_csv("cs_detail_results/reduction_0.50_low_avg_bound_loss05_var_q_steps.csv")

colnames(half_ndvi_low_avg_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(half_ndvi_low_avg_bound_loss05_var_q_steps))/20+0.75)

half_ndvi_low_avg_bound_loss05_var_q_steps_rate <-half_ndvi_low_avg_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(half_ndvi_low_avg_bound_loss05_var_q_steps)

half_ndvi_low_avg_bound_loss05_var_r_steps = read_csv("cs_detail_results/reduction_0.50_low_avg_bound_loss05_var_r_steps.csv")

colnames(half_ndvi_low_avg_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(half_ndvi_low_avg_bound_loss05_var_r_steps))/20+0.75)

half_ndvi_low_avg_bound_loss05_var_r_steps_rate <-half_ndvi_low_avg_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(half_ndvi_low_avg_bound_loss05_var_r_steps)

half_ndvi_low_avg_bound_loss05_var_ev_steps = read_csv("cs_detail_results/reduction_0.50_low_avg_bound_loss05_var_ev_steps.csv")

colnames(half_ndvi_low_avg_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(half_ndvi_low_avg_bound_loss05_var_ev_steps))/20+0.75)

half_ndvi_low_avg_bound_loss05_var_ev_steps_rate <-half_ndvi_low_avg_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(half_ndvi_low_avg_bound_loss05_var_ev_steps)

half_ndvi_low_avg_bound_loss05_var_cs_steps_rate  = cbind(half_ndvi_low_avg_bound_loss05_var_cs_steps_rate , demand_key)
half_ndvi_low_avg_bound_loss05_var_q_steps_rate = cbind(half_ndvi_low_avg_bound_loss05_var_q_steps_rate, demand_key)
half_ndvi_low_avg_bound_loss05_var_r_steps_rate = cbind(half_ndvi_low_avg_bound_loss05_var_r_steps_rate, demand_key)
half_ndvi_low_avg_bound_loss05_var_ev_steps_rate = cbind(half_ndvi_low_avg_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- half_ndvi_low_avg_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- half_ndvi_low_avg_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- half_ndvi_low_avg_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- half_ndvi_low_avg_bound_loss05_var_ev_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_by_bill_ym <- half_ndvi_low_avg_bound_loss05_var_ev_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  summarise(across(1:11, mean, na.rm = TRUE))

var_cs_rate_by_strata_low_ndvi <- var_cs_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_strata_low_ndvi <- var_r_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_strata_low_ndvi <- var_q_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_strata_low_ndvi <- var_ev_rate_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_by_strata_low_ndvi <- var_ev_by_bill_ym %>%
  #group_by(quantity_strata) %>%
  group_by(income_strata) %>%
  summarise(across(2:12, mean, na.rm = TRUE))

#var_cs_rate_by_strata = edit_strata_df_quantity(var_cs_rate_by_strata, "cs")
var_cs_rate_by_strata_low_ndvi = edit_strata_df(var_cs_rate_by_strata_low_ndvi, "cs")
var_cs_rate_by_strata_low_ndvi$cs_change_rate = as.numeric(var_cs_rate_by_strata_low_ndvi$cs_change_rate)

#var_r_rate_by_strata = edit_strata_df_quantity(var_r_rate_by_strata, "r")
var_r_rate_by_strata_low_ndvi = edit_strata_df(var_r_rate_by_strata_low_ndvi, "r")
var_r_rate_by_strata_low_ndvi$r_change_rate = as.numeric(var_r_rate_by_strata_low_ndvi$r_change_rate)

#var_q_rate_by_strata = edit_strata_df_quantity(var_q_rate_by_strata, "q")
var_q_rate_by_strata_low_ndvi = edit_strata_df(var_q_rate_by_strata_low_ndvi, "q")
var_q_rate_by_strata_low_ndvi$q_change_rate = as.numeric(var_q_rate_by_strata_low_ndvi$q_change_rate)

#var_ev_rate_by_strata = edit_strata_df_quantity(var_ev_rate_by_strata, "ev")
var_ev_rate_by_strata_low_ndvi = edit_strata_df(var_ev_rate_by_strata_low_ndvi, "ev")
var_ev_rate_by_strata_low_ndvi$ev_perct = as.numeric(var_ev_rate_by_strata_low_ndvi$ev_change_rate)
var_ev_rate_by_strata_low_ndvi$ev_change_rate = NULL

var_ev_by_strata_low_ndvi = edit_strata_df(var_ev_by_strata_low_ndvi, "ev")
var_ev_by_strata_low_ndvi$ev = as.numeric(var_ev_by_strata_low_ndvi$ev_change_rate)
var_ev_by_strata_low_ndvi$ev_change_rate = NULL

cs_half_ndvi = ggplot(var_cs_rate_by_strata_low_ndvi , aes(x = step, y = cs_change_rate, 
                                                color = income_strata, shape = income_strata
                                                #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  #geom_point(data = subset(var_cs_rate_by_strata, quantity_strata == "4"), size = 4.5, shape = 18) + # Adjust size of shape 18+
  geom_point(data = subset(var_cs_rate_by_strata_low_ndvi, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
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


r_half_ndvi = ggplot(var_r_rate_by_strata_low_ndvi , aes(x = step, y = r_change_rate, 
                                              color = income_strata, shape = income_strata
                                              #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_r_rate_by_strata_low_ndvi, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
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

q_half_ndvi = ggplot(var_q_rate_by_strata_low_ndvi , aes(x = step, y = q_change_rate, 
                                              color = income_strata, shape = income_strata
                                              #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_q_rate_by_strata_low_ndvi, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
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

ev_half_ndvi = ggplot(var_ev_rate_by_strata_low_ndvi , aes(x = step, y = ev_perct, 
                                                color = income_strata, shape = income_strata
                                                #color = quantity_strata, shape = quantity_strata
)) +
  geom_line(size = 1.5) +  # Add lines
  geom_point(size = 4, stroke = 1.2) + # Optional: Add points
  geom_point(data = subset(var_ev_rate_by_strata_low_ndvi, income_strata == "45k~100k"), size = 4.5, shape = 18) + # Adjust size of shape 18+
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

var_ev_rate_by_strata_low_ndvi$status = case_when(
  var_ev_rate_by_strata_low_ndvi$step<1 ~ "low",
  var_ev_rate_by_strata_low_ndvi$step>1 ~ "high"
)

var_cs_rate_by_strata_low_ndvi$status = case_when(
  var_cs_rate_by_strata_low_ndvi$step<1 ~ "low",
  var_cs_rate_by_strata_low_ndvi$step>1 ~ "high"
)

half_ndvi_bound_loss05_var_ev = var_ev_rate_by_strata_low_ndvi %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

half_ndvi_bound_loss05_var_cs = var_cs_rate_by_strata_low_ndvi %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


half_ndvi_bound_loss05_var_grid = ggarrange(cs_half_ndvi,r_half_ndvi,q_half_ndvi,ev_half_ndvi, ncol = 2, nrow = 2, 
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
half_ndvi_bound_loss05_var_grid_with_title <- half_ndvi_bound_loss05_var_grid +
  plot_annotation(
    title = "Concave Revenue Constraint", # Your desired title for this grid
    theme = theme(plot.title = element_text(size = 18, face = "bold", hjust = 0.5)) # Customize title appearance
  )

var_cs_rate_by_strata_low_ndvi[var_cs_rate_by_strata_low_ndvi$step>1,] %>% 
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

combined_grids_with_outer_labels <- ggarrange(sq_avg_bound_loss05_var_grid_with_title, half_ndvi_bound_loss05_var_grid_with_title,
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
half_ndvi_bound_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                       ncol = 1, nrow = 2,
                                       heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))



#### Combine plots ######

mean_r_rate_by_strata <- bind_rows(
  mutate(mean_r_rate_by_strata_low_ndvi, source = "Low NDVI - Low Stratum"),
  mutate(mean_r_rate_by_strata_sq, source = "Status Quo NDVI")
)

mean_cs_rate_by_strata <- bind_rows(
  mutate(mean_cs_rate_by_strata_low_ndvi, source = "Low NDVI - Low Stratum"),
  mutate(mean_cs_rate_by_strata_sq, source = "Status Quo NDVI")
)

mean_q_rate_by_strata <- bind_rows(
  mutate(mean_q_rate_by_strata_low_ndvi, source = "Low NDVI - Low Stratum"),
  mutate(mean_q_rate_by_strata_sq, source = "Status Quo NDVI")
)

mean_ev_rate_by_strata <- bind_rows(
  mutate(mean_ev_rate_by_strata_low_ndvi, source = "Low NDVI - Low Stratum"),
  mutate(mean_ev_rate_by_strata_sq, source = "Status Quo NDVI")
)

mean_ev_by_strata <- bind_rows(
  mutate(mean_ev_by_strata_low_ndvi, source = "Low NDVI - Low Stratum"),
  mutate(mean_ev_by_strata_sq, source = "Status Quo NDVI")
)

var_r_rate_by_strata <- bind_rows(
  mutate(var_r_rate_by_strata_low_ndvi, source = "Low NDVI - Low Stratum"),
  mutate(var_r_rate_by_strata_sq, source = "Status Quo NDVI")
)

var_cs_rate_by_strata <- bind_rows(
  mutate(var_cs_rate_by_strata_low_ndvi, source = "Low NDVI - Low Stratum"),
  mutate(var_cs_rate_by_strata_sq, source = "Status Quo NDVI")
)

var_q_rate_by_strata <- bind_rows(
  mutate(var_q_rate_by_strata_low_ndvi, source = "Low NDVI - Low Stratum"),
  mutate(var_q_rate_by_strata_sq, source = "Status Quo NDVI")
)

var_ev_rate_by_strata <- bind_rows(
  mutate(var_ev_rate_by_strata_low_ndvi, source = "Low NDVI - Low Stratum"),
  mutate(var_ev_rate_by_strata_sq, source = "Status Quo NDVI")
)

var_ev_by_strata <- bind_rows(
  mutate(var_ev_by_strata_low_ndvi, source = "Low NDVI - Low Stratum"),
  mutate(var_ev_by_strata_sq, source = "Status Quo NDVI")
)


#### 2. Define the Plotting Function ######
create_scenario_plot <- function(data, y_var, y_lab, plot_type, stratum, add_hline = TRUE) {
  
  # Filter the data for the specified income stratum
  plot_data <- data %>%
    filter(income_strata == stratum)
  
  # Get the specific color for the stratum
  stratum_color <- custom_colors[stratum]
  
  if (plot_type == "mean") {
    # Generate the plot for the 'mean' type
    p = ggplot(plot_data, aes(x = step, y = .data[[y_var]], linetype = source)) +
      geom_line(size = 1.5, color = stratum_color) +
      geom_point(size = 4, stroke = 1.2, shape = 21, fill = "white", color = stratum_color) +
      scale_linetype_manual(values = c("Status Quo NDVI" = "solid", "Low NDVI - Low Stratum" = "dotted")) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
      labs(x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"), y = y_lab, linetype = "Scenario") +
      academic_theme +
      guides(linetype = guide_legend(keywidth = unit(1.2, "cm")))
  } else {
    # Generate the plot for the alternate 'variance' type
    p = ggplot(plot_data, aes(x = step, y = .data[[y_var]], linetype = source)) +
      geom_line(size = 1.5, color = stratum_color) +
      geom_point(size = 4, stroke = 1.2, shape = 21, fill = "white", color = stratum_color) +
      scale_linetype_manual(values = c("Status Quo NDVI" = "solid", "Low NDVI - Low Stratum" = "dotted")) +
      geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
      labs(x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"), y = y_lab, linetype = "Scenario") +
      academic_theme +
      guides(linetype = guide_legend(keywidth = unit(1.2, "cm")))
  }
  
  # Conditionally add the horizontal line
  if (add_hline) {
    p <- p + geom_hline(yintercept = 0, linetype = "dashed", color = "black")
  }
  
  return(p)
}

#### 3. Create the Individual Plots ######

# Create the plot for PS % for the "0~6k" stratum
ps_mean <- create_scenario_plot(
  data = mean_r_rate_by_strata, 
  y_var = "r_change_rate", 
  y_lab = "PS %",
  plot_type = "mean",
  stratum = "0~6k"
)

# Create the plot for PS % for the "0~6k" stratum
cs_mean <- create_scenario_plot(
  data = mean_cs_rate_by_strata, 
  y_var = "cs_change_rate", 
  y_lab = "CS %",
  plot_type = "mean",
  stratum = "0~6k"
)

# Create the plot for Q % for the "0~6k" stratum
q_mean <- create_scenario_plot(
  data = mean_q_rate_by_strata, 
  y_var = "q_change_rate", 
  y_lab = "Q %",
  plot_type = "mean",
  stratum = "0~6k"
)

# Create the plot for EV/I % for the "0~6k" stratum (without the hline)
ev_mean <- create_scenario_plot(
  data = mean_ev_rate_by_strata, 
  y_var = "ev_perct", 
  y_lab = "EV/I %",
  plot_type = "mean",
  stratum = "0~6k",
  add_hline = T # Set this to FALSE to remove the line
)


#### 4. Arrange the Plots into a Single Figure ######

# Extract the legend from one of the plots
shared_legend <- get_legend(ps_mean)

# Arrange the three plots vertically and add the shared legend
final_plot_grid_mean <- ggarrange(
  ps_mean + theme(legend.position = "none"),
  q_mean + theme(legend.position = "none"),
  ev_mean + theme(legend.position = "none"),
  shared_legend,
  ncol = 2, # Arrange in two columns
  nrow = 2,  # Arrange in two rows
  labels = c("PS", "Q", "EV", ""),
  font.label = list(size = 15, face = "bold")
)

# Create the plot for PS % for the "0~6k" stratum
ps_var <- create_scenario_plot(
  data = var_r_rate_by_strata, 
  y_var = "r_change_rate", 
  y_lab = "PS %",
  plot_type = "var",
  stratum = "0~6k"
)

# Create the plot for Q % for the "0~6k" stratum
q_var <- create_scenario_plot(
  data = var_q_rate_by_strata, 
  y_var = "q_change_rate", 
  y_lab = "Q %",
  plot_type = "var",
  stratum = "0~6k"
)

# Create the plot for EV/I % for the "0~6k" stratum (without the hline)
ev_var <- create_scenario_plot(
  data = var_ev_rate_by_strata, 
  y_var = "ev_perct", 
  y_lab = "EV/I %",
  plot_type = "var",
  stratum = "0~6k",
  add_hline = T # Set this to FALSE to remove the line
)

#### 4. Arrange the Plots into a Single Figure ######

# Extract the legend from one of the plots
shared_legend <- get_legend(ps_var)

# Arrange the three plots vertically and add the shared legend
final_plot_grid_var <- ggarrange(
  ps_var + theme(legend.position = "none"),
  q_var + theme(legend.position = "none"),
  ev_var + theme(legend.position = "none"),
  shared_legend,
  ncol = 2, # Arrange in two columns
  nrow = 2,  # Arrange in two rows
  labels = c("PS", "Q", "EV", ""),
  font.label = list(size = 15, face = "bold")
)



create_faceted_scenario_plot <- function(data, y_var, y_lab, plot_type, 
                                            color_map = custom_colors, shape_map = custom_shapes, add_hline = TRUE) {
  
  # Base ggplot object with all aesthetics mapped
  base_plot <- ggplot(data, aes(x = step, y = .data[[y_var]], 
                                color = income_strata, 
                                shape = income_strata,
                                linetype = source)) # Removed fill from base aes
  
  # Plot-type specific layers (e.g., labels, vlines)
  if (plot_type == "mean") {
    p <- base_plot +
      geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
      labs(x = expression(zeta[1] ~ "=" ~ Delta* " E[Z] (Inch)"), y = y_lab)
    
  } else { # "variance" plot type
    p <- base_plot +
      geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
      labs(x = expression(zeta[2] ~ "=" ~ " Sd[Z] Ratio"), y = y_lab)
  }
  
  # Add layers common to both plot types
  p <- p +
    geom_line(size = 1.2) +
    # Updated geom_point for smaller, hollow dots with border
    geom_point(size = 3, stroke = 0.8, fill = "white") + 
    facet_wrap(~ income_strata, scales = "free_y") + # Use free y-scales
    scale_linetype_manual(values = c("Status Quo NDVI" = "solid", "Low NDVI - Low Stratum" = "dotted")) +
    scale_color_manual(values = color_map) +  # Use custom colors for lines and point borders
    scale_shape_manual(values = shape_map) +  # Use custom shapes
    academic_theme +
    # Tidy up the legend titles
    labs(linetype = "Scenario", color = "Income Stratum", 
         shape = "Income Stratum") + # Removed fill from labs
    guides(linetype = guide_legend(keywidth = unit(1.2, "cm")))
  
  # Conditionally add the horizontal line
  if (add_hline) {
    p <- p + geom_hline(yintercept = 0, linetype = "dashed", color = "black")
  }
  
  return(p)
}
  
mean_plot_faceted <- create_faceted_scenario_plot(
  data = mean_ev_rate_by_strata, 
  y_var = "ev_perct", 
  y_lab = "EV/I %",
  plot_type = "mean",
)

var_plot_faceted <- create_faceted_scenario_plot(
  data = var_ev_rate_by_strata, 
  y_var = "ev_perct", 
  y_lab = "EV/I %",
  plot_type = "var",
)



##### Why CS ia lower? with low q and low r #####
sq_ndvi_avg_bound_loss05_mean_cs_steps_rate_check = sq_ndvi_avg_bound_loss05_mean_cs_steps_rate %>% select(`0.25`, rate_0.25,income, cs_0, cs_0_bill_ym, cs_0_income_strata
                                                                                                                     , cs_0_income_strata_bill_ym, cs_0_quantity_strata)
half_ndvi_low_avg_bound_loss05_mean_cs_steps_rate_check= half_ndvi_low_avg_bound_loss05_mean_cs_steps_rate %>% select(`0.25`, rate_0.25,income, cs_0, cs_0_bill_ym, cs_0_income_strata
                                                                                                               , cs_0_income_strata_bill_ym, cs_0_quantity_strata)

cs_compare = sq_ndvi_avg_bound_loss05_mean_cs_steps_rate_check %>% rename(sq_0.25 = `0.25`,
                                                                               sq_rate_0.25 = rate_0.25 )

cs_compare = cbind(cs_compare, half_ndvi_low_avg_bound_loss05_mean_cs_steps_rate_check %>% select(`0.25`, rate_0.25) %>% rename(
  half_ndvi_0.25 = `0.25`,
  half_ndvi_rate_0.25 = rate_0.25 
))

sq_ndvi_avg_bound_loss05_mean_q_steps_rate_check = sq_ndvi_avg_bound_loss05_mean_q_steps_rate %>% select(`0.25`, rate_0.25,income, q_0, q_0_bill_ym, q_0_income_strata
                                                                                                                     , q_0_income_strata_bill_ym, q_0_quantity_strata)
half_ndvi_low_avg_bound_loss05_mean_q_steps_rate_check= half_ndvi_low_avg_bound_loss05_mean_q_steps_rate %>% select(`0.25`, rate_0.25,income, q_0, q_0_bill_ym, q_0_income_strata
                                                                                                                      , q_0_income_strata_bill_ym, q_0_quantity_strata)

q_compare = sq_ndvi_avg_bound_loss05_mean_q_steps_rate_check %>% rename(sq_0.25 = `0.25`,
                                                                               sq_rate_0.25 = rate_0.25 )

q_compare = cbind(q_compare, half_ndvi_low_avg_bound_loss05_mean_q_steps_rate_check %>% select(`0.25`, rate_0.25) %>% rename(
  half_ndvi_0.25 = `0.25`,
  half_ndvi_rate_0.25 = rate_0.25 
))

sq_ndvi_avg_bound_loss05_mean_r_steps_rate_check = sq_ndvi_avg_bound_loss05_mean_r_steps_rate %>% select(`0.25`, rate_0.25,income, r_0, r_0_bill_ym, r_0_income_strata
                                                                                                                   , r_0_income_strata_bill_ym, r_0_quantity_strata)
half_ndvi_low_avg_bound_loss05_mean_r_steps_rate_check= half_ndvi_low_avg_bound_loss05_mean_r_steps_rate %>% select(`0.25`, rate_0.25,income, r_0, r_0_bill_ym, r_0_income_strata
                                                                                                                    , r_0_income_strata_bill_ym, r_0_quantity_strata)

r_compare = sq_ndvi_avg_bound_loss05_mean_r_steps_rate_check %>% rename(sq_0.25 = `0.25`,
                                                                             sq_rate_0.25 = rate_0.25 )

r_compare = cbind(r_compare, half_ndvi_low_avg_bound_loss05_mean_r_steps_rate_check %>% select(`0.25`, rate_0.25) %>% rename(
  half_ndvi_0.25 = `0.25`,
  half_ndvi_rate_0.25 = rate_0.25 
))


sq_ndvi_avg_bound_loss05_mean_ev_steps_rate_check = sq_ndvi_avg_bound_loss05_mean_ev_steps_rate %>% select(`0.25`, rate_0.25,income)
half_ndvi_low_avg_bound_loss05_mean_ev_steps_rate_check= half_ndvi_low_avg_bound_loss05_mean_ev_steps_rate %>% select(`0.25`, rate_0.25,income)

ev_compare = sq_ndvi_avg_bound_loss05_mean_ev_steps_rate_check %>% rename(sq_0.25 = `0.25`,
                                                                             sq_rate_0.25 = rate_0.25 )

ev_compare = cbind(ev_compare, half_ndvi_low_avg_bound_loss05_mean_ev_steps_rate_check %>% select(`0.25`, rate_0.25) %>% rename(
  half_ndvi_0.25 = `0.25`,
  half_ndvi_rate_0.25 = rate_0.25 
))

cs_compare =cs_compare %>% mutate(
  diff_rate = half_ndvi_rate_0.25 - sq_rate_0.25,
  diff_cs = half_ndvi_0.25 - sq_0.25,
  income_strata = case_when(
    income < 6000 ~ "0~6k",
    income >= 6000 & income < 20000 ~ "6k~20k",
    income >= 20000 & income < 45000 ~ "20k~45k",
    income >= 45000 & income < 100000 ~ "45k~100k",
    income >= 100000 ~ ">100k"
  )) 

q_compare$diff_q = q_compare$half_ndvi_0.25 - q_compare$sq_0.25
q_compare$diff_q_rate = q_compare$half_ndvi_rate_0.25 - q_compare$sq_rate_0.25

r_compare$diff_r = r_compare$half_ndvi_0.25 - r_compare$sq_0.25
r_compare$diff_r_rate = r_compare$half_ndvi_rate_0.25 - r_compare$sq_rate_0.25

cs_compare$diff_q = q_compare$diff_q
cs_compare$diff_q_rate = q_compare$diff_q_rate

cs_compare$diff_r = r_compare$diff_r
cs_compare$diff_r_rate = r_compare$diff_r_rate

cs_compare_low = cs_compare %>% filter ( income_strata == "0~6k")

summary(cs_compare_low$diff_rate)

cs_compare_low_low = cs_compare_low %>% filter ( diff_rate < "-11")
