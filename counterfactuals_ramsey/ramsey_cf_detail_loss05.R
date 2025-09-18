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

# This single piped command creates the final, correctly ordered data frame
household_avg <- demand_key %>%
  group_by(prem_id) %>%
  summarise(
    avg_q = mean(q_0, na.rm = TRUE),
    income = first(income)
  ) %>%
  ungroup() %>%
  mutate(income_strata = case_when(
    income <= 6000 ~ "0~6k",
    income > 6000 & income <= 20000 ~ "6k~20k",
    income > 20000 & income <= 45000 ~ "20k~45k",
    income > 45000 & income <= 100000 ~ "45k~100k",
    TRUE ~ ">100k"
  )) %>%
  mutate(income_strata = factor(income_strata, levels = income_order))


# --- 3. Create a data frame for the labels ---
# We set x_pos to -Inf to anchor the label to the far-left edge.
label_df <- data.frame(
  income_strata = factor(c("0~6k", "45k~100k"), levels = income_order),
  x_pos = -Inf,       # Anchor to the far left of each panel
  y_pos = 20,         # Align perfectly with the line
  label = "20"
)


# --- 4. Generate the Plot with Margin Labels ---
ggplot(household_avg %>% filter(avg_q <= 250 & income < 2000000), 
       aes(x = income/1000, y = avg_q, color = income_strata)) +
  geom_point(size = 1, alpha = 0.3) +
  scale_color_manual(values = custom_colors) +
  geom_hline(yintercept = 20, color = "red", linetype = "dashed", size = 1) +
  
  # Use geom_text with new coordinates and justification
  geom_text(
    data = label_df,
    aes(x = x_pos, y = y_pos, label = label),
    color = "red",
    hjust = 1.5,  # Horizontally justify to push it *right* from the far-left edge
    inherit.aes = FALSE
  ) +
  
  # IMPORTANT: Turn clipping off to allow drawing outside the panel
  coord_cartesian(ylim = c(0, 250), clip = "off") +
  
  facet_wrap(~ income_strata, scales = "free_x") +
  labs(
    title = "",
    x = "Household Monthly Income (k$)",
    y = "Quantity (kGal)"
  ) +
  academic_theme + 
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none",
    # Add a left margin to the whole plot to make space for the label
    plot.margin = margin(5.5, 5.5, 5.5, 12, "pt") 
  )

# --- 3. Create a label data frame for ONLY the desired facets ---
label_df <- data.frame(
  income_strata = factor("0~6k", levels = income_order), # Only include the "0~6k" group
  x_pos = -Inf,
  y_pos = 20,
  label = "20"
)


# --- 4. Generate the Plot with Filtered Data ---
household_avg %>%
  # --- THIS IS THE KEY NEW LINE ---
  filter(income_strata %in% c("0~6k", ">100k") & avg_q <= 250 & income < 2000000) %>%
  
  ggplot(aes(x = income / 1000, y = avg_q, color = income_strata)) +
  geom_point(size = 1, alpha = 0.3) +
  scale_color_manual(values = custom_colors) +
  geom_hline(yintercept = 20, color = "red", linetype = "dashed", size = 1) +
  geom_text(
    data = label_df,
    aes(x = x_pos, y = y_pos, label = label),
    color = "red",
    hjust = 1.5,
    inherit.aes = FALSE
  ) +
  coord_cartesian(ylim = c(0, 250), clip = "off") +
  facet_wrap(~income_strata, scales = "free_x") +
  labs(
    title = "",
    x = "Household Monthly Income (k$)",
    y = "Quantity (kGal)"
  ) +
  academic_theme +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "none",
    plot.margin = margin(5.5, 5.5, 5.5, 12, "pt")
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





##### current_info_gamma045_bound_loss05_mean ########

current_info_gamma045_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/current_info_gamma045_bound_loss05_mean_cs_steps.csv")

colnames(current_info_gamma045_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(current_info_gamma045_bound_loss05_mean_cs_steps))*0.05-0.25)

current_info_gamma045_bound_loss05_mean_cs_steps_rate <-current_info_gamma045_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma045_bound_loss05_mean_cs_steps)

current_info_gamma045_bound_loss05_mean_q_steps = read_csv("cs_detail_results/current_info_gamma045_bound_loss05_mean_q_steps.csv")

colnames(current_info_gamma045_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(current_info_gamma045_bound_loss05_mean_q_steps))*0.05-0.25)

current_info_gamma045_bound_loss05_mean_q_steps_rate <-current_info_gamma045_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma045_bound_loss05_mean_q_steps)

current_info_gamma045_bound_loss05_mean_r_steps = read_csv("cs_detail_results/current_info_gamma045_bound_loss05_mean_r_steps.csv")

colnames(current_info_gamma045_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(current_info_gamma045_bound_loss05_mean_r_steps))*0.05-0.25)

current_info_gamma045_bound_loss05_mean_r_steps_rate <-current_info_gamma045_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma045_bound_loss05_mean_r_steps)

current_info_gamma045_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/current_info_gamma045_bound_loss05_mean_ev_steps.csv")

colnames(current_info_gamma045_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(current_info_gamma045_bound_loss05_mean_ev_steps))*0.05-0.25)

current_info_gamma045_bound_loss05_mean_ev_steps_rate <-current_info_gamma045_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma045_bound_loss05_mean_ev_steps)

current_info_gamma045_bound_loss05_mean_cs_steps_rate  = cbind(current_info_gamma045_bound_loss05_mean_cs_steps_rate , demand_key)
current_info_gamma045_bound_loss05_mean_q_steps_rate = cbind(current_info_gamma045_bound_loss05_mean_q_steps_rate, demand_key)
current_info_gamma045_bound_loss05_mean_r_steps_rate = cbind(current_info_gamma045_bound_loss05_mean_r_steps_rate, demand_key)
current_info_gamma045_bound_loss05_mean_ev_steps_rate = cbind(current_info_gamma045_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_gamma045_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_gamma045_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_gamma045_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- current_info_gamma045_bound_loss05_mean_ev_steps_rate %>%
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

cs_gamma045 = ggplot(mean_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma045 = ggplot(mean_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma045 = ggplot(mean_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma045 = ggplot(mean_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

gamma045_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma045_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

gamma045_bound_loss05_mean_grid = ggarrange(cs_gamma045,r_gamma045,q_gamma045,ev_gamma045, ncol = 2, nrow = 2, 
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
gamma045_bound_loss05_mean_grid_with_title <- gamma045_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_mean_grid_with_title, gamma045_bound_loss05_mean_grid_with_title,
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
avg_gamma045_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                     ncol = 1, nrow = 2,
                                     heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### current_info_gamma045_bound_loss05_var ########

current_info_gamma045_bound_loss05_var_cs_steps = read_csv("cs_detail_results/current_info_gamma045_bound_loss05_var_cs_steps.csv")

colnames(current_info_gamma045_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(current_info_gamma045_bound_loss05_var_cs_steps))/20+0.75)

current_info_gamma045_bound_loss05_var_cs_steps_rate <-current_info_gamma045_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma045_bound_loss05_var_cs_steps)

current_info_gamma045_bound_loss05_var_q_steps = read_csv("cs_detail_results/current_info_gamma045_bound_loss05_var_q_steps.csv")

colnames(current_info_gamma045_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(current_info_gamma045_bound_loss05_var_q_steps))/20+0.75)

current_info_gamma045_bound_loss05_var_q_steps_rate <-current_info_gamma045_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma045_bound_loss05_var_q_steps)

current_info_gamma045_bound_loss05_var_r_steps = read_csv("cs_detail_results/current_info_gamma045_bound_loss05_var_r_steps.csv")

colnames(current_info_gamma045_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(current_info_gamma045_bound_loss05_var_r_steps))/20+0.75)

current_info_gamma045_bound_loss05_var_r_steps_rate <-current_info_gamma045_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma045_bound_loss05_var_r_steps)

current_info_gamma045_bound_loss05_var_ev_steps = read_csv("cs_detail_results/current_info_gamma045_bound_loss05_var_ev_steps.csv")

colnames(current_info_gamma045_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(current_info_gamma045_bound_loss05_var_ev_steps))/20+0.75)

current_info_gamma045_bound_loss05_var_ev_steps_rate <-current_info_gamma045_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma045_bound_loss05_var_ev_steps)

current_info_gamma045_bound_loss05_var_cs_steps_rate  = cbind(current_info_gamma045_bound_loss05_var_cs_steps_rate , demand_key)
current_info_gamma045_bound_loss05_var_q_steps_rate = cbind(current_info_gamma045_bound_loss05_var_q_steps_rate, demand_key)
current_info_gamma045_bound_loss05_var_r_steps_rate = cbind(current_info_gamma045_bound_loss05_var_r_steps_rate, demand_key)
current_info_gamma045_bound_loss05_var_ev_steps_rate = cbind(current_info_gamma045_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_gamma045_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_gamma045_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_gamma045_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_gamma045_bound_loss05_var_ev_steps_rate %>%
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

cs_gamma045 = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma045 = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma045 = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma045 = ggplot(var_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

gamma045_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma045_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


gamma045_bound_loss05_var_grid = ggarrange(cs_gamma045,r_gamma045,q_gamma045,ev_gamma045, ncol = 2, nrow = 2, 
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
gamma045_bound_loss05_var_grid_with_title <- gamma045_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_var_grid_with_title, gamma045_bound_loss05_var_grid_with_title,
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
avg_gamma045_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
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



##### current_info_gamma035_bound_loss05_mean ########

current_info_gamma035_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/current_info_gamma035_bound_loss05_mean_cs_steps.csv")

colnames(current_info_gamma035_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(current_info_gamma035_bound_loss05_mean_cs_steps))*0.05-0.25)

current_info_gamma035_bound_loss05_mean_cs_steps_rate <-current_info_gamma035_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma035_bound_loss05_mean_cs_steps)

current_info_gamma035_bound_loss05_mean_q_steps = read_csv("cs_detail_results/current_info_gamma035_bound_loss05_mean_q_steps.csv")

colnames(current_info_gamma035_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(current_info_gamma035_bound_loss05_mean_q_steps))*0.05-0.25)

current_info_gamma035_bound_loss05_mean_q_steps_rate <-current_info_gamma035_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma035_bound_loss05_mean_q_steps)

current_info_gamma035_bound_loss05_mean_r_steps = read_csv("cs_detail_results/current_info_gamma035_bound_loss05_mean_r_steps.csv")

colnames(current_info_gamma035_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(current_info_gamma035_bound_loss05_mean_r_steps))*0.05-0.25)

current_info_gamma035_bound_loss05_mean_r_steps_rate <-current_info_gamma035_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma035_bound_loss05_mean_r_steps)

current_info_gamma035_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/current_info_gamma035_bound_loss05_mean_ev_steps.csv")

colnames(current_info_gamma035_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(current_info_gamma035_bound_loss05_mean_ev_steps))*0.05-0.25)

current_info_gamma035_bound_loss05_mean_ev_steps_rate <-current_info_gamma035_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma035_bound_loss05_mean_ev_steps)

current_info_gamma035_bound_loss05_mean_cs_steps_rate  = cbind(current_info_gamma035_bound_loss05_mean_cs_steps_rate , demand_key)
current_info_gamma035_bound_loss05_mean_q_steps_rate = cbind(current_info_gamma035_bound_loss05_mean_q_steps_rate, demand_key)
current_info_gamma035_bound_loss05_mean_r_steps_rate = cbind(current_info_gamma035_bound_loss05_mean_r_steps_rate, demand_key)
current_info_gamma035_bound_loss05_mean_ev_steps_rate = cbind(current_info_gamma035_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_gamma035_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_gamma035_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_gamma035_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- current_info_gamma035_bound_loss05_mean_ev_steps_rate %>%
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

cs_gamma035 = ggplot(mean_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma035 = ggplot(mean_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma035 = ggplot(mean_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma035 = ggplot(mean_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

gamma035_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma035_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

gamma035_bound_loss05_mean_grid = ggarrange(cs_gamma035,r_gamma035,q_gamma035,ev_gamma035, ncol = 2, nrow = 2, 
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
gamma035_bound_loss05_mean_grid_with_title <- gamma035_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_mean_grid_with_title, gamma035_bound_loss05_mean_grid_with_title,
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
avg_gamma035_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                     ncol = 1, nrow = 2,
                                     heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### current_info_gamma035_bound_loss05_var ########

current_info_gamma035_bound_loss05_var_cs_steps = read_csv("cs_detail_results/current_info_gamma035_bound_loss05_var_cs_steps.csv")

colnames(current_info_gamma035_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(current_info_gamma035_bound_loss05_var_cs_steps))/20+0.75)

current_info_gamma035_bound_loss05_var_cs_steps_rate <-current_info_gamma035_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma035_bound_loss05_var_cs_steps)

current_info_gamma035_bound_loss05_var_q_steps = read_csv("cs_detail_results/current_info_gamma035_bound_loss05_var_q_steps.csv")

colnames(current_info_gamma035_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(current_info_gamma035_bound_loss05_var_q_steps))/20+0.75)

current_info_gamma035_bound_loss05_var_q_steps_rate <-current_info_gamma035_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma035_bound_loss05_var_q_steps)

current_info_gamma035_bound_loss05_var_r_steps = read_csv("cs_detail_results/current_info_gamma035_bound_loss05_var_r_steps.csv")

colnames(current_info_gamma035_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(current_info_gamma035_bound_loss05_var_r_steps))/20+0.75)

current_info_gamma035_bound_loss05_var_r_steps_rate <-current_info_gamma035_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma035_bound_loss05_var_r_steps)

current_info_gamma035_bound_loss05_var_ev_steps = read_csv("cs_detail_results/current_info_gamma035_bound_loss05_var_ev_steps.csv")

colnames(current_info_gamma035_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(current_info_gamma035_bound_loss05_var_ev_steps))/20+0.75)

current_info_gamma035_bound_loss05_var_ev_steps_rate <-current_info_gamma035_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma035_bound_loss05_var_ev_steps)

current_info_gamma035_bound_loss05_var_cs_steps_rate  = cbind(current_info_gamma035_bound_loss05_var_cs_steps_rate , demand_key)
current_info_gamma035_bound_loss05_var_q_steps_rate = cbind(current_info_gamma035_bound_loss05_var_q_steps_rate, demand_key)
current_info_gamma035_bound_loss05_var_r_steps_rate = cbind(current_info_gamma035_bound_loss05_var_r_steps_rate, demand_key)
current_info_gamma035_bound_loss05_var_ev_steps_rate = cbind(current_info_gamma035_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_gamma035_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_gamma035_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_gamma035_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_gamma035_bound_loss05_var_ev_steps_rate %>%
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

cs_gamma035 = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma035 = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma035 = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma035 = ggplot(var_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

gamma035_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma035_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


gamma035_bound_loss05_var_grid = ggarrange(cs_gamma035,r_gamma035,q_gamma035,ev_gamma035, ncol = 2, nrow = 2, 
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
gamma035_bound_loss05_var_grid_with_title <- gamma035_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_var_grid_with_title, gamma035_bound_loss05_var_grid_with_title,
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
avg_gamma035_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
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



##### current_info_gamma015_bound_loss05_mean ########

current_info_gamma015_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/current_info_gamma015_bound_loss05_mean_cs_steps.csv")

colnames(current_info_gamma015_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(current_info_gamma015_bound_loss05_mean_cs_steps))*0.05-0.25)

current_info_gamma015_bound_loss05_mean_cs_steps_rate <-current_info_gamma015_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma015_bound_loss05_mean_cs_steps)

current_info_gamma015_bound_loss05_mean_q_steps = read_csv("cs_detail_results/current_info_gamma015_bound_loss05_mean_q_steps.csv")

colnames(current_info_gamma015_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(current_info_gamma015_bound_loss05_mean_q_steps))*0.05-0.25)

current_info_gamma015_bound_loss05_mean_q_steps_rate <-current_info_gamma015_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma015_bound_loss05_mean_q_steps)

current_info_gamma015_bound_loss05_mean_r_steps = read_csv("cs_detail_results/current_info_gamma015_bound_loss05_mean_r_steps.csv")

colnames(current_info_gamma015_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(current_info_gamma015_bound_loss05_mean_r_steps))*0.05-0.25)

current_info_gamma015_bound_loss05_mean_r_steps_rate <-current_info_gamma015_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma015_bound_loss05_mean_r_steps)

current_info_gamma015_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/current_info_gamma015_bound_loss05_mean_ev_steps.csv")

colnames(current_info_gamma015_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(current_info_gamma015_bound_loss05_mean_ev_steps))*0.05-0.25)

current_info_gamma015_bound_loss05_mean_ev_steps_rate <-current_info_gamma015_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma015_bound_loss05_mean_ev_steps)

current_info_gamma015_bound_loss05_mean_cs_steps_rate  = cbind(current_info_gamma015_bound_loss05_mean_cs_steps_rate , demand_key)
current_info_gamma015_bound_loss05_mean_q_steps_rate = cbind(current_info_gamma015_bound_loss05_mean_q_steps_rate, demand_key)
current_info_gamma015_bound_loss05_mean_r_steps_rate = cbind(current_info_gamma015_bound_loss05_mean_r_steps_rate, demand_key)
current_info_gamma015_bound_loss05_mean_ev_steps_rate = cbind(current_info_gamma015_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_gamma015_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_gamma015_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_gamma015_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- current_info_gamma015_bound_loss05_mean_ev_steps_rate %>%
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

cs_gamma015 = ggplot(mean_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma015 = ggplot(mean_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma015 = ggplot(mean_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma015 = ggplot(mean_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

gamma015_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma015_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

gamma015_bound_loss05_mean_grid = ggarrange(cs_gamma015,r_gamma015,q_gamma015,ev_gamma015, ncol = 2, nrow = 2, 
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
gamma015_bound_loss05_mean_grid_with_title <- gamma015_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_mean_grid_with_title, gamma015_bound_loss05_mean_grid_with_title,
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
avg_gamma015_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                     ncol = 1, nrow = 2,
                                     heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### current_info_gamma015_bound_loss05_var ########

current_info_gamma015_bound_loss05_var_cs_steps = read_csv("cs_detail_results/current_info_gamma015_bound_loss05_var_cs_steps.csv")

colnames(current_info_gamma015_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(current_info_gamma015_bound_loss05_var_cs_steps))/20+0.75)

current_info_gamma015_bound_loss05_var_cs_steps_rate <-current_info_gamma015_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma015_bound_loss05_var_cs_steps)

current_info_gamma015_bound_loss05_var_q_steps = read_csv("cs_detail_results/current_info_gamma015_bound_loss05_var_q_steps.csv")

colnames(current_info_gamma015_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(current_info_gamma015_bound_loss05_var_q_steps))/20+0.75)

current_info_gamma015_bound_loss05_var_q_steps_rate <-current_info_gamma015_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma015_bound_loss05_var_q_steps)

current_info_gamma015_bound_loss05_var_r_steps = read_csv("cs_detail_results/current_info_gamma015_bound_loss05_var_r_steps.csv")

colnames(current_info_gamma015_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(current_info_gamma015_bound_loss05_var_r_steps))/20+0.75)

current_info_gamma015_bound_loss05_var_r_steps_rate <-current_info_gamma015_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma015_bound_loss05_var_r_steps)

current_info_gamma015_bound_loss05_var_ev_steps = read_csv("cs_detail_results/current_info_gamma015_bound_loss05_var_ev_steps.csv")

colnames(current_info_gamma015_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(current_info_gamma015_bound_loss05_var_ev_steps))/20+0.75)

current_info_gamma015_bound_loss05_var_ev_steps_rate <-current_info_gamma015_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma015_bound_loss05_var_ev_steps)

current_info_gamma015_bound_loss05_var_cs_steps_rate  = cbind(current_info_gamma015_bound_loss05_var_cs_steps_rate , demand_key)
current_info_gamma015_bound_loss05_var_q_steps_rate = cbind(current_info_gamma015_bound_loss05_var_q_steps_rate, demand_key)
current_info_gamma015_bound_loss05_var_r_steps_rate = cbind(current_info_gamma015_bound_loss05_var_r_steps_rate, demand_key)
current_info_gamma015_bound_loss05_var_ev_steps_rate = cbind(current_info_gamma015_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_gamma015_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_gamma015_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_gamma015_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_gamma015_bound_loss05_var_ev_steps_rate %>%
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

cs_gamma015 = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma015 = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma015 = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma015 = ggplot(var_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

gamma015_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma015_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


gamma015_bound_loss05_var_grid = ggarrange(cs_gamma015,r_gamma015,q_gamma015,ev_gamma015, ncol = 2, nrow = 2, 
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
gamma015_bound_loss05_var_grid_with_title <- gamma015_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_var_grid_with_title, gamma015_bound_loss05_var_grid_with_title,
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
avg_gamma015_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
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









##### current_info_gamma005_bound_loss05_mean ########

current_info_gamma005_bound_loss05_mean_cs_steps = read_csv("cs_detail_results/current_info_gamma005_bound_loss05_mean_cs_steps.csv")

colnames(current_info_gamma005_bound_loss05_mean_cs_steps) = as.character(as.numeric(colnames(current_info_gamma005_bound_loss05_mean_cs_steps))*0.05-0.25)

current_info_gamma005_bound_loss05_mean_cs_steps_rate <-current_info_gamma005_bound_loss05_mean_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma005_bound_loss05_mean_cs_steps)

current_info_gamma005_bound_loss05_mean_q_steps = read_csv("cs_detail_results/current_info_gamma005_bound_loss05_mean_q_steps.csv")

colnames(current_info_gamma005_bound_loss05_mean_q_steps) = as.character(as.numeric(colnames(current_info_gamma005_bound_loss05_mean_q_steps))*0.05-0.25)

current_info_gamma005_bound_loss05_mean_q_steps_rate <-current_info_gamma005_bound_loss05_mean_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma005_bound_loss05_mean_q_steps)

current_info_gamma005_bound_loss05_mean_r_steps = read_csv("cs_detail_results/current_info_gamma005_bound_loss05_mean_r_steps.csv")

colnames(current_info_gamma005_bound_loss05_mean_r_steps) = as.character(as.numeric(colnames(current_info_gamma005_bound_loss05_mean_r_steps))*0.05-0.25)

current_info_gamma005_bound_loss05_mean_r_steps_rate <-current_info_gamma005_bound_loss05_mean_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma005_bound_loss05_mean_r_steps)

current_info_gamma005_bound_loss05_mean_ev_steps = read_csv("cs_detail_results/current_info_gamma005_bound_loss05_mean_ev_steps.csv")

colnames(current_info_gamma005_bound_loss05_mean_ev_steps) = as.character(as.numeric(colnames(current_info_gamma005_bound_loss05_mean_ev_steps))*0.05-0.25)

current_info_gamma005_bound_loss05_mean_ev_steps_rate <-current_info_gamma005_bound_loss05_mean_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma005_bound_loss05_mean_ev_steps)

current_info_gamma005_bound_loss05_mean_cs_steps_rate  = cbind(current_info_gamma005_bound_loss05_mean_cs_steps_rate , demand_key)
current_info_gamma005_bound_loss05_mean_q_steps_rate = cbind(current_info_gamma005_bound_loss05_mean_q_steps_rate, demand_key)
current_info_gamma005_bound_loss05_mean_r_steps_rate = cbind(current_info_gamma005_bound_loss05_mean_r_steps_rate, demand_key)
current_info_gamma005_bound_loss05_mean_ev_steps_rate = cbind(current_info_gamma005_bound_loss05_mean_ev_steps_rate, demand_key)

mean_cs_rate_by_bill_ym <- current_info_gamma005_bound_loss05_mean_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_q_rate_by_bill_ym <- current_info_gamma005_bound_loss05_mean_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_r_rate_by_bill_ym <- current_info_gamma005_bound_loss05_mean_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

mean_ev_rate_by_bill_ym <- current_info_gamma005_bound_loss05_mean_ev_steps_rate %>%
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

cs_gamma005 = ggplot(mean_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma005 = ggplot(mean_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma005 = ggplot(mean_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma005 = ggplot(mean_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

gamma005_bound_loss05_mean_ev = mean_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma005_bound_loss05_mean_cs = mean_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))

gamma005_bound_loss05_mean_grid = ggarrange(cs_gamma005,r_gamma005,q_gamma005,ev_gamma005, ncol = 2, nrow = 2, 
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
gamma005_bound_loss05_mean_grid_with_title <- gamma005_bound_loss05_mean_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_mean_grid_with_title, gamma005_bound_loss05_mean_grid_with_title,
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
avg_gamma005_mean_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                     ncol = 1, nrow = 2,
                                     heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))




##### current_info_gamma005_bound_loss05_var ########

current_info_gamma005_bound_loss05_var_cs_steps = read_csv("cs_detail_results/current_info_gamma005_bound_loss05_var_cs_steps.csv")

colnames(current_info_gamma005_bound_loss05_var_cs_steps) = as.character(as.numeric(colnames(current_info_gamma005_bound_loss05_var_cs_steps))/20+0.75)

current_info_gamma005_bound_loss05_var_cs_steps_rate <-current_info_gamma005_bound_loss05_var_cs_steps %>%
  mutate(across(everything(), ~ (. - demand_key$cs_0_income_strata) / abs(demand_key$cs_0_income_strata)*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$cs_0) / abs(demand_key$cs_0)*100, .names = "rate_{.col}"))

rm(current_info_gamma005_bound_loss05_var_cs_steps)

current_info_gamma005_bound_loss05_var_q_steps = read_csv("cs_detail_results/current_info_gamma005_bound_loss05_var_q_steps.csv")

colnames(current_info_gamma005_bound_loss05_var_q_steps) = as.character(as.numeric(colnames(current_info_gamma005_bound_loss05_var_q_steps))/20+0.75)

current_info_gamma005_bound_loss05_var_q_steps_rate <-current_info_gamma005_bound_loss05_var_q_steps %>%
  mutate(across(everything(), ~ (. - demand_key$q_0_income_strata) / demand_key$q_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$q_0) / demand_key$q_0*100, .names = "rate_{.col}"))

rm(current_info_gamma005_bound_loss05_var_q_steps)

current_info_gamma005_bound_loss05_var_r_steps = read_csv("cs_detail_results/current_info_gamma005_bound_loss05_var_r_steps.csv")

colnames(current_info_gamma005_bound_loss05_var_r_steps) = as.character(as.numeric(colnames(current_info_gamma005_bound_loss05_var_r_steps))/20+0.75)

current_info_gamma005_bound_loss05_var_r_steps_rate <-current_info_gamma005_bound_loss05_var_r_steps %>%
  mutate(across(everything(), ~ (. - demand_key$r_0_income_strata) / demand_key$r_0_income_strata*100, .names = "rate_{.col}"))
#mutate(across(everything(), ~ (. - demand_key$r_0) / demand_key$r_0*100, .names = "rate_{.col}"))

rm(current_info_gamma005_bound_loss05_var_r_steps)

current_info_gamma005_bound_loss05_var_ev_steps = read_csv("cs_detail_results/current_info_gamma005_bound_loss05_var_ev_steps.csv")

colnames(current_info_gamma005_bound_loss05_var_ev_steps) = as.character(as.numeric(colnames(current_info_gamma005_bound_loss05_var_ev_steps))/20+0.75)

current_info_gamma005_bound_loss05_var_ev_steps_rate <-current_info_gamma005_bound_loss05_var_ev_steps %>%
  mutate(across(everything(), ~ (. ) / income * 100, .names = "rate_{.col}"))

rm(current_info_gamma005_bound_loss05_var_ev_steps)

current_info_gamma005_bound_loss05_var_cs_steps_rate  = cbind(current_info_gamma005_bound_loss05_var_cs_steps_rate , demand_key)
current_info_gamma005_bound_loss05_var_q_steps_rate = cbind(current_info_gamma005_bound_loss05_var_q_steps_rate, demand_key)
current_info_gamma005_bound_loss05_var_r_steps_rate = cbind(current_info_gamma005_bound_loss05_var_r_steps_rate, demand_key)
current_info_gamma005_bound_loss05_var_ev_steps_rate = cbind(current_info_gamma005_bound_loss05_var_ev_steps_rate, demand_key)

var_cs_rate_by_bill_ym <- current_info_gamma005_bound_loss05_var_cs_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_q_rate_by_bill_ym <- current_info_gamma005_bound_loss05_var_q_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_r_rate_by_bill_ym <- current_info_gamma005_bound_loss05_var_r_steps_rate %>%
  #group_by(quantity_strata, bill_ym) %>%
  group_by(income_strata, bill_ym) %>%
  #group_by(prem_id) %>%
  summarise(across(starts_with("rate_"), mean, na.rm = TRUE))

var_ev_rate_by_bill_ym <- current_info_gamma005_bound_loss05_var_ev_steps_rate %>%
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

cs_gamma005 = ggplot(var_cs_rate_by_strata , aes(x = step, y = cs_change_rate, 
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


r_gamma005 = ggplot(var_r_rate_by_strata , aes(x = step, y = r_change_rate, 
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

q_gamma005 = ggplot(var_q_rate_by_strata , aes(x = step, y = q_change_rate, 
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

ev_gamma005 = ggplot(var_ev_rate_by_strata , aes(x = step, y = ev_perct, 
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

gamma005_bound_loss05_var_ev = var_ev_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_ev_perct = mean(ev_perct),
            sd_ev_perct = sd(ev_perct),
            ev_five_quantile = quantile(ev_perct, 0.05))

gamma005_bound_loss05_var_cs = var_cs_rate_by_strata %>% 
  group_by(income_strata, status) %>%
  summarise(mean_cs_change_rate = mean(cs_change_rate),
            sd_cs_change_rate = sd(cs_change_rate),
            cs_five_quantile = quantile(cs_change_rate, 0.05))


gamma005_bound_loss05_var_grid = ggarrange(cs_gamma005,r_gamma005,q_gamma005,ev_gamma005, ncol = 2, nrow = 2, 
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
gamma005_bound_loss05_var_grid_with_title <- gamma005_bound_loss05_var_grid +
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

combined_grids_with_outer_labels <- ggarrange(avg_bound_loss05_var_grid_with_title, gamma005_bound_loss05_var_grid_with_title,
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
avg_gamma005_var_loss05 <- ggarrange(combined_grids_with_outer_labels, common_legend,
                                    ncol = 1, nrow = 2,
                                    heights = c(1, 0.2)) # Adjust heights as needed
#grid.lines(x = unit(0.5, "npc"), y = unit(c(0.2, 1), "npc"), gp = gpar(col = "gray40", lwd = 1.5))







##### Individual Level ####

avg_mean_high_stratus = current_info_avg_bound_loss05_mean_ev_steps_rate[which(current_info_avg_bound_loss05_mean_ev_steps_rate$income_strata==">100k"),]
avg_mean_low_stratus = current_info_avg_bound_loss05_mean_ev_steps_rate[which(current_info_avg_bound_loss05_mean_ev_steps_rate$income_strata=="0~6k"),]

#gamma025_mean_high_stratus = current_info_gamma025_bound_loss05_mean_ev_steps_rate[which(current_info_gamma025_bound_loss05_mean_ev_steps_rate$income_strata==">100k"),]
#gamma025_mean_low_stratus = current_info_gamma025_bound_loss05_mean_ev_steps_rate[which(current_info_gamma025_bound_loss05_mean_ev_steps_rate$income_strata=="0~6k"),]

avg_mean_high_stratus_q = current_info_avg_bound_loss05_mean_q_steps_rate[which(current_info_avg_bound_loss05_mean_q_steps_rate$income_strata==">100k"),]
avg_mean_low_stratus_q = current_info_avg_bound_loss05_mean_q_steps_rate[which(current_info_avg_bound_loss05_mean_q_steps_rate$income_strata=="0~6k"),]


gamma025_low = ggplot(gamma025_mean_low_stratus, aes(x = q_0, y = rate_0.25)) +
  geom_hex(bins = 75, color = "white", size = 0.1) +
  scale_fill_viridis_c(trans = "log10",
                       option = "D",
                       direction = -1,
                       bquote(log[10](Count~of~Observations))) +
  coord_cartesian(xlim = c(0, 250),
                  ylim = c(-90, 10)
                  ) +
  
  labs(title = "Concave Constraint - 0~6k",
    x = "Initial Quantity (q_0) (kGal)",
    y =expression("EV/I (%) when "~zeta[1] ~ "=0.25"), # Updated y-axis label to reflect rate_0.25
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
    y =expression("EV/I (%) when "~zeta[1] ~ "=0.25"), # Updated y-axis label to reflect rate_0.25
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

avg_mean_low_stratus$`q_-0.25` = avg_mean_low_stratus_q$`-0.25`
avg_mean_high_stratus$`q_-0.25` = avg_mean_high_stratus_q$`-0.25`

avg_mean_low_stratus = avg_mean_low_stratus %>% 
  group_by(prem_id) %>%
  summarise(ev_dry = mean(`-0.25`),
            ev_rate_dry = mean(`rate_-0.25`),
            q_dry = mean(`q_-0.25`),
            q_0 = mean(q_0))

avg_mean_high_stratus = avg_mean_high_stratus %>% 
  group_by(prem_id) %>%
  summarise(ev_dry = mean(`-0.25`),
            ev_rate_dry = mean(`rate_-0.25`),
            q_dry = mean(`q_-0.25`),
            q_0 = mean(q_0))

# 1. Filter the dataframes based on your criteria
filtered_low_stratus <- avg_mean_low_stratus %>%
  filter(q_0 >= 0 & q_0 <= 250 & ev_rate_dry > -100)

filtered_high_stratus <- avg_mean_high_stratus %>%
  filter(q_0 >= 0 & q_0 <= 250 & ev_rate_dry > -100)

low_sample <- avg_mean_low_stratus %>% 
  filter(q_0 >= 0 & q_0 <= 250) %>% # Note: I removed the y-filter for a fuller picture
  #slice_sample(n = 50000) %>% 
  mutate(category = "0 - 6k")

high_sample <- avg_mean_high_stratus %>%
  filter(q_0 >= 0 & q_0 <= 250) %>% # Note: I removed the y-filter for a fuller picture
  #slice_sample(n = 50000) %>%
  mutate(category = ">100k")

combined_data <- bind_rows(low_sample, high_sample) %>%
  # 1. SET FACET ORDER: Convert category to a factor with a specific order
  mutate(category = factor(category, levels = c("0 - 6k", ">100k")))


# --- Plotting ---
final_simple_plot <- ggplot(combined_data, aes(x = q_dry, y = ev_rate_dry, color = category)) +
  
  geom_point(alpha = 0.4, size = 1.5) +
  
  scale_color_manual(values = c("0 - 6k" = "#e22959", ">100k" = "#234043")) +
  
  # 2. SYNCHRONIZE Y-AXIS: Change "free_y" to "fixed"
  facet_wrap(~ category, scales = "fixed") +
  
  labs(
    title = "",
    x = "Counterfactual Quantity (kGal)",
    y = expression("EV/I (%) when " ~ zeta[1] ~ "=-0.25")
  ) +
  my_plot_theme+
  theme(
    #plot.title = element_text(hjust = 0.5),
    legend.position = "none" 
  )

# Print the final plot
print(final_simple_plot)

##### Individual Level - Low Stratus ####

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

current_info_avg_bound_loss05_mean_ev_steps_rate = cbind(current_info_avg_bound_loss05_mean_ev_steps_rate, demand_char)

current_info_avg_bound_loss05_mean_ev_steps_rate$quantity_dry = current_info_avg_bound_loss05_mean_q_steps_rate$`-0.25`
current_info_avg_bound_loss05_mean_ev_steps_rate$quantity_wet = current_info_avg_bound_loss05_mean_q_steps_rate$`0.25`
current_info_avg_bound_loss05_mean_ev_steps_rate$payment_dry = current_info_avg_bound_loss05_mean_r_steps_rate$`-0.25`
current_info_avg_bound_loss05_mean_ev_steps_rate$payment_wet = current_info_avg_bound_loss05_mean_r_steps_rate$`0.25`
current_info_avg_bound_loss05_mean_ev_steps_rate$cs_dry = current_info_avg_bound_loss05_mean_cs_steps_rate$`-0.25`
current_info_avg_bound_loss05_mean_ev_steps_rate$cs_wet = current_info_avg_bound_loss05_mean_cs_steps_rate$`0.25`

current_info_avg_bound_loss05_mean_ev_steps_rate$cs_dry_diff = current_info_avg_bound_loss05_mean_ev_steps_rate$cs_dry - current_info_avg_bound_loss05_mean_ev_steps_rate$cs_0
current_info_avg_bound_loss05_mean_ev_steps_rate$cs_wet_diff = current_info_avg_bound_loss05_mean_ev_steps_rate$cs_wet - current_info_avg_bound_loss05_mean_ev_steps_rate$cs_0

avg_mean_high_stratus = current_info_avg_bound_loss05_mean_ev_steps_rate[which(current_info_avg_bound_loss05_mean_ev_steps_rate$income_strata==">100k"),]
avg_mean_low_stratus = current_info_avg_bound_loss05_mean_ev_steps_rate[which(current_info_avg_bound_loss05_mean_ev_steps_rate$income_strata=="0~6k"),]

cs_check = avg_mean_low_stratus %>%
  select(cs_dry_diff, `-0.25`, cs_wet_diff, `0.25`)

avg_mean_low_stratus_prem_id = avg_mean_low_stratus %>% 
  group_by(prem_id) %>%
  summarise(q0 = mean(q_0),
            r0 = mean(r_0),
            ev_dry = mean(`-0.25`),
            ev_wet = mean(`0.25`),
            quantity_dry = mean(quantity_dry),
            quantity_wet = mean(quantity_wet),
            payment_dry = mean(payment_dry),
            payment_wet = mean(payment_wet),
            ev_wet = mean(`0.25`),
            ev_rate_dry = mean(`rate_-0.25`),
            ev_rate_wet = mean(rate_0.25),
            income = mean(income)
            )

avg_mean_low_stratus_prem_id = avg_mean_low_stratus_prem_id %>%
  mutate(q_dry_diff = quantity_dry - q0,
         q_wet_diff = quantity_wet - q0,
         r_dry_diff = payment_dry - r0,
         r_wet_diff = payment_wet - r0)

long_data <- avg_mean_low_stratus_prem_id %>%
  pivot_longer(
    cols = ends_with("_dry") | ends_with("_wet"),
    # Use a regex pattern instead of names_sep
    # This correctly separates "evrate" from "dry"
    names_pattern = "(.*)_(dry|wet)",
    names_to = c(".value", "condition")
  )

plot_variable_by_condition <- function(data, x_variable, variable_name) {
  
  # Create a more readable title from the variable name
  pretty_title <- str_replace_all(variable_name, "_", " ") %>% str_to_title()
  
  ggplot(data, aes(x = x_variable, y = .data[[variable_name]], color = condition)) +
    geom_point(alpha = 0.7, size = 2) +
    scale_color_manual(
      name = "Condition",
      values = c("dry" = "#D55E00", "wet" = "#0072B2")
    ) +
    labs(
      title = paste(pretty_title, "vs. q0 by Condition"),
      x = as.character(x_variable),
      y = pretty_title
    ) +
    theme_minimal()
}

plot_variable_by_condition(long_data, long_data$q0, "ev")

ggplot(avg_mean_low_stratus_prem_id, aes(x = q_dry_diff, y = r_dry_diff, color = "#D55E00")) +
  geom_point(alpha = 0.7, size = 2) +
  theme_minimal()

#### In dry condition, there are people who consume more and there are people who consume less. but almost all of them pay more. 
#### They all use less water (due to the conservation constraint), but in wet condition, they all pay less, and in dry condition, they are pay more. 
### For the demand system CS go up, but why EV went down?

#### I am using less and paying less, get higher CS, why EV is lower?
#### EV is measuring the welfare impact of price change. Price either higher in marginal price or lower the cutoff point, making the price essentially higher.
#### The reason for the higher price is to satisfy the conservation constraint.

##### What is causing the heterogeneity?

current_info_avg_bound_loss05_mean_ev_steps_rate = current_info_avg_bound_loss05_mean_ev_steps_rate %>%
  mutate(high_q = case_when(
    q_0 >=20 ~ 1,
    q_0 < 20 ~ 0
  ))

# Create a new column where the unit is 100 sq ft
current_info_avg_bound_loss05_mean_ev_steps_rate$hvac_res_100sqft <- current_info_avg_bound_loss05_mean_ev_steps_rate$hvac_residential / 100
current_info_avg_bound_loss05_mean_ev_steps_rate$house_value_th <- current_info_avg_bound_loss05_mean_ev_steps_rate$total_housevalue / 1000

breaks <- quantile(current_info_avg_bound_loss05_mean_ev_steps_rate$hvac_res_100sqft, probs = c(0, 0.25, 0.5, 0.75, 1))

current_info_avg_bound_loss05_mean_ev_steps_rate <- current_info_avg_bound_loss05_mean_ev_steps_rate %>%
  mutate(hvac_category = cut(hvac_res_100sqft,
                             breaks = breaks,
                             labels = c("Low", "Medium-Low", "Medium-High", "High"),
                             include.lowest = TRUE)) # Ensures the minimum value is included

avg_mean_high_stratus = current_info_avg_bound_loss05_mean_ev_steps_rate[which(current_info_avg_bound_loss05_mean_ev_steps_rate$income_strata==">100k"),]
avg_mean_low_stratus = current_info_avg_bound_loss05_mean_ev_steps_rate[which(current_info_avg_bound_loss05_mean_ev_steps_rate$income_strata=="0~6k"),]

avg_mean_sum = current_info_avg_bound_loss05_mean_ev_steps_rate %>%
  group_by(prem_id) %>%
  summarise(q_0 = mean(q_0),
            total_housevalue = first(total_housevalue),
            prev_NDVI = mean(prev_NDVI),
            bathroom = first(bathroom),
            bedroom = first(bedroom),
            spa_area = first(spa_area),
            fountain_wtr_area = first(fountain_wtr_area),
            heavy_water_app_area = first(heavy_water_app_area),
            hvac_residential = first(hvac_residential),
            lawn_area = first(lawn_area),
            lawn_percentage = first(lawn_percentage),
            income = mean(income),
            incomne_strata = first(income_strata)
            )

avg_mean_sum = avg_mean_sum %>%
  mutate(high_q = case_when(
    q_0 >=20 ~ 1,
    q_0 < 20 ~ 0
  ))


avg_mean_high_stratus_sum = avg_mean_sum %>% filter (incomne_strata == ">100k")
avg_mean_low_stratus_sum = avg_mean_sum %>% filter (incomne_strata == "0~6k")

library(brglm2)
model_low <- glm(high_q ~ house_value_th*total_PRCP + prev_NDVI*total_PRCP + bathroom*total_PRCP + bedroom * total_PRCP
                 + spa_area*total_PRCP + heavy_water_app_area*total_PRCP + 
                   hvac_category*total_PRCP + lawn_percentage*total_PRCP, 
                    data = avg_mean_low_stratus, 
                    family = "binomial",
                    method = "brglmFit")

summary(model_low)
#Call:
#  glm(formula = high_q ~ house_value_th * total_PRCP + prev_NDVI * 
#        total_PRCP + bathroom * total_PRCP + bedroom * total_PRCP + 
#        spa_area * total_PRCP + heavy_water_app_area * total_PRCP + 
#        hvac_category * total_PRCP + lawn_percentage * total_PRCP, 
#      family = "binomial", data = avg_mean_low_stratus, method = "brglmFit")

#Deviance Residuals: 
#  Min       1Q   Median       3Q      Max  
#-2.7668  -0.5545  -0.1057  -0.0050   3.5464  

#Coefficients:
#  Estimate Std. Error z value Pr(>|z|)    
#(Intercept)                         -1.031e+00  1.383e-01  -7.458 8.81e-14 ***
#  house_value_th                      -4.073e-03  1.065e-04 -38.242  < 2e-16 ***
#  total_PRCP                          -2.345e+00  1.375e-01 -17.061  < 2e-16 ***
#  prev_NDVI                            5.294e+00  1.290e-01  41.048  < 2e-16 ***
#  bathroom                             2.093e-01  1.922e-02  10.892  < 2e-16 ***
#  bedroom                             -1.611e-02  1.618e-02  -0.996  0.31938    
#spa_area                             2.655e-03  2.011e-04  13.203  < 2e-16 ***
#  heavy_water_app_area                -9.095e-02  1.986e-01  -0.458  0.64702    
#hvac_categoryMedium-Low              4.144e-01  3.202e-02  12.944  < 2e-16 ***
#  hvac_categoryMedium-High             6.964e-01  4.165e-02  16.721  < 2e-16 ***
#  hvac_categoryHigh                    1.009e+00  5.689e-02  17.739  < 2e-16 ***
#  lawn_percentage                      2.117e-01  1.555e-01   1.361  0.17349    
#house_value_th:total_PRCP            1.971e-03  9.107e-05  21.639  < 2e-16 ***
#  total_PRCP:prev_NDVI                -3.692e-01  1.245e-01  -2.965  0.00303 ** 
#  total_PRCP:bathroom                 -6.639e-02  1.701e-02  -3.902 9.52e-05 ***
#  total_PRCP:bedroom                   1.654e-03  1.470e-02   0.112  0.91043    
#total_PRCP:spa_area                 -1.169e-03  1.144e-04 -10.222  < 2e-16 ***
#  total_PRCP:heavy_water_app_area      1.102e-01  1.524e-01   0.723  0.46945    
#total_PRCP:hvac_categoryMedium-Low  -1.037e-01  3.343e-02  -3.101  0.00193 ** 
#  total_PRCP:hvac_categoryMedium-High  3.744e-02  3.871e-02   0.967  0.33348    
#total_PRCP:hvac_categoryHigh         1.280e-01  4.755e-02   2.691  0.00712 ** 
#  total_PRCP:lawn_percentage           4.717e-01  1.554e-01   3.036  0.00240 ** 
#  ---
#  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#(Dispersion parameter for binomial family taken to be 1)

#Null deviance: 171816  on 171898  degrees of freedom
#Residual deviance: 107995  on 171877  degrees of freedom
#AIC:  108039

#Type of estimator: AS_mixed (mixed bias-reducing adjusted score equations)
#Number of Fisher Scoring iterations: 13

model_high <- glm(high_q ~ house_value_th*total_PRCP + prev_NDVI*total_PRCP + bathroom*total_PRCP + bedroom * total_PRCP
                 + spa_area*total_PRCP + heavy_water_app_area*total_PRCP + 
                   hvac_category*total_PRCP + lawn_percentage*total_PRCP, 
                 data = avg_mean_high_stratus, 
                 family = "binomial",
                 method = "brglmFit")

summary(model_high)

#Call:
#  glm(formula = high_q ~ house_value_th * total_PRCP + prev_NDVI * 
#        total_PRCP + bathroom * total_PRCP + bedroom * total_PRCP + 
#        spa_area * total_PRCP + heavy_water_app_area * total_PRCP + 
#        hvac_category * total_PRCP + lawn_percentage * total_PRCP, 
#      family = "binomial", data = avg_mean_high_stratus, method = "brglmFit")

#Deviance Residuals: 
#  Min       1Q   Median       3Q      Max  
#-3.9587  -0.4532  -0.0374   0.6490   3.9350  

#Coefficients:
#  Estimate Std. Error z value Pr(>|z|)    
#(Intercept)                         -9.509e-01  2.334e-01  -4.075 4.61e-05 ***
#  house_value_th                       3.193e-04  2.507e-05  12.739  < 2e-16 ***
#  total_PRCP                          -1.569e+00  1.893e-01  -8.291  < 2e-16 ***
#  prev_NDVI                            4.550e+00  2.265e-01  20.084  < 2e-16 ***
#  bathroom                            -1.155e-02  9.001e-03  -1.284  0.19925    
#bedroom                             -3.880e-02  8.667e-03  -4.477 7.57e-06 ***
#  spa_area                            -2.149e-03  8.812e-04  -2.439  0.01472 *  
#  heavy_water_app_area                -9.840e-05  2.207e-05  -4.458 8.26e-06 ***
#  hvac_categoryMedium-Low             -9.072e-02  2.439e-01  -0.372  0.70992    
#hvac_categoryMedium-High             8.660e-01  2.003e-01   4.323 1.54e-05 ***
#  hvac_categoryHigh                    1.332e+00  1.846e-01   7.213 5.46e-13 ***
#  lawn_percentage                      3.328e-02  1.838e-01   0.181  0.85635    
#house_value_th:total_PRCP           -3.041e-05  7.125e-06  -4.269 1.97e-05 ***
#  total_PRCP:prev_NDVI                -8.509e-01  1.349e-01  -6.308 2.84e-10 ***
#  total_PRCP:bathroom                  2.026e-02  4.367e-03   4.639 3.50e-06 ***
#  total_PRCP:bedroom                   1.336e-02  4.556e-03   2.934  0.00335 ** 
#  total_PRCP:spa_area                 -5.322e-04  5.151e-04  -1.033  0.30158    
#total_PRCP:heavy_water_app_area      2.505e-05  8.856e-06   2.828  0.00468 ** 
#  total_PRCP:hvac_categoryMedium-Low   2.147e-01  2.182e-01   0.984  0.32501    
#total_PRCP:hvac_categoryMedium-High -1.002e-01  1.801e-01  -0.557  0.57784    
#total_PRCP:hvac_categoryHigh         2.092e-01  1.691e-01   1.237  0.21604    
#total_PRCP:lawn_percentage           1.818e-01  1.076e-01   1.690  0.09110 .  
#---
#  Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#(Dispersion parameter for binomial family taken to be 1)

#Null deviance: 88883  on 65363  degrees of freedom
#Residual deviance: 48484  on 65342  degrees of freedom
#AIC:  48528

#Type of estimator: AS_mixed (mixed bias-reducing adjusted score equations)
#Number of Fisher Scoring iterations: 5

library(lme4)

avg_mean_low_stratus$prem_id <- factor(avg_mean_low_stratus$prem_id)

avg_mean_low_stratus$house_value_scaled <- scale(avg_mean_low_stratus$house_value_th)
avg_mean_low_stratus$NDVI_scaled <- scale(avg_mean_low_stratus$prev_NDVI)
avg_mean_low_stratus$lawn_percentage_c <- avg_mean_low_stratus$lawn_percentage - mean(avg_mean_low_stratus$lawn_percentage)

model_low <- glmer(high_q ~ house_value_scaled*total_PRCP + NDVI_scaled*total_PRCP + 
                              bathroom*total_PRCP  + spa_area*total_PRCP  + 
                              hvac_category*total_PRCP + 
                              (1 | prem_id), # Keep the random effect
                            data = avg_mean_low_stratus,
                            family = "binomial")

summary(model_low)

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
gamma005_bound_loss05_mean_ev$gamma = 0.05
gamma01_bound_loss05_mean_ev$gamma = 0.1
gamma015_bound_loss05_mean_ev$gamma = 0.15
gamma02_bound_loss05_mean_ev$gamma = 0.2
gamma025_bound_loss05_mean_ev$gamma = 0.25
gamma03_bound_loss05_mean_ev$gamma = 0.3
gamma035_bound_loss05_mean_ev$gamma = 0.35
gamma04_bound_loss05_mean_ev$gamma = 0.4
gamma045_bound_loss05_mean_ev$gamma = 0.45
gamma05_bound_loss05_mean_ev$gamma = 0.5
loss05_mean_ev = rbind(avg_bound_loss05_mean_ev,gamma005_bound_loss05_mean_ev,
                       gamma01_bound_loss05_mean_ev,gamma015_bound_loss05_mean_ev,
                       gamma02_bound_loss05_mean_ev,gamma025_bound_loss05_mean_ev, 
                       gamma03_bound_loss05_mean_ev,gamma035_bound_loss05_mean_ev,
                       gamma04_bound_loss05_mean_ev,gamma045_bound_loss05_mean_ev,
                       gamma05_bound_loss05_mean_ev)

avg_bound_loss05_var_ev$gamma = 0
gamma005_bound_loss05_var_ev$gamma = 0.05
gamma01_bound_loss05_var_ev$gamma = 0.1
gamma015_bound_loss05_var_ev$gamma = 0.15
gamma02_bound_loss05_var_ev$gamma = 0.2
gamma025_bound_loss05_var_ev$gamma = 0.25
gamma03_bound_loss05_var_ev$gamma = 0.3
gamma035_bound_loss05_var_ev$gamma = 0.35
gamma04_bound_loss05_var_ev$gamma = 0.4
gamma045_bound_loss05_var_ev$gamma = 0.45
gamma05_bound_loss05_var_ev$gamma = 0.5
loss05_var_ev = rbind(avg_bound_loss05_var_ev,gamma005_bound_loss05_var_ev,
                       gamma01_bound_loss05_var_ev,gamma015_bound_loss05_var_ev,
                       gamma02_bound_loss05_var_ev,gamma025_bound_loss05_var_ev, 
                       gamma03_bound_loss05_var_ev,gamma035_bound_loss05_var_ev,
                       gamma04_bound_loss05_var_ev,gamma045_bound_loss05_var_ev,
                       gamma05_bound_loss05_var_ev)

avg_bound_loss05_mean_cs$gamma = 0
gamma005_bound_loss05_mean_cs$gamma = 0.05
gamma01_bound_loss05_mean_cs$gamma = 0.1
gamma015_bound_loss05_mean_cs$gamma = 0.15
gamma02_bound_loss05_mean_cs$gamma = 0.2
gamma025_bound_loss05_mean_cs$gamma = 0.25
gamma03_bound_loss05_mean_cs$gamma = 0.3
gamma035_bound_loss05_mean_cs$gamma = 0.35
gamma04_bound_loss05_mean_cs$gamma = 0.4
gamma045_bound_loss05_mean_cs$gamma = 0.45
gamma05_bound_loss05_mean_cs$gamma = 0.5
loss05_mean_cs = rbind(avg_bound_loss05_mean_cs,gamma005_bound_loss05_mean_cs,
                       gamma01_bound_loss05_mean_cs,gamma015_bound_loss05_mean_cs,
                       gamma02_bound_loss05_mean_cs,gamma025_bound_loss05_mean_cs, 
                       gamma03_bound_loss05_mean_cs,gamma035_bound_loss05_mean_cs,
                       gamma04_bound_loss05_mean_cs,gamma045_bound_loss05_mean_cs,
                       gamma05_bound_loss05_mean_cs)

avg_bound_loss05_var_cs$gamma = 0
gamma005_bound_loss05_var_cs$gamma = 0.05
gamma01_bound_loss05_var_cs$gamma = 0.1
gamma015_bound_loss05_var_cs$gamma = 0.15
gamma02_bound_loss05_var_cs$gamma = 0.2
gamma025_bound_loss05_var_cs$gamma = 0.25
gamma03_bound_loss05_var_cs$gamma = 0.3
gamma035_bound_loss05_var_cs$gamma = 0.35
gamma04_bound_loss05_var_cs$gamma = 0.4
gamma045_bound_loss05_var_cs$gamma = 0.45
gamma05_bound_loss05_var_cs$gamma = 0.5
loss05_var_cs = rbind(avg_bound_loss05_var_cs,gamma005_bound_loss05_var_cs,
                      gamma01_bound_loss05_var_cs,gamma015_bound_loss05_var_cs,
                      gamma02_bound_loss05_var_cs,gamma025_bound_loss05_var_cs, 
                      gamma03_bound_loss05_var_cs,gamma035_bound_loss05_var_cs,
                      gamma04_bound_loss05_var_cs,gamma045_bound_loss05_var_cs,
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





