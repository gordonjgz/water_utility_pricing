setwd("~/Austin Water/NDVI_categorization")
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


academic_theme <- theme_minimal(base_size = 15) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(face = "bold", size = 10), # Added legend title styling
    legend.text = element_text(size = 13),
    axis.title.x = element_text(size = 13, face = "bold", margin = margin(t = 10)), # Bold, slightly larger, margin
    axis.title.y = element_text(size = 13, face = "bold", margin = margin(r = 10)), # Bold, slightly larger, margin
    axis.text = element_text(size = 13),
    panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
    panel.grid.minor = element_blank(), # Typically remove minor grids
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5) # Add subtle border
  )





##### Summary Stats #####
premise_segments_roster = read_csv(file = "../premise_segments_roster.csv")

demand_categorization = merge(demand_2018_using_new_small, premise_segments_roster, by.x = c("prem_id"),
                         by.y = c("prem_id"), all.x = T)

summary_stats <- demand_categorization %>%
  group_by(label) %>%
  reframe(across(
    c(charge, quantity, bathroom, heavy_water_spa, NDVI, income),
    list(
      #min = ~min(.x, na.rm = TRUE),
      #max = ~max(.x, na.rm = TRUE),
      mean = ~mean(.x, na.rm = TRUE),
      median = ~median(.x, na.rm = TRUE),
      q25 = ~quantile(.x, 0.25, na.rm = TRUE),
      q75 = ~quantile(.x, 0.75, na.rm = TRUE)
    ),
    .names = "{.col}_{.fn}"
  ))

# View the final summary table
print(t(summary_stats))

demand_categorization$elastic = case_when(demand_categorization$cluster)

##### price result ########
q = seq(0.5, 100, by = 0.1)

gen_p = function(p, t, A, q) {
  
  A1 = A[1]
  A2 = A[2]
  A3 = A[3]
  A4 = A[4]
  A5 = A[5]
  
  t1 = t[1]
  t2 = t[2]
  t3 = t[3]
  t4 = t[4]
  
  p1 = p[1]
  p2 = p[2]
  p3 = p[3]
  p4 = p[4]
  p5 = p[5]
  
  ap = case_when(
    q <= t1 ~ (A1 + p1 * q) / q,
    q > t1 & q <= t2 ~ (A2 + p1 * t1 + (p2 - 0.2) * (q - t1) + 0.2 * q) / q,
    q > t2 & q <= t3 ~ (A3 + (p1 - 0.2) * t1 + (p2 - 0.2) * (t2 - t1) + (p3 - 0.2) * (q - t2) + 0.2 * q) / q,
    q > t3 & q <= t4 ~ (A4 + (p1 - 0.2) * t1 + (p2 - 0.2) * (t2 - t1) + (p3 - 0.2) * (t3 - t2) + (p4 - 0.2) * (q - t3) + 0.2 * q) / q,
    q > t4 ~ (A5 + (p1 - 0.2) * t1 + (p2 - 0.2) * (t2 - t1) + (p3 - 0.2) * (t3 - t2) + (p4 - 0.2) * (t4 - t3) + (p5 - 0.2) * (q - t4) + 0.2 * q) / q
  )
  
  mp = case_when(
    q <= t1 ~ p1,
    q > t1 & q <= t2 ~ p2,
    q > t2 & q <= t3 ~ p3,
    q > t3 & q <= t4 ~ p4,
    q > t4 ~ p5
  )
  
  total = ap * q
  result = data.frame(q, ap, mp, total)
  colnames(result) = c("q", "ap", "mp", "total")
  
  return(result)
}

t = read_csv(file = "results/PD_kinks.csv")
t = t$`0`
mp_i = read_csv(file = "results/PD_price_inelastic.csv")
mp_i = mp_i$`0`
mp_e = read_csv(file = "results/PD_price_base.csv")
mp_e = mp_e$`0`
fp_i = read_csv(file = "results/PD_fixed_payments_inelastic.csv")
fp_i = fp_i$`0`
fp_e = read_csv(file = "results/PD_fixed_payments_elastic.csv")
fp_e = fp_e$`0`

mp0 = c(3.09,5.01,8.54,12.9,14.41)
t0 = c(2,6,11,20)
fp_0 = c(8.5,10.8,16.5,37,37)
statusquo = gen_p(mp0, t0, fp_0, q)
statusquo$step = "status_quo"

inelastic = gen_p(mp_i, t, fp_i, q)
elastic = gen_p(mp_e, t, fp_e, q)

elastic$step = "Elastic PD"
inelastic$step = "Inelastic PD"


mp_flat = rep(mean(c(3.09,5.01,8.54,12.9,14.41)),5 )
fp_flat = rep(mean(c(8.5,10.8,16.5,37,37)),5 )
flat = gen_p(mp_flat, t0, fp_flat, q)
flat$step = "flat"

color_palette = c(
  "Elastic PD" = "#0072B2",
  "Inelastic PD" = "#D55E00"
)

pd_all = bind_rows(elastic, inelastic)

total_payment = ggplot() +
  geom_line(
    data = pd_all,
    aes(x = q, y = total, color = step),
    linewidth = 1.2
  ) +
  geom_line(
    data = flat,
    aes(x = q, y = total, linetype = "Status Quo - Flat"),
    color = "#6e6e6e",
    linewidth = 0.8
  ) +
  geom_line(
    data = statusquo,
    aes(x = q, y = total, linetype = "Status Quo"),
    color = "#080808",
    linewidth = 0.8
  ) +
  xlab("Quantity (k Gallons)") +
  ylab("Total Payment ($)") +
  scale_color_manual(values = color_palette) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted",
               "Status Quo" = "dotdash")
  ) +
  labs(color = NULL, linetype = NULL) +  # ??? removes legend titles
  theme(legend.title = element_blank()) + # ??? extra safety
  #coord_cartesian(xlim = c(0, 75), ylim = c(0, 2500)) +
  academic_theme


marginal_price = ggplot() +
  geom_step(
    data = elastic,
    aes(x = q, y = mp, color = step),
    linewidth = 1
  ) +
  geom_step(
    data = inelastic,
    aes(x = q, y = mp, color = step),
    linewidth = 1
  ) +
  geom_step(
    data = statusquo,
    aes(x = q, y = mp, linetype = "Status Quo"),
    color = "#080808",
    linewidth = 0.8
  ) +
  xlab("Quantity (k Gallons)") +
  ylab("Marginal Price ($)") +
  scale_color_manual(values = color_palette) +
  scale_linetype_manual(
    values = c("Status Quo" = "dotdash")
  ) +
  labs(color = NULL, linetype = NULL) +
  theme(legend.title = element_blank()) +
  academic_theme

if (!dir.exists("plot")) {
  dir.create("plot")
}


ggsave(
  "plot/total_payment_PD.png",
  plot = total_payment,
  width = 8,
  height = 6,
  dpi = 300
)

ggsave(
  "plot/marginal_price_PD.png",
  plot =marginal_price,
  width = 8,
  height = 6,
  dpi = 300
)






##### welfare analysis ########

hh_level_results = read_csv(file = "results/PD_hh_level_results.csv")
hh_level_results$prem_id = NULL
hh_level_results = data.frame(cbind(demand_key, hh_level_results))

hh_level_results = merge(hh_level_results, premise_segments_roster, by.x = c("prem_id"),
                         by.y = c("prem_id"), all.x = T)

hh_level_results_prem_id = hh_level_results %>%
  group_by(prem_id) %>%
  summarise(welfare_difference = mean( (cs - cs_0) /cs_0),
            quantity_difference = mean( (q - q_0) /q_0),
            payment_difference = mean( (r - r_0) /r_0),
            mean_ev = mean(ev),
            mean_ev_percentage = mean(ev/income),
            income = mean(income),
            income_strata = first(income_strata),
            label = first(label)
            )

hh_level_results_income_strata = hh_level_results_prem_id %>%
  group_by(income_strata) %>%
  summarise(welfare_difference = mean( welfare_difference, na.rm = T),
            quantity_difference = mean( quantity_difference, na.rm = T),
            payment_difference = mean( payment_difference, na.rm = T),
            mean_ev = mean(mean_ev, na.rm = T),
            mean_ev_percentage = mean(mean_ev_percentage, na.rm = T))

hh_level_results_label_bill_ym = hh_level_results %>%
  group_by(bill_ym,label) %>%
  summarise(welfare_difference = mean( (cs - cs_0) /cs_0, na.rm = T),
            quantity_difference = mean( (q - q_0) /q_0, na.rm = T),
            payment_difference = mean( (r - r_0) /r_0, na.rm = T),
            mean_ev = mean(ev, na.rm = T),
            mean_ev_percentage = mean(ev/income, na.rm = T),
            income = mean(income)
  ) %>% ungroup()

hh_level_results_label = hh_level_results_prem_id %>%
  group_by(label) %>%
  summarise(welfare_difference = mean( welfare_difference, na.rm = T),
            quantity_difference = mean( quantity_difference, na.rm = T),
            payment_difference = mean( payment_difference, na.rm = T),
            mean_ev = mean(mean_ev, na.rm = T),
            mean_ev_percentage = mean(mean_ev_percentage, na.rm = T)) %>% ungroup()




#### Detail Analysis, Who's the winner and who's the loser #####

#### Quantity ####

hh_level_results = hh_level_results %>%
  mutate(
    welfare_difference = (cs - cs_0)/cs_0,
    quantity_difference = (q - q_0)/q_0,
    payment_difference = (r - r_0)/r_0
  )




