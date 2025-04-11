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

my_theme <- theme_minimal()+
  theme(title = element_text(hjust=0.5),legend.position='bottom')
theme_set(my_theme)
big_text <- theme(text = element_text(size=24))



#### Prepare #####
r_history_agg_0 = 26197351.09150049
cs_history_agg_0 = -1.13182266e+17
q_history_agg_0 = 1999930.37964376


r_agg_0 = 7258745.08835472
cs_agg_0 = -4.34379253e+17
q_agg_0 = 794810.84743965

q = seq(0.5, 75, by = 0.2)

gen_p = function(p_vector, q) {
  p = p_vector[1:5]
  t = p_vector[6:9]
  A = p_vector[10:14]
  
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

#####
avg_info_avg_bound_mean_pl = read_csv("../ramsey_price_result/price_detail_results/avg_info_avg_bound_mean_pl.csv")
avg_info_avg_bound_mean_ql = read_csv("../ramsey_price_result/price_detail_results/avg_info_avg_bound_mean_ql.csv")
avg_info_avg_bound_mean_fcl = read_csv("../ramsey_price_result/price_detail_results/avg_info_avg_bound_mean_fcl.csv")

avg_info_avg_bound_var_pl = read_csv("../ramsey_price_result/price_detail_results/avg_info_avg_bound_var_pl.csv")
avg_info_avg_bound_var_ql = read_csv("../ramsey_price_result/price_detail_results/avg_info_avg_bound_var_ql.csv")
avg_info_avg_bound_var_fcl = read_csv("../ramsey_price_result/price_detail_results/avg_info_avg_bound_var_fcl.csv")

avg_info_extreme_bound_mean_pl = read_csv("../ramsey_price_result/price_detail_results/avg_info_extreme_bound_mean_pl.csv")
avg_info_extreme_bound_mean_ql = read_csv("../ramsey_price_result/price_detail_results/avg_info_extreme_bound_mean_ql.csv")
avg_info_extreme_bound_mean_fcl = read_csv("../ramsey_price_result/price_detail_results/avg_info_extreme_bound_mean_fcl.csv")


avg_info_avg_bound_mean = rbind(avg_info_avg_bound_mean_pl, avg_info_avg_bound_mean_ql, avg_info_avg_bound_mean_fcl)
avg_info_avg_bound_var = rbind(avg_info_avg_bound_var_pl, avg_info_avg_bound_var_ql, avg_info_avg_bound_var_fcl)
avg_info_extreme_bound_mean = rbind(avg_info_extreme_bound_mean_pl, avg_info_extreme_bound_mean_ql, avg_info_extreme_bound_mean_fcl)


#####
current_info_avg_bound_mean_pl = read_csv("../ramsey_price_result/price_detail_results/current_info_avg_bound_mean_pl.csv")
current_info_avg_bound_mean_ql = read_csv("../ramsey_price_result/price_detail_results/current_info_avg_bound_mean_ql.csv")
current_info_avg_bound_mean_fcl = read_csv("../ramsey_price_result/price_detail_results/current_info_avg_bound_mean_fcl.csv")

current_info_avg_bound_var_pl = read_csv("../ramsey_price_result/price_detail_results/current_info_avg_bound_var_pl.csv")
current_info_avg_bound_var_ql = read_csv("../ramsey_price_result/price_detail_results/current_info_avg_bound_var_ql.csv")
current_info_avg_bound_var_fcl = read_csv("../ramsey_price_result/price_detail_results/current_info_avg_bound_var_fcl.csv")

current_info_logr_bound_mean_pl = read_csv("../ramsey_price_result/price_detail_results/current_info_logr_bound_mean_pl.csv")
current_info_logr_bound_mean_ql = read_csv("../ramsey_price_result/price_detail_results/current_info_logr_bound_mean_ql.csv")
current_info_logr_bound_mean_fcl = read_csv("../ramsey_price_result/price_detail_results/current_info_logr_bound_mean_fcl.csv")

current_info_logr_bound_var_pl = read_csv("../ramsey_price_result/price_detail_results/current_info_logr_bound_var_pl.csv")
current_info_logr_bound_var_ql = read_csv("../ramsey_price_result/price_detail_results/current_info_logr_bound_var_ql.csv")
current_info_logr_bound_var_fcl = read_csv("../ramsey_price_result/price_detail_results/current_info_logr_bound_var_fcl.csv")

current_info_gamma05_bound_mean_pl = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma05_bound_mean_pl.csv")
current_info_gamma05_bound_mean_ql = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma05_bound_mean_ql.csv")
current_info_gamma05_bound_mean_fcl = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma05_bound_mean_fcl.csv")

current_info_gamma05_bound_var_pl = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma05_bound_var_pl.csv")
current_info_gamma05_bound_var_ql = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma05_bound_var_ql.csv")
current_info_gamma05_bound_var_fcl = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma05_bound_var_fcl.csv")

#current_info_extreme_bound_mean_pl = read_csv("../ramsey_price_result/price_detail_results/current_info_extreme_bound_mean_pl.csv")
#current_info_extreme_bound_mean_ql = read_csv("../ramsey_price_result/price_detail_results/current_info_extreme_bound_mean_ql.csv")
#current_info_extreme_bound_mean_fcl = read_csv("../ramsey_price_result/price_detail_results/current_info_extreme_bound_mean_fcl.csv")

#current_info_extreme_bound_var_pl = read_csv("../ramsey_price_result/price_detail_results/current_info_extreme_bound_var_pl.csv")
#current_info_extreme_bound_var_ql = read_csv("../ramsey_price_result/price_detail_results/current_info_extreme_bound_var_ql.csv")
#current_info_extreme_bound_var_fcl = read_csv("../ramsey_price_result/price_detail_results/current_info_extreme_bound_var_fcl.csv")

pv0 = c(3.09,5.01,8.54,12.9,14.41,2,6,11,20,8.5,10.8,16.5,37,37)
statusquo = gen_p(pv0, q)
statusquo$step = "status_quo"

current_info_avg_bound_mean = rbind(current_info_avg_bound_mean_pl, current_info_avg_bound_mean_ql, current_info_avg_bound_mean_fcl)
current_info_avg_bound_var = rbind(current_info_avg_bound_var_pl, current_info_avg_bound_var_ql, current_info_avg_bound_var_fcl)

current_info_logr_bound_mean = rbind(current_info_logr_bound_mean_pl, current_info_logr_bound_mean_ql, current_info_logr_bound_mean_fcl)
current_info_logr_bound_var = rbind(current_info_logr_bound_var_pl, current_info_logr_bound_var_ql, current_info_logr_bound_var_fcl)

current_info_gamma05_bound_mean = rbind(current_info_gamma05_bound_mean_pl, current_info_gamma05_bound_mean_ql, current_info_gamma05_bound_mean_fcl)
current_info_gamma05_bound_var = rbind(current_info_gamma05_bound_var_pl, current_info_gamma05_bound_var_ql, current_info_gamma05_bound_var_fcl)

#current_info_extreme_bound_mean = rbind(current_info_extreme_bound_mean_pl, current_info_extreme_bound_mean_ql, current_info_extreme_bound_mean_fcl)
#current_info_extreme_bound_var = rbind(current_info_extreme_bound_var_pl, current_info_extreme_bound_var_ql, current_info_extreme_bound_var_fcl)


############################################################
################## avg_info_avg_bound_mean ##############################
#########################################################

avg_info_avg_bound_mean_info <- lapply(avg_info_avg_bound_mean, function(col) gen_p(col, q))

combined_avg_info_avg_bound_mean_info <- bind_rows(lapply(names(avg_info_avg_bound_mean_info), function(name) {
  df <- avg_info_avg_bound_mean_info[[name]]
  df$step <- as.numeric(name)/5-1  # Add a column to track the dataset
  return(df)
}))

# Ensure step is a factor for proper animation handling
combined_avg_info_avg_bound_mean_info$step <- factor(combined_avg_info_avg_bound_mean_info$step)

# Define color palette: Steps from brown (dry) to blue (wet), plus muted colors for static lines
num_steps <- length(unique(combined_avg_info_avg_bound_mean_info$step)) 
# Generate a gradient for steps from -1 to 0 (dark red to gray50)
gradient_neg <- colorRampPalette(c("darkred", "firebrick", "darkorange", "#e2a76d"))(ceiling((num_steps - 1) / 2))

# Generate a gradient for steps from 0 to 1 (gray50 to dark blue)
gradient_pos <- colorRampPalette(c("#c0d9e8","#7aa9d6", "royalblue", "darkblue"))(floor((num_steps - 1) / 2))

# Combine both gradients and assign gray50 to step 0
color_palette <- c(setNames(gradient_neg, as.character(seq(-1, -0.2, length.out = length(gradient_neg)))),
                   "0" = "#dcdcdc", 
                   setNames(gradient_pos, as.character(seq(0.2, 1, length.out = length(gradient_pos)))))


pvflat = c(rep(mean(avg_info_avg_bound_mean_pl$`5`),5 ), 2, 6, 11, 20, rep(mean(avg_info_avg_bound_mean_fcl$`5`),5 ))
flat = gen_p(pvflat, q)
flat$step = "flat"

ggplot() + 
  geom_line(data = combined_avg_info_avg_bound_mean_info, aes(x = q, y = total, color = step), linewidth = 1.5) +
  geom_line(data = flat, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dashed") +
  #geom_line(data = statusquo, aes(x = q, y = total), color = "#D8CFC4", linewidth = 1, linetype = "dashed") +
  xlab("Quantity (k Gallon)") +
  ylab("Payment ($)") +
  labs(color = "Δ E[Prcp]") +
  scale_color_manual(values = color_palette) +  
  guides(color = guide_legend(override.aes = list(size = 6))) + 
  ggtitle("Total Payment by Δ Mean Precipitation") +
  big_text

frames <- list()

# Now the loop can proceed without error
for (step_value in unique(combined_avg_info_avg_bound_mean_info$step)) {
  plot <- ggplot() + 
    geom_line(data = combined_avg_info_avg_bound_mean_info[combined_avg_info_avg_bound_mean_info$step == step_value, ], 
              aes(x = q, y = total, color = step), linewidth = 1.5) +
    geom_line(data = flat, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dashed") +
    xlab("Quantity (k Gallon)") +
    ylab("Payment ($)") +
    labs(color = "Δ E[Prcp]") +
    scale_color_manual(values = color_palette) +  
    guides(color = guide_legend(override.aes = list(size = 6))) + 
    ggtitle("Total Payment by Δ Mean Precipitation") +
    big_text+
    theme(plot.background = element_rect(fill = "white"))  # Set background to white
  
  temp_image <- tempfile(fileext = ".png")
  ggsave(temp_image, plot = plot, width = 10, height = 8, dpi = 300)
  frames[[length(frames) + 1]] <- image_read(temp_image)
}

# Combine all the images into a GIF
gif <- image_animate(image_join(frames), fps = 1, delay = 200)  # fps sets frames per second

# Save the GIF to a file
image_write(gif, "pics/avg_info_avg_bound_mean.gif")


############################################################
################## avg_info_avg_bound_var ##############################
#########################################################

avg_info_avg_bound_var_info <- lapply(avg_info_avg_bound_var, function(col) gen_p(col, q))

combined_avg_info_avg_bound_var_info <- bind_rows(lapply(names(avg_info_avg_bound_var_info), function(name) {
  df <- avg_info_avg_bound_var_info[[name]]
  df$step <- as.numeric(name)/2.5 -2  # Add a column to track the dataset
  return(df)
}))

pvflat = c(rep(mean(avg_info_avg_bound_var_pl$`5`),5 ), 2, 6, 11, 20, rep(mean(avg_info_avg_bound_var_fcl$`5`),5 ))
flat = gen_p(pvflat, q)
flat$step = "flat"

# Ensure step is a factor for proper animation handling
combined_avg_info_avg_bound_var_info$step <- factor(combined_avg_info_avg_bound_var_info$step)

# Define color palette: Steps from brown (dry) to blue (wet), plus muted colors for static lines
num_steps <- length(unique(combined_avg_info_avg_bound_var_info$step)) 
# Generate a gradient for steps from -1 to 0 (dark red to gray50)
gradient_neg <- colorRampPalette(c("darkred", "firebrick", "darkorange", "#e2a76d"))(ceiling((num_steps - 1) / 2))

# Generate a gradient for steps from 0 to 1 (gray50 to dark blue)
gradient_pos <- colorRampPalette(c("#c0d9e8","#7aa9d6", "royalblue", "darkblue"))(floor((num_steps - 1) / 2))

# Ensure step is a factor for proper animation handling
combined_avg_info_avg_bound_var_info$step <- factor(combined_avg_info_avg_bound_var_info$step)

# Combine both gradients and assign gray50 to step 0
color_palette <- c(setNames(gradient_neg, as.character(seq(-2, -0.4, length.out = length(gradient_neg)))),
                   "0" = "#dcdcdc", 
                   setNames(gradient_pos, as.character(seq(0.4, 2, length.out = length(gradient_pos)))))

combined_avg_info_avg_bound_var_info_small = combined_avg_info_avg_bound_var_info[abs(as.numeric(as.character(combined_avg_info_avg_bound_var_info$step)))<=1,]

ggplot() + 
  geom_line(data = combined_avg_info_avg_bound_var_info, aes(x = q, y = total, color = step), linewidth = 1.5) +
  geom_line(data = flat, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dashed") +
  #geom_line(data = statusquo, aes(x = q, y = total), color = "#D8CFC4", linewidth = 1, linetype = "dashed") +
  xlab("Quantity (k Gallon)") +
  ylab("Payment ($)") +
  labs(color = "Δ Min & Max[Prcp]") +
  scale_color_manual(values = color_palette) +  
  guides(color = guide_legend(override.aes = list(size = 6))) + 
  ggtitle("Total Payment by Δ Var Precipitation") +
  big_text

frames <- list()

# Now the loop can proceed without error
for (step_value in unique(combined_avg_info_avg_bound_var_info$step)) {
  plot <- ggplot() + 
    geom_line(data = combined_avg_info_avg_bound_var_info[combined_avg_info_avg_bound_var_info$step == step_value, ], 
              aes(x = q, y = total, color = step), linewidth = 1.5) +
    geom_line(data = flat, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dashed") +
    xlab("Quantity (k Gallon)") +
    ylab("Payment ($)") +
    labs(color = "Δ Min & Max[Prcp]") +
    scale_color_manual(values = color_palette) +  
    guides(color = guide_legend(override.aes = list(size = 6))) + 
    ggtitle("Total Payment by Δ Var Precipitation") +
    big_text+
    theme(plot.background = element_rect(fill = "white"))  # Set background to white
  
  temp_image <- tempfile(fileext = ".png")
  ggsave(temp_image, plot = plot, width = 10, height = 8, dpi = 300)
  frames[[length(frames) + 1]] <- image_read(temp_image)
}

# Combine all the images into a GIF
gif <- image_animate(image_join(frames), fps = 1, delay = 200)  # fps sets frames per second

# Save the GIF to a file
image_write(gif, "pics/avg_info_avg_bound_var.gif")




############################################################
################## current_info_avg_bound_mean ##############################
#########################################################

current_info_avg_bound_mean_info <- lapply(current_info_avg_bound_mean, function(col) gen_p(col, q))

combined_current_info_avg_bound_mean_info <- bind_rows(lapply(names(current_info_avg_bound_mean_info), function(name) {
  df <- current_info_avg_bound_mean_info[[name]]
  df$step <- as.numeric(name)/20-0.25  # Add a column to track the dataset
  return(df)
}))

# Ensure step is a factor for proper animation handling
combined_current_info_avg_bound_mean_info$step <- factor(combined_current_info_avg_bound_mean_info$step)

# Define color palette: Steps from brown (dry) to blue (wet), plus muted colors for static lines
num_steps <- length(unique(combined_current_info_avg_bound_mean_info$step)) 
# Generate a gradient for steps from -1 to 0 (dark red to gray50)
gradient_neg <- colorRampPalette(c("darkred", "firebrick", "darkorange", "#e2a76d"))(ceiling((num_steps - 1) / 2))

# Generate a gradient for steps from 0 to 1 (gray50 to dark blue)
gradient_pos <- colorRampPalette(c("#c0d9e8","#7aa9d6", "royalblue", "darkblue"))(floor((num_steps - 1) / 2))

# Combine both gradients and assign gray50 to step 0
color_palette <- c(setNames(gradient_neg, as.character(seq(-0.25, -0.05, length.out = length(gradient_neg)))),
                   "0" = "#dcdcdc", 
                   setNames(gradient_pos, as.character(seq(0.05, 0.25, length.out = length(gradient_pos)))))


pvflat = c(rep(mean(c(3.09,5.01,8.54,12.9,14.41)),5 ), 2, 6, 11, 20, rep(mean(c(8.5,10.8,16.5,37,37)),5 ))
#pvflat = c(rep(mean(current_info_avg_bound_mean_pl$`5`),5 ), 2, 6, 11, 20, rep(mean(current_info_avg_bound_mean_fcl$`5`),5 ))
flat = gen_p(pvflat, q)
flat$step = "flat"

ggplot() + 
  #geom_line(data = combined_current_info_avg_bound_mean_info[which(as.numeric(as.character((combined_current_info_avg_bound_mean_info$step)))>0),]
            #         , aes(x = q, y = total, color = step, linetype = "zero"), color = "#CFCFCF", linewidth = 1.5) +
   #         , aes(x = q, y = total, color = step), linewidth = 1.5) +
  geom_line(data = flat, aes(x = q, y = total, linetype = "linear"), color = "#6e6e6e", linewidth = 1) +
  geom_line(data = statusquo, aes(x = q, y = total, linetype = "statusquo"), color = "#080808", linewidth = 1) +
  #geom_line(data = flat, aes(x = q, y = total), color = "#6e6e6e", linewidth = 1, linetype = "dashed") +
  #geom_line(data = statusquo, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dotdash") +
  xlab("Quantity (k Gallon)") +
  ylab("Total Payment ($)") +
  labs(
    #color = "Δ E[Prcp] (Inch)", 
    linetype = NULL) +
  #scale_color_manual(values = color_palette) +
  scale_linetype_manual(values = c("zero" = "solid","linear" = "dashed", "statusquo" = "dotdash")) +
  coord_cartesian(ylim = c(0, 1750)) +
  guides(color = guide_legend(override.aes = list(size = 6))) + 
  #ggtitle("Total Payment") +
  #ggtitle("Total Payment by Δ Mean Precipitation") +
  big_text

ggplot() + 
  geom_line(data = combined_current_info_avg_bound_mean_info[which(as.numeric(as.character((combined_current_info_avg_bound_mean_info$step)))>0),]
   #         , aes(x = q, y = total, color = step, linetype = "zero"), color = "#CFCFCF", linewidth = 1.5) +
            , aes(x = q, y = total, color = step), linewidth = 1.5) +
  #geom_line(data = flat, aes(x = q, y = total, linetype = "linear"), color = "#6e6e6e", linewidth = 1) +
  #geom_line(data = statusquo, aes(x = q, y = total, linetype = "statusquo"), color = "#080808", linewidth = 1) +
  geom_line(data = flat, aes(x = q, y = total), color = "#6e6e6e", linewidth = 1, linetype = "dashed") +
  geom_line(data = statusquo, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dotdash") +
  xlab("Quantity (k Gallon)") +
  ylab("Total Payment ($)") +
  labs(
    #color = "Δ E[Prcp] (Inch)", 
       linetype = NULL) +
  scale_color_manual(values = color_palette) +
  #scale_linetype_manual(values = c("zero" = "solid","linear" = "dashed", "statusquo" = "dotdash")) +
  coord_cartesian(ylim = c(0, 1750)) +
  guides(color = guide_legend(override.aes = list(size = 6))) + 
  #ggtitle("Total Payment") +
  #ggtitle("Total Payment by Δ Mean Precipitation") +
  big_text

frames <- list()

weather_zero =combined_current_info_avg_bound_mean_info[combined_current_info_avg_bound_mean_info$step == 0, ]

# Now the loop can proceed without error
for (step_value in unique(combined_current_info_avg_bound_mean_info$step)) {
  plot <- ggplot() + 
    geom_line(data = combined_current_info_avg_bound_mean_info[combined_current_info_avg_bound_mean_info$step == step_value, ], 
              aes(x = q, y = total, color = step), linewidth = 1.5) +
    geom_line(data = flat, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dashed") +
    geom_line(data = weather_zero, aes(x = q, y = total), color = "#b0b0b0", linewidth = 1, linetype = "dashed") +
    xlab("Quantity (k Gallon)") +
    ylab("Payment ($)") +
    labs(color = "Δ E[Prcp]") +
    scale_color_manual(values = color_palette) +
    coord_cartesian(ylim = c(0, 1750)) +
    guides(color = guide_legend(override.aes = list(size = 6))) + 
    ggtitle("Total Payment by Δ Mean Precipitation") +
    big_text+
    theme(plot.background = element_rect(fill = "white"))  # Set background to white
  
  temp_image <- tempfile(fileext = ".png")
  ggsave(temp_image, plot = plot, width = 10, height = 8, dpi = 300)
  frames[[length(frames) + 1]] <- image_read(temp_image)
}

# Combine all the images into a GIF
gif <- image_animate(image_join(frames), fps = 1, delay = 200)  # fps sets frames per second

# Save the GIF to a file
image_write(gif, "pics/current_info_avg_bound_mean.gif")



############################################################
################## current_info_avg_bound_var ##############################
#########################################################

current_info_avg_bound_var_info <- lapply(current_info_avg_bound_var, function(col) gen_p(col, q))

combined_current_info_avg_bound_var_info <- bind_rows(lapply(names(current_info_avg_bound_var_info), function(name) {
  df <- current_info_avg_bound_var_info[[name]]
  df$step <- as.numeric(name)/20 +0.75  # Add a column to track the dataset
  return(df)
}))

# Ensure step is a factor for proper animation handling
combined_current_info_avg_bound_var_info$step <- factor(combined_current_info_avg_bound_var_info$step)

# Define color palette: Steps from brown (dry) to blue (wet), plus muted colors for static lines
num_steps <- length(unique(combined_current_info_avg_bound_var_info$step)) 
# Generate a gradient for steps from -1 to 0 (dark red to gray50)
gradient_neg <- colorRampPalette(c("darkred", "firebrick", "darkorange", "#e2a76d"))(ceiling((num_steps - 1) / 2))

# Generate a gradient for steps from 0 to 1 (gray50 to dark blue)
gradient_pos <- colorRampPalette(c("#c0d9e8","#7aa9d6", "royalblue", "darkblue"))(floor((num_steps - 1) / 2))

# Ensure step is a factor for proper animation handling
combined_current_info_avg_bound_var_info$step <- factor(combined_current_info_avg_bound_var_info$step)

# Combine both gradients and assign gray50 to step 0
color_palette <- c(setNames(gradient_neg, as.character(seq(0.75, 0.95, length.out = length(gradient_neg)))),
                   "1" = "#dcdcdc", 
                   setNames(gradient_pos, as.character(seq(1.05, 1.25, length.out = length(gradient_pos)))))

pvflat = c(rep(mean(c(3.09,5.01,8.54,12.9,14.41)),5 ), 2, 6, 11, 20, rep(mean(c(8.5,10.8,16.5,37,37)),5 ))
#pvflat = c(rep(mean(current_info_avg_bound_mean_pl$`5`),5 ), 2, 6, 11, 20, rep(mean(current_info_avg_bound_mean_fcl$`5`),5 ))
flat = gen_p(pvflat, q)
flat$step = "flat"

ggplot() + 
  geom_line(data = combined_current_info_avg_bound_var_info[which(as.numeric(as.character((combined_current_info_avg_bound_var_info$step)))>1),]
            , aes(x = q, y = total, color = step), linewidth = 1.5) +
  geom_line(data = flat, aes(x = q, y = total), color = "#6e6e6e", linewidth = 1, linetype = "dashed") +
  geom_line(data = statusquo, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dotdash") +
  xlab("Quantity (k Gallon)") +
  ylab("Payment ($)") +
  labs(color = "SD Ratio") +
  scale_color_manual(values = color_palette) +
  coord_cartesian(ylim = c(0, 1750)) +
  guides(color = guide_legend(override.aes = list(size = 6))) + 
  #ggtitle("Total Payment by Δ Var Precipitation") +
  big_text

frames <- list()

weather_zero =combined_current_info_avg_bound_var_info[combined_current_info_avg_bound_var_info$step == 0, ]

# Now the loop can proceed without error
for (step_value in unique(combined_current_info_avg_bound_var_info$step)) {
  plot <- ggplot() + 
    geom_line(data = combined_current_info_avg_bound_var_info[combined_current_info_avg_bound_var_info$step == step_value, ], 
              aes(x = q, y = total, color = step), linewidth = 1.5) +
    geom_line(data = flat, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dashed") +
    geom_line(data = weather_zero, aes(x = q, y = total), color = "#b0b0b0", linewidth = 1, linetype = "dashed") +
    xlab("Quantity (k Gallon)") +
    ylab("Payment ($)") +
    labs(color = "Δ Min & Max[Prcp]") +
    scale_color_manual(values = color_palette) +  
    guides(color = guide_legend(override.aes = list(size = 6))) + 
    ggtitle("Total Payment by Δ Var Precipitation") +
    coord_cartesian(ylim = c(0, 1750)) +
    big_text+
    theme(plot.background = element_rect(fill = "white"))  # Set background to white
  
  temp_image <- tempfile(fileext = ".png")
  ggsave(temp_image, plot = plot, width = 10, height = 8, dpi = 300)
  frames[[length(frames) + 1]] <- image_read(temp_image)
}

# Combine all the images into a GIF
gif <- image_animate(image_join(frames), fps = 1, delay = 200)  # fps sets frames per second

# Save the GIF to a file
image_write(gif, "pics/current_info_avg_bound_var.gif")


############################################################
################## current_info_extreme_bound_mean ##############################
#########################################################

current_info_extreme_bound_mean_info <- lapply(current_info_extreme_bound_mean, function(col) gen_p(col, q))

combined_current_info_extreme_bound_mean_info <- bind_rows(lapply(names(current_info_extreme_bound_mean_info), function(name) {
  df <- current_info_extreme_bound_mean_info[[name]]
  df$step <- as.numeric(name)/20-0.25  # Add a column to track the dataset
  return(df)
}))

# Ensure step is a factor for proper animation handling
combined_current_info_extreme_bound_mean_info$step <- factor(combined_current_info_extreme_bound_mean_info$step)

# Define color palette: Steps from brown (dry) to blue (wet), plus muted colors for static lines
num_steps <- length(unique(combined_current_info_extreme_bound_mean_info$step)) 
# Generate a gradient for steps from -1 to 0 (dark red to gray50)
gradient_neg <- colorRampPalette(c("darkred", "firebrick", "darkorange", "#e2a76d"))(ceiling((num_steps - 1) / 2))

# Generate a gradient for steps from 0 to 1 (gray50 to dark blue)
gradient_pos <- colorRampPalette(c("#c0d9e8","#7aa9d6", "royalblue", "darkblue"))(floor((num_steps - 1) / 2))

# Combine both gradients and assign gray50 to step 0
color_palette <- c(setNames(gradient_neg, as.character(seq(-0.25, -0.05, length.out = length(gradient_neg)))),
                   "0" = "#dcdcdc", 
                   setNames(gradient_pos, as.character(seq(0.05, 0.25, length.out = length(gradient_pos)))))


pvflat = c(rep(mean(current_info_extreme_bound_mean_pl$`5`),5 ), 2, 6, 11, 20, rep(mean(current_info_extreme_bound_mean_fcl$`5`),5 ))
flat = gen_p(pvflat, q)
flat$step = "flat"

ggplot() + 
  geom_line(data = combined_current_info_extreme_bound_mean_info[which(as.numeric(as.character((combined_current_info_avg_bound_mean_info$step)))>=0),]
            , aes(x = q, y = total, color = step), linewidth = 1.5) +
  #geom_line(data = flat, aes(x = q, y = total, linetype = "linear"), color = "#6e6e6e", linewidth = 1) +
  #geom_line(data = statusquo, aes(x = q, y = total, linetype = "statusquo"), color = "#080808", linewidth = 1) +
  geom_line(data = flat, aes(x = q, y = total), color = "#6e6e6e", linewidth = 1, linetype = "dashed") +
  geom_line(data = statusquo, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dotdash") +
  xlab("Quantity (k Gallon)") +
  ylab("Total Payment ($)") +
  labs(color = "Δ E[Prcp] (Inch)", linetype = NULL) +
  scale_color_manual(values = color_palette) +
  #scale_linetype_manual(values = c("linear" = "dashed", "statusquo" = "dotdash")) +
  coord_cartesian(ylim = c(0, 10500)) +
  guides(color = guide_legend(override.aes = list(size = 6))) + 
  #ggtitle("Total Payment") +
  #ggtitle("Total Payment by Δ Mean Precipitation") +
  big_text

frames <- list()

weather_zero =combined_current_info_extreme_bound_mean_info[combined_current_info_extreme_bound_mean_info$step == 0, ]

# Now the loop can proceed without error
for (step_value in unique(combined_current_info_extreme_bound_mean_info$step)) {
  plot <- ggplot() + 
    geom_line(data = combined_current_info_extreme_bound_mean_info[combined_current_info_extreme_bound_mean_info$step == step_value, ], 
              aes(x = q, y = total, color = step), linewidth = 1.5) +
    geom_line(data = flat, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dashed") +
    geom_line(data = weather_zero, aes(x = q, y = total), color = "#b0b0b0", linewidth = 1, linetype = "dashed") +
    xlab("Quantity (k Gallon)") +
    ylab("Payment ($)") +
    labs(color = "Δ E[Prcp]") +
    scale_color_manual(values = color_palette) +
    coord_cartesian(ylim = c(0, 1750)) +
    guides(color = guide_legend(override.aes = list(size = 6))) + 
    ggtitle("Total Payment by Δ Mean Precipitation") +
    big_text+
    theme(plot.background = element_rect(fill = "white"))  # Set background to white
  
  temp_image <- tempfile(fileext = ".png")
  ggsave(temp_image, plot = plot, width = 10, height = 8, dpi = 300)
  frames[[length(frames) + 1]] <- image_read(temp_image)
}

# Combine all the images into a GIF
gif <- image_animate(image_join(frames), fps = 1, delay = 200)  # fps sets frames per second

# Save the GIF to a file
image_write(gif, "pics/current_info_extreme_bound_mean.gif")



############################################################
################## current_info_extreme_bound_var ##############################
#########################################################

current_info_extreme_bound_var_info <- lapply(current_info_extreme_bound_var, function(col) gen_p(col, q))

combined_current_info_extreme_bound_var_info <- bind_rows(lapply(names(current_info_extreme_bound_var_info), function(name) {
  df <- current_info_extreme_bound_var_info[[name]]
  df$step <- as.numeric(name)/20 +0.75  # Add a column to track the dataset
  return(df)
}))

# Ensure step is a factor for proper animation handling
combined_current_info_extreme_bound_var_info$step <- factor(combined_current_info_extreme_bound_var_info$step)

# Define color palette: Steps from brown (dry) to blue (wet), plus muted colors for static lines
num_steps <- length(unique(combined_current_info_extreme_bound_var_info$step)) 
# Generate a gradient for steps from -1 to 0 (dark red to gray50)
gradient_neg <- colorRampPalette(c("darkred", "firebrick", "darkorange", "#e2a76d"))(ceiling((num_steps - 1) / 2))

# Generate a gradient for steps from 0 to 1 (gray50 to dark blue)
gradient_pos <- colorRampPalette(c("#c0d9e8","#7aa9d6", "royalblue", "darkblue"))(floor((num_steps - 1) / 2))

# Ensure step is a factor for proper animation handling
combined_current_info_extreme_bound_var_info$step <- factor(combined_current_info_extreme_bound_var_info$step)

# Combine both gradients and assign gray50 to step 0
color_palette <- c(setNames(gradient_neg, as.character(seq(0.75, 0.95, length.out = length(gradient_neg)))),
                   "1" = "#dcdcdc", 
                   setNames(gradient_pos, as.character(seq(1.05, 1.25, length.out = length(gradient_pos)))))

combined_current_info_extreme_bound_var_info_small = combined_current_info_extreme_bound_var_info[abs(as.numeric(as.character(combined_current_info_extreme_bound_var_info$step)))<=1,]

ggplot() + 
  geom_line(data = combined_current_info_extreme_bound_var_info[which(as.numeric(as.character((combined_current_info_extreme_bound_var_info$step)))>=1),]
            , aes(x = q, y = total, color = step), linewidth = 1.5) +
  geom_line(data = flat, aes(x = q, y = total), color = "#6e6e6e", linewidth = 1, linetype = "dashed") +
  geom_line(data = statusquo, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dotdash") +
  xlab("Quantity (k Gallon)") +
  ylab("Payment ($)") +
  labs(color = "SD Ratio") +
  scale_color_manual(values = color_palette) +
  coord_cartesian(ylim = c(0, 10500)) +
  guides(color = guide_legend(override.aes = list(size = 6))) + 
  #ggtitle("Total Payment by Δ Var Precipitation") +
  big_text

frames <- list()

weather_zero =combined_current_info_extreme_bound_var_info[combined_current_info_extreme_bound_var_info$step == 0, ]

# Now the loop can proceed without error
for (step_value in unique(combined_current_info_extreme_bound_var_info$step)) {
  plot <- ggplot() + 
    geom_line(data = combined_current_info_extreme_bound_var_info[combined_current_info_extreme_bound_var_info$step == step_value, ], 
              aes(x = q, y = total, color = step), linewidth = 1.5) +
    geom_line(data = flat, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dashed") +
    geom_line(data = weather_zero, aes(x = q, y = total), color = "#b0b0b0", linewidth = 1, linetype = "dashed") +
    xlab("Quantity (k Gallon)") +
    ylab("Payment ($)") +
    labs(color = "Δ Min & Max[Prcp]") +
    scale_color_manual(values = color_palette) +  
    guides(color = guide_legend(override.aes = list(size = 6))) + 
    ggtitle("Total Payment by Δ Var Precipitation") +
    coord_cartesian(ylim = c(0, 1750)) +
    big_text+
    theme(plot.background = element_rect(fill = "white"))  # Set background to white
  
  temp_image <- tempfile(fileext = ".png")
  ggsave(temp_image, plot = plot, width = 10, height = 8, dpi = 300)
  frames[[length(frames) + 1]] <- image_read(temp_image)
}

# Combine all the images into a GIF
gif <- image_animate(image_join(frames), fps = 1, delay = 200)  # fps sets frames per second

# Save the GIF to a file
image_write(gif, "pics/current_info_extreme_bound_var.gif")

############################################################
################## current_info_logr_bound_mean ##############################
#########################################################

current_info_logr_bound_mean_info <- lapply(current_info_logr_bound_mean, function(col) gen_p(col, q))

combined_current_info_logr_bound_mean_info <- bind_rows(lapply(names(current_info_logr_bound_mean_info), function(name) {
  df <- current_info_logr_bound_mean_info[[name]]
  df$step <- as.numeric(name)/20-0.25  # Add a column to track the dataset
  return(df)
}))

# Ensure step is a factor for proper animation handling
combined_current_info_logr_bound_mean_info$step <- factor(combined_current_info_logr_bound_mean_info$step)

# Define color palette: Steps from brown (dry) to blue (wet), plus muted colors for static lines
num_steps <- length(unique(combined_current_info_logr_bound_mean_info$step)) 
# Generate a gradient for steps from -1 to 0 (dark red to gray50)
gradient_neg <- colorRampPalette(c("darkred", "firebrick", "darkorange", "#e2a76d"))(ceiling((num_steps - 1) / 2))

# Generate a gradient for steps from 0 to 1 (gray50 to dark blue)
gradient_pos <- colorRampPalette(c("#c0d9e8","#7aa9d6", "royalblue", "darkblue"))(floor((num_steps - 1) / 2))

# Combine both gradients and assign gray50 to step 0
color_palette <- c(setNames(gradient_neg, as.character(seq(-0.25, -0.05, length.out = length(gradient_neg)))),
                   "0" = "#dcdcdc", 
                   setNames(gradient_pos, as.character(seq(0.05, 0.25, length.out = length(gradient_pos)))))


pvflat = c(rep(mean(c(3.09,5.01,8.54,12.9,14.41)),5 ), 2, 6, 11, 20, rep(mean(c(8.5,10.8,16.5,37,37)),5 ))
#pvflat = c(rep(mean(current_info_avg_bound_mean_pl$`5`),5 ), 2, 6, 11, 20, rep(mean(current_info_avg_bound_mean_fcl$`5`),5 ))
flat = gen_p(pvflat, q)
flat$step = "flat"

ggplot() + 
  geom_line(data = combined_current_info_logr_bound_mean_info[which(as.numeric(as.character((combined_current_info_logr_bound_mean_info$step)))<0),]
            , aes(x = q, y = total, color = step), linewidth = 1.5) +
  #geom_line(data = flat, aes(x = q, y = total, linetype = "linear"), color = "#6e6e6e", linewidth = 1) +
  #geom_line(data = statusquo, aes(x = q, y = total, linetype = "statusquo"), color = "#080808", linewidth = 1) +
  geom_line(data = flat, aes(x = q, y = total), color = "#6e6e6e", linewidth = 1, linetype = "dashed") +
  geom_line(data = statusquo, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dotdash") +
  xlab("Quantity (k Gallon)") +
  ylab("Total Payment ($)") +
  labs(color = "Δ E[Prcp] (Inch)", linetype = NULL) +
  scale_color_manual(values = color_palette) +
  #scale_linetype_manual(values = c("linear" = "dashed", "statusquo" = "dotdash")) +
  coord_cartesian(ylim = c(0, 2500)) +
  guides(color = guide_legend(override.aes = list(size = 6))) + 
  #ggtitle("Total Payment") +
  #ggtitle("Total Payment by Δ Mean Precipitation") +
  big_text

frames <- list()

weather_zero =combined_current_info_logr_bound_mean_info[combined_current_info_logr_bound_mean_info$step == 0, ]

# Now the loop can proceed without error
for (step_value in unique(combined_current_info_logr_bound_mean_info$step)) {
  plot <- ggplot() + 
    geom_line(data = combined_current_info_logr_bound_mean_info[combined_current_info_logr_bound_mean_info$step == step_value, ], 
              aes(x = q, y = total, color = step), linewidth = 1.5) +
    geom_line(data = flat, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dashed") +
    geom_line(data = weather_zero, aes(x = q, y = total), color = "#b0b0b0", linewidth = 1, linetype = "dashed") +
    xlab("Quantity (k Gallon)") +
    ylab("Payment ($)") +
    labs(color = "Δ E[Prcp]") +
    scale_color_manual(values = color_palette) +
    coord_cartesian(ylim = c(0, 1750)) +
    guides(color = guide_legend(override.aes = list(size = 6))) + 
    ggtitle("Total Payment by Δ Mean Precipitation") +
    big_text+
    theme(plot.background = element_rect(fill = "white"))  # Set background to white
  
  temp_image <- tempfile(fileext = ".png")
  ggsave(temp_image, plot = plot, width = 10, height = 8, dpi = 300)
  frames[[length(frames) + 1]] <- image_read(temp_image)
}

# Combine all the images into a GIF
gif <- image_animate(image_join(frames), fps = 1, delay = 200)  # fps sets frames per second

# Save the GIF to a file
image_write(gif, "pics/current_info_logr_bound_mean.gif")



############################################################
################## current_info_logr_bound_var ##############################
#########################################################

current_info_logr_bound_var_info <- lapply(current_info_logr_bound_var, function(col) gen_p(col, q))

combined_current_info_logr_bound_var_info <- bind_rows(lapply(names(current_info_logr_bound_var_info), function(name) {
  df <- current_info_logr_bound_var_info[[name]]
  df$step <- as.numeric(name)/20 +0.75  # Add a column to track the dataset
  return(df)
}))

# Ensure step is a factor for proper animation handling
combined_current_info_logr_bound_var_info$step <- factor(combined_current_info_logr_bound_var_info$step)

# Define color palette: Steps from brown (dry) to blue (wet), plus muted colors for static lines
num_steps <- length(unique(combined_current_info_logr_bound_var_info$step)) 
# Generate a gradient for steps from -1 to 0 (dark red to gray50)
gradient_neg <- colorRampPalette(c("darkred", "firebrick", "darkorange", "#e2a76d"))(ceiling((num_steps - 1) / 2))

# Generate a gradient for steps from 0 to 1 (gray50 to dark blue)
gradient_pos <- colorRampPalette(c("#c0d9e8","#7aa9d6", "royalblue", "darkblue"))(floor((num_steps - 1) / 2))

# Ensure step is a factor for proper animation handling
combined_current_info_logr_bound_var_info$step <- factor(combined_current_info_logr_bound_var_info$step)

# Combine both gradients and assign gray50 to step 0
color_palette <- c(setNames(gradient_neg, as.character(seq(0.75, 0.95, length.out = length(gradient_neg)))),
                   "1" = "#dcdcdc", 
                   setNames(gradient_pos, as.character(seq(1.05, 1.25, length.out = length(gradient_pos)))))

combined_current_info_logr_bound_var_info_small = combined_current_info_logr_bound_var_info[abs(as.numeric(as.character(combined_current_info_logr_bound_var_info$step)))<=1,]

pvflat = c(rep(mean(c(3.09,5.01,8.54,12.9,14.41)),5 ), 2, 6, 11, 20, rep(mean(c(8.5,10.8,16.5,37,37)),5 ))
#pvflat = c(rep(mean(current_info_avg_bound_mean_pl$`5`),5 ), 2, 6, 11, 20, rep(mean(current_info_avg_bound_mean_fcl$`5`),5 ))
flat = gen_p(pvflat, q)
flat$step = "flat"

ggplot() + 
  geom_line(data = combined_current_info_logr_bound_var_info[which(as.numeric(as.character((combined_current_info_logr_bound_var_info$step)))>1),]
            , aes(x = q, y = total, color = step), linewidth = 1.5) +
  geom_line(data = flat, aes(x = q, y = total), color = "#6e6e6e", linewidth = 1, linetype = "dashed") +
  geom_line(data = statusquo, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dotdash") +
  xlab("Quantity (k Gallon)") +
  ylab("Payment ($)") +
  labs(color = "SD Ratio") +
  scale_color_manual(values = color_palette) +
  coord_cartesian(ylim = c(0, 2500)) +
  guides(color = guide_legend(override.aes = list(size = 6))) + 
  #ggtitle("Total Payment by Δ Var Precipitation") +
  big_text

frames <- list()

weather_zero =combined_current_info_logr_bound_var_info[combined_current_info_logr_bound_var_info$step == 0, ]

# Now the loop can proceed without error
for (step_value in unique(combined_current_info_logr_bound_var_info$step)) {
  plot <- ggplot() + 
    geom_line(data = combined_current_info_logr_bound_var_info[combined_current_info_logr_bound_var_info$step == step_value, ], 
              aes(x = q, y = total, color = step), linewidth = 1.5) +
    geom_line(data = flat, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dashed") +
    geom_line(data = weather_zero, aes(x = q, y = total), color = "#b0b0b0", linewidth = 1, linetype = "dashed") +
    xlab("Quantity (k Gallon)") +
    ylab("Payment ($)") +
    labs(color = "Δ Min & Max[Prcp]") +
    scale_color_manual(values = color_palette) +  
    guides(color = guide_legend(override.aes = list(size = 6))) + 
    ggtitle("Total Payment by Δ Var Precipitation") +
    coord_cartesian(ylim = c(0, 1750)) +
    big_text+
    theme(plot.background = element_rect(fill = "white"))  # Set background to white
  
  temp_image <- tempfile(fileext = ".png")
  ggsave(temp_image, plot = plot, width = 10, height = 8, dpi = 300)
  frames[[length(frames) + 1]] <- image_read(temp_image)
}

# Combine all the images into a GIF
gif <- image_animate(image_join(frames), fps = 1, delay = 200)  # fps sets frames per second

# Save the GIF to a file
image_write(gif, "pics/current_info_logr_bound_var.gif")

############################################################
################## current_info_gamma05_bound_mean ##############################
#########################################################

current_info_gamma05_bound_mean_info <- lapply(current_info_gamma05_bound_mean, function(col) gen_p(col, q))

combined_current_info_gamma05_bound_mean_info <- bind_rows(lapply(names(current_info_gamma05_bound_mean_info), function(name) {
  df <- current_info_gamma05_bound_mean_info[[name]]
  df$step <- as.numeric(name)/20-0.25  # Add a column to track the dataset
  return(df)
}))

# Ensure step is a factor for proper animation handling
combined_current_info_gamma05_bound_mean_info$step <- factor(combined_current_info_gamma05_bound_mean_info$step)

# Define color palette: Steps from brown (dry) to blue (wet), plus muted colors for static lines
num_steps <- length(unique(combined_current_info_gamma05_bound_mean_info$step)) 
# Generate a gradient for steps from -1 to 0 (dark red to gray50)
gradient_neg <- colorRampPalette(c("darkred", "firebrick", "darkorange", "#e2a76d"))(ceiling((num_steps - 1) / 2))

# Generate a gradient for steps from 0 to 1 (gray50 to dark blue)
gradient_pos <- colorRampPalette(c("#c0d9e8","#7aa9d6", "royalblue", "darkblue"))(floor((num_steps - 1) / 2))

# Combine both gradients and assign gray50 to step 0
color_palette <- c(setNames(gradient_neg, as.character(seq(-0.25, -0.05, length.out = length(gradient_neg)))),
                   "0" = "#dcdcdc", 
                   setNames(gradient_pos, as.character(seq(0.05, 0.25, length.out = length(gradient_pos)))))


pvflat = c(rep(mean(c(3.09,5.01,8.54,12.9,14.41)),5 ), 2, 6, 11, 20, rep(mean(c(8.5,10.8,16.5,37,37)),5 ))
#pvflat = c(rep(mean(current_info_avg_bound_mean_pl$`5`),5 ), 2, 6, 11, 20, rep(mean(current_info_avg_bound_mean_fcl$`5`),5 ))
flat = gen_p(pvflat, q)
flat$step = "flat"

ggplot() + 
  geom_line(data = combined_current_info_gamma05_bound_mean_info[which(as.numeric(as.character((combined_current_info_gamma05_bound_mean_info$step)))>0),]
            , aes(x = q, y = total, color = step), linewidth = 1.5) +
  #geom_line(data = flat, aes(x = q, y = total, linetype = "linear"), color = "#6e6e6e", linewidth = 1) +
  #geom_line(data = statusquo, aes(x = q, y = total, linetype = "statusquo"), color = "#080808", linewidth = 1) +
  geom_line(data = flat, aes(x = q, y = total), color = "#6e6e6e", linewidth = 1, linetype = "dashed") +
  geom_line(data = statusquo, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dotdash") +
  xlab("Quantity (k Gallon)") +
  ylab("Total Payment ($)") +
  labs(color = "Δ E[Prcp] (Inch)", linetype = NULL) +
  scale_color_manual(values = color_palette) +
  #scale_linetype_manual(values = c("linear" = "dashed", "statusquo" = "dotdash")) +
  coord_cartesian(ylim = c(0, 2500)) +
  guides(color = guide_legend(override.aes = list(size = 6))) + 
  #ggtitle("Total Payment") +
  #ggtitle("Total Payment by Δ Mean Precipitation") +
  big_text

frames <- list()

weather_zero =combined_current_info_gamma05_bound_mean_info[combined_current_info_gamma05_bound_mean_info$step == 0, ]

# Now the loop can proceed without error
for (step_value in unique(combined_current_info_gamma05_bound_mean_info$step)) {
  plot <- ggplot() + 
    geom_line(data = combined_current_info_gamma05_bound_mean_info[combined_current_info_gamma05_bound_mean_info$step == step_value, ], 
              aes(x = q, y = total, color = step), linewidth = 1.5) +
    geom_line(data = flat, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dashed") +
    geom_line(data = weather_zero, aes(x = q, y = total), color = "#b0b0b0", linewidth = 1, linetype = "dashed") +
    xlab("Quantity (k Gallon)") +
    ylab("Payment ($)") +
    labs(color = "Δ E[Prcp]") +
    scale_color_manual(values = color_palette) +
    coord_cartesian(ylim = c(0, 1750)) +
    guides(color = guide_legend(override.aes = list(size = 6))) + 
    ggtitle("Total Payment by Δ Mean Precipitation") +
    big_text+
    theme(plot.background = element_rect(fill = "white"))  # Set background to white
  
  temp_image <- tempfile(fileext = ".png")
  ggsave(temp_image, plot = plot, width = 10, height = 8, dpi = 300)
  frames[[length(frames) + 1]] <- image_read(temp_image)
}

# Combine all the images into a GIF
gif <- image_animate(image_join(frames), fps = 1, delay = 200)  # fps sets frames per second

# Save the GIF to a file
image_write(gif, "pics/current_info_gamma05_bound_mean.gif")



############################################################
################## current_info_gamma05_bound_var ##############################
#########################################################

current_info_gamma05_bound_var_info <- lapply(current_info_gamma05_bound_var, function(col) gen_p(col, q))

combined_current_info_gamma05_bound_var_info <- bind_rows(lapply(names(current_info_gamma05_bound_var_info), function(name) {
  df <- current_info_gamma05_bound_var_info[[name]]
  df$step <- as.numeric(name)/20 +0.75  # Add a column to track the dataset
  return(df)
}))

# Ensure step is a factor for proper animation handling
combined_current_info_gamma05_bound_var_info$step <- factor(combined_current_info_gamma05_bound_var_info$step)

# Define color palette: Steps from brown (dry) to blue (wet), plus muted colors for static lines
num_steps <- length(unique(combined_current_info_gamma05_bound_var_info$step)) 
# Generate a gradient for steps from -1 to 0 (dark red to gray50)
gradient_neg <- colorRampPalette(c("darkred", "firebrick", "darkorange", "#e2a76d"))(ceiling((num_steps - 1) / 2))

# Generate a gradient for steps from 0 to 1 (gray50 to dark blue)
gradient_pos <- colorRampPalette(c("#c0d9e8","#7aa9d6", "royalblue", "darkblue"))(floor((num_steps - 1) / 2))

# Ensure step is a factor for proper animation handling
combined_current_info_gamma05_bound_var_info$step <- factor(combined_current_info_gamma05_bound_var_info$step)

# Combine both gradients and assign gray50 to step 0
color_palette <- c(setNames(gradient_neg, as.character(seq(0.75, 0.95, length.out = length(gradient_neg)))),
                   "1" = "#dcdcdc", 
                   setNames(gradient_pos, as.character(seq(1.05, 1.25, length.out = length(gradient_pos)))))

combined_current_info_gamma05_bound_var_info_small = combined_current_info_gamma05_bound_var_info[abs(as.numeric(as.character(combined_current_info_gamma05_bound_var_info$step)))<=1,]

pvflat = c(rep(mean(c(3.09,5.01,8.54,12.9,14.41)),5 ), 2, 6, 11, 20, rep(mean(c(8.5,10.8,16.5,37,37)),5 ))
#pvflat = c(rep(mean(current_info_avg_bound_mean_pl$`5`),5 ), 2, 6, 11, 20, rep(mean(current_info_avg_bound_mean_fcl$`5`),5 ))
flat = gen_p(pvflat, q)
flat$step = "flat"

ggplot() + 
  geom_line(data = combined_current_info_gamma05_bound_var_info[which(as.numeric(as.character((combined_current_info_gamma05_bound_var_info$step)))>1),]
            , aes(x = q, y = total, color = step), linewidth = 1.5) +
  geom_line(data = flat, aes(x = q, y = total), color = "#6e6e6e", linewidth = 1, linetype = "dashed") +
  geom_line(data = statusquo, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dotdash") +
  xlab("Quantity (k Gallon)") +
  ylab("Payment ($)") +
  labs(color = "SD Ratio") +
  scale_color_manual(values = color_palette) +
  coord_cartesian(ylim = c(0, 2000)) +
  guides(color = guide_legend(override.aes = list(size = 6))) + 
  #ggtitle("Total Payment by Δ Var Precipitation") +
  big_text

frames <- list()

weather_zero =combined_current_info_gamma05_bound_var_info[combined_current_info_gamma05_bound_var_info$step == 0, ]

# Now the loop can proceed without error
for (step_value in unique(combined_current_info_gamma05_bound_var_info$step)) {
  plot <- ggplot() + 
    geom_line(data = combined_current_info_gamma05_bound_var_info[combined_current_info_gamma05_bound_var_info$step == step_value, ], 
              aes(x = q, y = total, color = step), linewidth = 1.5) +
    geom_line(data = flat, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dashed") +
    geom_line(data = weather_zero, aes(x = q, y = total), color = "#b0b0b0", linewidth = 1, linetype = "dashed") +
    xlab("Quantity (k Gallon)") +
    ylab("Payment ($)") +
    labs(color = "Δ Min & Max[Prcp]") +
    scale_color_manual(values = color_palette) +  
    guides(color = guide_legend(override.aes = list(size = 6))) + 
    ggtitle("Total Payment by Δ Var Precipitation") +
    coord_cartesian(ylim = c(0, 1750)) +
    big_text+
    theme(plot.background = element_rect(fill = "white"))  # Set background to white
  
  temp_image <- tempfile(fileext = ".png")
  ggsave(temp_image, plot = plot, width = 10, height = 8, dpi = 300)
  frames[[length(frames) + 1]] <- image_read(temp_image)
}

# Combine all the images into a GIF
gif <- image_animate(image_join(frames), fps = 1, delay = 200)  # fps sets frames per second

# Save the GIF to a file
image_write(gif, "pics/current_info_gamma05_bound_var.gif")

#####
mp1 = ggplot() + 
  geom_step(data = noweather_history, aes(x = q, y = mp, color = "0", linetype = "0"), size = 1.5) +
  geom_step(data = avg_exp, aes(x = q, y = mp, color = "1", linetype = "1"), size = 1.5) +
  geom_step(data = worst_extreme, aes(x = q, y = mp, color = "1.5", linetype = "1.5"), size = 1.5) +
  geom_step(data = statuosquo, aes(x = q, y = mp, color = "og", linetype = "og"), size = 0.8) +
  xlab("Quantity (k Gallon)") +
  ylab("Price ($)") +
  labs(color = "Price & Type", linetype = "Price & Type") +
  scale_color_manual(
    values = c("0" = "#35618f", "1" = "#8c334c", "1.5" = "#2a8476", "og" = "#080808"),
    labels = c("0" = "0", "1" = "1", "1.5" = "1.5", "og" = "Status Quo")
  ) +
  scale_linetype_manual(
    values = c("0" = "solid", "1" = "longdash", "1.5" = "dotdash", "og" = "solid"),
    labels = c("0" = "0", "1" = "1", "1.5" = "1.5", "og" = "Status Quo")
  ) +
  guides(
    color = guide_legend(override.aes = list(size = 1.5), keywidth = 4, keyheight = 1),
    linetype = guide_legend(override.aes = list(size = 1.5), keywidth = 4, keyheight = 1)
  ) +
  ggtitle("Marginal Price")+big_text

A1 = ggplot() + 
  geom_step(data = noweather_history, aes(x = q, y = A, color = "0", linetype = "0"), size = 1.5) +
  geom_step(data = avg_exp, aes(x = q, y = A, color = "1", linetype = "1"), size = 1.5) +
  geom_step(data = worst_extreme, aes(x = q, y = A, color = "1.5", linetype = "1.5"), size = 1.5) +
  geom_step(data = statuosquo, aes(x = q, y = A, color = "og", linetype = "og"), size = 0.8) +
  xlab("Quantity (k Gallon)") +
  ylab("Price ($)") +
  labs(color = "Price & Type", linetype = "Price & Type") +
  scale_color_manual(
    values = c("0" = "#35618f", "1" = "#8c334c", "1.5" = "#2a8476", "og" = "#080808"),
    labels = c("0" = "0", "1" = "1", "1.5" = "1.5", "og" = "Status Quo")
  ) +
  scale_linetype_manual(
    values = c("0" = "solid", "1" = "longdash", "1.5" = "dotdash", "og" = "solid"),
    labels = c("0" = "0", "1" = "1", "1.5" = "1.5", "og" = "Status Quo")
  ) +
  guides(
    color = guide_legend(override.aes = list(size = 1.5), keywidth = 4, keyheight = 1),
    linetype = guide_legend(override.aes = list(size = 1.5), keywidth = 4, keyheight = 1)
  ) +
  ggtitle("Fixed Payment")+big_text

price_1 <- ggarrange(
  mp1,  A1,
  ncol = 2,                      # Arrange in 2 columns
  nrow = 1,                       # Arrange in 1 row
  common.legend = TRUE,                  # Single legend for both plots
  legend = "bottom"                      # Position the legend at the bottom
)

ggplot() + 
  geom_line(data = noweather_history, aes(x=q,y = ap, color = "0", linetype = "0"), size = 1) +
  geom_line(data = avg_exp, aes(x=q,y = ap, color = "1", linetype = "1"), size = 1) +
  geom_line(data = worst_extreme, aes(x=q,y = ap, color = "1.5", linetype = "1.5"), size = 1) +
  geom_line(data = statuosquo, aes(x=q, y = ap, color = "og", linetype = "og"), size = 0.8) +
  xlab("Quantity (k Gallon)")+
  ylab("Price ($)")+
  labs(color = "Price") +
  scale_color_manual(values = c("0" = "#35618f","1" = "#8c334c","1.5" = "#2a8476",
                                "og" = "#080808"
  ),
  labels = c(
    "0" = "0","1" = "1","1.5" = "1.5",
    "og" = "Status Quo"
  )) +
  scale_linetype_manual(values = c("0" = "solid","1" = "longdash","1.5" = "dotdash",
                                   "og" = "solid"
  ),
  labels = c(
    "0" = "0","1" = "1","1.5" = "1.5",
    "og" = "Status Quo"
  )) +
  guides(color = guide_legend(override.aes = list(size = 6))) + 
  ggtitle("Average Price")+
  big_text


###


create_abs_diff <- function(df) {
  # Ensure the input is a data frame
  if (!is.data.frame(df)) stop("Input must be a data frame")
  
  # Extract values where steps == 0
  r_step_agg_mean_0 <- df[which(df$steps == 0),]$r_step_agg_mean
  cs_step_agg_mean_0 <- df[which(df$steps == 0),]$cs_step_agg_mean
  r_step_agg_var_0 <- df[which(abs(df$s_steps - 0) < 1e-16),]$r_step_agg_var
  cs_step_agg_var_0 <- df[which(abs(df$s_steps - 0) < 1e-16),]$cs_step_agg_var
  
  # Assign these values to new columns
  df$r_step_agg_mean_0 <- r_step_agg_mean_0
  df$cs_step_agg_mean_0 <- cs_step_agg_mean_0
  df$r_step_agg_var_0 <- r_step_agg_var_0
  df$cs_step_agg_var_0 <- cs_step_agg_var_0
  
  # Calculate absolute differences
  df$r_step_agg_mean_abs_diff <- (df$r_step_agg_mean - df$r_step_agg_mean_0) / df$r_step_agg_mean_0
  df$cs_step_agg_mean_abs_diff <- (df$cs_step_agg_mean - df$cs_step_agg_mean_0) / abs(df$cs_step_agg_mean_0)
  df$r_step_agg_var_abs_diff <- (df$r_step_agg_var - df$r_step_agg_var_0) / df$r_step_agg_var_0
  df$cs_step_agg_var_abs_diff <- (df$cs_step_agg_var - df$cs_step_agg_var_0) / abs(df$cs_step_agg_var_0)
  rownames(df) <- NULL
  
  return(df)
}

############################################
########### Welfare Data #################
############################################

##### avg_info_avg_bound ####

avg_info_avg_bound_mean_cs = read_csv("avg_info_avg_bound_mean_cs.csv")
avg_info_avg_bound_mean_q = read_csv("avg_info_avg_bound_mean_q.csv")
avg_info_avg_bound_mean_r = read_csv("avg_info_avg_bound_mean_r.csv")

avg_info_avg_bound_var_cs = read_csv("avg_info_avg_bound_var_cs.csv")
avg_info_avg_bound_var_q = read_csv("avg_info_avg_bound_var_q.csv")
avg_info_avg_bound_var_r = read_csv("avg_info_avg_bound_var_r.csv")

avg_info_avg_bound_welfare = cbind(avg_info_avg_bound_mean_cs,avg_info_avg_bound_mean_q,avg_info_avg_bound_mean_r,
                                   avg_info_avg_bound_var_cs,avg_info_avg_bound_var_q,avg_info_avg_bound_var_r)
colnames(avg_info_avg_bound_welfare) = c("mean_cs", "mean_q", "mean_r","var_cs", "var_q", "var_r")

avg_info_avg_bound_welfare$steps = seq(-1, 1, by = 0.2)
avg_info_avg_bound_welfare$s_steps = seq(-2, 2, by = 0.4)

avg_info_avg_bound_welfare$mean_r_diff = (avg_info_avg_bound_welfare$mean_r - r_history_agg_0)/abs(r_history_agg_0)*100
avg_info_avg_bound_welfare$var_r_diff = (avg_info_avg_bound_welfare$var_r - r_history_agg_0)/abs(r_history_agg_0)*100

avg_info_avg_bound_welfare$mean_q_diff = (avg_info_avg_bound_welfare$mean_q - q_history_agg_0)/abs(q_history_agg_0)*100
avg_info_avg_bound_welfare$var_q_diff = (avg_info_avg_bound_welfare$var_q - q_history_agg_0)/abs(q_history_agg_0)*100

avg_info_avg_bound_welfare$mean_cs_diff = (avg_info_avg_bound_welfare$mean_cs - cs_history_agg_0)/abs(cs_history_agg_0)*100
avg_info_avg_bound_welfare$var_cs_diff = (avg_info_avg_bound_welfare$var_cs - cs_history_agg_0)/abs(cs_history_agg_0)*100

avg_info_avg_bound_welfare_small = avg_info_avg_bound_welfare[abs(avg_info_avg_bound_welfare$steps)<=0.5,]
library(scales)  # For label formatting

r_avg_info_avg_bound = ggplot(data = avg_info_avg_bound_welfare_small) +
  geom_line(aes(x = steps, y = mean_r_diff, color = "mean", linetype = "mean"), linewidth = 2)+
  geom_line(aes(x = steps, y = var_r_diff, color = "var", linetype = "var"), linewidth = 2)+
  scale_color_manual(
    values = c("mean" = "#35618f", "var" = "#8c334c"),
    labels = c("mean" = "mean", "var" = "var")
  ) +
  scale_linetype_manual(
    values = c("mean" = "solid", "var" = "longdash"),
    labels = c("mean" = "mean", "var" = "var")
  ) +
  scale_x_continuous(
      name = "E[Prcp]",
      sec.axis = sec_axis(~ rescale(., from = range(avg_info_avg_bound_welfare$steps), to = range(avg_info_avg_bound_welfare$s_steps )), 
                          name = "Min & Max[Prcp]")
    ) +
  coord_cartesian(xlim = c(min(range(avg_info_avg_bound_welfare_small$steps))
                           , max(range(avg_info_avg_bound_welfare_small$steps)))
  ) +
  labs(
    title = "Producer Surplus Change (%)",
    color = "",
    linetype = "",
    x = NULL,
    y = NULL
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )

cs_avg_info_avg_bound = ggplot(data = avg_info_avg_bound_welfare_small) +
  geom_line(aes(x = steps, y = mean_cs_diff, color = "mean", linetype = "mean"), linewidth = 2)+
  geom_line(aes(x = steps, y = var_cs_diff, color = "var", linetype = "var"), linewidth = 2)+
  scale_color_manual(
    values = c("mean" = "#35618f", "var" = "#8c334c"),
    labels = c("mean" = "mean", "var" = "var")
  ) +
  scale_linetype_manual(
    values = c("mean" = "solid", "var" = "longdash"),
    labels = c("mean" = "mean", "var" = "var")
  ) +
  scale_x_continuous(
    name = "E[Prcp]",
    sec.axis = sec_axis(~ rescale(., from = range(avg_info_avg_bound_welfare$steps), to = range(avg_info_avg_bound_welfare$s_steps )), 
                        name = "Min & Max[Prcp]")
  ) +
  coord_cartesian(xlim = c(min(range(avg_info_avg_bound_welfare_small$steps))
                           , max(range(avg_info_avg_bound_welfare_small$steps)))
  ) +
  labs(
    title = "Consumer Surplus Change (%)",
    color = "",
    linetype = "",
    x = NULL,
    y = NULL
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )

q_avg_info_avg_bound = ggplot(data = avg_info_avg_bound_welfare_small) +
  geom_line(aes(x = steps, y = mean_q_diff, color = "mean", linetype = "mean"), linewidth = 2)+
  geom_line(aes(x = steps, y = var_q_diff, color = "var", linetype = "var"), linewidth = 2)+
  scale_color_manual(
    values = c("mean" = "#35618f", "var" = "#8c334c"),
    labels = c("mean" = "mean", "var" = "var")
  ) +
  scale_linetype_manual(
    values = c("mean" = "solid", "var" = "longdash"),
    labels = c("mean" = "mean", "var" = "var")
  ) +
  scale_x_continuous(
    name = "E[Prcp]",
    sec.axis = sec_axis(~ rescale(., from = range(avg_info_avg_bound_welfare$steps), to = range(avg_info_avg_bound_welfare$s_steps )), 
                        name = "Min & Max[Prcp]")
  ) +
  coord_cartesian(xlim = c(min(range(avg_info_avg_bound_welfare_small$steps))
                           , max(range(avg_info_avg_bound_welfare_small$steps)))
  ) +
  labs(
    title = "Total Quantity Change (%)",
    color = "",
    linetype = "",
    x = NULL,
    y = NULL
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )

ggplot(data = avg_info_avg_bound_welfare) +
  geom_line(aes(x = steps, y = mean_r_diff, color = "r"), linetype = "dotdash",linewidth = 2)+
  geom_line(aes(x = steps, y = mean_cs_diff, color = "cs"), linetype = "longdash", linewidth = 2)+
  geom_line(aes(x = steps, y = mean_q_diff, color = "q"), linetype = "solid", linewidth = 2)+
  scale_color_manual(
    values = c("r" = "#316387", "cs" = "#842411", "q" = "#256b33"),
    labels = c("r" = "PS", "cs" = "CS", "q" = "Q")
  ) +
  coord_cartesian(xlim = c(min(range(avg_info_avg_bound_welfare$steps))
                           , max(range(avg_info_avg_bound_welfare$steps)))
  ) +
  labs(
    title = "Welfare Change (%)",
    color = "Type",
    linetype = "Type",
    x = NULL,
    y = NULL
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 1.2) +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )


ggplot(data = avg_info_avg_bound_welfare) +
  geom_line(aes(x = s_steps, y = var_r_diff, color = "r"), linetype = "dotdash",linewidth = 2)+
  geom_line(aes(x = s_steps, y = var_cs_diff, color = "cs"), linetype = "longdash", linewidth = 2)+
  geom_line(aes(x = s_steps, y = var_q_diff, color = "q"), linetype = "solid", linewidth = 2)+
  scale_color_manual(
    values = c("r" = "#316387", "cs" = "#842411", "q" = "#256b33"),
    labels = c("r" = "PS", "cs" = "CS", "q" = "Q")
  ) +
  coord_cartesian(xlim = c(min(range(avg_info_avg_bound_welfare$s_steps))
                           , max(range(avg_info_avg_bound_welfare$s_steps)))
  ) +
  labs(
    title = "Welfare Change (%)",
    color = "Type",
    linetype = "Type",
    x = NULL,
    y = NULL
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 1.2) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 1.2) +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )


##### current_info_avg_bound ####

current_info_avg_bound_mean_cs = read_csv("current_info_avg_bound_mean_cs.csv")
current_info_avg_bound_mean_q = read_csv("current_info_avg_bound_mean_q.csv")
current_info_avg_bound_mean_r = read_csv("current_info_avg_bound_mean_r.csv")

current_info_avg_bound_var_cs = read_csv("current_info_avg_bound_var_cs.csv")
current_info_avg_bound_var_q = read_csv("current_info_avg_bound_var_q.csv")
current_info_avg_bound_var_r = read_csv("current_info_avg_bound_var_r.csv")

current_info_avg_bound_welfare = cbind(current_info_avg_bound_mean_cs,current_info_avg_bound_mean_q,current_info_avg_bound_mean_r,
                                   current_info_avg_bound_var_cs,current_info_avg_bound_var_q,current_info_avg_bound_var_r)
colnames(current_info_avg_bound_welfare) = c("mean_cs", "mean_q", "mean_r","var_cs", "var_q", "var_r")

current_info_avg_bound_welfare$steps = seq(-0.25, 0.25, by = 0.05)
current_info_avg_bound_welfare$s_steps = seq(0.75, 1.25, by = 0.05)

current_info_avg_bound_welfare$mean_r_diff = (current_info_avg_bound_welfare$mean_r - r_agg_0)/abs(r_agg_0)*100
current_info_avg_bound_welfare$var_r_diff = (current_info_avg_bound_welfare$var_r - r_agg_0)/abs(r_agg_0)*100

current_info_avg_bound_welfare$mean_q_diff = (current_info_avg_bound_welfare$mean_q - q_agg_0)/abs(q_agg_0)*100
current_info_avg_bound_welfare$var_q_diff = (current_info_avg_bound_welfare$var_q - q_agg_0)/abs(q_agg_0)*100

current_info_avg_bound_welfare$mean_cs_diff = (current_info_avg_bound_welfare$mean_cs - cs_agg_0)/abs(cs_agg_0)*100
current_info_avg_bound_welfare$var_cs_diff = (current_info_avg_bound_welfare$var_cs - cs_agg_0)/abs(cs_agg_0)*100

library(scales)  # For label formatting

r_current_info_avg_bound = ggplot(data = current_info_avg_bound_welfare) +
  geom_line(aes(x = steps, y = mean_r_diff, color = "mean", linetype = "mean"), linewidth = 2)+
  geom_line(aes(x = steps, y = var_r_diff, color = "var", linetype = "var"), linewidth = 2)+
  scale_color_manual(
    values = c("mean" = "#35618f", "var" = "#8c334c"),
    labels = c("mean" = "mean", "var" = "var")
  ) +
  scale_linetype_manual(
    values = c("mean" = "solid", "var" = "longdash"),
    labels = c("mean" = "mean", "var" = "var")
  ) +
  scale_x_continuous(
    name = "E[Prcp]",
    sec.axis = sec_axis(~ rescale(., from = range(current_info_avg_bound_welfare$steps), to = range(current_info_avg_bound_welfare$s_steps )), 
                        name = "Min & Max[Prcp]")
  ) +
  coord_cartesian(xlim = c(min(range(current_info_avg_bound_welfare$steps))
                           , max(range(current_info_avg_bound_welfare$steps)))
  ) +
  labs(
    title = "Producer Surplus Change (%)",
    color = "",
    linetype = "",
    x = NULL,
    y = NULL
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )

cs_current_info_avg_bound = ggplot(data = current_info_avg_bound_welfare) +
  geom_line(aes(x = steps, y = mean_cs_diff, color = "mean", linetype = "mean"), linewidth = 2)+
  geom_line(aes(x = steps, y = var_cs_diff, color = "var", linetype = "var"), linewidth = 2)+
  scale_color_manual(
    values = c("mean" = "#35618f", "var" = "#8c334c"),
    labels = c("mean" = "mean", "var" = "var")
  ) +
  scale_linetype_manual(
    values = c("mean" = "solid", "var" = "longdash"),
    labels = c("mean" = "mean", "var" = "var")
  ) +
  scale_x_continuous(
    name = "E[Prcp]",
    sec.axis = sec_axis(~ rescale(., from = range(current_info_avg_bound_welfare$steps), to = range(current_info_avg_bound_welfare$s_steps )), 
                        name = "Min & Max[Prcp]")
  ) +
  coord_cartesian(xlim = c(min(range(current_info_avg_bound_welfare$steps))
                           , max(range(current_info_avg_bound_welfare$steps)))
  ) +
  labs(
    title = "Consumer Surplus Change (%)",
    color = "",
    linetype = "",
    x = NULL,
    y = NULL
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )

q_current_info_avg_bound = ggplot(data = current_info_avg_bound_welfare) +
  geom_line(aes(x = steps, y = mean_q_diff, color = "mean", linetype = "mean"), linewidth = 2)+
  geom_line(aes(x = steps, y = var_q_diff, color = "var", linetype = "var"), linewidth = 2)+
  scale_color_manual(
    values = c("mean" = "#35618f", "var" = "#8c334c"),
    labels = c("mean" = "mean", "var" = "var")
  ) +
  scale_linetype_manual(
    values = c("mean" = "solid", "var" = "longdash"),
    labels = c("mean" = "mean", "var" = "var")
  ) +
  scale_x_continuous(
    name = "E[Prcp]",
    sec.axis = sec_axis(~ rescale(., from = range(current_info_avg_bound_welfare$steps), to = range(current_info_avg_bound_welfare$s_steps )), 
                        name = "Min & Max[Prcp]")
  ) +
  coord_cartesian(xlim = c(min(range(current_info_avg_bound_welfare$steps))
                           , max(range(current_info_avg_bound_welfare$steps)))
  ) +
  labs(
    title = "Total Quantity Change (%)",
    color = "",
    linetype = "",
    x = NULL,
    y = NULL
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )

ggplot(data = current_info_avg_bound_welfare) +
  geom_line(aes(x = steps, y = mean_r_diff, color = "r"), linetype = "dotdash",linewidth = 2)+
  geom_line(aes(x = steps, y = mean_cs_diff, color = "cs"), linetype = "longdash", linewidth = 2)+
  geom_line(aes(x = steps, y = mean_q_diff, color = "q"), linetype = "solid", linewidth = 2)+
  scale_color_manual(
    values = c("r" = "#316387", "cs" = "#842411", "q" = "#256b33"),
    labels = c("r" = "PS", "cs" = "CS", "q" = "Q")
  ) +
  coord_cartesian(xlim = c(min(range(current_info_avg_bound_welfare$steps))
                           , max(range(current_info_avg_bound_welfare$steps)))
  ) +
  labs(
    #title = "Welfare Change (%)",
    color = "Type",
    linetype = "Type",
    x = "Δ E[Prcp] (Inches)",
    y = "%"
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )


ggplot(data = current_info_avg_bound_welfare) +
  geom_line(aes(x = s_steps, y = var_r_diff, color = "r"), linetype = "dotdash",linewidth = 2)+
  geom_line(aes(x = s_steps, y = var_cs_diff, color = "cs"), linetype = "longdash", linewidth = 2)+
  geom_line(aes(x = s_steps, y = var_q_diff, color = "q"), linetype = "solid", linewidth = 2)+
  scale_color_manual(
    values = c("r" = "#316387", "cs" = "#842411", "q" = "#256b33"),
    labels = c("r" = "PS", "cs" = "CS", "q" = "Q")
  ) +
  coord_cartesian(xlim = c(min(range(current_info_avg_bound_welfare$s_steps))
                           , max(range(current_info_avg_bound_welfare$s_steps)))
  ) +
  labs(
    #title = "Welfare Change (%)",
    color = "Type",
    linetype = "Type",
    x = "SD Ratio",
    y = "%"
  ) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )

mean(current_info_avg_bound_welfare[which(current_info_avg_bound_welfare$s_steps<1),]$var_cs_diff)
#16.01809
mean(current_info_avg_bound_welfare[which(current_info_avg_bound_welfare$s_steps>1),]$var_cs_diff)
#-24.77576

##### current_info_logr_bound ####

current_info_logr_bound_mean_cs = read_csv("current_info_logr_bound_mean_cs.csv")
current_info_logr_bound_mean_q = read_csv("current_info_logr_bound_mean_q.csv")
current_info_logr_bound_mean_r = read_csv("current_info_logr_bound_mean_r.csv")

current_info_logr_bound_var_cs = read_csv("current_info_logr_bound_var_cs.csv")
current_info_logr_bound_var_q = read_csv("current_info_logr_bound_var_q.csv")
current_info_logr_bound_var_r = read_csv("current_info_logr_bound_var_r.csv")

current_info_logr_bound_welfare = cbind(current_info_logr_bound_mean_cs,current_info_logr_bound_mean_q,current_info_logr_bound_mean_r,
                                        current_info_logr_bound_var_cs,current_info_logr_bound_var_q,current_info_logr_bound_var_r)
colnames(current_info_logr_bound_welfare) = c("mean_cs", "mean_q", "mean_r","var_cs", "var_q", "var_r")

current_info_logr_bound_welfare$steps = seq(-0.25, 0.25, by = 0.05)
current_info_logr_bound_welfare$s_steps = seq(0.75, 1.25, by = 0.05)

current_info_logr_bound_welfare$mean_r_diff = (current_info_logr_bound_welfare$mean_r - r_agg_0)/abs(r_agg_0)*100
current_info_logr_bound_welfare$var_r_diff = (current_info_logr_bound_welfare$var_r - r_agg_0)/abs(r_agg_0)*100

current_info_logr_bound_welfare$mean_q_diff = (current_info_logr_bound_welfare$mean_q - q_agg_0)/abs(q_agg_0)*100
current_info_logr_bound_welfare$var_q_diff = (current_info_logr_bound_welfare$var_q - q_agg_0)/abs(q_agg_0)*100

current_info_logr_bound_welfare$mean_cs_diff = (current_info_logr_bound_welfare$mean_cs - cs_agg_0)/abs(cs_agg_0)*100
current_info_logr_bound_welfare$var_cs_diff = (current_info_logr_bound_welfare$var_cs - cs_agg_0)/abs(cs_agg_0)*100

ggplot(data = current_info_logr_bound_welfare) +
  geom_line(aes(x = steps, y = mean_r_diff, color = "r"), linetype = "dotdash",linewidth = 2)+
  geom_line(aes(x = steps, y = mean_cs_diff, color = "cs"), linetype = "longdash", linewidth = 2)+
  geom_line(aes(x = steps, y = mean_q_diff, color = "q"), linetype = "solid", linewidth = 2)+
  scale_color_manual(
    values = c("r" = "#316387", "cs" = "#842411", "q" = "#256b33"),
    labels = c("r" = "PS", "cs" = "CS", "q" = "Q")
  ) +
  coord_cartesian(xlim = c(min(range(current_info_logr_bound_welfare$steps))
                           , max(range(current_info_logr_bound_welfare$steps)))
  ) +
  labs(
    #title = "Welfare Change (%)",
    color = "Type",
    linetype = "Type",
    x = "Δ E[Prcp] (Inches)",
    y = "%"
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )


ggplot(data = current_info_logr_bound_welfare) +
  geom_line(aes(x = s_steps, y = var_r_diff, color = "r"), linetype = "dotdash",linewidth = 2)+
  geom_line(aes(x = s_steps, y = var_cs_diff, color = "cs"), linetype = "longdash", linewidth = 2)+
  geom_line(aes(x = s_steps, y = var_q_diff, color = "q"), linetype = "solid", linewidth = 2)+
  scale_color_manual(
    values = c("r" = "#316387", "cs" = "#842411", "q" = "#256b33"),
    labels = c("r" = "PS", "cs" = "CS", "q" = "Q")
  ) +
  coord_cartesian(xlim = c(min(range(current_info_logr_bound_welfare$s_steps))
                           , max(range(current_info_logr_bound_welfare$s_steps)))
  ) +
  labs(
    #title = "Welfare Change (%)",
    color = "Type",
    linetype = "Type",
    x = "SD Ratio",
    y = "%"
  ) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )

mean(current_info_logr_bound_welfare[which(current_info_logr_bound_welfare$s_steps<1),]$var_cs_diff)
#21.91156
mean(current_info_logr_bound_welfare[which(current_info_logr_bound_welfare$s_steps>1),]$var_cs_diff)
#-47.01439

##### current_info_gamma05_bound ####

current_info_gamma05_bound_mean_cs = read_csv("current_info_gamma05_bound_mean_cs.csv")
current_info_gamma05_bound_mean_q = read_csv("current_info_gamma05_bound_mean_q.csv")
current_info_gamma05_bound_mean_r = read_csv("current_info_gamma05_bound_mean_r.csv")

current_info_gamma05_bound_var_cs = read_csv("current_info_gamma05_bound_var_cs.csv")
current_info_gamma05_bound_var_q = read_csv("current_info_gamma05_bound_var_q.csv")
current_info_gamma05_bound_var_r = read_csv("current_info_gamma05_bound_var_r.csv")

current_info_gamma05_bound_welfare = cbind(current_info_gamma05_bound_mean_cs,current_info_gamma05_bound_mean_q,current_info_gamma05_bound_mean_r,
                                        current_info_gamma05_bound_var_cs,current_info_gamma05_bound_var_q,current_info_gamma05_bound_var_r)
colnames(current_info_gamma05_bound_welfare) = c("mean_cs", "mean_q", "mean_r","var_cs", "var_q", "var_r")

current_info_gamma05_bound_welfare$steps = seq(-0.25, 0.25, by = 0.05)
current_info_gamma05_bound_welfare$s_steps = seq(0.75, 1.25, by = 0.05)

current_info_gamma05_bound_welfare$mean_r_diff = (current_info_gamma05_bound_welfare$mean_r - r_agg_0)/abs(r_agg_0)*100
current_info_gamma05_bound_welfare$var_r_diff = (current_info_gamma05_bound_welfare$var_r - r_agg_0)/abs(r_agg_0)*100

current_info_gamma05_bound_welfare$mean_q_diff = (current_info_gamma05_bound_welfare$mean_q - q_agg_0)/abs(q_agg_0)*100
current_info_gamma05_bound_welfare$var_q_diff = (current_info_gamma05_bound_welfare$var_q - q_agg_0)/abs(q_agg_0)*100

current_info_gamma05_bound_welfare$mean_cs_diff = (current_info_gamma05_bound_welfare$mean_cs - cs_agg_0)/abs(cs_agg_0)*100
current_info_gamma05_bound_welfare$var_cs_diff = (current_info_gamma05_bound_welfare$var_cs - cs_agg_0)/abs(cs_agg_0)*100

ggplot(data = current_info_gamma05_bound_welfare) +
  geom_line(aes(x = steps, y = mean_r_diff, color = "r"), linetype = "dotdash",linewidth = 2)+
  geom_line(aes(x = steps, y = mean_cs_diff, color = "cs"), linetype = "longdash", linewidth = 2)+
  geom_line(aes(x = steps, y = mean_q_diff, color = "q"), linetype = "solid", linewidth = 2)+
  scale_color_manual(
    values = c("r" = "#316387", "cs" = "#842411", "q" = "#256b33"),
    labels = c("r" = "PS", "cs" = "CS", "q" = "Q")
  ) +
  coord_cartesian(xlim = c(min(range(current_info_gamma05_bound_welfare$steps))
                           , max(range(current_info_gamma05_bound_welfare$steps)))
  ) +
  labs(
    #title = "Welfare Change (%)",
    color = "Type",
    linetype = "Type",
    x = "Δ E[Prcp] (Inches)",
    y = "%"
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )


ggplot(data = current_info_gamma05_bound_welfare) +
  geom_line(aes(x = s_steps, y = var_r_diff, color = "r"), linetype = "dotdash",linewidth = 2)+
  geom_line(aes(x = s_steps, y = var_cs_diff, color = "cs"), linetype = "longdash", linewidth = 2)+
  geom_line(aes(x = s_steps, y = var_q_diff, color = "q"), linetype = "solid", linewidth = 2)+
  scale_color_manual(
    values = c("r" = "#316387", "cs" = "#842411", "q" = "#256b33"),
    labels = c("r" = "PS", "cs" = "CS", "q" = "Q")
  ) +
  coord_cartesian(xlim = c(min(range(current_info_gamma05_bound_welfare$s_steps))
                           , max(range(current_info_gamma05_bound_welfare$s_steps)))
  ) +
  labs(
    #title = "Welfare Change (%)",
    color = "Type",
    linetype = "Type",
    x = "SD Ratio",
    y = "%"
  ) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )

mean(current_info_gamma05_bound_welfare[which(current_info_gamma05_bound_welfare$s_steps<1),]$var_cs_diff)
#20.67548
mean(current_info_gamma05_bound_welfare[which(current_info_gamma05_bound_welfare$s_steps>1),]$var_cs_diff)
#-37.66772

##### current_info_extreme_bound ####

current_info_extreme_bound_mean_cs = read_csv("current_info_extreme_bound_mean_cs.csv")
current_info_extreme_bound_mean_q = read_csv("current_info_extreme_bound_mean_q.csv")
current_info_extreme_bound_mean_r = read_csv("current_info_extreme_bound_mean_r.csv")

current_info_extreme_bound_var_cs = read_csv("current_info_extreme_bound_var_cs.csv")
current_info_extreme_bound_var_q = read_csv("current_info_extreme_bound_var_q.csv")
current_info_extreme_bound_var_r = read_csv("current_info_extreme_bound_var_r.csv")

current_info_extreme_bound_welfare = cbind(current_info_extreme_bound_mean_cs,current_info_extreme_bound_mean_q,current_info_extreme_bound_mean_r,
                                       current_info_extreme_bound_var_cs,current_info_extreme_bound_var_q,current_info_extreme_bound_var_r)
colnames(current_info_extreme_bound_welfare) = c("mean_cs", "mean_q", "mean_r","var_cs", "var_q", "var_r")

current_info_extreme_bound_welfare$steps = seq(-0.25, 0.25, by = 0.05)
current_info_extreme_bound_welfare$s_steps = seq(0.75, 1.25, by = 0.05)

current_info_extreme_bound_welfare$mean_r_diff = (current_info_extreme_bound_welfare$mean_r - r_agg_0)/abs(r_agg_0)*100
current_info_extreme_bound_welfare$var_r_diff = (current_info_extreme_bound_welfare$var_r - r_agg_0)/abs(r_agg_0)*100

current_info_extreme_bound_welfare$mean_q_diff = (current_info_extreme_bound_welfare$mean_q - q_agg_0)/abs(q_agg_0)*100
current_info_extreme_bound_welfare$var_q_diff = (current_info_extreme_bound_welfare$var_q - q_agg_0)/abs(q_agg_0)*100

current_info_extreme_bound_welfare$mean_cs_diff = (current_info_extreme_bound_welfare$mean_cs - cs_agg_0)/abs(cs_agg_0)*100
current_info_extreme_bound_welfare$var_cs_diff = (current_info_extreme_bound_welfare$var_cs - cs_agg_0)/abs(cs_agg_0)*100

ggplot(data = current_info_extreme_bound_welfare) +
  geom_line(aes(x = steps, y = mean_r_diff, color = "r"), linetype = "dotdash",linewidth = 2)+
  geom_line(aes(x = steps, y = mean_cs_diff, color = "cs"), linetype = "longdash", linewidth = 2)+
  geom_line(aes(x = steps, y = mean_q_diff, color = "q"), linetype = "solid", linewidth = 2)+
  scale_color_manual(
    values = c("r" = "#316387", "cs" = "#842411", "q" = "#256b33"),
    labels = c("r" = "PS", "cs" = "CS", "q" = "Q")
  ) +
  coord_cartesian(xlim = c(min(range(current_info_extreme_bound_welfare$steps))
                           , max(range(current_info_extreme_bound_welfare$steps)))
  ) +
  labs(
    #title = "Welfare Change (%)",
    color = "Type",
    linetype = "Type",
    x = "Δ E[Prcp] (Inches)",
    y = "%"
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )


ggplot(data = current_info_extreme_bound_welfare) +
  geom_line(aes(x = s_steps, y = var_r_diff, color = "r"), linetype = "dotdash",linewidth = 2)+
  geom_line(aes(x = s_steps, y = var_cs_diff, color = "cs"), linetype = "longdash", linewidth = 2)+
  geom_line(aes(x = s_steps, y = var_q_diff, color = "q"), linetype = "solid", linewidth = 2)+
  scale_color_manual(
    values = c("r" = "#316387", "cs" = "#842411", "q" = "#256b33"),
    labels = c("r" = "PS", "cs" = "CS", "q" = "Q")
  ) +
  coord_cartesian(xlim = c(min(range(current_info_extreme_bound_welfare$s_steps))
                           , max(range(current_info_extreme_bound_welfare$s_steps)))
  ) +
  labs(
    #title = "Welfare Change (%)",
    color = "Type",
    linetype = "Type",
    x = "SD Ratio",
    y = "%"
  ) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )

##### current_info_soft_bound ####

current_info_soft_bound_mean_cs = read_csv("current_info_soft_bound_mean_cs.csv")
current_info_soft_bound_mean_q = read_csv("current_info_soft_bound_mean_q.csv")
current_info_soft_bound_mean_r = read_csv("current_info_soft_bound_mean_r.csv")

current_info_soft_bound_var_cs = read_csv("current_info_soft_bound_var_cs.csv")
current_info_soft_bound_var_q = read_csv("current_info_soft_bound_var_q.csv")
current_info_soft_bound_var_r = read_csv("current_info_soft_bound_var_r.csv")

current_info_soft_bound_welfare = cbind(current_info_soft_bound_mean_cs,current_info_soft_bound_mean_q,current_info_soft_bound_mean_r,
                                           current_info_soft_bound_var_cs,current_info_soft_bound_var_q,current_info_soft_bound_var_r)
colnames(current_info_soft_bound_welfare) = c("mean_cs", "mean_q", "mean_r","var_cs", "var_q", "var_r")

current_info_soft_bound_welfare$steps = seq(-0.25, 0.25, by = 0.05)
current_info_soft_bound_welfare$s_steps = seq(0.75, 1.25, by = 0.05)

current_info_soft_bound_welfare$mean_r_diff = (current_info_soft_bound_welfare$mean_r - r_agg_0)/abs(r_agg_0)*100
current_info_soft_bound_welfare$var_r_diff = (current_info_soft_bound_welfare$var_r - r_agg_0)/abs(r_agg_0)*100

current_info_soft_bound_welfare$mean_q_diff = (current_info_soft_bound_welfare$mean_q - q_agg_0)/abs(q_agg_0)*100
current_info_soft_bound_welfare$var_q_diff = (current_info_soft_bound_welfare$var_q - q_agg_0)/abs(q_agg_0)*100

current_info_soft_bound_welfare$mean_cs_diff = (current_info_soft_bound_welfare$mean_cs - cs_agg_0)/abs(cs_agg_0)*100
current_info_soft_bound_welfare$var_cs_diff = (current_info_soft_bound_welfare$var_cs - cs_agg_0)/abs(cs_agg_0)*100

ggplot(data = current_info_soft_bound_welfare) +
  geom_line(aes(x = steps, y = mean_r_diff, color = "r"), linetype = "dotdash",linewidth = 2)+
  geom_line(aes(x = steps, y = mean_cs_diff, color = "cs"), linetype = "longdash", linewidth = 2)+
  geom_line(aes(x = steps, y = mean_q_diff, color = "q"), linetype = "solid", linewidth = 2)+
  scale_color_manual(
    values = c("r" = "#316387", "cs" = "#842411", "q" = "#256b33"),
    labels = c("r" = "PS", "cs" = "CS", "q" = "Q")
  ) +
  coord_cartesian(xlim = c(min(range(current_info_soft_bound_welfare$steps))
                           , max(range(current_info_soft_bound_welfare$steps)))
  ) +
  labs(
    #title = "Welfare Change (%)",
    color = "Type",
    linetype = "Type",
    x = "Δ E[Prcp] (Inches)",
    y = "%"
  ) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )


ggplot(data = current_info_soft_bound_welfare) +
  geom_line(aes(x = s_steps, y = var_r_diff, color = "r"), linetype = "dotdash",linewidth = 2)+
  geom_line(aes(x = s_steps, y = var_cs_diff, color = "cs"), linetype = "longdash", linewidth = 2)+
  geom_line(aes(x = s_steps, y = var_q_diff, color = "q"), linetype = "solid", linewidth = 2)+
  scale_color_manual(
    values = c("r" = "#316387", "cs" = "#842411", "q" = "#256b33"),
    labels = c("r" = "PS", "cs" = "CS", "q" = "Q")
  ) +
  coord_cartesian(xlim = c(min(range(current_info_soft_bound_welfare$s_steps))
                           , max(range(current_info_soft_bound_welfare$s_steps)))
  ) +
  labs(
    #title = "Welfare Change (%)",
    color = "Type",
    linetype = "Type",
    x = "SD Ratio",
    y = "%"
  ) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.8) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", linewidth = 0.8) +
  big_text +
  theme(
    legend.position = "bottom",
    legend.key.width = unit(2, "cm"),
    plot.title = element_text(hjust = 0.5)
  )

