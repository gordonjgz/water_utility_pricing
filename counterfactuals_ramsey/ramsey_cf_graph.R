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

q = seq(0.5, 25, by = 0.1)

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

# --- Plot Definitions ---
my_plot_theme <- theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
    axis.title = element_text(face = "bold", size = 10),
    axis.text = element_text(size = 9),
    legend.position = "none", # IMPORTANT: Hide legend for individual plots
    panel.grid.major = element_line(color = "grey90", linewidth = 0.2),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA),
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5)
  )
#####
current_info_avg_bound_mean_pl = read_csv("../ramsey_price_result/price_detail_results/current_info_avg_bound_loss1_mean_pl.csv")
current_info_avg_bound_mean_ql = read_csv("../ramsey_price_result/price_detail_results/current_info_avg_bound_loss1_mean_ql.csv")
current_info_avg_bound_mean_fcl = read_csv("../ramsey_price_result/price_detail_results/current_info_avg_bound_loss1_mean_fcl.csv")
current_info_avg_bound_var_pl = read_csv("../ramsey_price_result/price_detail_results/current_info_avg_bound_loss1_var_pl.csv")
current_info_avg_bound_var_ql = read_csv("../ramsey_price_result/price_detail_results/current_info_avg_bound_loss1_var_ql.csv")
current_info_avg_bound_var_fcl = read_csv("../ramsey_price_result/price_detail_results/current_info_avg_bound_loss1_var_fcl.csv")

current_info_gamma1_bound_mean_pl = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma1_bound_loss1_mean_pl.csv")
current_info_gamma1_bound_mean_ql = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma1_bound_loss1_mean_ql.csv")
current_info_gamma1_bound_mean_fcl = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma1_bound_loss1_mean_fcl.csv")
current_info_gamma1_bound_var_pl = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma1_bound_loss1_var_pl.csv")
current_info_gamma1_bound_var_ql = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma1_bound_loss1_var_ql.csv")
current_info_gamma1_bound_var_fcl = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma1_bound_loss1_var_fcl.csv")


current_info_gamma025_bound_mean_pl = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma025_bound_loss05_mean_pl.csv")
current_info_gamma025_bound_mean_ql = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma025_bound_loss05_mean_ql.csv")
current_info_gamma025_bound_mean_fcl = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma025_bound_loss05_mean_fcl.csv")
current_info_gamma025_bound_var_pl = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma025_bound_loss05_var_pl.csv")
current_info_gamma025_bound_var_ql = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma025_bound_loss05_var_ql.csv")
current_info_gamma025_bound_var_fcl = read_csv("../ramsey_price_result/price_detail_results/current_info_gamma025_bound_loss05_var_fcl.csv")


pv0 = c(3.09,5.01,8.54,12.9,14.41,2,6,11,20,8.5,10.8,16.5,37,37)
statusquo = gen_p(pv0, q)
statusquo$step = "status_quo"

current_info_avg_bound_mean = rbind(current_info_avg_bound_mean_pl, current_info_avg_bound_mean_ql, current_info_avg_bound_mean_fcl)
current_info_avg_bound_var = rbind(current_info_avg_bound_var_pl, current_info_avg_bound_var_ql, current_info_avg_bound_var_fcl)

current_info_gamma1_bound_mean = rbind(current_info_gamma1_bound_mean_pl, current_info_gamma1_bound_mean_ql, current_info_gamma1_bound_mean_fcl)
current_info_gamma1_bound_var = rbind(current_info_gamma1_bound_var_pl, current_info_gamma1_bound_var_ql, current_info_gamma1_bound_var_fcl)

current_info_gamma025_bound_mean = rbind(current_info_gamma025_bound_mean_pl, current_info_gamma025_bound_mean_ql, current_info_gamma025_bound_mean_fcl)
current_info_gamma025_bound_var = rbind(current_info_gamma025_bound_var_pl, current_info_gamma025_bound_var_ql, current_info_gamma025_bound_var_fcl)


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
                   "0" = "#808080", 
                   setNames(gradient_pos, as.character(seq(0.05, 0.25, length.out = length(gradient_pos)))))


pvflat = c(rep(mean(c(3.09,5.01,8.54,12.9,14.41)),5 ), 2, 6, 11, 20, rep(mean(c(8.5,10.8,16.5,37,37)),5 ))
#pvflat = c(rep(mean(current_info_avg_bound_mean_pl$`5`),5 ), 2, 6, 11, 20, rep(mean(current_info_avg_bound_mean_fcl$`5`),5 ))
flat = gen_p(pvflat, q)
flat$step = "flat"

ggplot() + 
  #geom_line(data = combined_current_info_avg_bound_mean_info[which(as.numeric(as.character((combined_current_info_avg_bound_mean_info$step)))>0),]
            #         , aes(x = q, y = total, color = step, linetype = "zero"), color = "#CFCFCF", linewidth = 1.5) +
   #         , aes(x = q, y = total, color = step), linewidth = 1.5) +
  geom_line(data = flat, aes(x = q, y = total, linetype = "Status Quo - Flat"), color = "#6e6e6e", linewidth = 1) +
  geom_line(data = statusquo, aes(x = q, y = total, linetype = "statusquo"), color = "#080808", linewidth = 1) +
  #geom_line(data = flat, aes(x = q, y = total), color = "#6e6e6e", linewidth = 1, linetype = "dotted") +
  #geom_line(data = statusquo, aes(x = q, y = total), color = "#080808", linewidth = 1, linetype = "dotdash") +
  xlab("Quantity (k Gallon)") +
  ylab("Total Payment ($)") +
  labs(
    #color = "Δ E[Prcp] (Inch)", 
    linetype = NULL) +
  #scale_color_manual(values = color_palette) +
  scale_linetype_manual(values = c("zero" = "solid","Status Quo - Flat" = "dotted", "statusquo" = "dotdash")) +
  coord_cartesian(ylim = c(0, 500)) +
  guides(color = guide_legend(override.aes = list(size = 6))) + 
  #ggtitle("Total Payment") +
  #ggtitle("Total Payment by Δ Mean Precipitation") +
  big_text


# Dry Condition Plot
dry_condition <- ggplot() +
  geom_line(
    data = combined_current_info_avg_bound_mean_info %>% filter(as.numeric(as.character(step)) <= 0),
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    title = expression(zeta[1] <= 0) # Label as title
  ) +
  scale_color_manual(
    values = color_palette # Uses the global color palette
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash")
  ) +
  coord_cartesian(ylim = c(0, 500)) +
  my_plot_theme # Apply the theme that hides legend


# Wet Condition Plot
wet_condition <- ggplot() +
  geom_line(
    data = combined_current_info_avg_bound_mean_info %>% filter(as.numeric(as.character(step)) >= 0),
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    title =  expression(zeta[1] >= 0) # Label as title
  ) +
  scale_color_manual(
    values = color_palette # Uses the global color palette
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash")
  ) +
  coord_cartesian(ylim = c(0, 500)) +
  my_plot_theme # Apply the theme that hides legend


# --- Create a 'full' plot temporarily to extract the complete legend ---
# This plot contains all possible 'step' values and linetypes to ensure a comprehensive legend.
full_legend_plot <- ggplot() +
  geom_line(
    data = combined_current_info_avg_bound_mean_info, # Use ALL step values for legend
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    color = expression(zeta[1] ~ "=" ~ Delta * " E[Prcp] (Inch)"), # Legend title for color
    linetype = "Benchmark" # Legend title for linetype
  ) +
  scale_color_manual(
    values = color_palette,
    guide = guide_legend(override.aes = list(linewidth = 1))
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash"),
    guide = guide_legend(override.aes = list(linewidth = 0.8))
  ) +
  # Theme for this temp plot only for legend extraction
  theme(
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 9),
    legend.position = "bottom", # Make sure legend is visible here
    legend.box = "horizontal", # Arrange multiple legends horizontally
    # Hide plot elements as we only need the legend
    axis.title = element_blank(),
    axis.text = element_blank(),
    plot.background = element_blank(),
    panel.background = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_blank(),
    plot.margin = unit(c(0,0,0,0), "cm") # Reduce margins
  )

plot_grob <- ggplotGrob(full_legend_plot)

legend_index <- which(sapply(plot_grob$grobs, function(x) x$name) == "guide-box")

common_legend_zeta_1 <- plot_grob$grobs[[legend_index]]


# --- Arrange Plots with the extracted common legend ---

# First, arrange the plots themselves (which have no legends)
plots_grid <- ggarrange(dry_condition, wet_condition,
                        ncol = 2, nrow = 1,
                        # No need for labels, they are now titles within plots
                        common.legend = FALSE, # Explicitly no common legend from ggarrange
                        legend = "none") # Ensure no legend is generated by ggarrange

# Then, combine the grid of plots with the extracted common legend
avg_bound_mean <- ggarrange(plots_grid, common_legend_zeta_1,
                                 ncol = 1, nrow = 2,
                                 heights = c(1, 0.2)) # Adjust heights to give space for legend

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
                   "1" = "#808080", 
                   setNames(gradient_pos, as.character(seq(1.05, 1.25, length.out = length(gradient_pos)))))

pvflat = c(rep(mean(c(3.09,5.01,8.54,12.9,14.41)),5 ), 2, 6, 11, 20, rep(mean(c(8.5,10.8,16.5,37,37)),5 ))
#pvflat = c(rep(mean(current_info_avg_bound_mean_pl$`5`),5 ), 2, 6, 11, 20, rep(mean(current_info_avg_bound_mean_fcl$`5`),5 ))
flat = gen_p(pvflat, q)
flat$step = "flat"

# Dry Condition Plot
low_condition <- ggplot() +
  geom_line(
    data = combined_current_info_avg_bound_var_info %>% filter(as.numeric(as.character(step)) <= 1),
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    title = expression(zeta[2] <= 1) # Label as title
  ) +
  scale_color_manual(
    values = color_palette # Uses the global color palette
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash")
  ) +
  coord_cartesian(ylim = c(0, 600)) +
  my_plot_theme # Apply the theme that hides legend


# Wet Condition Plot
high_condition <- ggplot() +
  geom_line(
    data = combined_current_info_avg_bound_var_info %>% filter(as.numeric(as.character(step)) >= 1),
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    title =  expression(zeta[2] >= 1) # Label as title
  ) +
  scale_color_manual(
    values = color_palette # Uses the global color palette
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash")
  ) +
  coord_cartesian(ylim = c(0, 600)) +
  my_plot_theme # Apply the theme that hides legend


# --- Create a 'full' plot temporarily to extract the complete legend ---
# This plot contains all possible 'step' values and linetypes to ensure a comprehensive legend.
full_legend_plot <- ggplot() +
  geom_line(
    data = combined_current_info_avg_bound_var_info, # Use ALL step values for legend
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    color = expression(zeta[2] ~ "=" ~ " Sd Ratio"), # Legend title for color
    linetype = "Benchmark" # Legend title for linetype
  ) +
  scale_color_manual(
    values = color_palette,
    guide = guide_legend(override.aes = list(linewidth = 1))
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash"),
    guide = guide_legend(override.aes = list(linewidth = 0.8))
  ) +
  # Theme for this temp plot only for legend extraction
  theme(
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 9),
    legend.position = "bottom", # Make sure legend is visible here
    legend.box = "horizontal", # Arrange multiple legends horizontally
    # Hide plot elements as we only need the legend
    axis.title = element_blank(),
    axis.text = element_blank(),
    plot.background = element_blank(),
    panel.background = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_blank(),
    plot.margin = unit(c(0,0,0,0), "cm") # Reduce margins
  )

plot_grob <- ggplotGrob(full_legend_plot)

legend_index <- which(sapply(plot_grob$grobs, function(x) x$name) == "guide-box")

common_legend_zeta_2 <- plot_grob$grobs[[legend_index]]



# --- Arrange Plots with the extracted common legend ---

# First, arrange the plots themselves (which have no legends)
plots_grid <- ggarrange(low_condition, high_condition,
                        ncol = 2, nrow = 1,
                        # No need for labels, they are now titles within plots
                        common.legend = FALSE, # Explicitly no common legend from ggarrange
                        legend = "none") # Ensure no legend is generated by ggarrange

# Then, combine the grid of plots with the extracted common legend
avg_bound_var <- ggarrange(plots_grid, common_legend_zeta_2,
                            ncol = 1, nrow = 2,
                            heights = c(1, 0.2)) # Adjust heights to give space for legend





############################################################
################## current_info_gamma1_bound_mean ##############################
#########################################################

current_info_gamma1_bound_mean_info <- lapply(current_info_gamma1_bound_mean, function(col) gen_p(col, q))

combined_current_info_gamma1_bound_mean_info <- bind_rows(lapply(names(current_info_gamma1_bound_mean_info), function(name) {
  df <- current_info_gamma1_bound_mean_info[[name]]
  df$step <- as.numeric(name)/20-0.25  # Add a column to track the dataset
  return(df)
}))

# Ensure step is a factor for proper animation handling
combined_current_info_gamma1_bound_mean_info$step <- factor(combined_current_info_gamma1_bound_mean_info$step)

# Define color palette: Steps from brown (dry) to blue (wet), plus muted colors for static lines
num_steps <- length(unique(combined_current_info_gamma1_bound_mean_info$step)) 
# Generate a gradient for steps from -1 to 0 (dark red to gray50)
gradient_neg <- colorRampPalette(c("darkred", "firebrick", "darkorange", "#e2a76d"))(ceiling((num_steps - 1) / 2))

# Generate a gradient for steps from 0 to 1 (gray50 to dark blue)
gradient_pos <- colorRampPalette(c("#c0d9e8","#7aa9d6", "royalblue", "darkblue"))(floor((num_steps - 1) / 2))

# Combine both gradients and assign gray50 to step 0
color_palette <- c(setNames(gradient_neg, as.character(seq(-0.25, -0.05, length.out = length(gradient_neg)))),
                   "0" = "#808080", 
                   setNames(gradient_pos, as.character(seq(0.05, 0.25, length.out = length(gradient_pos)))))


pvflat = c(rep(mean(c(3.09,5.01,8.54,12.9,14.41)),5 ), 2, 6, 11, 20, rep(mean(c(8.5,10.8,16.5,37,37)),5 ))
#pvflat = c(rep(mean(current_info_gamma1_bound_mean_pl$`5`),5 ), 2, 6, 11, 20, rep(mean(current_info_gamma1_bound_mean_fcl$`5`),5 ))
flat = gen_p(pvflat, q)
flat$step = "flat"

# Dry Condition Plot
dry_condition <- ggplot() +
  geom_line(
    data = combined_current_info_gamma1_bound_mean_info %>% filter(as.numeric(as.character(step)) <= 0),
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    title = expression(zeta[1] <= 0) # Label as title
  ) +
  scale_color_manual(
    values = color_palette # Uses the global color palette
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash")
  ) +
  coord_cartesian(ylim = c(0, 1000)) +
  my_plot_theme # Apply the theme that hides legend


# Wet Condition Plot
wet_condition <- ggplot() +
  geom_line(
    data = combined_current_info_gamma1_bound_mean_info %>% filter(as.numeric(as.character(step)) >= 0),
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    title =  expression(zeta[1] >= 0) # Label as title
  ) +
  scale_color_manual(
    values = color_palette # Uses the global color palette
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash")
  ) +
  coord_cartesian(ylim = c(0, 1000)) +
  my_plot_theme # Apply the theme that hides legend


# --- Create a 'full' plot temporarily to extract the complete legend ---
# This plot contains all possible 'step' values and linetypes to ensure a comprehensive legend.
full_legend_plot <- ggplot() +
  geom_line(
    data = combined_current_info_gamma1_bound_mean_info, # Use ALL step values for legend
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    color = expression(zeta[1] ~ "=" ~ Delta * " E[Prcp] (Inch)"), # Legend title for color
    linetype = "Benchmark" # Legend title for linetype
  ) +
  scale_color_manual(
    values = color_palette,
    guide = guide_legend(override.aes = list(linewidth = 1))
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash"),
    guide = guide_legend(override.aes = list(linewidth = 0.8))
  ) +
  # Theme for this temp plot only for legend extraction
  theme(
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 9),
    legend.position = "bottom", # Make sure legend is visible here
    legend.box = "horizontal", # Arrange multiple legends horizontally
    # Hide plot elements as we only need the legend
    axis.title = element_blank(),
    axis.text = element_blank(),
    plot.background = element_blank(),
    panel.background = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_blank(),
    plot.margin = unit(c(0,0,0,0), "cm") # Reduce margins
  )

plot_grob <- ggplotGrob(full_legend_plot)

legend_index <- which(sapply(plot_grob$grobs, function(x) x$name) == "guide-box")

common_legend_zeta_1 <- plot_grob$grobs[[legend_index]]


# --- Arrange Plots with the extracted common legend ---

# First, arrange the plots themselves (which have no legends)
plots_grid <- ggarrange(dry_condition, wet_condition,
                        ncol = 2, nrow = 1,
                        # No need for labels, they are now titles within plots
                        common.legend = FALSE, # Explicitly no common legend from ggarrange
                        legend = "none") # Ensure no legend is generated by ggarrange

# Then, combine the grid of plots with the extracted common legend
gamma1_bound_mean <- ggarrange(plots_grid, common_legend_zeta_1,
                            ncol = 1, nrow = 2,
                            heights = c(1, 0.2)) # Adjust heights to give space for legend

############################################################
################## current_info_gamma1_bound_var ##############################
#########################################################

current_info_gamma1_bound_var_info <- lapply(current_info_gamma1_bound_var, function(col) gen_p(col, q))

combined_current_info_gamma1_bound_var_info <- bind_rows(lapply(names(current_info_gamma1_bound_var_info), function(name) {
  df <- current_info_gamma1_bound_var_info[[name]]
  df$step <- as.numeric(name)/20 +0.75  # Add a column to track the dataset
  return(df)
}))

# Ensure step is a factor for proper animation handling
combined_current_info_gamma1_bound_var_info$step <- factor(combined_current_info_gamma1_bound_var_info$step)

# Define color palette: Steps from brown (dry) to blue (wet), plus muted colors for static lines
num_steps <- length(unique(combined_current_info_gamma1_bound_var_info$step)) 
# Generate a gradient for steps from -1 to 0 (dark red to gray50)
gradient_neg <- colorRampPalette(c("darkred", "firebrick", "darkorange", "#e2a76d"))(ceiling((num_steps - 1) / 2))

# Generate a gradient for steps from 0 to 1 (gray50 to dark blue)
gradient_pos <- colorRampPalette(c("#c0d9e8","#7aa9d6", "royalblue", "darkblue"))(floor((num_steps - 1) / 2))

# Ensure step is a factor for proper animation handling
combined_current_info_gamma1_bound_var_info$step <- factor(combined_current_info_gamma1_bound_var_info$step)

# Combine both gradients and assign gray50 to step 0
color_palette <- c(setNames(gradient_neg, as.character(seq(0.75, 0.95, length.out = length(gradient_neg)))),
                   "1" = "#808080", 
                   setNames(gradient_pos, as.character(seq(1.05, 1.25, length.out = length(gradient_pos)))))

pvflat = c(rep(mean(c(3.09,5.01,8.54,12.9,14.41)),5 ), 2, 6, 11, 20, rep(mean(c(8.5,10.8,16.5,37,37)),5 ))
#pvflat = c(rep(mean(current_info_gamma1_bound_mean_pl$`5`),5 ), 2, 6, 11, 20, rep(mean(current_info_gamma1_bound_mean_fcl$`5`),5 ))
flat = gen_p(pvflat, q)
flat$step = "flat"

# Dry Condition Plot
low_condition <- ggplot() +
  geom_line(
    data = combined_current_info_gamma1_bound_var_info %>% filter(as.numeric(as.character(step)) <= 1),
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    title = expression(zeta[2] <= 1) # Label as title
  ) +
  scale_color_manual(
    values = color_palette # Uses the global color palette
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash")
  ) +
  coord_cartesian(ylim = c(0, 1100)) +
  my_plot_theme # Apply the theme that hides legend


# Wet Condition Plot
high_condition <- ggplot() +
  geom_line(
    data = combined_current_info_gamma1_bound_var_info %>% filter(as.numeric(as.character(step)) >= 1),
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    title =  expression(zeta[2] >= 1) # Label as title
  ) +
  scale_color_manual(
    values = color_palette # Uses the global color palette
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash")
  ) +
  coord_cartesian(ylim = c(0, 1100)) +
  my_plot_theme # Apply the theme that hides legend


# --- Create a 'full' plot temporarily to extract the complete legend ---
# This plot contains all possible 'step' values and linetypes to ensure a comprehensive legend.
full_legend_plot <- ggplot() +
  geom_line(
    data = combined_current_info_gamma1_bound_var_info, # Use ALL step values for legend
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    color = expression(zeta[2] ~ "=" ~ " Sd Ratio"), # Legend title for color
    linetype = "Benchmark" # Legend title for linetype
  ) +
  scale_color_manual(
    values = color_palette,
    guide = guide_legend(override.aes = list(linewidth = 1))
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash"),
    guide = guide_legend(override.aes = list(linewidth = 0.8))
  ) +
  # Theme for this temp plot only for legend extraction
  theme(
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 9),
    legend.position = "bottom", # Make sure legend is visible here
    legend.box = "horizontal", # Arrange multiple legends horizontally
    # Hide plot elements as we only need the legend
    axis.title = element_blank(),
    axis.text = element_blank(),
    plot.background = element_blank(),
    panel.background = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_blank(),
    plot.margin = unit(c(0,0,0,0), "cm") # Reduce margins
  )

plot_grob <- ggplotGrob(full_legend_plot)

legend_index <- which(sapply(plot_grob$grobs, function(x) x$name) == "guide-box")

common_legend_zeta_2 <- plot_grob$grobs[[legend_index]]



# --- Arrange Plots with the extracted common legend ---

# First, arrange the plots themselves (which have no legends)
plots_grid <- ggarrange(low_condition, high_condition,
                        ncol = 2, nrow = 1,
                        # No need for labels, they are now titles within plots
                        common.legend = FALSE, # Explicitly no common legend from ggarrange
                        legend = "none") # Ensure no legend is generated by ggarrange

# Then, combine the grid of plots with the extracted common legend
gamma1_bound_var <- ggarrange(plots_grid, common_legend_zeta_2,
                           ncol = 1, nrow = 2,
                           heights = c(1, 0.2)) # Adjust heights to give space for legend






############################################################
################## current_info_gamma025_bound_mean ##############################
#########################################################

current_info_gamma025_bound_mean_info <- lapply(current_info_gamma025_bound_mean, function(col) gen_p(col, q))

combined_current_info_gamma025_bound_mean_info <- bind_rows(lapply(names(current_info_gamma025_bound_mean_info), function(name) {
  df <- current_info_gamma025_bound_mean_info[[name]]
  df$step <- as.numeric(name)/20-0.25  # Add a column to track the dataset
  return(df)
}))

# Ensure step is a factor for proper animation handling
combined_current_info_gamma025_bound_mean_info$step <- factor(combined_current_info_gamma025_bound_mean_info$step)

# Define color palette: Steps from brown (dry) to blue (wet), plus muted colors for static lines
num_steps <- length(unique(combined_current_info_gamma025_bound_mean_info$step)) 
# Generate a gradient for steps from -1 to 0 (dark red to gray50)
gradient_neg <- colorRampPalette(c("darkred", "firebrick", "darkorange", "#e2a76d"))(ceiling((num_steps - 1) / 2))

# Generate a gradient for steps from 0 to 1 (gray50 to dark blue)
gradient_pos <- colorRampPalette(c("#c0d9e8","#7aa9d6", "royalblue", "darkblue"))(floor((num_steps - 1) / 2))

# Combine both gradients and assign gray50 to step 0
color_palette <- c(setNames(gradient_neg, as.character(seq(-0.25, -0.05, length.out = length(gradient_neg)))),
                   "0" = "#808080", 
                   setNames(gradient_pos, as.character(seq(0.05, 0.25, length.out = length(gradient_pos)))))


pvflat = c(rep(mean(c(3.09,5.01,8.54,12.9,14.41)),5 ), 2, 6, 11, 20, rep(mean(c(8.5,10.8,16.5,37,37)),5 ))
#pvflat = c(rep(mean(current_info_gamma025_bound_mean_pl$`5`),5 ), 2, 6, 11, 20, rep(mean(current_info_gamma025_bound_mean_fcl$`5`),5 ))
flat = gen_p(pvflat, q)
flat$step = "flat"

# Dry Condition Plot
dry_condition <- ggplot() +
  geom_line(
    data = combined_current_info_gamma025_bound_mean_info %>% filter(as.numeric(as.character(step)) <= 0),
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    title = expression(zeta[1] <= 0) # Label as title
  ) +
  scale_color_manual(
    values = color_palette # Uses the global color palette
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash")
  ) +
  coord_cartesian(ylim = c(0, 600)) +
  my_plot_theme # Apply the theme that hides legend


# Wet Condition Plot
wet_condition <- ggplot() +
  geom_line(
    data = combined_current_info_gamma025_bound_mean_info %>% filter(as.numeric(as.character(step)) >= 0),
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    title =  expression(zeta[1] >= 0) # Label as title
  ) +
  scale_color_manual(
    values = color_palette # Uses the global color palette
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash")
  ) +
  coord_cartesian(ylim = c(0, 600)) +
  my_plot_theme # Apply the theme that hides legend


# --- Create a 'full' plot temporarily to extract the complete legend ---
# This plot contains all possible 'step' values and linetypes to ensure a comprehensive legend.
full_legend_plot <- ggplot() +
  geom_line(
    data = combined_current_info_gamma025_bound_mean_info, # Use ALL step values for legend
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    color = expression(zeta[1] ~ "=" ~ Delta * " E[Prcp] (Inch)"), # Legend title for color
    linetype = "Benchmark" # Legend title for linetype
  ) +
  scale_color_manual(
    values = color_palette,
    guide = guide_legend(override.aes = list(linewidth = 1))
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash"),
    guide = guide_legend(override.aes = list(linewidth = 0.8))
  ) +
  # Theme for this temp plot only for legend extraction
  theme(
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 9),
    legend.position = "bottom", # Make sure legend is visible here
    legend.box = "horizontal", # Arrange multiple legends horizontally
    # Hide plot elements as we only need the legend
    axis.title = element_blank(),
    axis.text = element_blank(),
    plot.background = element_blank(),
    panel.background = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_blank(),
    plot.margin = unit(c(0,0,0,0), "cm") # Reduce margins
  )

plot_grob <- ggplotGrob(full_legend_plot)

legend_index <- which(sapply(plot_grob$grobs, function(x) x$name) == "guide-box")

common_legend_zeta_1 <- plot_grob$grobs[[legend_index]]


# --- Arrange Plots with the extracted common legend ---

# First, arrange the plots themselves (which have no legends)
plots_grid <- ggarrange(dry_condition, wet_condition,
                        ncol = 2, nrow = 1,
                        # No need for labels, they are now titles within plots
                        common.legend = FALSE, # Explicitly no common legend from ggarrange
                        legend = "none") # Ensure no legend is generated by ggarrange

# Then, combine the grid of plots with the extracted common legend
gamma025_bound_mean <- ggarrange(plots_grid, common_legend_zeta_1,
                               ncol = 1, nrow = 2,
                               heights = c(1, 0.2)) # Adjust heights to give space for legend

############################################################
################## current_info_gamma025_bound_var ##############################
#########################################################

current_info_gamma025_bound_var_info <- lapply(current_info_gamma025_bound_var, function(col) gen_p(col, q))

combined_current_info_gamma025_bound_var_info <- bind_rows(lapply(names(current_info_gamma025_bound_var_info), function(name) {
  df <- current_info_gamma025_bound_var_info[[name]]
  df$step <- as.numeric(name)/20 +0.75  # Add a column to track the dataset
  return(df)
}))

# Ensure step is a factor for proper animation handling
combined_current_info_gamma025_bound_var_info$step <- factor(combined_current_info_gamma025_bound_var_info$step)

# Define color palette: Steps from brown (dry) to blue (wet), plus muted colors for static lines
num_steps <- length(unique(combined_current_info_gamma025_bound_var_info$step)) 
# Generate a gradient for steps from -1 to 0 (dark red to gray50)
gradient_neg <- colorRampPalette(c("darkred", "firebrick", "darkorange", "#e2a76d"))(ceiling((num_steps - 1) / 2))

# Generate a gradient for steps from 0 to 1 (gray50 to dark blue)
gradient_pos <- colorRampPalette(c("#c0d9e8","#7aa9d6", "royalblue", "darkblue"))(floor((num_steps - 1) / 2))

# Ensure step is a factor for proper animation handling
combined_current_info_gamma025_bound_var_info$step <- factor(combined_current_info_gamma025_bound_var_info$step)

# Combine both gradients and assign gray50 to step 0
color_palette <- c(setNames(gradient_neg, as.character(seq(0.75, 0.95, length.out = length(gradient_neg)))),
                   "1" = "#808080", 
                   setNames(gradient_pos, as.character(seq(1.05, 1.25, length.out = length(gradient_pos)))))

pvflat = c(rep(mean(c(3.09,5.01,8.54,12.9,14.41)),5 ), 2, 6, 11, 20, rep(mean(c(8.5,10.8,16.5,37,37)),5 ))
#pvflat = c(rep(mean(current_info_gamma025_bound_mean_pl$`5`),5 ), 2, 6, 11, 20, rep(mean(current_info_gamma025_bound_mean_fcl$`5`),5 ))
flat = gen_p(pvflat, q)
flat$step = "flat"

# Dry Condition Plot
low_condition <- ggplot() +
  geom_line(
    data = combined_current_info_gamma025_bound_var_info %>% filter(as.numeric(as.character(step)) <= 1),
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    title = expression(zeta[2] <= 1) # Label as title
  ) +
  scale_color_manual(
    values = color_palette # Uses the global color palette
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash")
  ) +
  coord_cartesian(ylim = c(0, 600)) +
  my_plot_theme # Apply the theme that hides legend


# Wet Condition Plot
high_condition <- ggplot() +
  geom_line(
    data = combined_current_info_gamma025_bound_var_info %>% filter(as.numeric(as.character(step)) >= 1),
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    title =  expression(zeta[2] >= 1) # Label as title
  ) +
  scale_color_manual(
    values = color_palette # Uses the global color palette
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash")
  ) +
  coord_cartesian(ylim = c(0, 600)) +
  my_plot_theme # Apply the theme that hides legend


# --- Create a 'full' plot temporarily to extract the complete legend ---
# This plot contains all possible 'step' values and linetypes to ensure a comprehensive legend.
full_legend_plot <- ggplot() +
  geom_line(
    data = combined_current_info_gamma025_bound_var_info, # Use ALL step values for legend
    aes(x = q, y = total, color = step),
    linewidth = 1
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
  labs(
    color = expression(zeta[2] ~ "=" ~ " Sd Ratio"), # Legend title for color
    linetype = "Benchmark" # Legend title for linetype
  ) +
  scale_color_manual(
    values = color_palette,
    guide = guide_legend(override.aes = list(linewidth = 1))
  ) +
  scale_linetype_manual(
    values = c("Status Quo - Flat" = "dotted", "Status Quo" = "dotdash"),
    guide = guide_legend(override.aes = list(linewidth = 0.8))
  ) +
  # Theme for this temp plot only for legend extraction
  theme(
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 9),
    legend.position = "bottom", # Make sure legend is visible here
    legend.box = "horizontal", # Arrange multiple legends horizontally
    # Hide plot elements as we only need the legend
    axis.title = element_blank(),
    axis.text = element_blank(),
    plot.background = element_blank(),
    panel.background = element_blank(),
    panel.grid = element_blank(),
    panel.border = element_blank(),
    plot.margin = unit(c(0,0,0,0), "cm") # Reduce margins
  )

plot_grob <- ggplotGrob(full_legend_plot)

legend_index <- which(sapply(plot_grob$grobs, function(x) x$name) == "guide-box")

common_legend_zeta_2 <- plot_grob$grobs[[legend_index]]



# --- Arrange Plots with the extracted common legend ---

# First, arrange the plots themselves (which have no legends)
plots_grid <- ggarrange(low_condition, high_condition,
                        ncol = 2, nrow = 1,
                        # No need for labels, they are now titles within plots
                        common.legend = FALSE, # Explicitly no common legend from ggarrange
                        legend = "none") # Ensure no legend is generated by ggarrange

# Then, combine the grid of plots with the extracted common legend
gamma025_bound_var <- ggarrange(plots_grid, common_legend_zeta_2,
                              ncol = 1, nrow = 2,
                              heights = c(1, 0.2)) # Adjust heights to give space for legend






############################################
########### Welfare Data #################
############################################

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

