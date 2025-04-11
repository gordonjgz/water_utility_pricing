setwd("~/Austin Water")
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

my_theme <- theme_minimal()+
  theme(title = element_text(hjust=0.5),legend.position='bottom')
theme_set(my_theme)
big_text <- theme(text = element_text(size=24))

demand_2018 = read_csv(file = "demand_2018_using_new.csv")

#alpha = jnp.exp(
#  jnp.dot(A, beta_4) +
#    c_a
#)
  
demand_r = read_csv(file = "result.csv", col_names = F)
beta_4 = as.matrix(c(demand_r$X11[1], demand_r$X12[1],demand_r$X13[1]))
c_a = demand_r$X14[1]

beta_4_se = c(demand_r$X11[2], demand_r$X12[2],demand_r$X13[2])
c_a_se = demand_r$X14[2]

A = cbind(demand_2018$heavy_water_spa, demand_2018$bedroom, demand_2018$lawn_areaxNDVI)

alpha = exp(A %*% beta_4 +c_a)

alpha_df = demand_2018 %>% select(prem_id, bill_ym)

alpha_df$alpha = alpha

colnames(alpha_df) = c("prem_id", "bill_ym", "alpha")

rm(demand_2018)

#Nonlinear pricing of water and the assumed stochastic structure gives rise to the joint density of Wi, 
# the vector of billing cycle-level consumption values for household i, implies
#that the coeffcient on the logarithm of price for a given household cannot be interpreted
#as a price elasticity of demand. The same logic applies to the coeffcient on logarithm of
#household-level income. Nevertheless, as shown in Section 7, analogues to price and income
#elasticities can be computed with respect to the expected water demand of the household.

