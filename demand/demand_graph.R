setwd("C:/Users/Gordon Ji/Dropbox/UTAustin/Research/Austin Water/data")
setwd("~/Austin Water")
setwd("C:/Users/gj5378/Box/Water_Data_Share")
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

demand_2018 = read_csv(file = "demand_2018.csv")

demand_2018_using = demand_2018[which(demand_2018$CAP == 0 & demand_2018$wastewateravg == 0),]

ggplot(demand_2018_using, aes(x = quantity)) +
  geom_histogram(binwidth = 0.05, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Quantity Histogram",
       x = "Quantity (1000 Gallon)",
       y = "Frequency") +
  my_theme

ggplot(demand_2018_using, aes(x = log(quantity))) +
  geom_histogram(binwidth = 0.05, fill = "blue", color = "black", alpha = 0.7) +
  geom_vline(xintercept = log(2))+
  geom_vline(xintercept = log(6))+
  geom_vline(xintercept = log(11))+
  geom_vline(xintercept = log(20))+
  labs(title = "Quantity Histogram",
       x = "Log(Quantity) (1000 Gallon)",
       y = "Frequency") +
  my_theme

ggplot(demand_2018_using, aes(x = charge)) +
  geom_histogram(binwidth = 0.05, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Charge Histogram",
       x = "Charge ($)",
       y = "Frequency") +
  my_theme

ggplot(demand_2018_using, aes(x = log(charge) )) +
  geom_histogram(binwidth = 0.05, fill = "blue", color = "black", alpha = 0.7) +
  labs(title = "Charge Histogram",
       x = "Log(Charge) ($)",
       y = "Frequency") +
  my_theme