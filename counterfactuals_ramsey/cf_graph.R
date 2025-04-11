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

cf_cs_p_change = read_csv("cf_cs_p_change.csv")

cf_cs_p_change_mean = cf_cs_p_change %>%
  group_by(prem_id) %>%
  summarise(
    across(-c(bill_ym), mean, na.rm = TRUE)
  )

cf_ps_p_change = read_csv("cf_ps_p_change.csv")