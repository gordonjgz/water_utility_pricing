setwd("C:/Users/Gordon Ji/Dropbox/UTAustin/Research/Austin Water/data")
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

my_theme <- theme_minimal()+
  theme(title = element_text(hjust=0.5),legend.position='bottom')
theme_set(my_theme)
big_text <- theme(text = element_text(size=24))

final_ndvi <- read_csv("prem_key/final_ndvi.csv")

final_ndvi = final_ndvi %>% arrange(prem_id, bill_ym)  # Ascending order

final_ndvi$Date <- as.Date(paste0(final_ndvi$bill_ym, "01"), format = "%Y%m%d")

breaks <- c(1, 100, 500, 2000, 5000)  # Adjust these breaks based on your data range

ggplot(final_ndvi, aes(x = Date, y = NDVI)) +
  geom_bin2d(bins = 50) +  
  scale_fill_gradientn(
    colors  = c("#f7fbff", "#a6cbe3", "#67a9cf", "#1c6f6f"),
    breaks = breaks,  # Apply custom breaks
    trans = "log",  # Keep log scale for contrast
    name = "Households"
  ) +  
  labs(title = "NDVI Over Time", 
       x = "Date", 
       y = "NDVI") +
  theme_minimal()+big_text

prem_id_end_date = read_csv(file = "prem_key/prem_id_end_date.csv")

prem_id_end_date$total_day = case_when(
  prem_id_end_date$month == 1 ~ 31,
  prem_id_end_date$month == 2 ~ 28,
  prem_id_end_date$month == 3 ~ 31,
  prem_id_end_date$month == 4 ~ 30,
  prem_id_end_date$month == 5 ~ 31,
  prem_id_end_date$month == 6 ~ 30,
  prem_id_end_date$month == 7 ~ 31,
  prem_id_end_date$month == 8 ~ 31,
  prem_id_end_date$month == 9 ~ 30,
  prem_id_end_date$month == 10 ~ 31,
  prem_id_end_date$month == 11 ~ 30,
  prem_id_end_date$month == 12 ~ 31
)

prem_id_end_date$current_p = prem_id_end_date$bseg_end_dt/prem_id_end_date$total_day

prem_id_end_date <- prem_id_end_date %>%
  arrange(prem_id,bill_ym) %>%  
  group_by(prem_id) %>%  
  mutate(spillover_p = lag(1-current_p, 1))

prem_id_end_date$p_thismonth = prem_id_end_date$current_p / (prem_id_end_date$current_p + prem_id_end_date$spillover_p)
prem_id_end_date$p_lastmonth = prem_id_end_date$spillover_p / (prem_id_end_date$current_p + prem_id_end_date$spillover_p)

write.csv(prem_id_end_date, file = "prem_key/prem_id_end_date.csv", row.names = F)

final_ndvi <- final_ndvi %>%
  arrange(prem_id, Date) %>%  # Ensure the data is ordered by prim_id and Date
  group_by(prem_id) %>%  # Group by prim_id
  mutate(prev_NDVI = lag(NDVI, 1))  # Create the lag 1 variable

final_ndvi = merge(final_ndvi, prem_id_end_date, by.x = c("prem_id", "bill_ym"), by.y = c("prem_id", "bill_ym"), all.x = T)

final_ndvi$NDVI_final = final_ndvi$NDVI*final_ndvi$p_thismonth + final_ndvi$prev_NDVI*final_ndvi$p_lastmonth

final_ndvi <- final_ndvi %>%
  arrange(prem_id, Date) %>%  # Ensure the data is ordered by prim_id and Date
  group_by(prem_id) %>%  # Group by prim_id
  mutate(prev_NDVI_final = lag(NDVI_final, 1))  # Create the lag 1 variable

final_ndvi_small = final_ndvi %>%
  select(prem_id, bill_ym, prev_NDVI_final, NDVI_final) %>%
  drop_na()

summary(final_ndvi_small$NDVI_final)

####Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#-0.3564  0.3267  0.4038  0.3979  0.4747  0.7857 

summary(final_ndvi_small$prev_NDVI_final)

####Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
#-0.3564  0.3185  0.3997  0.3925  0.4728  0.7857 

write.csv(final_ndvi_small, file = "prem_key/final_ndvi_small.csv", row.names = F)
