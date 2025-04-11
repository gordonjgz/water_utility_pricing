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


my_theme <- theme_minimal()+
  theme(title = element_text(hjust=0.5),legend.position='bottom')
theme_set(my_theme)
big_text <- theme(text = element_text(size=24))

myd0119 = read_csv(file = "NDVI/myd0119.csv")
myd0219 = read_csv(file = "NDVI/myd0219.csv")
myd0319 = read_csv(file = "NDVI/myd0319.csv")
myd0418 = read_csv(file = "NDVI/myd0418.csv")
myd0419 = read_csv(file = "NDVI/myd0419.csv")
myd0518 = read_csv(file = "NDVI/myd0518.csv")
myd0519 = read_csv(file = "NDVI/myd0519.csv")
myd0618 = read_csv(file = "NDVI/myd0618.csv")
myd0619 = read_csv(file = "NDVI/myd0619.csv")
myd0718 = read_csv(file = "NDVI/myd0718.csv")
myd0719 = read_csv(file = "NDVI/myd0719.csv")
myd0818 = read_csv(file = "NDVI/myd0818.csv")
myd0819 = read_csv(file = "NDVI/myd0819.csv")
myd0918 = read_csv(file = "NDVI/myd0918.csv")
myd0919 = read_csv(file = "NDVI/myd0919.csv")
myd1018 = read_csv(file = "NDVI/myd1018.csv")
myd1019 = read_csv(file = "NDVI/myd1019.csv")
myd1118 = read_csv(file = "NDVI/myd1118.csv")
myd1119 = read_csv(file = "NDVI/myd1119.csv")
myd1218 = read_csv(file = "NDVI/myd1218.csv")
myd1219 = read_csv(file = "NDVI/myd1219.csv")
demand_2018 = read_csv(file = "demand_2018.csv")

myd_l = list(myd0418, myd0518, myd0618, myd0718, myd0818, myd0918, myd1018, myd1118, myd1218,
          myd0119, myd0219, myd0319,myd0419, myd0519, myd0619, myd0719, myd0819, myd0919,
          myd1019, myd1119, myd1219)

myd0418 = myd0418[order(myd0418$prem_id),]

myd_df = cbind(myd0418[, 2], myd0418[, 3], myd0418[, 6])

for (i in 2: length(myd_l)) {
  myd = myd_l[[i]]
  myd = myd[order(myd$prem_id),]
  myd_df = cbind(myd_df, myd[, 6])
}

myd_df <- myd_df %>%
  pivot_longer(
    cols = `NDVI_0418`:`NDVI_1219`, 
    names_to = "bill_ym",
    values_to = "NDVI_MYD"
  )

edit_string_billym = function(x){# x is the string in NDVI_mo_year
  x = sub("NDVI_", "", x)
  x = paste(c("20", substr(x, 3, 4), substr(x, 1, 2)), collapse = "")
  return(as.numeric(x))
}

myd_df <- myd_df %>%
  group_by(prem_id) %>%
  mutate(bill_ym =  sapply(bill_ym,edit_string_billym ))

write.csv(myd_df, file = "prem_key/myd_df.csv", row.names = F)

prem_key = read_csv(file = "prem_key/prem_key.csv")

prem_myd = merge(myd_df,prem_key, by.x = c("prem_id"), by.y = c("prem_id"), all.x = T)

weather =  read_csv(file = "Percipitation_revenue.csv")

#prem_myd_agg = merge(prem_myd_agg, weather, by.x = c("bill_ym"), by.y = c("bill_ym"), all.x = T)

prem_myd = merge(prem_myd, weather, by.x = c("bill_ym"), by.y = c("bill_ym"), all.x = T)

one.way = aov(NDVI_MYD ~ lake_water_level, data = prem_myd )

summary(one.way)

all.way = aov(NDVI_MYD ~ percipitation+MaxT+AvgT+MinT + `Dew Point` + lake_water_level, data = prem_myd )

summary(all.way)

interaction= aov(NDVI_MYD ~ percipitation*AvgT  + percipitation*lake_water_level + AvgT*lake_water_level, data = prem_myd )

summary(interaction)

blocking= aov(NDVI_MYD ~ percipitation+MaxT+AvgT+MinT + `Dew Point` + lake_water_level + usage, data = prem_myd )

summary(blocking)

library(AICcmodavg)

model.set <- list(one.way, all.way, interaction, blocking)
model.names <- c("one.way", "all.way", "interaction", "blocking")

aictab(model.set, modnames = model.names)

prem_myd$residuals = blocking$residuals

prem_myd_agg = prem_myd %>%
  group_by(bill_ym, usage) %>%
  summarise(NDVI_MYD = mean(NDVI_MYD, na.rm = T),
            residuals = mean(residuals, na.rm = T),
            category = unique(category))

prem_myd_agg$bill_ymd = as.yearmon(as.character(prem_myd_agg$bill_ym),"%Y%m")
prem_myd_agg$bill_ymd = as.Date(prem_myd_agg$bill_ymd)

ggplot(prem_myd_agg, aes(x = bill_ymd)) + 
  geom_line(aes(y = residuals, color = as.factor(usage) ), linewidth = 1.5) +
  #geom_line(aes(y = NDVI_MYD, color = as.factor(usage) ), linewidth = 1.5) +
  xlab("Year Month")+
  ylab("NDVI")+
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("NDVI") +
  #scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "Low Var", "High Var"))+
  big_text

################################
######## Weather Data ###########
################################
austin_weather = read_csv(file = "weather/3630735.csv")
austin_weather$TAVG = NULL
other_weather = read_csv(file = "weather/3630742.csv")
weather = rbind(austin_weather, other_weather)

weather$bill_ym = as.numeric(substr(weather$DATE, 1, 4))*100 + as.numeric(substr(weather$DATE, 6, 7))

weather = weather %>%
  group_by(bill_ym, STATION) %>%
  summarise(NAME = unique(NAME),
            LATITUDE = unique(LATITUDE),
            LONGITUDE = unique(LONGITUDE),
            ELEVATION = unique(ELEVATION),
            total_PRCP = sum(PRCP, na.rm = T),
            IQR_PRCP = IQR(PRCP, na.rm = T),
            mean_TMAX = mean(TMAX, na.rm = T),
            IQR_TMAX = IQR(TMAX, na.rm = T),
            TMIN = mean(TMIN, na.rm = T),
            TOBS = mean(TOBS, na.rm = T))

weather$IQR_PRCP[which(is.na(weather$IQR_PRCP) == T)] = 0

write.csv(weather, file = "weather/weather.csv", row.names = F)

austin_hh_weather = read_csv(file = "weather/Austin_hh_weather.csv")

austin_hh_weather$prem_id = austin_hh_weather$ID
austin_hh_weather$ID = NULL
austin_hh_weather$Latitude = NULL
austin_hh_weather$Longitude = NULL
austin_hh_weather$STATION = NULL
austin_hh_weather$NAME = NULL
austin_hh_weather$LATITUDE_1 = NULL
austin_hh_weather$LONGITUDE_1 = NULL
austin_hh_weather$mean_TMAX = austin_hh_weather$mean_TMAX_1
austin_hh_weather$IQR_TMAX = austin_hh_weather$IQR_TMAX_1
austin_hh_weather$IQR_TMAX_1 = NULL
austin_hh_weather$mean_TMAX_1 = NULL

percipitation_revenue = read_csv(file = "percipitation_revenue.csv")

lake_level = data.frame(cbind(percipitation_revenue$bill_ym,percipitation_revenue$lake_water_level, percipitation_revenue$Drought_stage))
colnames(lake_level) = c("bill_ym", "lake_level", "dr_stage")

austin_hh_weather = merge(austin_hh_weather, lake_level, by.x = c("bill_ym"),  by.y = c("bill_ym"), all.x = T)

austin_hh_weather$lake_level = as.numeric(austin_hh_weather$lake_level)

austin_hh_weather$dr_stage[which(austin_hh_weather$dr_stage == "None")] = "-1"

usage_charge_2018 = read_csv(file = "usage_charge_2018.csv")

usage_charge_2018$cur_read_month = month(usage_charge_2018$cur_read_dttm)
usage_charge_2018$cur_read_day = day(usage_charge_2018$cur_read_dttm)

usage_charge_2018$prv_read_month = month(usage_charge_2018$prv_read_dttm)
usage_charge_2018$prv_read_day = day(usage_charge_2018$prv_read_dttm)

prem_id_bill_cycle = usage_charge_2018 %>%
  group_by(prem_id, bill_ym) %>%
  summarize(cur_read_month = median(cur_read_month),
            cur_read_day = median(cur_read_day),
            prv_read_month = median(prv_read_month),
            prv_read_day = median(prv_read_day))

prem_id_bill_cycle$cur_read_full = case_when(
  prem_id_bill_cycle$cur_read_month %in% c(1, 3, 5, 7, 8, 10, 12) ~ 31,
  prem_id_bill_cycle$cur_read_month %in% c(4, 6, 9, 11) ~ 30,
  prem_id_bill_cycle$cur_read_month %in% c(2) ~ 28
)

prem_id_bill_cycle$prv_read_full = case_when(
  prem_id_bill_cycle$prv_read_month %in% c(1, 3, 5, 7, 8, 10, 12) ~ 31,
  prem_id_bill_cycle$prv_read_month %in% c(4, 6, 9, 11) ~ 30,
  prem_id_bill_cycle$prv_read_month %in% c(2) ~ 28
)

prem_id_bill_cycle$cur_ratio = prem_id_bill_cycle$cur_read_day / prem_id_bill_cycle$cur_read_full
prem_id_bill_cycle$prv_ratio = 1-prem_id_bill_cycle$prv_read_day / prem_id_bill_cycle$prv_read_full
prem_id_bill_cycle$sum_ratio = prem_id_bill_cycle$cur_ratio + prem_id_bill_cycle$prv_ratio
prem_id_bill_cycle$cur_ratio = prem_id_bill_cycle$cur_ratio/prem_id_bill_cycle$sum_ratio
prem_id_bill_cycle$prv_ratio = prem_id_bill_cycle$prv_ratio/prem_id_bill_cycle$sum_ratio

prem_id_bill_cycle = data.frame(cbind(prem_id_bill_cycle$prem_id, prem_id_bill_cycle$bill_ym,
                                      prem_id_bill_cycle$cur_ratio, prem_id_bill_cycle$prv_ratio))
colnames(prem_id_bill_cycle) = c("prem_id", "bill_ym", "cur_ratio", "prv_ratio")

prem_id_end_date = read_csv(file = "prem_key/prem_id_end_date.csv")

prem_id_end_date = prem_id_end_date %>%
  group_by(prem_id) %>%
  mutate(prv_end_dt = lag(bseg_end_dt, 1))

prem_id_end_date = prem_id_end_date %>%
  group_by(prem_id) %>%
  mutate(prv_month = lag(month, 1))

prem_id_end_date$cur_read_full = case_when(
  prem_id_end_date$month %in% c(1, 3, 5, 7, 8, 10, 12) ~ 31,
  prem_id_end_date$month %in% c(4, 6, 9, 11) ~ 30,
  prem_id_end_date$month %in% c(2) ~ 28
)

prem_id_end_date$prv_read_full = case_when(
  prem_id_end_date$prv_month %in% c(1, 3, 5, 7, 8, 10, 12) ~ 31,
  prem_id_end_date$prv_month %in% c(4, 6, 9, 11) ~ 30,
  prem_id_end_date$prv_month %in% c(2) ~ 28
)

prem_id_end_date$cur_ratio = prem_id_end_date$bseg_end_dt/ prem_id_end_date$cur_read_full
prem_id_end_date$prv_ratio = 1-prem_id_end_date$prv_end_dt / prem_id_end_date$prv_read_full
prem_id_end_date$sum_ratio = prem_id_end_date$cur_ratio + prem_id_end_date$prv_ratio
prem_id_end_date$cur_ratio = prem_id_end_date$cur_ratio/prem_id_end_date$sum_ratio
prem_id_end_date$prv_ratio = prem_id_end_date$prv_ratio/prem_id_end_date$sum_ratio

write.csv(prem_id_end_date, file = "prem_key/prem_id_end_date.csv",row.names = F)

austin_hh_weather = merge(austin_hh_weather, prem_id_bill_cycle, by.x = c("prem_id", "bill_ym"),
                          by.y = c("prem_id", "bill_ym"), all.x = T)

austin_hh_weather = austin_hh_weather %>%
  group_by(prem_id) %>%
  mutate(prv_total_PRCP = lag(total_PRCP, 1),
         prv_IQR_PRCP = lag(IQR_PRCP, 1),
         prv_mean_TMAX_1 = lag(mean_TMAX, 1),
         prv_IQR_TMAX_1 = lag(IQR_TMAX, 1),
         prv_lake_level = lag(lake_level, 1),
         prv_dr_stage = lag(dr_stage, 1))

austin_hh_weather = austin_hh_weather %>%
  group_by(prem_id) %>%
  mutate(final_total_PRCP = total_PRCP*cur_ratio + prv_total_PRCP*prv_ratio,
         final_IQR_PRCP = IQR_PRCP*cur_ratio + prv_IQR_PRCP*prv_ratio,
         final_mean_TMAX = mean_TMAX*cur_ratio + prv_mean_TMAX_1*prv_ratio,
         final_IQR_TMAX = IQR_TMAX*cur_ratio + prv_IQR_TMAX_1*prv_ratio,
         final_lake_level = lake_level*cur_ratio + prv_lake_level*prv_ratio)

austin_hh_weather = data.frame(cbind(austin_hh_weather$prem_id, austin_hh_weather$bill_ym, 
                                     austin_hh_weather$final_total_PRCP, 
                                     austin_hh_weather$final_IQR_PRCP,
                                     austin_hh_weather$final_mean_TMAX,
                                     austin_hh_weather$final_IQR_TMAX,
                                     austin_hh_weather$final_lake_level))
colnames(austin_hh_weather) = c("prem_id", "bill_ym", "total_PRCP", "IQR_PRCP", 
                                "mean_TMAX_1", "IQR_TMAX_1", "lake_level")

write.csv(austin_hh_weather, file = "austin_hh_weather.csv", row.names = F)

myd_df = read_csv(file = "prem_key/myd_df.csv")

myd_df = merge(myd_df, prem_id_end_date, by.x = c("prem_id", "bill_ym"),
               by.y = c("prem_id", "bill_ym"), all.x = T)

myd_df = myd_df %>%
  group_by(prem_id) %>%
  mutate(prv_NDVI_MYD = lag(NDVI_MYD, 1))

myd_df = myd_df %>%
  group_by(prem_id) %>%
  mutate(final_NDVI_MYD = NDVI_MYD*cur_ratio/sum_ratio + prv_NDVI_MYD*prv_ratio/sum_ratio)

myd_df <- myd_df %>%
  select(prem_id, bill_ym, PROP_ID, final_NDVI_MYD) 

myd_df$NDVI_MYD = myd_df$final_NDVI_MYD
myd_df$final_NDVI_MYD = NULL
write.csv(myd_df, file = "prem_key/myd_df.csv", row.names = F)