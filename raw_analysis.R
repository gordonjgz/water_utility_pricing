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

charge = read_csv("raw/UT_DataRequest_BilledCharges.csv")
charge$PREM_ID = as.numeric(charge$PREM_ID)
charge$MTR_ID = as.numeric(charge$MTR_ID)

usage = read_csv("raw/UT_DataRequest_BilledUsage.csv")
usage$PREM_ID = as.numeric(usage$PREM_ID)
usage$MTR_ID = as.numeric(usage$MTR_ID)

#PREM_ID = Premise unique identifier
#LAT & LONG = Premise latitude and longitude
#MTR_ID = Meter unique identifier
#BADGE_NBR = Meter unique identifier
#VIRTUAL_CHAN_ID = Change ID for compound meters, likely none included with this data
#BILL_YM = Bill year and month yyyymm
#COMPLETE_DTTM = Bill date
#BSEG_END_DT = Bill segment end date
#CUR_READ_DTTM = Current meter read date, should match bill segment end date
#PRV_READ_DTTM = Previous meter read date
#FINAL_REG_QTY = Billed consumption in hundreds of gallons between the current and previous meter read
#UOM_CD = Units for FINAL_REG_QTY, should all be HGAL
#RC_TYPE_FLG = Charge type (F = fixed, Q = variable, A = adjustment)
#CHARGE = Charge amount in dollars
#RS_CD = Rate code (only included W-RES in the data for single-family residential)

usage$BADGE_NBR = NULL
usage$VIRTUAL_CHAN_ID = NULL
usage$UOM_CD = NULL
usage$RS_CD = NULL
charge$RS_CD = NULL

charge_usage = merge(charge, usage, by.x = c("PREM_ID", "BILL_YM"), 
                     by.y = c("PREM_ID", "BILL_YM"), all.x = TRUE, all.y = T)

charge_usage = charge_usage[order(charge_usage$PREM_ID, charge_usage$BILL_YM),]
charge_usage$COMPLETE_DTTM = charge_usage$COMPLETE_DTTM.x
charge_usage$COMPLETE_DTTM.x = NULL
charge_usage$COMPLETE_DTTM.y = NULL

charge_usage$BSEG_END_DT = charge_usage$BSEG_END_DT.x
charge_usage$BSEG_END_DT.x = NULL
charge_usage$BSEG_END_DT.y = NULL

charge_usage$MTR_ID = charge_usage$MTR_ID.x
charge_usage$MTR_ID.x = NULL
charge_usage$MTR_ID.y = NULL

write.csv(charge_usage, file = "data/usage_charge_merge.csv", row.names = F)

## Some months (e.g. 2012.01) does not have quantity, filter these out

charge_usage = read_csv(file = "data/usage_charge_merge.csv")

charge_usage = charge_usage[which(is.na(charge_usage$FINAL_REG_QTY) == F ),]

usage_charge_2018 = read_csv(file = "usage_charge_2018.csv") # from 201602 ~ 202002

#usage_2018$length = aggregate(rate_change~bill_ym+prem_id+ mtr_id, FUN = length, data = usage_charge_2018)$rate_change

#usage_2018 = aggregate(rate_change~rc_type_flg + bill_ym+prem_id+ mtr_id, FUN = unique, data = usage_charge_2018)

usage_2018= usage_charge_2018   %>%
 group_by(rc_type_flg, bill_ym, prem_id) %>%
  summarize(quantity = sum(quantity, na.rm = T), 
            length = n())

charge_2018= usage_charge_2018   %>%
  group_by(bill_ym, prem_id) %>%
  summarize(charge = sum(charge, na.rm = T), 
            length = n())

#usage_2018 = usage_2018[which(usage_2018$quantity !=0),]

agg_quantity = function(q){# q is a vector of quantities
  if(length(unique(q)) == 1 ){
    return(unique(q))
  }else{
    return(mean(q))
  }
}

usage_2018= usage_2018   %>%
  group_by(prem_id,bill_ym) %>%
  summarize(quantity = agg_quantity(quantity),
            length = mean(length))

usage_2018$rate_change= case_when(
  usage_2018$bill_ym < 201805 ~ 0,
  usage_2018$bill_ym >= 201805 ~ 1)

charge_2018$rate_change= case_when(
  charge_2018$bill_ym < 201805 ~ 0,
  charge_2018$bill_ym >= 201805 ~ 1)

write.csv(usage_2018, file = "usage_2018.csv", row.names = F)
write.csv(charge_2018, file = "charge_2018.csv", row.names = F)

usage_2018_before = usage_2018[which(usage_2018$rate_change == 0),]
usage_2018_after = usage_2018[which(usage_2018$rate_change == 1),]

charge_2018_before = charge_2018[which(charge_2018$rate_change == 0),]
charge_2018_after = charge_2018[which(charge_2018$rate_change == 1),]

usage_2018_before= usage_2018_before   %>%
  group_by(prem_id) %>%
  summarize(mean_quantity = mean(quantity),
            sd_quantity = sd(quantity),
            length = n())

charge_2018_before= charge_2018_before   %>%
  group_by(prem_id) %>%
  summarize(mean_charge = mean(charge),
            sd_charge = sd(charge),
            length = n())

usage_2018_before = usage_2018_before[which(usage_2018_before$sd_quantity<5000),]
usage_2018_before = usage_2018_before[which(usage_2018_before$mean_quantity>0),]
usage_2018_before = usage_2018_before[which(usage_2018_before$length>=20),]

charge_2018_before = charge_2018_before[which(charge_2018_before$length>=20),]

usage_2018_before$usage= case_when(
  usage_2018_before$mean_quantity <=20 ~ 1,
  usage_2018_before$mean_quantity >20 & usage_2018_before$mean_quantity <=60  ~ 2,
  usage_2018_before$mean_quantity >60 &usage_2018_before$mean_quantity <=110 ~ 3,
  usage_2018_before$mean_quantity >110 &usage_2018_before$mean_quantity <=200 ~ 4,
  usage_2018_before$mean_quantity >200 ~ 5)

usage_2018_before$coeff = usage_2018_before$sd_quantity / usage_2018_before$mean_quantity

ggplot(usage_2018_before, aes(x = mean_quantity, y = coeff)) + 
  geom_point(aes(color = usage),alpha = 0.3) +
  #geom_segment(aes(x = 0, y = 0, xend = 0, yend = 10)) +
  #geom_segment(aes(x = 0.5, y = 0, xend = 0.5, yend = 10))+
  #coord_cartesian(xlim = c(0, 0.5), ylim = c(0, 10), expand = FALSE) +
  scale_color_gradient(low = "white", high = "firebrick4") + 
  big_text

usage_2018_before$coeff_level= case_when(
  #usage_2018_before$coeff <=2 ~ 1,
  #usage_2018_before$coeff >2 ~ 2)
  usage_2018_before$coeff <=1 ~ 1,  
  usage_2018_before$coeff >1 ~ 2)

usage_2018_before$category= case_when(
  usage_2018_before$usage ==1 & usage_2018_before$coeff_level == 1 ~ "1_low",
  usage_2018_before$usage ==1 & usage_2018_before$coeff_level == 2 ~ "1_high",
  usage_2018_before$usage ==2 & usage_2018_before$coeff_level == 1 ~ "2_low",
  usage_2018_before$usage ==2 & usage_2018_before$coeff_level == 2 ~ "2_high",
  usage_2018_before$usage ==3 & usage_2018_before$coeff_level == 1 ~ "3_low",
  usage_2018_before$usage ==3 & usage_2018_before$coeff_level == 2 ~ "3_high",
  usage_2018_before$usage ==4 & usage_2018_before$coeff_level == 1 ~ "4_low",
  usage_2018_before$usage ==4 & usage_2018_before$coeff_level == 2 ~ "4_high",
  usage_2018_before$usage ==5 & usage_2018_before$coeff_level == 1 ~ "5_low",
  usage_2018_before$usage ==5 & usage_2018_before$coeff_level == 2 ~ "5_high")

usage_2018_before   %>%
  group_by(category) %>%
  summarize(percentage = n()/nrow(usage_2018_before))

# A tibble: 10 x 2
#category percentage
#<chr>         <dbl>
 # 1 1_high      0.0177 
#2 1_low       0.0846 31.4%
#3 2_high      0.0386 
#4 2_low       0.505  
#5 3_high      0.0238 
#6 3_low       0.246  
#7 4_high      0.00901
#8 4_low       0.0620 
#9 5_high      0.00243
#10 5_low       0.0103 

charge_2018_before = merge(charge_2018_before, usage_2018_before, by.x = c("prem_id"), by.y = c("prem_id"), all.x = F, all.y = T)

charge_2018_before %>%
  group_by(category) %>%
  summarize(charge_sum_percentage = sum(mean_charge) / sum(charge_2018_before$mean_charge))

# A tibble: 10 x 2
#category charge_sum_percentage
#<chr>                    <dbl>
#1 1_high                 0.00518
#2 1_low                  0.0237 
#3 2_high                 0.0287 
#4 2_low                  0.288  
#5 3_high                 0.0398 
#6 3_low                  0.314  
#7 4_high                 0.0304 
#8 4_low                  0.175  
#9 5_high                 0.0205 
#10 5_low                  0.0752 

usage_2018_before_slim = data.frame(cbind(usage_2018_before$prem_id, usage_2018_before$usage, usage_2018_before$coeff_level))
colnames(usage_2018_before_slim) = c ("prem_id", "before_usage_level", "before_coeff_level")

usage_2018_merge = merge(usage_2018, usage_2018_before_slim, by.x = c("prem_id"), by.y = c("prem_id"), all.x = F, all.y = T)

usage_2018_merge$category= case_when(
  usage_2018_merge$before_usage_level ==1 & usage_2018_merge$before_coeff_level == 1 ~ "1_low",
  usage_2018_merge$before_usage_level ==1 & usage_2018_merge$before_coeff_level == 2 ~ "1_high",
  usage_2018_merge$before_usage_level ==2 & usage_2018_merge$before_coeff_level == 1 ~ "2_low",
  usage_2018_merge$before_usage_level ==2 & usage_2018_merge$before_coeff_level == 2 ~ "2_high",
  usage_2018_merge$before_usage_level ==3 & usage_2018_merge$before_coeff_level == 1 ~ "3_low",
  usage_2018_merge$before_usage_level ==3 & usage_2018_merge$before_coeff_level == 2 ~ "3_high",
  usage_2018_merge$before_usage_level ==4 & usage_2018_merge$before_coeff_level == 1 ~ "4_low",
  usage_2018_merge$before_usage_level ==4 & usage_2018_merge$before_coeff_level == 2 ~ "4_high",
  usage_2018_merge$before_usage_level ==5 & usage_2018_merge$before_coeff_level == 1 ~ "5_low",
  usage_2018_merge$before_usage_level ==5 & usage_2018_merge$before_coeff_level == 2 ~ "5_high")

usage_2018_merge$months_after= case_when(
  usage_2018_merge$bill_ym == 201602 ~ -27,
  usage_2018_merge$bill_ym == 201603 ~ -26,
  usage_2018_merge$bill_ym == 201604 ~ -25,
  usage_2018_merge$bill_ym == 201605 ~ -24,
  usage_2018_merge$bill_ym == 201606 ~ -23,
  usage_2018_merge$bill_ym == 201607 ~ -22,
  usage_2018_merge$bill_ym == 201608 ~ -21,
  usage_2018_merge$bill_ym == 201609 ~ -20,
  usage_2018_merge$bill_ym == 201610 ~ -19,
  usage_2018_merge$bill_ym == 201611 ~ -18,
  usage_2018_merge$bill_ym == 201612 ~ -17,
  usage_2018_merge$bill_ym == 201701 ~ -16,
  usage_2018_merge$bill_ym == 201702 ~ -15,
  usage_2018_merge$bill_ym == 201703 ~ -14,
  usage_2018_merge$bill_ym == 201704 ~ -13,
  usage_2018_merge$bill_ym == 201705 ~ -12,
  usage_2018_merge$bill_ym == 201706 ~ -11,
  usage_2018_merge$bill_ym == 201707 ~ -10,
  usage_2018_merge$bill_ym == 201708 ~ -9,
  usage_2018_merge$bill_ym == 201709 ~ -8,
  usage_2018_merge$bill_ym == 201710 ~ -7,
  usage_2018_merge$bill_ym == 201711 ~ -6,
  usage_2018_merge$bill_ym == 201712 ~ -5,
  usage_2018_merge$bill_ym == 201801 ~ -4,
  usage_2018_merge$bill_ym == 201802 ~ -3,
  usage_2018_merge$bill_ym == 201803 ~ -2,
  usage_2018_merge$bill_ym == 201804 ~ -1,
  usage_2018_merge$bill_ym == 201805 ~ 0,
  usage_2018_merge$bill_ym == 201806 ~ 1,
  usage_2018_merge$bill_ym == 201807 ~ 2,
  usage_2018_merge$bill_ym == 201808 ~ 3,
  usage_2018_merge$bill_ym == 201809 ~ 4,
  usage_2018_merge$bill_ym == 201810 ~ 5,
  usage_2018_merge$bill_ym == 201811 ~ 6,
  usage_2018_merge$bill_ym == 201812 ~ 7,
  usage_2018_merge$bill_ym == 201901 ~ 8,
  usage_2018_merge$bill_ym == 201902 ~ 9,
  usage_2018_merge$bill_ym == 201903 ~ 10,
  usage_2018_merge$bill_ym == 201904 ~ 11,
  usage_2018_merge$bill_ym == 201905 ~ 12,
  usage_2018_merge$bill_ym == 201906 ~ 13,
  usage_2018_merge$bill_ym == 201907 ~ 14,
  usage_2018_merge$bill_ym == 201908 ~ 15,
  usage_2018_merge$bill_ym == 201909 ~ 16,
  usage_2018_merge$bill_ym == 201910 ~ 17,
  usage_2018_merge$bill_ym == 201911 ~ 18,
  usage_2018_merge$bill_ym == 201912 ~ 19,
  usage_2018_merge$bill_ym == 202001 ~ 20,
  usage_2018_merge$bill_ym == 202002 ~ 21
  )

#charge_2018_merge = merge(charge_2018, usage_2018_merge, by.x = c("prem_id", "bill_ym"), by.y = c("prem_id", "bill_ym"), all.x = T, all.y = T)


write.csv(charge_2018_merge, file = "charge_2018_merge.csv", row.names = F)


charge_2018_merge = read_csv(file = "charge_2018_merge.csv")
usage_2018_merge = read_csv(file = "usage_2018_merge.csv")

usage_2018_level= usage_2018_merge   %>%
  group_by(category, months_after) %>%
  summarize(before_usage_level = unique(before_usage_level),
    sum_quantity = sum(quantity),
            quantity = mean(quantity))

percipitation_revenue = read_csv(file = "Percipitation_revenue.csv")

 percipitation_revenue$months_after= case_when(
   percipitation_revenue$bill_ym == 201601 ~ -28,
   percipitation_revenue$bill_ym == 201602 ~ -27,
   percipitation_revenue$bill_ym == 201603 ~ -26,
   percipitation_revenue$bill_ym == 201604 ~ -25,
   percipitation_revenue$bill_ym == 201605 ~ -24,
   percipitation_revenue$bill_ym == 201606 ~ -23,
   percipitation_revenue$bill_ym == 201607 ~ -22,
   percipitation_revenue$bill_ym == 201608 ~ -21,
   percipitation_revenue$bill_ym == 201609 ~ -20,
   percipitation_revenue$bill_ym == 201610 ~ -19,
   percipitation_revenue$bill_ym == 201611 ~ -18,
   percipitation_revenue$bill_ym == 201612 ~ -17,
   percipitation_revenue$bill_ym == 201701 ~ -16,
   percipitation_revenue$bill_ym == 201702 ~ -15,
   percipitation_revenue$bill_ym == 201703 ~ -14,
   percipitation_revenue$bill_ym == 201704 ~ -13,
   percipitation_revenue$bill_ym == 201705 ~ -12,
   percipitation_revenue$bill_ym == 201706 ~ -11,
   percipitation_revenue$bill_ym == 201707 ~ -10,
   percipitation_revenue$bill_ym == 201708 ~ -9,
   percipitation_revenue$bill_ym == 201709 ~ -8,
   percipitation_revenue$bill_ym == 201710 ~ -7,
   percipitation_revenue$bill_ym == 201711 ~ -6,
   percipitation_revenue$bill_ym == 201712 ~ -5,
   percipitation_revenue$bill_ym == 201801 ~ -4,
   percipitation_revenue$bill_ym == 201802 ~ -3,
   percipitation_revenue$bill_ym == 201803 ~ -2,
   percipitation_revenue$bill_ym == 201804 ~ -1,
   percipitation_revenue$bill_ym == 201805 ~ 0,
   percipitation_revenue$bill_ym == 201806 ~ 1,
   percipitation_revenue$bill_ym == 201807 ~ 2,
   percipitation_revenue$bill_ym == 201808 ~ 3,
   percipitation_revenue$bill_ym == 201809 ~ 4,
   percipitation_revenue$bill_ym == 201810 ~ 5,
   percipitation_revenue$bill_ym == 201811 ~ 6,
   percipitation_revenue$bill_ym == 201812 ~ 7,
   percipitation_revenue$bill_ym == 201901 ~ 8,
   percipitation_revenue$bill_ym == 201902 ~ 9,
   percipitation_revenue$bill_ym == 201903 ~ 10,
   percipitation_revenue$bill_ym == 201904 ~ 11,
   percipitation_revenue$bill_ym == 201905 ~ 12,
   percipitation_revenue$bill_ym == 201906 ~ 13,
   percipitation_revenue$bill_ym == 201907 ~ 14,
   percipitation_revenue$bill_ym == 201908 ~ 15,
   percipitation_revenue$bill_ym == 201909 ~ 16,
   percipitation_revenue$bill_ym == 201910 ~ 17,
   percipitation_revenue$bill_ym == 201911 ~ 18,
   percipitation_revenue$bill_ym == 201912 ~ 19,
   percipitation_revenue$bill_ym == 202001 ~ 20,
   percipitation_revenue$bill_ym == 202002 ~ 21
)

usage_2018_level = merge(usage_2018_level,  percipitation_revenue, by.x = c("months_after"), by.y = c("months_after"), all.x = T, all.y = F)
usage_2018_level$sum_quantity = usage_2018_level$sum_quantity*100/1000000

q1 = ggplot(usage_2018_level[which(usage_2018_level$category == "1_low" | usage_2018_level$category == "1_high"),], aes(x = months_after)) + 
  geom_line(aes(y = quantity, color = category), linewidth = 1.5) +
  #geom_line(aes(y = percipitation/ (50/50), color = "#247672"), linewidth = 1.5, linetype = "longdash") +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 50), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Mean Usage (hundreds gallon)")+
  geom_text(aes(x = 19, y = 1, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 1, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 1, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 1, label = "2017"), color = "black") +
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Avg Usage") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  #scale_color_manual(values = c("#104507","#ba0951", "#25738b"),labels = c("Percipitation", "High Var", "Low Var"))+
  #scale_y_continuous(
   # "Mean Usage (hundreds gallon)", 
    #sec.axis = sec_axis(~ . * 50/50, name = "Percipitation (Inches)")
  #) +
  big_text

q2 = ggplot(usage_2018_level[which(usage_2018_level$category == "2_low" | usage_2018_level$category == "2_high"),], aes(x = months_after)) + 
  geom_line(aes(y = quantity, color = category), linewidth = 1.5) +
  #geom_line(aes(y = percipitation/ (50/100), color = "#247672"), linewidth = 1.5, linetype = "longdash") +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 100), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Mean Usage (hundreds gallon)")+
  geom_text(aes(x = 19, y = 2, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 2, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 2, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 2, label = "2017"), color = "black") +
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Avg Usage") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  #scale_color_manual(values = c("#104507","#ba0951", "#25738b"),labels = c("Percipitation", "High Var", "Low Var"))+
  #scale_y_continuous(
   # "Mean Usage (hundreds gallon)", 
  #  sec.axis = sec_axis(~ . * 50/100, name = "Percipitation (Inches)")
  #) +
  big_text

q3 = ggplot(usage_2018_level[which(usage_2018_level$category == "3_low" | usage_2018_level$category == "3_high"),], aes(x = months_after)) + 
  geom_line(aes(y = quantity, color = category), linewidth = 1.5) +
  #geom_line(aes(y = percipitation/ (50/200), color = "#247672"), linewidth = 1.5, linetype = "longdash") +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 200), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Mean Usage (hundreds gallon)")+
  geom_text(aes(x = 19, y = 5, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 5, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 5, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 5, label = "2017"), color = "black") +
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Avg Usage") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  #scale_color_manual(values = c("#104507","#ba0951", "#25738b"),labels = c("Percipitation", "High Var", "Low Var"))+
  #scale_y_continuous(
   #"Mean Usage (hundreds gallon)", 
   #sec.axis = sec_axis(~ . * 50/200, name = "Percipitation (Inches)")
  #) +
  big_text

q4 = ggplot(usage_2018_level[which(usage_2018_level$category == "4_low" | usage_2018_level$category == "4_high"),], aes(x = months_after)) + 
  geom_line(aes(y = quantity, color = category), linewidth = 1.5) +
  #geom_line(aes(y = percipitation/ (50/300), color = "#247672"), linewidth = 1.5, linetype = "longdash") +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 300), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Mean Usage (hundreds gallon)")+
  geom_text(aes(x = 19, y = 10, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 10, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 10, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 10, label = "2017"), color = "black") +
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Avg Usage") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  #scale_color_manual(values = c("#104507","#ba0951", "#25738b"),labels = c("Percipitation", "High Var", "Low Var"))+
  #scale_y_continuous(
   # "Mean Usage (hundreds gallon)", 
   # sec.axis = sec_axis(~ . * 50/300, name = "Percipitation (Inches)")
  #) +
  big_text

q5 = ggplot(usage_2018_level[which(usage_2018_level$category == "5_low" | usage_2018_level$category == "5_high"),], aes(x = months_after)) + 
  geom_line(aes(y = quantity, color = category), linewidth = 1.5) +
  #geom_line(aes(y = percipitation/ (50/700), color = "#247672"), linewidth = 1.5, linetype = "longdash") +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 700), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Mean Usage (hundreds gallon)")+
  geom_text(aes(x = 19, y = 10, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 10, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 10, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 10, label = "2017"), color = "black") +
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Avg Usage") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  #scale_color_manual(values = c("#104507","#ba0951", "#25738b"),labels = c("Percipitation", "High Var", "Low Var"))+
  #scale_y_continuous(
   # "Mean Usage (hundreds gallon)", 
  #  sec.axis = sec_axis(~ . * 50/700, name = "Percipitation (Inches)")
  #) +
  big_text

usage_2018_level$bill_ym = as.yearmon(as.character(usage_2018_level$bill_ym),"%Y%m")
usage_2018_level$bill_ym = as.Date(usage_2018_level$bill_ym)

usage_2018_tier = usage_2018_merge   %>%
  group_by(before_usage_level, months_after) %>%
  summarize(
            sum_quantity = sum(quantity),
            quantity = mean(quantity))

ggplot(usage_2018_tier[which(usage_2018_tier$before_usage_level == 2),], aes(x = months_after)) + 
  geom_line(aes(y = quantity, color = "#ba0951"), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 100), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Mean Usage (hundreds gallon)")+
  geom_text(aes(x = 19, y = 1, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 1, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 1, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 1, label = "2017"), color = "black") +
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Consumer Monthly Avg Water Usage - Tier 2") +
  #scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  big_text

ggplot(usage_2018_tier[which(usage_2018_tier$before_usage_level == 3),], aes(x = months_after)) + 
  geom_line(aes(y = quantity, color = "#ba0951"), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 200), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Mean Usage (hundreds gallon)")+
  geom_text(aes(x = 19, y = 1, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 1, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 1, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 1, label = "2017"), color = "black") +
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Consumer Monthly Avg Water Usage - Tier 3") +
  #scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  big_text

ggplot(usage_2018_tier[which(usage_2018_tier$before_usage_level == 5),], aes(x = months_after)) + 
  geom_line(aes(y = quantity, color = "#ba0951"), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 600), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Mean Usage (hundreds gallon)")+
  geom_text(aes(x = 19, y = 1, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 1, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 1, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 1, label = "2017"), color = "black") +
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Consumer Monthly Avg Water Usage - Tier 5") +
  #scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  big_text

##Revenue Plot
ggplot(usage_2018_level[which(usage_2018_level$category == "4_low" | usage_2018_level$category == "4_high"),], aes(x = bill_ym)) + 
  geom_line(aes(y = revenue, color = "#ba0951"), linewidth = 1.5) +
  #geom_line(aes(y = percipitation/ (50/300), color = "#247672"), linewidth = 1.5, linetype = "longdash") +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  #coord_cartesian(xlim = c(-27, 21), ylim = c(0, 300), expand = FALSE) +
  xlab("Year Month")+
  ylab("Water Usage Revenue (Millons)")+
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Water Usage Revenue") +
  big_text


prem_key= usage_charge_2018   %>%
  group_by(prem_id) %>%
  summarize(longtitude = mean(longtitude, na.rm = T), 
            latitude = mean(latitude, na.rm = T))

prem_key = merge(prem_key, usage_2018_before, by.x = c("prem_id"), by.y = c("prem_id"), all.x = F, all.y = T)
prem_key$length = prem_key$length.x
prem_key$length.x = NULL
prem_key$length.y = NULL
write.csv(prem_key, file = "prem_key.csv", row.names = F)
prem_key = read_csv(file = "prem_key.csv")

prem_key_high = prem_key[which(prem_key$coeff_level == 2),]

## 18437 High variance household

usage_2018_after_high = merge(usage_2018_after, prem_key_high,by.x = c("prem_id"), 
                              by.y = c("prem_id"),all.x = F, all.y = T)

usage_2018_after_high = usage_2018_after_high %>%
  group_by(prem_id) %>%
  mutate(quantity_mean_after = mean(quantity, na.rm = T), 
            length = n())

usage_2018_after_high = usage_2018_after_high[which(usage_2018_after_high$length>=5),]

usage_2018_after_high_level = usage_2018_after_high %>%
  group_by(bill_ym, category) %>%
  summarize(quantity = mean(quantity, na.rm = T), 
         mean_quantity = mean(mean_quantity, na.rm = T),
         sd_quantity = mean(sd_quantity, na.rm = T))

usage_2018_after_high_level$months_after= case_when(
  usage_2018_after_high_level$bill_ym == 201805 ~ 0,
  usage_2018_after_high_level$bill_ym == 201806 ~ 1,
  usage_2018_after_high_level$bill_ym == 201807 ~ 2,
  usage_2018_after_high_level$bill_ym == 201808 ~ 3,
  usage_2018_after_high_level$bill_ym == 201809 ~ 4,
  usage_2018_after_high_level$bill_ym == 201810 ~ 5,
  usage_2018_after_high_level$bill_ym == 201811 ~ 6,
  usage_2018_after_high_level$bill_ym == 201812 ~ 7,
  usage_2018_after_high_level$bill_ym == 201901 ~ 8,
  usage_2018_after_high_level$bill_ym == 201902 ~ 9,
  usage_2018_after_high_level$bill_ym == 201903 ~ 10,
  usage_2018_after_high_level$bill_ym == 201904 ~ 11,
  usage_2018_after_high_level$bill_ym == 201905 ~ 12,
  usage_2018_after_high_level$bill_ym == 201906 ~ 13,
  usage_2018_after_high_level$bill_ym == 201907 ~ 14,
  usage_2018_after_high_level$bill_ym == 201908 ~ 15,
  usage_2018_after_high_level$bill_ym == 201909 ~ 16,
  usage_2018_after_high_level$bill_ym == 201910 ~ 17,
  usage_2018_after_high_level$bill_ym == 201911 ~ 18,
  usage_2018_after_high_level$bill_ym == 201912 ~ 19,
  usage_2018_after_high_level$bill_ym == 202001 ~ 20,
  usage_2018_after_high_level$bill_ym == 202002 ~ 21
)

usage_2018_after_high_level$category = factor(usage_2018_after_high_level$category, levels=c('1_high', '2_high', '3_high', '4_high', '5_high'))

ggplot(usage_2018_level[which(usage_2018_level$category == "1_high" | 
                                usage_2018_level$category == "2_high"|
                                usage_2018_level$category == "3_high" |
                                usage_2018_level$category == "4_high" |
                                usage_2018_level$category == "5_high"),], aes(x = months_after, y = quantity)) + 
  geom_line(aes(color = category), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 800), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Mean Usage (hundreds gallon)")+
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  #scale_color_manual(values = c("steelblue","darkred"))+
  big_text

ggplot(usage_2018_after_high_level, aes(x = months_after, y = quantity)) + 
  geom_line(aes(color = category)) +
  #geom_segment(aes(x = 0, y = 0, xend = 0, yend = 10)) +
  #geom_segment(aes(x = 0.5, y = 0, xend = 0.5, yend = 10))+
  #coord_cartesian(xlim = c(0, 0.5), ylim = c(0, 10), expand = FALSE) +
  big_text

usage_2018_after_high_level = merge(usage_2018_after_high_level, percipitation, by.x = c("bill_ym"), by.y = c("bill_ym"), all.x = T, all.y = F)


#### Construct Revenue Summaries

charge_2018_sum = charge_2018 %>%
  group_by(bill_ym) %>%
  summarize(charge= sum(charge, na.rm = T))

charge_2018_sum = merge(charge_2018_sum, percipitation_revenue, by.x = c("bill_ym"), by.y = c("bill_ym"), all.x = T,all.y = T)

charge_2018_sum$bill_ym = as.yearmon(as.character(charge_2018_sum$bill_ym),"%Y%m")
charge_2018_sum$bill_ym = as.Date(charge_2018_sum$bill_ym)
charge_2018_sum$charge = charge_2018_sum$charge/1000000

r = ggplot(charge_2018_sum, aes(x = months_after)) + 
  geom_line(aes(y = charge, color = "#ba0951"), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  #geom_line(aes(y = percipitation/2, color = "#247672"), linewidth = 1.5, linetype = "longdash") +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 20), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Revenue (millons $)")+
  geom_text(aes(x = 19, y = 0.5, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 0.5, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 0.5, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 0.5, label = "2017"), color = "black") +
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Residential Revenue") +
  scale_color_manual(values = c("#498538"),labels = c( "Revenue"))+
  #scale_color_manual(values = c("#104507","#ba0951"),labels = c("Percipitation", "Revenue"))+
  #scale_y_continuous(
   # "Revenue (millons)", 
    #sec.axis = sec_axis(~ . *2, name = "Percipitation (Inches)")
  #) +
  big_text

usage_2018_sum = usage_2018 %>%
  group_by(bill_ym) %>%
  summarize(mean_quantity = mean(quantity, na.rm = T),
            quantity= sum(quantity, na.rm = T))

usage_2018_sum = merge(usage_2018_sum, percipitation_revenue, 
                       by.x = c("bill_ym"), by.y = c("bill_ym"), all.x = T,all.y = T)

usage_2018_sum$bill_ym = as.yearmon(as.character(usage_2018_sum$bill_ym),"%Y%m")
usage_2018_sum$bill_ym = as.Date(usage_2018_sum$bill_ym)
usage_2018_sum$quantity = usage_2018_sum$quantity*100/1000000000

ggplot(usage_2018_sum, aes(x = months_after)) + 
  geom_line(aes(y = quantity, color = "#ba0951"), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  geom_line(aes(y = percipitation/20, color = "#247672"), linewidth = 1.5, linetype = "longdash") +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 2), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Total Q (billons gallon)")+
  geom_text(aes(x = 19, y = 0.05, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 0.05, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 0.05, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 0.05, label = "2017"), color = "black") +
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Residential Usage") +
  scale_color_manual(values = c("#104507","#ba0951"),labels = c("Percipitation", "Quantity"))+
  scale_y_continuous(
    "Quantity (billons gallon)", 
    sec.axis = sec_axis(~ . *20, name = "Percipitation (Inches)")
  ) +
  big_text

ggplot(usage_2018_sum, aes(x = months_after)) + 
  geom_line(aes(y = mean_quantity, color = "#ba0951"), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  #geom_line(aes(y = percipitation/0.4, color = "#247672"), linewidth = 1.5, linetype = "longdash") +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 100), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Q Per HHs (hundreds gallon)")+
  geom_text(aes(x = 19, y = 1.5, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 1.5, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 1.5, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 1.5, label = "2017"), color = "black") +
  ggtitle("Residential Usage Per Households") +
  scale_color_manual(values = c("#ba0951"),labels = c("Quantity"))+
  #scale_y_continuous(
   # "Quantity (hundreds gallon)", 
  #  sec.axis = sec_axis(~ . *0.4, name = "Percipitation (Inches)")
  #) +
  big_text

charge_2018_merge = merge(charge_2018, prem_key, by.x = c("prem_id"), by.y = c("prem_id"), all.x = F, all.y = T)

#calculate_fare = function(q, rate_change){ ## q in 1,000 gallons
 # if(rate_change == 0){
  #  f = case_when(
   #   q>=0 & q<=2 ~ 7.10+ 3.18*q + 1.2,
    #  q>2 & q<=6 ~ 7.10 + 3.55 + 3.18*2 + 5.05*(q-2),
     # q>6 & q<=11 ~ 7.10 + 9.25 + 3.18*2 + 5.05 *4 + 8.56 * (q-6), 
      #q>11 & q<=20 ~ 7.10 + 29.75 + 3.18*2 + 5.05 * 4 + 8.56 * 5 + 12.92*(q-11),
      #q>20 ~ 7.10 + 29.75 + 3.18*2 + 5.05 * 4 + 8.56 * 5 + 12.92*9 + 14.43 * (q-20),
    #)
  #}else{
   # f = case_when(
    #  q>=0 & q<=2 ~ 2.89*q+1.25,
     # q>2 & q<=6 ~ 4.81*q-0.29,
      #q>6 & q<=11 ~ 8.34*q-15.77 ,
      #q>11 & q<=20 ~ 12.7*q-43.23,
      #q>20 ~ 14.21*q-73.43
    #)
  #}
  #return(f)
#}

usage_2018 = usage_2018 %>%
  group_by(bill_ym) %>%
  mutate(fare = calculate_fare(quantity/10, unique(rate_change) ))

charge_2018_merge = merge(charge_2018_merge, usage_2018, by.x = c("prem_id", "bill_ym"), by.y = c("prem_id", "bill_ym"), all = T)

charge_2018_merge$length_f_type = charge_2018_merge$length.x
charge_2018_merge$length.x = NULL
charge_2018_merge$length_f_unique = charge_2018_merge$length.y
charge_2018_merge$length.y = NULL
charge_2018_merge$rate_change = charge_2018_merge$rate_change.x
charge_2018_merge$rate_change.x = NULL
charge_2018_merge$rate_change.y = NULL
charge_2018_merge = charge_2018_merge[which(is.na( charge_2018_merge$category)==F ),]

write.csv(usage_2018_merge, file = "usage_2018_merge.csv", row.names = F)

write.csv(charge_2018_merge, file = "charge_2018_merge.csv", row.names = F)

charge_2018_level = charge_2018_merge %>%
  group_by(category, bill_ym) %>%
  summarize(charge_sum = sum(charge),
            charge_mean = mean(charge),
            fare_sum = sum(fare, na.rm = T),
            fare_mean = mean(fare, na.rm = T))

charge_2018_level$months_after= case_when(
  charge_2018_level$bill_ym == 201602 ~ -27,
  charge_2018_level$bill_ym == 201603 ~ -26,
  charge_2018_level$bill_ym == 201604 ~ -25,
  charge_2018_level$bill_ym == 201605 ~ -24,
  charge_2018_level$bill_ym == 201606 ~ -23,
  charge_2018_level$bill_ym == 201607 ~ -22,
  charge_2018_level$bill_ym == 201608 ~ -21,
  charge_2018_level$bill_ym == 201609 ~ -20,
  charge_2018_level$bill_ym == 201610 ~ -19,
  charge_2018_level$bill_ym == 201611 ~ -18,
  charge_2018_level$bill_ym == 201612 ~ -17,
  charge_2018_level$bill_ym == 201701 ~ -16,
  charge_2018_level$bill_ym == 201702 ~ -15,
  charge_2018_level$bill_ym == 201703 ~ -14,
  charge_2018_level$bill_ym == 201704 ~ -13,
  charge_2018_level$bill_ym == 201705 ~ -12,
  charge_2018_level$bill_ym == 201706 ~ -11,
  charge_2018_level$bill_ym == 201707 ~ -10,
  charge_2018_level$bill_ym == 201708 ~ -9,
  charge_2018_level$bill_ym == 201709 ~ -8,
  charge_2018_level$bill_ym == 201710 ~ -7,
  charge_2018_level$bill_ym == 201711 ~ -6,
  charge_2018_level$bill_ym == 201712 ~ -5,
  charge_2018_level$bill_ym == 201801 ~ -4,
  charge_2018_level$bill_ym == 201802 ~ -3,
  charge_2018_level$bill_ym == 201803 ~ -2,
  charge_2018_level$bill_ym == 201804 ~ -1,
  charge_2018_level$bill_ym == 201805 ~ 0,
  charge_2018_level$bill_ym == 201806 ~ 1,
  charge_2018_level$bill_ym == 201807 ~ 2,
  charge_2018_level$bill_ym == 201808 ~ 3,
  charge_2018_level$bill_ym == 201809 ~ 4,
  charge_2018_level$bill_ym == 201810 ~ 5,
  charge_2018_level$bill_ym == 201811 ~ 6,
  charge_2018_level$bill_ym == 201812 ~ 7,
  charge_2018_level$bill_ym == 201901 ~ 8,
  charge_2018_level$bill_ym == 201902 ~ 9,
  charge_2018_level$bill_ym == 201903 ~ 10,
  charge_2018_level$bill_ym == 201904 ~ 11,
  charge_2018_level$bill_ym == 201905 ~ 12,
  charge_2018_level$bill_ym == 201906 ~ 13,
  charge_2018_level$bill_ym == 201907 ~ 14,
  charge_2018_level$bill_ym == 201908 ~ 15,
  charge_2018_level$bill_ym == 201909 ~ 16,
  charge_2018_level$bill_ym == 201910 ~ 17,
  charge_2018_level$bill_ym == 201911 ~ 18,
  charge_2018_level$bill_ym == 201912 ~ 19,
  charge_2018_level$bill_ym == 202001 ~ 20,
  charge_2018_level$bill_ym == 202002 ~ 21
)

charge_2018_level$charge_sum = charge_2018_level$charge_sum / 1000
charge_2018_level$fare_sum = charge_2018_level$fare_sum / 1000

r1 = ggplot(charge_2018_level[which(charge_2018_level$category == "1_low" | charge_2018_level$category == "1_high"),], aes(x = months_after)) + 
  #geom_line(aes(y = charge_mean, color = category), linewidth = 1.5) +
  geom_line(aes(y = fare_mean, color = category), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 50), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Mean Charge ($)")+
  geom_text(aes(x = 19, y = 1, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 1, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 1, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 1, label = "2017"), color = "black") +
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Avg Charge") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  big_text

r2 = ggplot(charge_2018_level[which(charge_2018_level$category == "2_low" | charge_2018_level$category == "2_high"),], aes(x = months_after)) + 
  #geom_line(aes(y = charge_sum, color = category), linewidth = 1.5) +
  geom_line(aes(y = fare_mean, color = category), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 100), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Mean Charge ($)")+
  geom_text(aes(x = 19, y = 2, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 2, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 2, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 2, label = "2017"), color = "black") +
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Avg Charge") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  big_text

r3 = ggplot(charge_2018_level[which(charge_2018_level$category == "3_low" | charge_2018_level$category == "3_high"),], aes(x = months_after)) + 
  #geom_line(aes(y = charge_mean, color = category), linewidth = 1.5) +
  geom_line(aes(y = fare_mean, color = category), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 200), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Mean Charge ($)")+
  geom_text(aes(x = 19, y = 5, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 5, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 5, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 5, label = "2017"), color = "black") +
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Avg Charge") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  big_text

r4 = ggplot(charge_2018_level[which(charge_2018_level$category == "4_low" | charge_2018_level$category == "4_high"),], aes(x = months_after)) + 
  #geom_line(aes(y = charge_sum, color = category), linewidth = 1.5) +
  geom_line(aes(y = fare_mean, color = category), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 500), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Mean Charge ($)")+
  geom_text(aes(x = 19, y = 10, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 10, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 10, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 10, label = "2017"), color = "black") +
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Avg Charge") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  big_text

r5 = ggplot(charge_2018_level[which(charge_2018_level$category == "5_low" | charge_2018_level$category == "5_high"),], aes(x = months_after)) + 
  #geom_line(aes(y = charge_sum, color = category), linewidth = 1.5) +
  geom_line(aes(y = fare_mean, color = category), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 1000), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Mean Charge ($)")+
  geom_text(aes(x = 19, y = 25, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 25, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 25, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 25, label = "2017"), color = "black") +
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 1)))+
  ggtitle("Avg Charge") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  big_text

qr1 = ggarrange(q1, r1, ncol = 2, nrow = 1, common.legend = T, legend = "bottom")
annotate_figure(qr1, top = text_grob("Monthly Average Quantity and Charge - Tier 1", 
                                  color = "Black", face = "bold", size = 26))

qr2 = ggarrange(q2, r2, ncol = 2, nrow = 1, common.legend = T, legend = "bottom")
annotate_figure(qr2, top = text_grob("Monthly Average Quantity and Charge - Tier 2", 
                                     color = "Black", face = "bold", size = 26))

qr3 = ggarrange(q3, r3, ncol = 2, nrow = 1, common.legend = T, legend = "bottom")
annotate_figure(qr3, top = text_grob("Monthly Average Quantity and Charge - Tier 3", 
                                     color = "Black", face = "bold", size = 26))

qr4 = ggarrange(q4, r4, ncol = 2, nrow = 1, common.legend = T, legend = "bottom")
annotate_figure(qr4, top = text_grob("Monthly Average Quantity and Charge - Tier 4", 
                                     color = "Black", face = "bold", size = 26))

qr5 = ggarrange(q5, r5, ncol = 2, nrow = 1, common.legend = T, legend = "bottom")
annotate_figure(qr5, top = text_grob("Monthly Average Quantity and Charge - Tier 5", 
                                     color = "Black", face = "bold", size = 26))

w = ggplot(usage_2018_sum, aes(x = months_after)) + 
  #geom_line(aes(y = mean_quantity, color = "#ba0951"), linewidth = 1.5) +
  geom_line(aes(y = AvgT, color = "#ba0951"), linewidth = 1.5) +
  #geom_line(aes(y = MaxT, color = "#ba0951"), linewidth = 1.2,linetype = "dashed") +
  #geom_line(aes(y = MinT, color = "#ba0951"), linewidth = 1.2,linetype = "dashed") +
  #geom_line(aes(y = `Dew Point`, color = "#ba0951"), linewidth = 1.2,linetype = "dashed") +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  geom_line(aes(y = percipitation/0.25, color = "#247672"), linewidth = 1.5, linetype = "longdash") +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 100), expand = FALSE) +
  xlab("Months since rate changed")+
  #ylab("Q Per HHs (hundreds gallon)")+
  geom_text(aes(x = 19, y = 1.5, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 1.5, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 1.5, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 1.5, label = "2017"), color = "black") +
  ggtitle("Weather Condition") +
  scale_color_manual(values = c("#256676", "#e23209"),labels = c("Percipitation", "Avg Temperature"))+
  scale_y_continuous(
   "Temperature (F)", 
    sec.axis = sec_axis(~ . *0.25, name = "Percipitation (Inches)")
  ) +
  big_text

rw = ggarrange(r, w, ncol = 2, nrow = 1, common.legend = F, legend = "bottom")
#annotate_figure(rw, top = text_grob("Monthly Average Quantity and Charge - Tier 5", 
                           #         color = "Black", face = "bold", size = 26))


##### Type changes:
usage_2018_after = merge(usage_2018_after, prem_key, by.x = c("prem_id"), by.y = c("prem_id"), all.x = F, all.y = T)

usage_2018_after_category= usage_2018_after   %>%
  group_by(prem_id) %>%
  summarize(mean_quantity_before = unique(mean_quantity),
            sd_quantity_before = unique(sd_quantity),
            mean_quantity_after = mean(quantity),
            sd_quantity_after = sd(quantity),
            length = n(),
            usage_before = unique(usage),
            coeff_before = unique(coeff)
            )


#usage_2018_after_category = usage_2018_after_category[which(usage_2018_after_category$sd_quantity_after<5000),]
usage_2018_after_category = usage_2018_after_category[which(usage_2018_after_category$mean_quantity_after>0),]
usage_2018_after_category = usage_2018_after_category[which(usage_2018_after_category$length>=15),]

usage_2018_after_category$usage= case_when(
  usage_2018_after_category$mean_quantity_after <=20 ~ 1,
  usage_2018_after_category$mean_quantity_after >20 & usage_2018_after_category$mean_quantity_after <=60  ~ 2,
  usage_2018_after_category$mean_quantity_after >60 &usage_2018_after_category$mean_quantity_after <=110 ~ 3,
  usage_2018_after_category$mean_quantity_after >110 &usage_2018_after_category$mean_quantity_after <=200 ~ 4,
  usage_2018_after_category$mean_quantity_after >200 ~ 5)

usage_2018_after_category$coeff = usage_2018_after_category$sd_quantity_after / usage_2018_after_category$mean_quantity_after

usage_2018_after_category$coeff_level_before= case_when(
  usage_2018_after_category$coeff_before <=1 ~ 1,  
  usage_2018_after_category$coeff_before >1 ~ 2)

usage_2018_after_category$coeff_level_after= case_when(
  usage_2018_after_category$coeff <=1 ~ 1,  
  usage_2018_after_category$coeff >1 ~ 2)

usage_2018_after_category$coeff_level_before = as.character(usage_2018_after_category$coeff_level_before)
usage_2018_after_category$coeff_level_after = as.character(usage_2018_after_category$coeff_level_after)

ggplot(usage_2018_after_category, aes(x = usage_before)) + 
  geom_jitter(aes(y = usage, color = coeff_level_before, size = mean_quantity_after, shape = coeff_level_before), alpha = 0.15, width = 0.25, height = 0.25) +
  coord_cartesian(xlim = c(0, 6), ylim = c(0, 6), expand = FALSE) +
  xlab("Tier Before")+
  ylab("Tier After")+
  scale_color_manual(values = c("#25738b", "#ba0951"),labels = c( "Low Var", "High Var"), name = "Category")+
  scale_shape_manual(values = c(1,16), guide = "none")+
  ggtitle("Tier Change for Both Categories")+
  big_text + theme(panel.background = element_rect(fill = 'gray90'))

ggplot(usage_2018_after_category, aes(x = usage_before)) + 
  geom_point(aes(y = coeff_before), alpha = 0.1) +
  coord_cartesian(xlim = c(0, 6), ylim = c(0, 7), expand = FALSE) +
  xlab("Tier Before")+
  ylab("Sd/mean Before")+
  #scale_color_gradient2(low = "#0000FF", mid = "#FFFFFF", high ="#FF0000", 
   #                     midpoint = 3, guide = "colourbar")+
  #scale_color_manual(values = c("#5e904e", "#7c117e", "#378dae", "#08315c", "#c551dc"),labels = c( "2", "4", "1", "3", "5"), name = "Tier")+
  ggtitle("sd/mean before")+
  big_text + theme(panel.background = element_rect(fill = 'gray90'))

ggplot(usage_2018_after_category, aes(x = usage)) + 
  geom_point(aes(y = coeff), alpha = 0.1) +
  coord_cartesian(xlim = c(0, 6), ylim = c(0, 7), expand = FALSE) +
  xlab("Tier After")+
  ylab("Sd/mean After")+
  #scale_color_gradient2(low = "#0000FF", mid = "#FFFFFF", high ="#FF0000", 
  #                     midpoint = 3, guide = "colourbar")+
  #scale_color_manual(values = c("#5e904e", "#7c117e", "#378dae", "#08315c", "#c551dc"),labels = c( "2", "4", "1", "3", "5"), name = "Tier")+
  ggtitle("sd/mean after")+
  big_text + theme(panel.background = element_rect(fill = 'gray90'))


###### NDVI

myd13q11_ndvi = read_csv("MYD13Q11_16-19_all.csv")

myd13q11_ndvi_agg = myd13q11_ndvi %>%
  group_by(Category) %>%
  summarize(ndvi_2016 = mean(NDVI_MYD13Q11_201612, na.rm = T),
            ndvi_2018_after = mean(NDVI_MYD13Q11_201810, na.rm = T),
            ndvi_2019 = mean(NDVI_MYD13Q11_201912, na.rm = T),
            ndvi_2018_before = mean(NDVI_MYD13Q11_2018089, na.rm = T))

myd13q11_ndvi$NDVI_2018_change = myd13q11_ndvi$NDVI_MYD13Q11_201810 - myd13q11_ndvi$NDVI_MYD13Q11_2018089

charge_2018_merge = merge(charge_2018_merge, myd13q11_ndvi, by.x = c("prem_id"), by.y = c("ID"), all.x = T, all.y = T)
charge_2018_merge$category = NULL
charge_2018_merge$longtitude = NULL
charge_2018_merge$latitude = NULL

mod13q11_ndvi = read_csv("MOD13Q11_16-19_all.csv")

mod13q11_ndvi_agg = mod13q11_ndvi %>%
  group_by(Category) %>%
  summarize(ndvi_2016 = mean(ndvi_2016353_Mean, na.rm = T),
            ndvi_2018_after = mean(ndvi_2018289_Mean, na.rm = T),
            ndvi_2019 = mean(ndvi_2020001_Mean, na.rm = T),
            ndvi_2018_before = mean(ndvi_2018097_Mean, na.rm = T))

charge_2018_merge = merge(charge_2018_merge, mod13q11_ndvi, by.x = c("prem_id"), by.y = c("ID"), all.x = T, all.y = T)
charge_2018_merge$Category= charge_2018_merge$Category.x
charge_2018_merge$Category.x= NULL
charge_2018_merge$Category.y= NULL
charge_2018_merge$Longitude= charge_2018_merge$Longitude.x
charge_2018_merge$Longitude.x= NULL
charge_2018_merge$Longitude.y= NULL
charge_2018_merge$Latitude= charge_2018_merge$Latitude.x
charge_2018_merge$Latitude.x= NULL
charge_2018_merge$Latitude.y= NULL

charge_2018_merge_small = charge_2018_merge[which(charge_2018_merge$bill_ym == 201612| charge_2018_merge$bill_ym == 201804|
                                                    charge_2018_merge$bill_ym == 201810 | charge_2018_merge$bill_ym == 201912),]

charge_2018_merge_small_ndvi = charge_2018_merge_small %>%
  group_by(prem_id) %>%
  summarize(category = unique(category),
            q_201810_201804 = quantity[3] - quantity[2],
            q_201912_201804 = quantity[4] - quantity[2],
            f_201810_201804 = fare[3] - fare[2],
            f_201912_201804 = fare[4] - fare[2],
            #ndvi_2016_myd = mean(NDVI_MYD13Q11_201612, na.rm = T),
            #ndvi_2018_after_myd = mean(NDVI_MYD13Q11_201810, na.rm = T),
            #ndvi_2019_myd = mean(NDVI_MYD13Q11_201912, na.rm = T),
            #ndvi_2018_before_myd = mean(NDVI_MYD13Q11_2018089, na.rm = T),
            ndvi_2016_mod = mean(ndvi_2016353_Mean, na.rm = T),
            ndvi_2018_before_mod = mean(ndvi_2018097_Mean, na.rm = T),
            ndvi_2018_after_mod = mean(ndvi_2018289_Mean, na.rm = T),
            ndvi_2019_mod = mean(ndvi_2020001_Mean, na.rm = T))

#charge_2018_merge_small_ndvi$ndvi_201810_201804_myd = charge_2018_merge_small_ndvi$ndvi_2018_after_myd - charge_2018_merge_small_ndvi$ndvi_2018_before_myd
#charge_2018_merge_small_ndvi$ndvi_201912_201804_myd = charge_2018_merge_small_ndvi$ndvi_2019_myd - charge_2018_merge_small_ndvi$ndvi_2018_before_myd
charge_2018_merge_small_ndvi$ndvi_201810_201804_mod = charge_2018_merge_small_ndvi$ndvi_2018_after_mod - charge_2018_merge_small_ndvi$ndvi_2018_before_mod
charge_2018_merge_small_ndvi$ndvi_201912_201804_mod = charge_2018_merge_small_ndvi$ndvi_2019_mod - charge_2018_merge_small_ndvi$ndvi_2018_before_mod

charge_2018_merge_small_ndvi = charge_2018_merge_small_ndvi[which(charge_2018_merge_small_ndvi$q_201810_201804<=1000),]
charge_2018_merge_small_ndvi = charge_2018_merge_small_ndvi[which(charge_2018_merge_small_ndvi$q_201810_201804>=-1000),]

#ggplot(charge_2018_merge_small_ndvi[which(charge_2018_merge_small_ndvi$category == "2_low" | charge_2018_merge_small_ndvi$category == "2_high"),], 
 #      aes(x = q_201810_201804, y = ndvi_201810_201804_myd)) + 
  #geom_point(aes(color = category),alpha = 0.3) +
  #xlab("Quantity Change (100 Gallons)")+
  #ylab("NDVI Change")+
  #guides(colour = guide_legend(override.aes = list(size=5, alpha = 5)))+
  #ggtitle("Q vs. NDVI - Tier 2") +
  #scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  #big_text

qndvi1 = ggplot(charge_2018_merge_small_ndvi[which(charge_2018_merge_small_ndvi$category == "1_low" | charge_2018_merge_small_ndvi$category == "1_high"),], 
       aes(x = q_201810_201804, y = ndvi_201810_201804_mod)) + 
  geom_point(aes(color = category),alpha = 0.3) +
  geom_vline(xintercept=0,linetype = "solid", linewidth = 1.5) +
  geom_hline(yintercept=0,linetype = "solid", linewidth = 1.5) +
  xlab("Quantity Change (100 Gallons)")+
  ylab("NDVI Change")+
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 5)))+
  ggtitle("Tier 1") +
  coord_cartesian(xlim = c(-1000, 1000), ylim = c(-7000, 2500), expand = FALSE) +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  big_text

ggplot(charge_2018_merge_small_ndvi[which(charge_2018_merge_small_ndvi$category == "2_low" | charge_2018_merge_small_ndvi$category == "2_high"),], 
       aes(x = q_201810_201804, y = ndvi_201810_201804_mod)) + 
  geom_point(aes(color = category),alpha = 0.3) +
  geom_vline(xintercept=0,linetype = "solid", linewidth = 1.5) +
  geom_hline(yintercept=0,linetype = "solid", linewidth = 1.5) +
  xlab("Quantity Change (100 Gallons)")+
  ylab("NDVI Change")+
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 5)))+
  ggtitle("Q vs. NDVI - Tier 2") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  big_text

ggplot(charge_2018_merge_small_ndvi[which(charge_2018_merge_small_ndvi$category == "3_low" | charge_2018_merge_small_ndvi$category == "3_high"),], 
       aes(x = q_201810_201804, y = ndvi_201810_201804_mod)) + 
  geom_point(aes(color = category),alpha = 0.3) +
  geom_vline(xintercept=0,linetype = "solid", linewidth = 1.5) +
  geom_hline(yintercept=0,linetype = "solid", linewidth = 1.5) +
  xlab("Quantity Change (100 Gallons)")+
  ylab("NDVI Change")+
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 5)))+
  ggtitle("Q vs. NDVI - Tier 3") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  big_text

ggplot(charge_2018_merge_small_ndvi[which(charge_2018_merge_small_ndvi$category == "4_low" | charge_2018_merge_small_ndvi$category == "4_high"),], 
       aes(x = q_201810_201804, y = ndvi_201810_201804_mod)) + 
  geom_point(aes(color = category),alpha = 0.3) +
  geom_vline(xintercept=0,linetype = "solid", linewidth = 1.5) +
  geom_hline(yintercept=0,linetype = "solid", linewidth = 1.5) +
  xlab("Quantity Change (100 Gallons)")+
  ylab("NDVI Change")+
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 5)))+
  ggtitle("Q vs. NDVI - Tier 4") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  big_text

qndvi5 = ggplot(charge_2018_merge_small_ndvi[which(charge_2018_merge_small_ndvi$category == "5_low" | charge_2018_merge_small_ndvi$category == "5_high"),], 
       aes(x = q_201810_201804, y = ndvi_201810_201804_mod)) + 
  geom_point(aes(color = category),alpha = 0.3) +
  geom_vline(xintercept=0,linetype = "solid", linewidth = 1.5) +
  geom_hline(yintercept=0,linetype = "solid", linewidth = 1.5) +
  xlab("Quantity Change (100 Gallons)")+
  ylab("NDVI Change")+
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 5)))+
  ggtitle("Tier 5") +
  coord_cartesian(xlim = c(-1000, 1000), ylim = c(-7000, 2500), expand = FALSE) +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  big_text

qndvi15 = ggarrange(qndvi1, qndvi5, ncol = 2, nrow = 1, common.legend = T, legend = "bottom")
annotate_figure(qndvi15, top = text_grob("NDVI and Q Difference b/w Apr and Oct.2018", 
         color = "Black", face = "bold", size = 26))


#ggplot(charge_2018_merge_small_ndvi[which(charge_2018_merge_small_ndvi$category == "5_low" | charge_2018_merge_small_ndvi$category == "5_high"),], 
 #      aes(x = q_201810_201804, y = ndvi_201810_201804_myd)) + 
#  geom_point(aes(color = category),alpha = 0.3) +
 # xlab("Quantity Change (100 Gallons)")+
  #ylab("NDVI Change")+
  #guides(colour = guide_legend(override.aes = list(size=5, alpha = 5)))+
  #ggtitle("Q vs. NDVI - Tier 5") +
  #scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  #big_text

ggplot(charge_2018_merge_small_ndvi[which(charge_2018_merge_small_ndvi$category == "2_low" | charge_2018_merge_small_ndvi$category == "2_high"),], 
       aes(x = f_201810_201804, y = ndvi_201810_201804_myd)) + 
  geom_point(aes(color = category),alpha = 0.3) +
  xlab("Fare Change ($)")+
  ylab("NDVI Change")+
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 5)))+
  ggtitle("Fare vs. NDVI - Tier 2") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  big_text

ggplot(charge_2018_merge_small_ndvi[which(charge_2018_merge_small_ndvi$category == "5_low" | charge_2018_merge_small_ndvi$category == "5_high"),], 
       aes(x = f_201810_201804, y = ndvi_201810_201804_myd)) + 
  geom_point(aes(color = category),alpha = 0.3) +
  xlab("Fare Change ($)")+
  ylab("NDVI Change")+
  guides(colour = guide_legend(override.aes = list(size=5, alpha = 5)))+
  ggtitle("Fare vs. NDVI - Tier 5") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "High Var", "Low Var"))+
  big_text

###################################### ######################################
######################Check for Bunching ######################################
###################################### #####################################

usage_2018_merge = usage_2018_merge[order(usage_2018_merge$prem_id, usage_2018_merge$bill_ym),]

ggplot(usage_2018_merge[which(usage_2018_merge$months_after<=0 & usage_2018_merge$quantity>0),], aes(x=quantity)) + 
  geom_histogram(color="red", fill="white", binwidth=2, boundary = 0) +
  geom_vline(xintercept=20,linetype = "solid", linewidth = 0.5) +
  geom_vline(xintercept=60,linetype = "solid", linewidth = 0.5) +
  geom_vline(xintercept=110,linetype = "solid", linewidth = 0.5) +
  geom_vline(xintercept=200,linetype = "solid", linewidth = 0.5) +
  geom_text(aes(x = 20, y = -1.5, label = "20"), color = "black") +
  geom_text(aes(x = 60, y = -1.5, label = "60"), color = "black") +
  geom_text(aes(x = 110, y = -1.5, label = "110"), color = "black") +
  geom_text(aes(x = 200, y = -1.5, label = "200"), color = "black") +
  xlim(c(0, 200)) + big_text

###################################### ######################################
######################Waste Water Averaging ######################################
###################################### #####################################

usage_charge_2018$end_dt = substr(usage_charge_2018$bseg_end_dt, 9, 10)
usage_charge_2018$end_dt = as.numeric(usage_charge_2018$end_dt)

prem_id_end_date= usage_charge_2018   %>%
  group_by(prem_id, bill_ym) %>%
  summarize(bseg_end_dt =  unique(end_dt))

prem_id_end_date$month = prem_id_end_date$bill_ym %% 100

write.csv(prem_id_end_date, file = "prem_id_end_date.csv", row.names = F)

prem_id_end_date_winter = prem_id_end_date[which(prem_id_end_date$month>=11 | prem_id_end_date$month <=3),]

prem_id_end_date_winter$wasterwasteravg = case_when(
  prem_id_end_date_winter$month == 11 ~ 0,
  prem_id_end_date_winter$month == 12 & prem_id_end_date_winter$bseg_end_dt<=13 ~ 0 ,
  prem_id_end_date_winter$month == 12 & prem_id_end_date_winter$bseg_end_dt>13 ~ 1 ,
  prem_id_end_date_winter$month == 1 ~ 1,
  prem_id_end_date_winter$month == 2 ~ 1,
  prem_id_end_date_winter$month == 3 & prem_id_end_date_winter$bseg_end_dt<=13 ~ 1 ,
  prem_id_end_date_winter$month == 3 & prem_id_end_date_winter$bseg_end_dt>13 ~ 0
)

usage_2018_merge = merge(usage_2018_merge, prem_id_end_date_winter, by.x = c("prem_id", "bill_ym"), by.y = c("prem_id", "bill_ym"), all = T)
usage_2018_merge$bseg_end_dt = NULL
usage_2018_merge$month = NULL
usage_2018_merge$wasterwasteravg[which(is.na(usage_2018_merge$wasterwasteravg)==T )] = 0


#m1 = lm(quantity ~ category * wasterwasteravg, data = usage_2018_merge)
#m2 = lm(quantity ~ category + wasterwasteravg, data = usage_2018_merge)

#Call:
 # lm(formula = quantity ~ category + wasterwasteravg, data = usage_2018_merge)

#Residuals:
 # Min      1Q  Median      3Q     Max 
#-1243.6   -21.5    -7.1    10.5 18253.7 

#Coefficients:
 # Estimate Std. Error  t value Pr(>|t|)    
#(Intercept)      20.60749    0.16080  128.157   <2e-16 ***
 # category1_low     0.52220    0.17599    2.967    0.003 ** 
  #category2_high   24.66977    0.19318  127.702   <2e-16 ***
  #category2_low    24.90788    0.16304  152.770   <2e-16 ***
  #category3_high   60.27053    0.21097  285.679   <2e-16 ***
  #category3_low    63.00366    0.16585  379.877   <2e-16 ***
  #category4_high  111.64117    0.27436  406.914   <2e-16 ***
  #category4_low   119.22839    0.18132  657.556   <2e-16 ***
  #category5_high  232.64678    0.45817  507.771   <2e-16 ***
  #category5_low   267.52198    0.26242 1019.444   <2e-16 ***
  #wasterwasteravg -16.60320    0.04832 -343.634   <2e-16 ***
  #---
  #Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#Residual standard error: 66.18 on 9835486 degrees of freedom
#(211917 observations deleted due to missingness)
#Multiple R-squared:  0.259,	Adjusted R-squared:  0.259 
#F-statistic: 3.437e+05 on 10 and 9835486 DF,  p-value: < 2.2e-16

anova(m1, m2)
### Both the category and the interaction with wasterwateravg are significant

one_third_cumsum = function(x){
  l = length(x)
  r = c()
  sum = 0
  while (l>0) {
    sum = sum + x[length(x) - l + 1]/3 
    r = c(r, sum)
    l = l-1
  }
  return(ceiling(r+0.1))
}

usage_2018_merge = usage_2018_merge%>%
  group_by(prem_id) %>%
  mutate(wastewt_year =  
           one_third_cumsum (wasterwasteravg))

wastewtavg = function(x){
  x = x[which(x>0)]
  if(length(x)<=1){
    return(50)
  }else{
    return(mean(x))
  }
}

usage_2018_merge = usage_2018_merge%>%
  group_by(prem_id, wastewt_year) %>%
  mutate(wastew_q = wastewtavg(quantity * wasterwasteravg))

usage_2018_merge = usage_2018_merge[which( is.na(usage_2018_merge$quantity) == F),]

usage_2018_merge = usage_2018_merge%>%
  group_by(prem_id) %>%
  mutate(wastew_q = lag(wastew_q, n = wasterwasteravg[1] + wasterwasteravg[2] + 1))

usage_2018_merge$extra_usage = pmax (usage_2018_merge$quantity - usage_2018_merge$wastew_q, 0)
usage_2018_merge$essential_usage = usage_2018_merge$quantity - usage_2018_merge$extra_usage

usage_2018_ww = data.frame(cbind(usage_2018_merge$prem_id, usage_2018_merge$bill_ym, usage_2018_merge$quantity, usage_2018_merge$wasterwasteravg,
                                 usage_2018_merge$wastew_q,
                                 usage_2018_merge$extra_usage, usage_2018_merge$essential_usage))
colnames(usage_2018_ww) = c("prem_id", "bill_ym","quantity","wastewateravg", "wastew_q",
                            "extra_usage", "essential_usage")

write.csv(usage_2018_ww, file = "usage_2018_ww.csv", row.names = F)

usage_2018_ww = read_csv(file = "usage_2018_ww.csv")

#usage_2018_ww$quantity = usage_2018_ww$extra_usage + usage_2018_ww$essential_usage

calculate_wastew_mp = function(q, bill_ym, wastewateravg, waste_q){
  if(bill_ym<201611){
    mp = case_when(
      wastewateravg==0 & q <= waste_q & q<=20~ 0.49,
      wastewateravg==0 & q <= waste_q & q>20~ 0.994,
      wastewateravg==0 & q > waste_q~ 0,
      ### when in the wwa months, the mp is in expectation terms
      wastewateravg==1 & q <= waste_q & q<=20 ~ 0.49 + 0.49*12/3,
      wastewateravg==1 & q <= waste_q & q>20~ 0.994 + 0.994*12/3,
      wastewateravg==1 & q > waste_q & q<=20~ 0.49*12/3,
      wastewateravg==1 & q > waste_q & q>20~ 0.994*12/3)
  }else if (bill_ym <201801 & bill_ym >= 201611){
    mp = case_when(
      wastewateravg==0 & q <= waste_q & q<=20~ 0.53,
      wastewateravg==0 & q <= waste_q & q>20~ 1.035,
      wastewateravg==0 & q > waste_q~ 0,
      ### when in the wwa months, the mp is in expectation terms
      wastewateravg==1 & q <= waste_q & q<=20 ~ 0.53 + 0.53*12/3,
      wastewateravg==1 & q <= waste_q & q>20~ 1.035 + 1.035*12/3,
      wastewateravg==1 & q > waste_q & q<=20~ 0.53*12/3,
      wastewateravg==1 & q > waste_q & q>20~ 1.035*12/3)
  }else if (bill_ym <201805 & bill_ym >= 201801){
    mp = case_when(
      wastewateravg==0 & q <= waste_q & q<=20~ 0.53,
      wastewateravg==0 & q <= waste_q & q>20~ 1.035,
      wastewateravg==0 & q > waste_q~ 0,
      ### when in the wwa months, the mp is in expectation terms
      wastewateravg==1 & q <= waste_q & q<=20 ~ 0.53 + 0.53*12/3,
      wastewateravg==1 & q <= waste_q & q>20~ 1.035 + 1.035*12/3,
      wastewateravg==1 & q > waste_q & q<=20~ 0.53*12/3,
      wastewateravg==1 & q > waste_q & q>20~ 1.035*12/3)
  }else{
    mp = case_when(
      wastewateravg==0 & q <= waste_q & q<=20~ 0.5,
      wastewateravg==0 & q <= waste_q & q>20~ 1.009,
      wastewateravg==0 & q > waste_q~ 0,
      ### when in the wwa months, the mp is in expectation terms
      wastewateravg==1 & q <= waste_q & q<=20 ~ 0.5 + 0.5*12/3,
      wastewateravg==1 & q <= waste_q & q>20~ 1.009 + 1.009*12/3,
      wastewateravg==1 & q > waste_q & q<=20~ 0.5*12/3,
      wastewateravg==1 & q > waste_q & q>20~ 1.009*12/3)
  }
}

usage_2018_ww = usage_2018_ww %>%
  group_by(bill_ym) %>%
  mutate(wastew_mp = calculate_wastew_mp(quantity, unique(bill_ym),wastewateravg, wastew_q))

usage_2018_ww$wastew_q = pmin(usage_2018_ww$wastew_q, usage_2018_ww$quantity)

usage_2018_ww$wastew_f = case_when(
 usage_2018_ww$bill_ym<201611 & usage_2018_ww$wastew_q<=20 ~ 10.30 + 0.49*usage_2018_ww$wastew_q,
 usage_2018_ww$bill_ym<201611 & usage_2018_ww$wastew_q>20 ~ 10.30+ 20*0.49 + 0.994*(usage_2018_ww$wastew_q-20),
 usage_2018_ww$bill_ym<201801 & usage_2018_ww$bill_ym>=201611 & usage_2018_ww$wastew_q<=20 ~ 10.30 + 0.53*usage_2018_ww$wastew_q,
 usage_2018_ww$bill_ym<201801 & usage_2018_ww$bill_ym>=201611 & usage_2018_ww$wastew_q>20 ~ 10.30+ 20*0.53 + 1.035*(usage_2018_ww$wastew_q-20),
 usage_2018_ww$bill_ym<201805 & usage_2018_ww$bill_ym>=201801 & usage_2018_ww$wastew_q<=20 ~ 10.30 + 0.515*usage_2018_ww$wastew_q + 0.015*usage_2018_ww$wastew_q ,
 usage_2018_ww$bill_ym<201805 & usage_2018_ww$bill_ym>=201801 & usage_2018_ww$wastew_q>20 ~ 10.30+ 20*0.515 + 1.020*(usage_2018_ww$wastew_q-20) + 0.015*usage_2018_ww$wastew_q,
 usage_2018_ww$bill_ym>=201805 & usage_2018_ww$wastew_q<=20 ~ 10.30 + 0.485*usage_2018_ww$wastew_q + usage_2018_ww$wastew_q * 0.015,
 usage_2018_ww$bill_ym>=201805 & usage_2018_ww$wastew_q>20 ~ 10.30+ 20*0.485 + 0.994*(usage_2018_ww$wastew_q-20) + usage_2018_ww$wastew_q*0.015
)

write.csv(usage_2018_ww, file = "usage_2018_ww.csv", row.names = F)

usage_2018_wwa = usage_2018_merge%>%
  group_by(before_usage_level, months_after) %>%
  summarize(extra_usage =mean(extra_usage, na.rm = T),
         essential_usage = mean(essential_usage, na.rm = T))

ggplot(usage_2018_wwa[which(usage_2018_wwa$before_usage_level == 1),], aes(x = months_after)) + 
  geom_line(aes(y = extra_usage, color = "#ba0951"), linewidth = 1.5) +
  geom_line(aes(y = essential_usage, color = "#25738b"), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 20), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Essential vs. Extra (hundreds gallon)")+
  geom_text(aes(x = 19, y = 1, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 1, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 1, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 1, label = "2017"), color = "black") +
  ggtitle("Residential Usage Per Households") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "essential", "extra"))+
  big_text

ggplot(usage_2018_wwa[which(usage_2018_wwa$before_usage_level == 2),], aes(x = months_after)) + 
  geom_line(aes(y = extra_usage, color = "#ba0951"), linewidth = 1.5) +
  geom_line(aes(y = essential_usage, color = "#25738b"), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 50), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Essential vs. Extra (hundreds gallon)")+
  geom_text(aes(x = 19, y = 1, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 1, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 1, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 1, label = "2017"), color = "black") +
  ggtitle("Residential Usage Per Households") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "essential", "extra"))+
  big_text

ggplot(usage_2018_wwa[which(usage_2018_wwa$before_usage_level == 3),], aes(x = months_after)) + 
  geom_line(aes(y = extra_usage, color = "#ba0951"), linewidth = 1.5) +
  geom_line(aes(y = essential_usage, color = "#25738b"), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 100), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Essential vs. Extra (hundreds gallon)")+
  geom_text(aes(x = 19, y = 1, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 1, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 1, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 1, label = "2017"), color = "black") +
  ggtitle("Residential Usage Per Households") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "essential", "extra"))+
  big_text

ggplot(usage_2018_wwa[which(usage_2018_wwa$before_usage_level == 4),], aes(x = months_after)) + 
  geom_line(aes(y = extra_usage, color = "#ba0951"), linewidth = 1.5) +
  geom_line(aes(y = essential_usage, color = "#25738b"), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 200), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Essential vs. Extra (hundreds gallon)")+
  geom_text(aes(x = 19, y = 1, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 1, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 1, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 1, label = "2017"), color = "black") +
  ggtitle("Residential Usage Per Households") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "essential", "extra"))+
  big_text

ggplot(usage_2018_wwa[which(usage_2018_wwa$before_usage_level == 5),], aes(x = months_after)) + 
  geom_line(aes(y = extra_usage, color = "#ba0951"), linewidth = 1.5) +
  geom_line(aes(y = essential_usage, color = "#25738b"), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 400), expand = FALSE) +
  xlab("Months since rate changed")+
  ylab("Essential vs. Extra (hundreds gallon)")+
  geom_text(aes(x = 19, y = 1, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 1, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 1, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 1, label = "2017"), color = "black") +
  ggtitle("Residential Usage Per Households") +
  scale_color_manual(values = c("#ba0951", "#25738b"),labels = c( "essential", "extra"))+
  big_text

ggplot(usage_2018_wwa, aes(x = months_after)) + 
  #geom_line(aes(y = essential_usage, color = as.factor(before_usage_level) ), linewidth = 1.5) +
  geom_line(aes(y = extra_usage, color = as.factor(before_usage_level) ), linewidth = 1.5) +
  geom_vline(xintercept=0,linetype = "dashed", linewidth = 2) +
  coord_cartesian(xlim = c(-27, 21), ylim = c(0, 350), expand = FALSE) +
  xlab("Months since rate changed")+
  #ylab("Essential (hundreds gallon)")+
  ylab("Extra (hundreds gallon)")+
  geom_text(aes(x = 19, y = 1, label = "2020"), color = "black") +
  geom_text(aes(x = 7, y = 1, label = "2019"), color = "black") +
  geom_text(aes(x = -5, y = 1, label = "2018"), color = "black") +
  geom_text(aes(x = -17, y = 1, label = "2017"), color = "black") +
  ggtitle("Residential Usage Per Households") +
  scale_color_manual(values = c("#256676", "#a1d832", "#bf11af", "#75db96", "#863563"),labels = c( "1", "2", "3", "4", "5"))+
  big_text

write.csv(usage_2018_merge, file = "usage_2018_merge.csv", row.names = F)

##########
########## ########## ##########
##### Ito 2012 ##########
########## ########## ##########

charge_2018_merge = read_csv(file = "charge_2018_merge.csv")

calculate_fare = function(q, bill_ym){ ## q in 1,000 gallons
  if(bill_ym < 201611){
    f = case_when(
      q>=0 & q<=2 ~ 7.10+ 3.16*q + 1.2 + 0.19*q,
      q>2 & q<=6 ~ 7.10 + 3.45 + 3.16*2 + 4.84*(q-2)+ 0.19*q,
      q>6 & q<=11 ~ 7.10 + 8.75 + 3.16*2 + 4.84 *4 + 7.88 * (q-6)+ 0.19*q, 
      q>11 & q<=20 ~ 7.10 + 27.35 + 3.16*2 + 4.84 * 4 + 7.88 * 5 + 11.90*(q-11)+ 0.19*q,
      q>20 ~ 7.10 + 27.35 + 3.16*2 + 4.84 * 4 + 7.88 * 5 + 11.90*9 + 14.16 * (q-20)+ 0.19*q,
    )
  }else if (bill_ym >= 201611 & bill_ym < 201801){
    f = case_when(
      q>=0 & q<=2 ~ 7.10+ 3.18*q + 1.25+ 0.19*q,
      q>2 & q<=6 ~ 7.10 + 3.55 + 3.18*2 + 5.05*(q-2)+ 0.19*q,
      q>6 & q<=11 ~ 7.10 + 9.25 + 3.18*2 + 5.05 *4 + 8.56 * (q-6)+ 0.19*q, 
      q>11 & q<=20 ~ 7.10 + 29.75 + 3.18*2 + 5.05 * 4 + 8.56 * 5 + 12.92*(q-11)+ 0.19*q,
      q>20 ~ 7.10 + 29.75 + 3.18*2 + 5.05 * 4 + 8.56 * 5 + 12.92*9 + 14.43 * (q-20)+ 0.19*q,
    )
  }else if (bill_ym >= 201801 & bill_ym < 201805){
    f = case_when(
      q>=0 & q<=2 ~ 7.10+ 3.03*q + 1.25+ 0.34*q,
      q>2 & q<=6 ~ 7.10 + 3.55 + 3.03*2 + 4.90*(q-2)+ 0.34*q,
      q>6 & q<=11 ~ 7.10 + 9.25 + 3.03*2 + 4.90 *4 + 8.41 * (q-6)+ 0.34*q, 
      q>11 & q<=20 ~ 7.10 + 29.75 + 3.03*2 + 4.90 * 4 + 8.41 * 5 + 12.77*(q-11)+ 0.34*q,
      q>20 ~ 7.10 + 29.75 + 3.03*2 + 4.90 * 4 + 8.41 * 5 + 12.77*9 + 14.28 * (q-20)+ 0.34*q,
    )
  }else{
    f = case_when(
      q>=0 & q<=2 ~ 7.25+ 2.89*q + 1.25+ 0.2*q,
      q>2 & q<=6 ~ 7.25 + 3.55 + 2.89*2 + 4.81*(q-2)+ 0.2*q,
      q>6 & q<=11 ~ 7.25 + 9.25 + 2.89*2 + 4.81 *4 + 8.34 * (q-6)+ 0.2*q, 
      q>11 & q<=20 ~ 7.25 + 29.75 + 2.89*2 + 4.81 * 4 + 8.34 * 5 + 12.70*(q-11)+ 0.2*q,
      q>20 ~ 7.10 + 29.75 + 2.89*2 + 4.81 * 4 + 8.34 * 5 + 12.70*9 + 14.21 * (q-20)+ 0.2*q,
    )
  }
  return(f)
}

charge_2018_merge = charge_2018_merge %>%
  group_by(bill_ym) %>%
  mutate(fare = calculate_fare(quantity/10, unique(bill_ym)))

charge_2018_merge = merge(charge_2018_merge, usage_2018_ww, by.x = c("prem_id", "bill_ym"), by.y = c("prem_id", "bill_ym"), all.x = T)
charge_2018_merge$quantity = charge_2018_merge$quantity.x
charge_2018_merge$quantity.x = NULL
charge_2018_merge$quantity.y = NULL

calculate_marginal_p_water = function(q, bill_ym){ ## q in 1,000 gallons
  if(bill_ym < 201611){
    f = case_when(
      q>=0 & q<=2 ~ 3.16 + 0.19,
      q>2 & q<=6 ~ 4.84+ 0.19,
      q>6 & q<=11 ~ 7.88 + 0.19, 
      q>11 & q<=20 ~ 11.90+ 0.19,
      q>20 ~ 14.16 + 0.19,
    )
  }else if (bill_ym >= 201611 & bill_ym < 201801){
    f = case_when(
      q>=0 & q<=2 ~ 3.18+ 0.19,
      q>2 & q<=6 ~ 5.05+ 0.19,
      q>6 & q<=11 ~ 8.56 + 0.19, 
      q>11 & q<=20 ~ 12.92+ 0.19,
      q>20 ~ 14.43 + 0.19,
    )
  }else if (bill_ym >= 201801 & bill_ym < 201805){
    f = case_when(
      q>=0 & q<=2 ~ 3.03+0.34,
      q>2 & q<=6 ~ 4.90+ 0.34,
      q>6 & q<=11 ~ 8.41+ 0.34, 
      q>11 & q<=20 ~ 12.77+ 0.34,
      q>20 ~ 14.28+ 0.34,
    )
  }else{
    f = case_when(
      q>=0 & q<=2 ~ 2.89+ 0.2,
      q>2 & q<=6 ~ 4.81+ 0.2,
      q>6 & q<=11 ~ 8.34 + 0.2, 
      q>11 & q<=20 ~ 12.70+ 0.2,
      q>20 ~ 14.21 + 0.2,
    )
  }
  return(f)
}

charge_2018_merge = charge_2018_merge %>%
  group_by(bill_ym) %>%
  mutate(marginal_p_water = calculate_marginal_p_water(quantity/10, unique(bill_ym)))

charge_2018_merge$marginal_p_water = charge_2018_merge$marginal_p_water/10

calculate_marginal_p_ww = function(q, bill_ym){
  if(bill_ym<201611){
    mp = case_when(
      q<=20~ 0.49,
      q>20~ 0.994)
  }else if (bill_ym <201801 & bill_ym >= 201611){
    mp = case_when(
      q<=20~ 0.53,
     q>20~ 1.035)
  }else if (bill_ym <201805 & bill_ym >= 201801){
    mp = case_when(
      q<=20~ 0.53,
     q>20~ 1.035)
  }else{
    mp = case_when(
     q<=20~ 0.5,
     q>20~ 1.009)
  }
}

charge_2018_merge = charge_2018_merge %>%
  group_by(bill_ym) %>%
  mutate(marginal_p_ww = calculate_marginal_p_ww(quantity, unique(bill_ym)))


charge_2018_merge$avg_p_water_wadjustment = (charge_2018_merge$charge) / charge_2018_merge$quantity
charge_2018_merge$avg_p_water = (charge_2018_merge$fare) / charge_2018_merge$quantity

charge_2018_merge$avg_p_ww = (charge_2018_merge$wastew_f) / charge_2018_merge$wastew_q

charge_2018_merge$month = charge_2018_merge$bill_ym %% 100

charge_2018_merge = charge_2018_merge[which(charge_2018_merge$quantity>0),]

demand_2018 = charge_2018_merge[which(charge_2018_merge$rate_change == 1),]

write.csv(demand_2018, file = "demand_2018_og.csv", row.names = F)

#percipitation_revenue = read_csv(file = "percipitation_revenue.csv")

#charge_2018_merge = merge(charge_2018_merge, percipitation_revenue, by.x = c("bill_ym"), by.y = c("bill_ym"), all.x = T)

#charge_2018_merge = merge(charge_2018_merge, usage_2018_ww, by.x = c("prem_id", "bill_ym"), by.y = c("prem_id", "bill_ym"), all.x = T)

prem_key_hh_char = read_csv(file = "TCAD/prem_key_hh_char.csv")

charge_2018_merge= merge(charge_2018_merge, prem_key_hh_char, by.x = c("prem_id"), by.y = c("prem_id"), all.y = T)

calculate_d = function(q, bill_ym){ ## q in 1,000 gallons
  if(bill_ym < 201611){
    f = case_when(
      q>=0 & q<=2 ~ 7.10+ 1.2 ,
      q>2 & q<=6 ~ 7.10 + 3.45 + 3.16*2,
      q>6 & q<=11 ~ 7.10 + 8.75 + 3.16*2 + 4.84 *4, 
      q>11 & q<=20 ~ 7.10 + 27.35 + 3.16*2 + 4.84 * 4 + 7.88 * 5,
      q>20 ~ 7.10 + 27.35 + 3.16*2 + 4.84 * 4 + 7.88 * 5 + 11.90*9,
    )
  }else if (bill_ym >= 201611 & bill_ym < 201801){
    f = case_when(
      q>=0 & q<=2 ~ 7.10 + 1.25,
      q>2 & q<=6 ~ 7.10 + 3.55 + 3.18*2 ,
      q>6 & q<=11 ~ 7.10 + 9.25 + 3.18*2 + 5.05 *4, 
      q>11 & q<=20 ~ 7.10 + 29.75 + 3.18*2 + 5.05 * 4 + 8.56 * 5,
      q>20 ~ 7.10 + 29.75 + 3.18*2 + 5.05 * 4 + 8.56 * 5 + 12.92*9,
    )
  }else if (bill_ym >= 201801 & bill_ym < 201805){
    f = case_when(
      q>=0 & q<=2 ~ 7.10+ 1.25,
      q>2 & q<=6 ~ 7.10 + 3.55 + 3.03*2 ,
      q>6 & q<=11 ~ 7.10 + 9.25 + 3.03*2 + 4.90 *4, 
      q>11 & q<=20 ~ 7.10 + 29.75 + 3.03*2 + 4.90 * 4 + 8.41 * 5,
      q>20 ~ 7.10 + 29.75 + 3.03*2 + 4.90 * 4 + 8.41 * 5 + 12.77*9,
    )
  }else{
    f = case_when(
      q>=0 & q<=2 ~ 7.25 + 1.25,
      q>2 & q<=6 ~ 7.25 + 3.55 + 2.89*2,
      q>6 & q<=11 ~ 7.25 + 9.25 + 2.89*2 + 4.81 *4, 
      q>11 & q<=20 ~ 7.25 + 29.75 + 2.89*2 + 4.81 * 4 + 8.34 * 5,
      q>20 ~ 7.10 + 29.75 + 2.89*2 + 4.81 * 4 + 8.34 * 5 + 12.70*9,
    )
  }
  return(f)
}

charge_2018_merge = charge_2018_merge %>%
  group_by(bill_ym) %>%
  mutate(d = -1*calculate_d(quantity/10, unique(bill_ym)))

charge_2018_merge$house_deflator = case_when(
  charge_2018_merge$bill_ym>=201601 & charge_2018_merge$bill_ym<201604 ~ 89.22714/172.29451,
  charge_2018_merge$bill_ym>=201604 & charge_2018_merge$bill_ym<201607 ~ 91.77714/172.29451,
  charge_2018_merge$bill_ym>=201607 & charge_2018_merge$bill_ym<201610 ~ 93.86232/172.29451,
  charge_2018_merge$bill_ym>=201610 & charge_2018_merge$bill_ym<201701 ~ 94.36315/172.29451,
  charge_2018_merge$bill_ym>=201701 & charge_2018_merge$bill_ym<201704 ~ 95.79364/172.29451,
  charge_2018_merge$bill_ym>=201704 & charge_2018_merge$bill_ym<201707 ~ 98.98524/172.29451,
  charge_2018_merge$bill_ym>=201707 & charge_2018_merge$bill_ym<201710 ~ 100/172.29451,
  charge_2018_merge$bill_ym>=201710 & charge_2018_merge$bill_ym<201801 ~ 100.3208/172.29451,
  charge_2018_merge$bill_ym>=201801 & charge_2018_merge$bill_ym<201804 ~ 101.72182/172.29451,
  charge_2018_merge$bill_ym>=201804 & charge_2018_merge$bill_ym<201807 ~ 104.43222/172.29451,
  charge_2018_merge$bill_ym>=201807 & charge_2018_merge$bill_ym<201810 ~ 105.84635/172.29451,
  charge_2018_merge$bill_ym>=201810 & charge_2018_merge$bill_ym<201901 ~ 105.75142/172.29451,
  charge_2018_merge$bill_ym>=201901 & charge_2018_merge$bill_ym<201904 ~ 108.35707/172.29451,
  charge_2018_merge$bill_ym>=201904 & charge_2018_merge$bill_ym<201907 ~ 110.7794/172.29451,
  charge_2018_merge$bill_ym>=201907 & charge_2018_merge$bill_ym<201910 ~ 112.33101/172.29451,
  charge_2018_merge$bill_ym>=201910 & charge_2018_merge$bill_ym<202001 ~ 113.36869/172.29451,
  charge_2018_merge$bill_ym>=202001 & charge_2018_merge$bill_ym<=202002 ~ 115.31638/172.29451
)

#charge_2018_merge$total_housevalue = charge_2018_merge$total_housevalue* charge_2018_merge$house_deflator

charge_2018_merge$virtual_income=charge_2018_merge$d + charge_2018_merge$total_housevalue

austin_hh_weather = read_csv(file = "weather/Austin_hh_weather.csv")
austin_hh_weather$STATION = NULL
austin_hh_weather$NAME = NULL
austin_hh_weather$Latitude = NULL
austin_hh_weather$Longitude = NULL
austin_hh_weather$LATITUDE_1 = NULL
austin_hh_weather$LONGITUDE_1 = NULL


charge_2018_sum = charge_2018_merge[which(charge_2018_merge$rate_change == 0),] %>%
  group_by(prem_id, month, rate_change) %>%
  summarize(quantity = mean(quantity, na.rm = T),
            wastew_q = mean(wastew_q, na.rm = T),
            extra_usage = mean(extra_usage, na.rm = T),
            essential_usage = mean(essential_usage, na.rm = T),
            marginal_p_water = mean(marginal_p_water, na.rm = T),
            marginal_p_ww = mean(marginal_p_ww, na.rm = T),
            wastew_mp = mean(wastew_mp, na.rm = T),
            avg_p_water_wadjustment = mean(avg_p_water_wadjustment, na.rm = T),
            avg_p_water = mean(avg_p_water, na.rm = T),
            avg_p_ww = mean(avg_p_ww, na.rm = T),
            total_housevalue = mean(total_housevalue, na.rm = T),
            virtual_income = mean(virtual_income, na.rm = T),
            d = mean(d, na.rm = T)
            #percipitation = mean(percipitation, na.rm = T),
            #MaxT = mean(MaxT, na.rm = T),
            #AvgT = mean(AvgT, na.rm = T),
            #MinT = mean(MinT, na.rm = T)
            )

#charge_2018_sum = charge_2018_sum[which(charge_2018_sum$rate_change == 0),]

charge_2018_after = charge_2018_merge[which(charge_2018_merge$rate_change == 1),]

charge_2018_sum$rate_change = NULL

charge_2018_after = merge(charge_2018_after, charge_2018_sum, by.x = c("prem_id", "month"), by.y = c("prem_id", "month"), all.x = T)

charge_2018_after$quantity_base = charge_2018_after$quantity.y
charge_2018_after$wastew_q_base = charge_2018_after$wastew_q.y
charge_2018_after$marginal_p_water_base = charge_2018_after$marginal_p_water.y
charge_2018_after$marginal_p_ww_base = charge_2018_after$marginal_p_ww.y
charge_2018_after$wastew_mp_base = charge_2018_after$wastew_mp.y
charge_2018_after$avg_p_water_wadjustment_base = charge_2018_after$avg_p_water_wadjustment.y
charge_2018_after$avg_p_water_base = charge_2018_after$avg_p_water.y
charge_2018_after$avg_p_ww_base = charge_2018_after$avg_p_ww.y
charge_2018_after$extra_usage_base = charge_2018_after$extra_usage.y
charge_2018_after$essential_usage_base = charge_2018_after$essential_usage.y
charge_2018_after$virtual_income_base = charge_2018_after$virtual_income.y
charge_2018_after$total_housevalue_base = charge_2018_after$total_housevalue.y
charge_2018_after$d_base = charge_2018_after$d.y
#charge_2018_after$percipitation_base = charge_2018_after$percipitation.y
#charge_2018_after$MaxT_base = charge_2018_after$MaxT.y
#charge_2018_after$AvgT_base = charge_2018_after$AvgT.y
#charge_2018_after$MinT_base = charge_2018_after$MinT.y

charge_2018_after$quantity = charge_2018_after$quantity.x
charge_2018_after$wastew_q = charge_2018_after$wastew_q.x
charge_2018_after$marginal_p_water = charge_2018_after$marginal_p_water.x
charge_2018_after$marginal_p_ww = charge_2018_after$marginal_p_ww.x
charge_2018_after$wastew_mp = charge_2018_after$wastew_mp.x
charge_2018_after$avg_p_water_wadjustment = charge_2018_after$avg_p_water_wadjustment.x
charge_2018_after$avg_p_water = charge_2018_after$avg_p_water.x
charge_2018_after$avg_p_ww = charge_2018_after$avg_p_ww.x
charge_2018_after$extra_usage = charge_2018_after$extra_usage.x
charge_2018_after$essential_usage = charge_2018_after$essential_usage.x
charge_2018_after$virtual_income = charge_2018_after$virtual_income.x
charge_2018_after$total_housevalue = charge_2018_after$total_housevalue.x
charge_2018_after$d = charge_2018_after$d.x
#charge_2018_after$percipitation = charge_2018_after$percipitation.x
#charge_2018_after$MaxT = charge_2018_after$MaxT.x
#charge_2018_after$AvgT = charge_2018_after$AvgT.x
#charge_2018_after$MinT = charge_2018_after$MinT.x

charge_2018_after$quantity.x = NULL
charge_2018_after$wastew_q.x = NULL
charge_2018_after$marginal_p_water.x = NULL
charge_2018_after$marginal_p_ww.x = NULL
charge_2018_after$wastew_mp.x = NULL
charge_2018_after$avg_p_water_wadjustment.x = NULL
charge_2018_after$avg_p_water.x = NULL
charge_2018_after$avg_p_ww.x = NULL
charge_2018_after$essential_usage.x = NULL
charge_2018_after$extra_usage.x = NULL
charge_2018_after$virtual_income.x = NULL
charge_2018_after$total_housevalue.x = NULL
charge_2018_after$d.x = NULL
#charge_2018_after$percipitation.x = NULL
#charge_2018_after$MaxT.x = NULL
#charge_2018_after$AvgT.x = NULL
#charge_2018_after$MinT.x = NULL

charge_2018_after$quantity.y = NULL
charge_2018_after$wastew_q.y = NULL
charge_2018_after$marginal_p_water.y = NULL
charge_2018_after$marginal_p_ww.y = NULL
charge_2018_after$wastew_mp.y = NULL
charge_2018_after$avg_p_water_wadjustment.y = NULL
charge_2018_after$avg_p_water.y = NULL
charge_2018_after$avg_p_ww.y = NULL
charge_2018_after$essential_usage.y = NULL
charge_2018_after$extra_usage.y = NULL
charge_2018_after$virtual_income.y = NULL
charge_2018_after$total_housevalue.y = NULL
charge_2018_after$d.y = NULL
#charge_2018_after$percipitation.y = NULL
#charge_2018_after$MaxT.y = NULL
#charge_2018_after$AvgT.y = NULL
#charge_2018_after$MinT.y = NULL

charge_2018_after = charge_2018_after[order(charge_2018_after$prem_id, charge_2018_after$bill_ym),]
charge_2018_after = charge_2018_after[which(charge_2018_after$charge>0),]

charge_2018_after$essential_usage = pmax(charge_2018_after$essential_usage, 0.01)
charge_2018_after$essential_usage_base = pmax(charge_2018_after$essential_usage_base, 0.01)
charge_2018_after$extra_usage = pmax(charge_2018_after$extra_usage, 0.01)
charge_2018_after$extra_usage_base = pmax(charge_2018_after$extra_usage_base, 0.01)

charge_2018_after$did_quantity = log(charge_2018_after$quantity) - log(charge_2018_after$quantity_base)
charge_2018_after$did_essential_usage = log(charge_2018_after$essential_usage) - log(charge_2018_after$essential_usage_base)
charge_2018_after$did_extra_usage = log(charge_2018_after$extra_usage) - log(charge_2018_after$extra_usage_base)
charge_2018_after$did_wastew_q = log(charge_2018_after$wastew_q) - log(charge_2018_after$wastew_q_base)

charge_2018_after$did_mp_w = log(charge_2018_after$marginal_p_water) - log(charge_2018_after$marginal_p_water_base)
charge_2018_after$did_mp_ww = log(charge_2018_after$marginal_p_ww) - log(charge_2018_after$marginal_p_ww_base)
charge_2018_after$did_wastew_ww = log(charge_2018_after$wastew_mp) - log(charge_2018_after$wastew_mp_base)
charge_2018_after$did_ap_wadj = log(charge_2018_after$avg_p_water_wadjustment) - log(charge_2018_after$avg_p_water_wadjustment_base)
charge_2018_after$did_ap = log(charge_2018_after$avg_p_water) - log(charge_2018_after$avg_p_water_base)
charge_2018_after$did_ap_ww = log(charge_2018_after$avg_p_ww) - log(charge_2018_after$avg_p_ww_base)

charge_2018_after$did_virtual_i = log(charge_2018_after$virtual_income) - log(charge_2018_after$virtual_income_base)
charge_2018_after$did_total_i = log(charge_2018_after$total_housevalue) - log(charge_2018_after$total_housevalue_base)

#charge_2018_after$did_d = log(charge_2018_after$d) - log(charge_2018_after$d_base)

#charge_2018_after$did_percipitation = log(charge_2018_after$percipitation) - log(charge_2018_after$percipitation_base)
#charge_2018_after$did_MaxT = log(charge_2018_after$MaxT) - log(charge_2018_after$MaxT_base)
#charge_2018_after$did_AvgT = log(charge_2018_after$AvgT) - log(charge_2018_after$AvgT_base)
#charge_2018_after$did_MinT = log(charge_2018_after$MinT) - log(charge_2018_after$MinT_base)

write.csv(charge_2018_after, file = "charge_2018_after.csv", row.names = F)

#charge_2018_after_category = charge_2018_after %>%
 # group_by(prem_id) %>%
  #summarize(mean_quantity_after = mean(quantity, na.rm = T),
   #         sd_quantity_after = sd(quantity, na.rm = T))

#charge_2018_after_category$usage_after= case_when(
 # charge_2018_after_category$mean_quantity_after <=20 ~ 1,
  #charge_2018_after_category$mean_quantity_after >20 & charge_2018_after_category$mean_quantity_after <=60  ~ 2,
  #charge_2018_after_category$mean_quantity_after >60 &charge_2018_after_category$mean_quantity_after <=110 ~ 3,
  #charge_2018_after_category$mean_quantity_after >110 &charge_2018_after_category$mean_quantity_after <=200 ~ 4,
  #charge_2018_after_category$mean_quantity_after >200 ~ 5)

#charge_2018_after_category$coeff_after = charge_2018_after_category$sd_quantity_after / charge_2018_after_category$mean_quantity_after

#charge_2018_after_category$coeff_level_after= case_when(
 # charge_2018_after_category$coeff_after <=1 ~ 1,  
  #charge_2018_after_category$coeff_after >1 ~ 2)

#charge_2018_after = merge(charge_2018_after, charge_2018_after_category, by.x = c("prem_id"), 
 #                         by.y = c("prem_id"), all.x = T)

charge_2018_after=  charge_2018_after %>%
  group_by(prem_id) %>%
  mutate(mean_virtual_i = mean(virtual_income, na.rm = T))



charge_2018_after$income_tier = case_when(
  charge_2018_after$mean_virtual_i<=150000~"1",
  charge_2018_after$mean_virtual_i>150000 & charge_2018_after$mean_virtual_i<=250000~"2",
  charge_2018_after$mean_virtual_i>250000 & charge_2018_after$mean_virtual_i<=400000~"3",
  charge_2018_after$mean_virtual_i>400000~"4"
)

charge_2018_did = charge_2018_after %>%
  group_by(bill_ym, income_tier) %>%
  summarize(quantity = mean(did_quantity, na.rm = T),
            essential_usage = mean(did_essential_usage, na.rm = T),
            extra_usage = mean(did_extra_usage, na.rm = T),
            wastew_q = mean(did_wastew_q, na.rm = T),
            marginal_p_w = mean(did_mp_w, na.rm = T),
            marginal_p_ww = mean(did_mp_ww, na.rm = T),
            ap = mean(did_ap, na.rm = T),
            ap_wadj = mean(did_ap_wadj, na.rm = T),
            ap_ww = mean(did_ap_ww, na.rm = T),
            virtual_income = mean(did_virtual_i, na.rm = T),
            total_housevalue = mean(did_total_i, na.rm = T),
            d = mean(did_d, na.rm = T)
            #percipitation = mean(did_percipitation, na.rm = T),
            #MinT = mean(did_MinT, na.rm = T),
            #AvgT = mean(did_AvgT, na.rm = T),
            #MaxT = mean(did_MaxT, na.rm = T)
            )

#percipitation_revenue = read_csv(file = "percipitation_revenue.csv")

#charge_2018_did = merge(charge_2018_did, percipitation_revenue, by.x = c("bill_ym"), by.y = c("bill_ym")
# , all.x = T)

charge_2018_did$bill_ym = as.yearmon(as.character(charge_2018_did$bill_ym),"%Y%m")
charge_2018_did$bill_ym = as.Date(charge_2018_did$bill_ym)

# 0 - 150000
## 150000 - 250000
### 250000 - 400000
#### > 400000

ratio = 0.00005
vi1 = ggplot(charge_2018_did[which(charge_2018_did$income_tier == "1"),], aes(x = bill_ym)) + 
  geom_line(aes(y = quantity, color = "Quantity"), linewidth = 1.5) +
  geom_line(aes(y = extra_usage, color = "Quantity"), linewidth = 1.5, linetype = "dotdash") +
  geom_line(aes(y = essential_usage, color = "Quantity"), linewidth = 1.5, linetype = "longdash") +
  geom_line(aes(y = virtual_income/ratio, color = "Virtual Income"), linewidth = 1.5) +
  #geom_line(aes(y = ap/ratio, color = "A_P"), linewidth = 1.5) +
  coord_cartesian(ylim = c(-3.8, 3.8), expand = FALSE) +
  xlab("Year-Mon")+
  ylab("DiD Log Quantity")+
  ggtitle("House Value < 150k") +
  scale_color_manual(values = c("#023880", "#1c875c"),labels = c("Quantity","Virtual Income"))+
  scale_y_continuous(
    "DiD Log Quantity", 
    sec.axis = sec_axis(~ . *ratio, name = "DiD Log Virtual Income")
  ) +
  big_text

vi2 = ggplot(charge_2018_did[which(charge_2018_did$income_tier == "2"),], aes(x = bill_ym)) + 
  geom_line(aes(y = quantity, color = "Quantity"), linewidth = 1.5) +
  geom_line(aes(y = extra_usage, color = "Quantity"), linewidth = 1.5, linetype = "dotdash") +
  geom_line(aes(y = essential_usage, color = "Quantity"), linewidth = 1.5, linetype = "longdash") +
  geom_line(aes(y = virtual_income/ratio, color = "Virtual Income"), linewidth = 1.5) +
  #geom_line(aes(y = ap/ratio, color = "A_P"), linewidth = 1.5) +
  coord_cartesian(ylim = c(-3.8, 3.8), expand = FALSE) +
  xlab("Year-Mon")+
  ylab("DiD Log Quantity")+
  ggtitle("House Value: [150k, 250k]") +
  scale_color_manual(values = c("#023880", "#1c875c"),labels = c("Quantity","Virtual Income"))+
  scale_y_continuous(
    "DiD Log Quantity", 
    sec.axis = sec_axis(~ . *ratio, name = "DiD Log Virtual Income")
  ) +
  big_text

vi3 = ggplot(charge_2018_did[which(charge_2018_did$income_tier == "3"),], aes(x = bill_ym)) + 
  geom_line(aes(y = quantity, color = "Quantity"), linewidth = 1.5) +
  geom_line(aes(y = extra_usage, color = "Quantity"), linewidth = 1.5, linetype = "dotdash") +
  geom_line(aes(y = essential_usage, color = "Quantity"), linewidth = 1.5, linetype = "longdash") +
  geom_line(aes(y = virtual_income/ratio, color = "Virtual Income"), linewidth = 1.5) +
  #geom_line(aes(y = ap/ratio, color = "A_P"), linewidth = 1.5) +
  coord_cartesian(ylim = c(-3.8, 3.8), expand = FALSE) +
  xlab("Year-Mon")+
  ylab("DiD Log Quantity")+
  ggtitle("House Value: [250k, 400k]") +
  scale_color_manual(values = c("#023880", "#1c875c"),labels = c("Quantity","Virtual Income"))+
  scale_y_continuous(
    "DiD Log Quantity", 
    sec.axis = sec_axis(~ . *ratio, name = "DiD Log Virtual Income")
  ) +
  big_text

vi4 = ggplot(charge_2018_did[which(charge_2018_did$income_tier == "4"),], aes(x = bill_ym)) + 
  geom_line(aes(y = quantity, color = "Quantity"), linewidth = 1.5) +
  geom_line(aes(y = extra_usage, color = "Quantity"), linewidth = 1.5, linetype = "dotdash") +
  geom_line(aes(y = essential_usage, color = "Quantity"), linewidth = 1.5, linetype = "longdash") +
  geom_line(aes(y = virtual_income/ratio, color = "Virtual Income"), linewidth = 1.5) +
  #geom_line(aes(y = ap/ratio, color = "A_P"), linewidth = 1.5) +
  coord_cartesian(ylim = c(-3.8, 3.8), expand = FALSE) +
  xlab("Year-Mon")+
  ylab("DiD Log Quantity")+
  ggtitle("House Value: >400k") +
  scale_color_manual(values = c("#023880", "#1c875c"),labels = c("Quantity","Virtual Income"))+
  scale_y_continuous(
    "DiD Log Quantity", 
    sec.axis = sec_axis(~ . *ratio, name = "DiD Log Virtual Income")
  ) +
  big_text

vi1234 = ggarrange(vi1, vi2, vi3, vi4, ncol = 2, nrow = 2, common.legend = T, legend = "bottom")
annotate_figure(vi1234, top = text_grob("DiD Quantity and V-Income", 
                                         color = "Black", face = "bold", size = 26))


############################################
################ Income Effect?################################
####################################




### Using Geopgraphical

zipcode = read_csv(file = "zip-code.csv")

charge_2018_after = merge(charge_2018_after, zipcode, by.x = c("prem_id"), by.y = c("prem_id"), all.x = T)

income_extreme = charge_2018_after[which(charge_2018_after$zipcode == 78739 | charge_2018_after$zipcode == 78748),]

income_extreme_small = income_extreme[which(income_extreme$extra_usage_base>0.01),]
income_extreme_small = income_extreme_small[which(income_extreme_small$extra_usage>0.01),]

income_extreme_sum = income_extreme_small %>%
  #income_extreme_sum = income_extreme %>% 
  group_by(bill_ym, zipcode) %>%
  summarize(quantity = mean(did_quantity, na.rm = T),
            essential_usage = mean(did_essential_usage, na.rm = T),
            extra_usage = mean(did_extra_usage, na.rm = T),
            wastew_q = mean(did_wastew_q, na.rm = T),
            marginal_p_w = mean(did_mp_w, na.rm = T),
            marginal_p_ww = mean(did_mp_ww, na.rm = T),
            ap = mean(did_ap, na.rm = T),
            ap_wadj = mean(did_ap_wadj, na.rm = T),
            ap_ww = mean(did_ap_ww, na.rm = T),
            percipitation = mean(did_percipitation, na.rm = T),
            MinT = mean(did_MinT, na.rm = T),
            AvgT = mean(did_AvgT, na.rm = T),
            MaxT = mean(did_MaxT, na.rm = T))

income_extreme_sum$bill_ym = as.yearmon(as.character(income_extreme_sum$bill_ym),"%Y%m")
income_extreme_sum$bill_ym = as.Date(income_extreme_sum$bill_ym)

ggplot(income_extreme_sum, aes(x = bill_ym, color = as.factor(zipcode ))) + 
  #geom_line(aes(y = quantity), linewidth = 1.5) +
  geom_line(aes(y = essential_usage), linewidth = 1.5, linetype = "longdash") +
  geom_line(aes(y = extra_usage), linewidth = 1.5, linetype = "dotdash") +
  coord_cartesian(ylim = c(-1, 1), expand = FALSE) +
  xlab("Year-Mon")+
  ylab("DiD Log Quantity")+
  ggtitle("Income Effect ?") + 
  big_text


#### Different hhs with different incomes have different effect on extra water usage. This is highly correlated to weather. 

### Check households in the higher quantiles of consumption will have zero effects from a change in
#the price at lower levels of consumption. This is because income effects in this
#model are null, and the substitution effects will not change, as the marginal price
#of the higher levels doesn't change.

charge_2018_merge_small = charge_2018_merge[which( (charge_2018_merge$bill_ym < 201805 & charge_2018_merge$bill_ym >= 201801) | 
                                                     (charge_2018_merge$bill_ym < 201705 & charge_2018_merge$bill_ym >= 201701) ),]

charge_2018_merge_small$longtitude = NULL
charge_2018_merge_small$latitude = NULL
charge_2018_merge_small$mean_quantity = NULL
charge_2018_merge_small$sd_quantity = NULL
charge_2018_merge_small$usage = NULL
charge_2018_merge_small$coeff = NULL
charge_2018_merge_small$coeff_level = NULL
charge_2018_merge_small$category = NULL
charge_2018_merge_small$length_f_type = NULL
charge_2018_merge_small$length_f_unique = NULL
charge_2018_merge_small$rate_change = NULL

charge_2018_sum_small = charge_2018_merge_small[which(charge_2018_merge_small$bill_ym < 201801),]
charge_2018_after_small = charge_2018_merge_small[which(charge_2018_merge_small$bill_ym >= 201801),]

charge_2018_sum_small$bill_ym = NULL
charge_2018_after_small$bill_ym = NULL

charge_2018_after_small = merge(charge_2018_after_small, charge_2018_sum_small, by.x = c("prem_id","month"), by.y = c("prem_id","month"),
                                all.x = T)

charge_2018_after_small$quantity_base = charge_2018_after_small$quantity.y
charge_2018_after_small$wastew_q_base = charge_2018_after_small$wastew_q.y
charge_2018_after_small$marginal_p_water_base = charge_2018_after_small$marginal_p_water.y
charge_2018_after_small$marginal_p_ww_base = charge_2018_after_small$marginal_p_ww.y
charge_2018_after_small$wastew_mp_base = charge_2018_after_small$wastew_mp.y
charge_2018_after_small$avg_p_water_wadjustment_base = charge_2018_after_small$avg_p_water_wadjustment.y
charge_2018_after_small$avg_p_water_base = charge_2018_after_small$avg_p_water.y
charge_2018_after_small$avg_p_ww_base = charge_2018_after_small$avg_p_ww.y
charge_2018_after_small$extra_usage_base = charge_2018_after_small$extra_usage.y
charge_2018_after_small$essential_usage_base = charge_2018_after_small$essential_usage.y
charge_2018_after_small$percipitation_base = charge_2018_after_small$percipitation.y
charge_2018_after_small$MaxT_base = charge_2018_after_small$MaxT.y
charge_2018_after_small$AvgT_base = charge_2018_after_small$AvgT.y
charge_2018_after_small$MinT_base = charge_2018_after_small$MinT.y

charge_2018_after_small$quantity = charge_2018_after_small$quantity.x
charge_2018_after_small$wastew_q = charge_2018_after_small$wastew_q.x
charge_2018_after_small$marginal_p_water = charge_2018_after_small$marginal_p_water.x
charge_2018_after_small$marginal_p_ww = charge_2018_after_small$marginal_p_ww.x
charge_2018_after_small$wastew_mp = charge_2018_after_small$wastew_mp.x
charge_2018_after_small$avg_p_water_wadjustment = charge_2018_after_small$avg_p_water_wadjustment.x
charge_2018_after_small$avg_p_water = charge_2018_after_small$avg_p_water.x
charge_2018_after_small$avg_p_ww = charge_2018_after_small$avg_p_ww.x
charge_2018_after_small$extra_usage = charge_2018_after_small$extra_usage.x
charge_2018_after_small$essential_usage = charge_2018_after_small$essential_usage.x
charge_2018_after_small$percipitation = charge_2018_after_small$percipitation.x
charge_2018_after_small$MaxT = charge_2018_after_small$MaxT.x
charge_2018_after_small$AvgT = charge_2018_after_small$AvgT.x
charge_2018_after_small$MinT = charge_2018_after_small$MinT.x

charge_2018_after_small$quantity.x = NULL
charge_2018_after_small$wastew_q.x = NULL
charge_2018_after_small$marginal_p_water.x = NULL
charge_2018_after_small$marginal_p_ww.x = NULL
charge_2018_after_small$wastew_mp.x = NULL
charge_2018_after_small$avg_p_water_wadjustment.x = NULL
charge_2018_after_small$avg_p_water.x = NULL
charge_2018_after_small$avg_p_ww.x = NULL
charge_2018_after_small$essential_usage.x = NULL
charge_2018_after_small$extra_usage.x = NULL
charge_2018_after_small$percipitation.x = NULL
charge_2018_after_small$MaxT.x = NULL
charge_2018_after_small$AvgT.x = NULL
charge_2018_after_small$MinT.x = NULL

charge_2018_after_small$quantity.y = NULL
charge_2018_after_small$wastew_q.y = NULL
charge_2018_after_small$marginal_p_water.y = NULL
charge_2018_after_small$marginal_p_ww.y = NULL
charge_2018_after_small$wastew_mp.y = NULL
charge_2018_after_small$avg_p_water_wadjustment.y = NULL
charge_2018_after_small$avg_p_water.y = NULL
charge_2018_after_small$avg_p_ww.y = NULL
charge_2018_after_small$essential_usage.y = NULL
charge_2018_after_small$extra_usage.y = NULL
charge_2018_after_small$percipitation.y = NULL
charge_2018_after_small$MaxT.y = NULL
charge_2018_after_small$AvgT.y = NULL
charge_2018_after_small$MinT.y = NULL

charge_2018_after_small = charge_2018_after_small[which(charge_2018_after_small$charge.x>0),]
charge_2018_after_small = charge_2018_after_small[which(charge_2018_after_small$charge.y>0),]

charge_2018_after_small$essential_usage = pmax(charge_2018_after_small$essential_usage, 0.01)
charge_2018_after_small$essential_usage_base = pmax(charge_2018_after_small$essential_usage_base, 0.01)
charge_2018_after_small$extra_usage = pmax(charge_2018_after_small$extra_usage, 0.01)
charge_2018_after_small$extra_usage_base = pmax(charge_2018_after_small$extra_usage_base, 0.01)

charge_2018_after_small$did_quantity = log(charge_2018_after_small$quantity) - log(charge_2018_after_small$quantity_base)
charge_2018_after_small$did_essential_usage = log(charge_2018_after_small$essential_usage) - log(charge_2018_after_small$essential_usage_base)
charge_2018_after_small$did_extra_usage = log(charge_2018_after_small$extra_usage) - log(charge_2018_after_small$extra_usage_base)
charge_2018_after_small$did_wastew_q = log(charge_2018_after_small$wastew_q) - log(charge_2018_after_small$wastew_q_base)

charge_2018_after_small$did_mp_w = log(charge_2018_after_small$marginal_p_water) - log(charge_2018_after_small$marginal_p_water_base)
charge_2018_after_small$did_mp_ww = log(charge_2018_after_small$marginal_p_ww) - log(charge_2018_after_small$marginal_p_ww_base)
charge_2018_after_small$did_wastew_ww = log(charge_2018_after_small$wastew_mp) - log(charge_2018_after_small$wastew_mp_base)
charge_2018_after_small$did_ap_wadj = log(charge_2018_after_small$avg_p_water_wadjustment) - log(charge_2018_after_small$avg_p_water_wadjustment_base)
charge_2018_after_small$did_ap = log(charge_2018_after_small$avg_p_water) - log(charge_2018_after_small$avg_p_water_base)
charge_2018_after_small$did_ap_ww = log(charge_2018_after_small$avg_p_ww) - log(charge_2018_after_small$avg_p_ww_base)

charge_2018_after_small$did_percipitation = log(charge_2018_after_small$percipitation) - log(charge_2018_after_small$percipitation_base)
charge_2018_after_small$did_MaxT = log(charge_2018_after_small$MaxT) - log(charge_2018_after_small$MaxT_base)
charge_2018_after_small$did_AvgT = log(charge_2018_after_small$AvgT) - log(charge_2018_after_small$AvgT_base)
charge_2018_after_small$did_MinT = log(charge_2018_after_small$MinT) - log(charge_2018_after_small$MinT_base)


charge_2018_after_small_category = charge_2018_after_small %>%
  group_by(prem_id) %>%
  summarize(mean_quantity_after = mean(quantity, na.rm = T),
            sd_quantity_after = sd(quantity, na.rm = T))

charge_2018_after_small_category$usage_after= case_when(
  charge_2018_after_small_category$mean_quantity_after <=20 ~ 1,
  charge_2018_after_small_category$mean_quantity_after >20 & charge_2018_after_small_category$mean_quantity_after <=60  ~ 2,
  charge_2018_after_small_category$mean_quantity_after >60 &charge_2018_after_small_category$mean_quantity_after <=110 ~ 3,
  charge_2018_after_small_category$mean_quantity_after >110 &charge_2018_after_small_category$mean_quantity_after <=200 ~ 4,
  charge_2018_after_small_category$mean_quantity_after >200 ~ 5)


charge_2018_after_small = merge(charge_2018_after_small, charge_2018_after_small_category, by.x = c("prem_id"), 
                          by.y = c("prem_id"), all.x = T)

charge_2018_did_small = charge_2018_after_small %>%
  group_by(month, usage_after) %>%
  summarize(quantity = mean(did_quantity, na.rm = T),
            essential_usage = mean(did_essential_usage, na.rm = T),
            extra_usage = mean(did_extra_usage, na.rm = T),
            wastew_q = mean(did_wastew_q, na.rm = T),
            marginal_p_w = mean(did_mp_w, na.rm = T),
            marginal_p_ww = mean(did_mp_ww, na.rm = T),
            ap = mean(did_ap, na.rm = T),
            ap_wadj = mean(did_ap_wadj, na.rm = T),
            ap_ww = mean(did_ap_ww, na.rm = T),
            percipitation = mean(did_percipitation, na.rm = T),
            MinT = mean(did_MinT, na.rm = T),
            AvgT = mean(did_AvgT, na.rm = T),
            MaxT = mean(did_MaxT, na.rm = T))

ratio = 0.12
ggplot(charge_2018_did_small[which(charge_2018_did_small$usage_after==5),], aes(x = month)) + 
  #geom_line(aes(y = quantity, color = "Quantity"), linewidth = 1.5) +
  #geom_line(aes(y = essential_usage, color = "Quantity"), linewidth = 1.5, linetype = "longdash") +
  geom_line(aes(y = extra_usage, color = "Quantity"), linewidth = 1.5, linetype = "dotdash") +
  geom_line(aes(y = marginal_p_w/ratio, color = "M_P"), linewidth = 1.5) +
  geom_line(aes(y = ap/ratio, color = "A_P"), linewidth = 1.5) +
  geom_line(aes(y = MaxT/ratio, color = "Temp"), linewidth = 1, linetype = "dotdash") +
  geom_line(aes(y = percipitation, color = "Percipitation"), linewidth = 1, linetype = "dotdash") +
  coord_cartesian(ylim = c(-3.5, 3.5), expand = FALSE) +
  xlab("Year-Mon")+
  ylab("DiD Log Quantity/Rainfall")+
  ggtitle("DiD in P, Q, Weather") + 
  scale_color_manual(values = c("#4553c2", "#2d7a2c", "#841ea4", "#467ab2", "#163719"),labels = c("A_P", "M_P", "Percipitation", "Quantity", "Temp"))+
  scale_y_continuous(
    "DiD Log Quantity", 
    sec.axis = sec_axis(~ . *ratio, name = "DiD Log Price/T")
  ) +
  big_text

###


##################################################
########## ITO 2012 Weight Estimation ############
##################################################


summary(ivreg(did_quantity ~ did_ap2 + bill_ym + bseg_end_dt | 
                appi + bill_ym +bseg_end_dt, data = charge_2018_after))

summary(lm(did_quantity ~ did_mp, data = charge_2018_after))

#Call:
 # lm(formula = did_quantity ~ did_mp + did_ap, data = charge_2018_after)

#Residuals:
 # Min       1Q   Median       3Q      Max 
#-17.7313  -0.2054   0.0144   0.2273   4.7860 

#Coefficients:
 # Estimate Std. Error t value Pr(>|t|)    
#(Intercept) -0.0592768  0.0001884  -314.7   <2e-16 ***
 # did_mp       1.6214158  0.0004885  3319.1   <2e-16 ***
  #did_ap      -0.4129277  0.0003766 -1096.5   <2e-16 ***
  #---
  #Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#Residual standard error: 0.3886 on 4312727 degrees of freedom
#(8100 observations deleted due to missingness)
#Multiple R-squared:  0.7447,	Adjusted R-squared:  0.7447 
#F-statistic: 6.291e+06 on 2 and 4312727 DF,  p-value: < 2.2e-16

find_mid_point = function(x){
  result=c()
  for (i in 1:length(x)) {
    result[i] = x[ceiling(i/2)]
  }
  return(result)
}

charge_2018_after = charge_2018_after %>%
  group_by(prem_id) %>%
  mutate(x_m = find_mid_point(quantity))

#charge_2018_after = charge_2018_after %>%
 # group_by(prem_id) %>%
  #mutate(essential_m = find_mid_point(essential_usage))

#charge_2018_after = charge_2018_after %>%
 # group_by(prem_id) %>%
  #mutate(extra_m = find_mid_point(extra_usage))

simulate_m_p_log_d = function(x){
  return(case_when(
    x<=20 ~ log(0.289+0.485+0.015) - log(0.318 + 0.53),
    x>20 & x<=60 ~ log(0.481+0.994+0.015) - log(0.505+1.035),
    x>60 & x<=110 ~ log(0.834+0.994+0.015) - log(0.856+1.035),
    x>110 & x<=200 ~ log(1.27+0.994+0.015) - log(1.292+1.035),
    x>200 ~ log(1.421+0.994+0.015) - log(1.443+1.035)
  ))
}

simulate_v_p_log_d = function(q){
  return(case_when(
    q<=20 ~ log(( 7.25+ 0.289*q + 1.25)/q ) - log( (7.10+ 0.318*q + 1.2)/q),
    q>20 & q<=60 ~ log( (7.25 + 3.55 + 0.289*20 + 0.481*(q-20))/q) - log( (7.10 + 3.55 + 0.318*20 + 0.505*(q-20))/q),
    q>60 & q<=110 ~ log((7.25 + 9.25 + 0.289*20 + 0.481*40 + 0.834 * (q-60))/q) - log( (7.10 + 9.25 + 0.318*20 + 0.505 *40 + 0.856 * (q-60))/q),
    q>110 & q<=200 ~ log( (7.25 + 29.75 + 0.289*20 + 0.481 * 40 + 0.834 * 50 + 1.270*(q-110))/q) - log( (7.10 + 29.75 + 3.18*2 + 5.05 * 4 + 8.56 * 5 + 1.292*(q-110))/q),
    q>200 ~ log((7.25 + 29.75 + 2.89*2 + 4.81 * 4 + 8.34 * 5 + 12.70*9 + 1.421 * (q-200))/q) - log((7.10 + 29.75 + 3.18*2 + 5.05 * 4 + 8.56 * 5 + 12.92*9 + 1.443 * (q-200))/q)
  ))
}

charge_2018_after$mppi = case_when(
  charge_2018_after$x_m<=20 ~ log(0.289+0.485+0.015) - log(0.318 + 0.53),
  charge_2018_after$x_m>20 & charge_2018_after$x_m<=60 ~ log(0.481+0.994+0.015) - log(0.505+1.035),
  charge_2018_after$x_m>60 & charge_2018_after$x_m<=110 ~ log(0.834+0.994+0.015) - log(0.856+1.035),
  charge_2018_after$x_m>110 & charge_2018_after$x_m<=200 ~ log(1.27+0.994+0.015) - log(1.292+1.035),
  charge_2018_after$x_m>200 ~ log(1.421+0.994+0.015) - log(1.443+1.035)
)
  
  
charge_2018_after$appi = case_when(
  charge_2018_after$x_m<=20 ~ log(( 7.25+ 0.289*charge_2018_after$x_m + 1.25)/charge_2018_after$x_m ) - 
    log( (7.10+ 0.318*charge_2018_after$x_m + 1.2)/charge_2018_after$x_m),
  charge_2018_after$x_m>20 & charge_2018_after$x_m<=60 ~ log( (7.25 + 3.55 + 0.289*20 + 0.481*(charge_2018_after$x_m-20))/charge_2018_after$x_m) - 
    log( (7.10 + 3.55 + 0.318*20 + 0.505*(charge_2018_after$x_m-20))/charge_2018_after$x_m),
  charge_2018_after$x_m>60 & charge_2018_after$x_m<=110 ~ log((7.25 + 9.25 + 0.289*20 + 0.481*40 + 0.834 * (charge_2018_after$x_m-60))/charge_2018_after$x_m) - 
    log( (7.10 + 9.25 + 0.318*20 + 0.505 *40 + 0.856 * (charge_2018_after$x_m-60))/charge_2018_after$x_m),
  charge_2018_after$x_m>110 & charge_2018_after$x_m<=200 ~ log( (7.25 + 29.75 + 0.289*20 + 0.481 * 40 + 0.834 * 50 + 1.270*(charge_2018_after$x_m-110))/charge_2018_after$x_m) - 
    log( (7.10 + 29.75 + 3.18*2 + 5.05 * 4 + 8.56 * 5 + 1.292*(charge_2018_after$x_m-110))/charge_2018_after$x_m),
  charge_2018_after$x_m>200 ~ log((7.25 + 29.75 + 2.89*2 + 4.81 * 4 + 8.34 * 5 + 12.70*9 + 1.421 * (charge_2018_after$x_m-200))/charge_2018_after$x_m) - 
    log((7.10 + 29.75 + 3.18*2 + 5.05 * 4 + 8.56 * 5 + 12.92*9 + 1.443 * (charge_2018_after$x_m-200))/charge_2018_after$x_m)
)

#charge_2018_after = merge (charge_2018_after, percipitation_revenue, by.x = c("bill_ym"), by.y = c("bill_ym"), all.x = T)

charge_2018_after = merge (charge_2018_after, prem_id_end_date, 
                           by.x = c("prem_id","bill_ym"), by.y = c("prem_id","bill_ym"), all.x = T)

charge_2018_after$month = charge_2018_after$month.x
charge_2018_after$month.x = NULL
charge_2018_after$month.y = NULL

write.csv(charge_2018_after, file = "charge_2018_after.csv", row.names = F)

summary(ivreg(did_quantity ~ did_mp + bill_ym + bseg_end_dt | 
                mppi + bill_ym + bseg_end_dt, data = charge_2018_after))

#Call:
 # ivreg(formula = did_quantity ~ did_mp + bill_ym + bseg_end_dt | 
  #        mppi + bill_ym + bseg_end_dt, data = charge_2018_after)

#Residuals:
 # Min       1Q   Median       3Q      Max 
#-6.18735 -0.21009  0.01475  0.23913  4.94181 

#Coefficients:
 # Estimate Std. Error t value Pr(>|t|)    
#(Intercept) -7.055e+00  7.184e-01  -9.820   <2e-16 ***
 # did_mp       1.539e+00  3.729e-03 412.709   <2e-16 ***
  #bill_ym      3.471e-05  3.559e-06   9.753   <2e-16 ***
  #bseg_end_dt  2.670e-05  2.455e-05   1.088    0.277    
#---
 # Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#Residual standard error: 0.4451 on 4371046 degrees of freedom
#Multiple R-Squared: 0.6759,	Adjusted R-squared: 0.6759 
#Wald test: 5.697e+04 on 3 and 4371046 DF,  p-value: < 2.2e-16 

summary(ivreg(did_quantity ~ did_ap2 + bill_ym + bseg_end_dt | 
                appi + bill_ym +bseg_end_dt, data = charge_2018_after))

#Call:
 # ivreg(formula = did_quantity ~ did_ap2 + bill_ym + bseg_end_dt | 
  #        appi + bill_ym + bseg_end_dt, data = charge_2018_after)

#Residuals:
 # Min       1Q   Median       3Q      Max 
#-8.66677 -0.17502  0.04001  0.24261  7.41670 

#Coefficients:
 # Estimate Std. Error t value Pr(>|t|)    
#(Intercept) -4.481e+00  1.132e+00  -3.960 7.48e-05 ***
 # did_ap2      1.446e+00  1.075e-02 134.461  < 2e-16 ***
  #bill_ym      2.190e-05  5.605e-06   3.908 9.32e-05 ***
  #bseg_end_dt  1.212e-04  3.877e-05   3.127  0.00177 ** 
  #---
  #Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#Residual standard error: 0.7011 on 4371046 degrees of freedom
#Multiple R-Squared: 0.1962,	Adjusted R-squared: 0.1962 
#Wald test:  6104 on 3 and 4371046 DF,  p-value: < 2.2e-16

summary(ivreg(did_quantity ~ did_mp + did_ap + bill_ym + bseg_end_dt | 
                mppi + appi + bill_ym + bseg_end_dt, data = charge_2018_after))

#Call:
 # ivreg(formula = did_quantity ~ did_mp + did_ap + bill_ym + bseg_end_dt | 
  #        mppi + appi + bill_ym + bseg_end_dt, data = charge_2018_after)

#Residuals:
 # Min        1Q    Median        3Q       Max 
#-37.84693  -0.19539   0.02654   0.22053   6.51275 

#Coefficients:
 # Estimate Std. Error  t value Pr(>|t|)    
#(Intercept)  1.213e+01  8.408e-01   14.425   <2e-16 ***
 # did_mp       1.164e+00  5.290e-03  220.110   <2e-16 ***
  #did_ap      -9.772e-01  6.888e-03 -141.875   <2e-16 ***
  #bill_ym     -6.051e-05  4.165e-06  -14.529   <2e-16 ***
  #bseg_end_dt  2.521e-04  2.840e-05    8.874   <2e-16 ***
  #---
  #Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#Residual standard error: 0.514 on 4371045 degrees of freedom
#Multiple R-Squared: 0.568,	Adjusted R-squared: 0.568 
#Wald test: 4.033e+04 on 4 and 4371045 DF,  p-value: < 2.2e-16 

summary(ivreg(did_quantity ~ did_mp + did_ap2 + bill_ym + bseg_end_dt | 
                mppi + appi + bill_ym + bseg_end_dt, data = charge_2018_after))

#Call:
 # ivreg(formula = did_quantity ~ did_mp + did_ap2 + bill_ym + bseg_end_dt | 
  #        mppi + appi + bill_ym + bseg_end_dt, data = charge_2018_after)

#Residuals:
 # Min       1Q   Median       3Q      Max 
#-5.36606 -0.23111  0.01131  0.23614  5.15251 

#Coefficients:
 # Estimate Std. Error  t value Pr(>|t|)    
#(Intercept) -5.518e+00  6.510e-01   -8.476   <2e-16 ***
 # did_mp       2.215e+00  4.758e-03  465.493   <2e-16 ***
  #did_ap2     -1.583e+00  8.755e-03 -180.779   <2e-16 ***
  #bill_ym      2.697e-05  3.225e-06    8.364   <2e-16 ***
  #bseg_end_dt  2.908e-04  2.231e-05   13.037   <2e-16 ***
  #---
  #Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

#Residual standard error: 0.4034 on 4371045 degrees of freedom
#Multiple R-Squared: 0.7339,	Adjusted R-squared: 0.7339 
#Wald test: 6.548e+04 on 4 and 4371045 DF,  p-value: < 2.2e-16

charge_2018_after = charge_2018_after[which(is.na(charge_2018_after$did_mp)==F),]
charge_2018_after = charge_2018_after[which(is.na(charge_2018_after$did_ap)==F),]
charge_2018_after = charge_2018_after[which(is.na(charge_2018_after$did_quantity)==F),]
charge_2018_after = charge_2018_after[which(is.na(charge_2018_after$did_extra_usage)==F),]
charge_2018_after = charge_2018_after[which(is.na(charge_2018_after$did_essential_usage)==F),]

tsls1_didmp = lm(did_mp~
                   bill_ym + bseg_end_dt+
                   mppi, data = charge_2018_after)
charge_2018_after$did_mp_iv<-fitted.values(tsls1_didmp)

tsls1_didap = lm(did_ap~
                   bill_ym + bseg_end_dt+
                   appi, data = charge_2018_after)
charge_2018_after$did_ap_iv<-fitted.values(tsls1_didap)

tsls1_didap2 = lm(did_ap2~
                    bill_ym + bseg_end_dt+
                    appi, data = charge_2018_after)
charge_2018_after$did_ap2_iv<-fitted.values(tsls1_didap2)

