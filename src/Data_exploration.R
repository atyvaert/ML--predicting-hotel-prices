##############################################################
##############################################################
# 1. Data Import
##############################################################
##############################################################
rm(list = ls())

setwd(dir = "./GitHub/ml22-team10")



train <- read.csv('./data/bronze_data/train.csv', fileEncoding = 'latin1')
test_X <- read.csv('./data/bronze_data/test.csv')


# packages and libraries needed
library(stringr)

# Look at the data
summary(train)
summary(test_X)
train$average_daily_rate

# separate dependent and independent variables
train_X <- subset(train, select = -c(average_daily_rate))
train_y <- train$average_daily_rate

# inspect
str(train_X)
str(train_y)
# We have a problem with train_y, as it is a character type and it is accompanied
# with some random nr each time that does not belong to this variable
# Therefore, we keep the first part of the string character as a numeric variable
train_y <- as.numeric(word(train_y, 1))

# Again inspect the structure of train_y
str(train_y) 



##############################################################
##############################################################
# 2. Data exploration
##############################################################
##############################################################

# 1 Arrival dates
train_X$arrival_date
# IMPORTANT: WEEKDAY NEEDS TO BE ADDED
# parse date with posixct: 

# 2 assigned_room_type
barplot(table(train_X$assigned_room_type))
table(train_X$assigned_room_type)
# 2 categorieën met bijna geen enkele observatie
# Wat doen we hiermee?
# Op zich kunnen we erin laten want geen misssing values
# nominal feature: dummy encoding

# 3 booking_agent
barplot(table(train_X$booking_agent))
table(train_X$booking_agent)
# lots of different values or NULL values (11 361): is niet gelijk aan Na values
# Maak variabele aan die aangeeft als booking agent wel (=1) of niet (= 0) aanwezig was:
# booking_agent_present
# => binary feature: integer encoding (indicator variable)


# 4 booking_company
barplot(table(train_X$booking_company))
table(train_X$booking_company)
# lots of different values or NULL values (78 303): is niet gelijk aan Na values
# Maak variabele aan die aangeeft als booking company wel (=1) of niet (= 0) aanwezig was:
# booking_company_present
# => binary feature: integer encoding (indicator variable)


# 5 booking_distribution_channel
barplot(table(train_X$booking_distribution_channel))
table(train_X$booking_distribution_channel)
# only 2 undefined dus zou ditlaten, op zich geen Na?
# other option: impute with modus and treat undefined as Na
# nominal feature: dummy encoding

# 6 canceled
barplot(table(train_X$canceled))
table(train_X$canceled)
# geen problemen met variabelen, enkel nog dummies ofzo van maken
# # nominal feature: dummy encoding OF binary feature: integer encoding (denk dit)

# 7 car_parking_spaces
hist(train_X$car_parking_spaces)
quantile(train_X$car_parking_spaces, na.rm = T, probs = seq(0, 1, 0.01))
summary(train_X$car_parking_spaces)
# van 0 tot 3 (informatie voor encoding)
# 93% die nul hebben als value, 6% heeft 1 en 1% die 2 of 3 hebben
# zal zich waarschijnlijk oplossen bij behandelen van outliers
# dus binary: integer encoding

# 8 country
barplot(table(train_X$country))
table(train_X$country)
unique(train_X$country)
# heel wat verschillende landen (164)
# nominal feature: dummy encoding maar beperkt tot x aantal belangrijkste
# zou persoonlijk voor 11 gaan obv die barplot
cut_off_point <- as.numeric(table(train_X$country))
cut_off_point_perc <- cut_off_point/(sum(cut_off_point))
sort(cut_off_point_perc, decreasing  = T)
# 13 dummies hierop gebaseerd (tot het ten minste 0.01% van de data voorsteld)

#  9 customer type 
train_X$customer_type
barplot(table(train_X$customer_type))
# categorical variable

# 10 days in waiting list 
train_X$days_in_waiting_list
hist(train_X$days_in_waiting_list)
# numerical variable

# 11 deposit 
train_X$deposit
barplot(table(train_X$deposit))
# categorical variable

# 12 hotel type 
train_X$hotel_type
barplot(table(train_X$hotel_type))
# categorical variable

# 13 is repeated guest 
train_X$is_repeated_guest
barplot(table(train_X$is_repeated_guest))
# binary variable

# 14 last_status
train_X$last_status
barplot(table(train_X$last_status))
# categorical variable

#15 last_status_date
train_X$last_status_date
sum(is.na(train_X$last_status_date))
# what to do with 2615 missing values (where are we going to use this column)
# Na: look at average time since last_status_update for the others
# use variable for hint but not in prediction model I think

#16 lead_time
train_X$lead_time
boxplot(train_X$lead_time)
hist(train_X$lead_time, breaks = 1000)
#0 when cancelled

#17 market_segment
table(train_X$market_segment) # no problems

#18 meal_booked
table(train_X$meal_booked)
#No NA

#19 nr_adults
table(train_X$nr_adults)
#values 26, 27 -> large groups (cancelled)
# Vraag: wat hiermee doen?

#20 nr_babies
table(train_X$nr_babies)
#string n/ = 0
# 9 als valid of invalid outlier?

#21 nr_booking_changes  
barplot(table(train_X$nr_booking_changes))
table(train_X$nr_booking_changes)
# frequency decreases drastically, and after 5 it becomes really small. Lots of Na which = 0!

#22 nr_children        
barplot(table(train_X$nr_children))
table(train_X$nr_children)
# Vast majority has 0 zero children, 1 and 2 also prevalent (>2000), 3 (44 times), and 1 observation of 10 children => mistake?

#23 nr_nights  
barplot(table(train_X$nr_nights))
table(train_X$nr_nights)
#(0 to 10 is well present, >10 less observations), lot of different amount of days with 1 observations...

#24 nr_previous_bookings  
barplot(table(train_X$nr_previous_bookings))
table(train_X$nr_previous_bookings)
#lots of hifg values but extreme low frequencies (1)

#25 previous_bookings_not_canceled  
barplot(table(train_X$previous_bookings_not_canceled))
table(train_X$previous_bookings_not_canceled)

#26 previous_cancellations  
barplot(table(train_X$previous_cancellations))
table(train_X$previous_cancellations)
# 0 - 8 captures most of observations, but goes through 72...

#27 reserved_room_type  
barplot(table(train_X$reserved_room_type))
table(train_X$reserved_room_type)
#L and P have few observations
# I and K were NEVER reserved, but assigned. A lot of A reservations are not assigned to A 

#28 special_requests  
barplot(table(train_X$special_requests))
table(train_X$special_requests)
#decreases exponentially






# Na overview
# 83045 rows
sum(is.na((train_X$nr_booking_changes))) #72192, a lot!
sum(is.na((train_X$nr_children))) #16894
sum(is.na((train_X$nr_nights))) #0
sum(is.na((train_X$nr_previous_bookings))) #18394
sum(is.na((train_X$previous_bookings_not_canceled))) #5027
sum(is.na((train_X$previous_cancellations))) #4992
sum(is.na((train_X$reserved_room_type))) #0
sum(is.na((train_X$special_requests))) #0







