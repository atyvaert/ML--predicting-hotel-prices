##############################################################
##############################################################
# Data Import
##############################################################
##############################################################
setwd(dir = '/Users/Artur/Desktop/uni jaar 6 sem 1/machine learning/groepswerk/R/data')
train <- read.csv('./train.csv')
test_X <- read.csv('./test.csv')

# Look at the data
summary(train)
summary(test_X)
train$average_daily_rate

# separate dependent and independent variables
train_X <- subset(train, select = -c(average_daily_rate))
train_y <- train$average_daily_rate

# look at the structure of train datasets
str(train_X)
str(train_y) # moet nog omgezet worden naar numeric vector



##############################################################
##############################################################
# Data exploration
##############################################################
##############################################################

# 1) Arrival dates
# have not looked at dates yet on dodona
# get month and day of the week out of it

# 2) assigned_room_type
barplot(table(train_X$assigned_room_type))
table(train_X$assigned_room_type)
# 2 categorie??n met bijna geen enkele observatie
# Wat doen we hiermee?

# 3) booking agent
barplot(table(train_X$booking_agent))
table(train_X$booking_agent)
# lots of different values, should be made numeric?

# 4) booking company
barplot(table(train_X$booking_company))
table(train_X$booking_company)
# Bijna allemaal NULL values

# 5) booking_distribution_channel
barplot(table(train_X$booking_distribution_channel))
table(train_X$booking_distribution_channel)
# only 2 undefined dus zou met modus imputen of toch laten?

# 6) canceled
barplot(table(train_X$canceled))
table(train_X$canceled)
# geen problemen met variabelen, enkel nog dummies ofzo van maken

# 7) car_parking_spaces
hist(train_X$car_parking_spaces)
boxplot(train_X$car_parking_spaces)
# enkel 0 of 1 (informatie voor encoding)

#8 country
table(train_X$country)

#15 last_status_date
sum(is.na(train$last_status_date))
# what to do with 2615 missing values (where are we going to use this column)

#16 lead_time
boxplot(train_X$lead_time)
hist(train_X$lead_time, breaks = 1000)
#0 when cancelled

#17 market_segment
table(train_X$market_segment)

#18 meal_booked
table(train_X$meal_booked)
#No NA

#19 nr_adults
table(train_X$nr_adults)
#values 26, 27 -> large groups (cancelled)

#20 nr_babies
table(train_X$nr_babies)
#string n/ = 0

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







