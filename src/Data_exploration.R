##############################################################
##############################################################
# 0. Introduction of File
##############################################################
##############################################################
# According to the CRISP-DM framework,there needs to be a good connection between business understanding and data understanding.
# After having carefully read the assignment, the authors have developed a good business understanding and look to improve their data understanding.
# Therefore, the first step after the business understanding is Data Exploration.


##############################################################
##############################################################
# 1. Data Import
##############################################################
##############################################################

# The authors follow the Bronze/Silver/Gold data structure.
# In this step, the raw (Bronze) data is used.

rm(list = ls())

# The needed packages and libraries are loaded
library(stringr)
library(miscTools)
library(readr)

#The Bronze data is read
train <- read_csv('./data/bronze_data/train.csv')
test_X <- read_csv('./data/bronze_data/test.csv')

# First look at the data
summary(train)
summary(test_X)
train$average_daily_rate

# Separate dependent and independent variables
train_X <- subset(train, select = -c(average_daily_rate))
train_y <- train$average_daily_rate

# Inspect structure
str(train_X)
str(train_y)

# There is a problem in train_y as it is a character type and it is accompanied
# with some random number each time that does not belong to this variable.
# Therefor,only the first part of the string character is kept as a numeric variable
train_y <- as.numeric(word(train_y, 1))

# Again inspect the structure of train_y
str(train_y) 



##############################################################
##############################################################
# 2. Data exploration
##############################################################
##############################################################

# The authors now start with the actual exploration by creating boxplots, barplots, and histograms.
# Afterwards, a first missing values check was provided (which will be needed in "data_cleaning.R").


# 1 Arrival dates
train_X$arrival_date
# These dates will need to be parsed with posixct in the feature engineering section

# 2 assigned_room_type
barplot(table(train_X$assigned_room_type))
table(train_X$assigned_room_type)
# It can be noted that there are 2 categories with almost no observations

# 3 booking_agent
barplot(table(train_X$booking_agent))
table(train_X$booking_agent)
# There are lots of different values or NULL values (11 263): This is not NA, but informative missing!
# Create variable that show if booking agent is available (=1) or not (= 0):
# booking_agent_present
# => binary feature: integer encoding (indicator variable)

# 4 booking_company
barplot(table(train_X$booking_company))
table(train_X$booking_company)
# lots of different values or NULL values (78 331): This is not NA, but informative missing!
# Create variable that show if booking company is available (=1) or not (= 0):
# booking_company_present
# => binary feature: integer encoding (indicator variable)


# 5 booking_distribution_channel
barplot(table(train_X$booking_distribution_channel))
table(train_X$booking_distribution_channel)
# Only 2 undefined values which will be kept since it is not really Na!
# other option: impute with modus and treat undefined as Na
# nominal feature: dummy encoding

# 6 canceled
barplot(table(train_X$canceled))
table(train_X$canceled)
# There are no issues with this variable
# Nominal feature: dummy encoding

# 7 car_parking_spaces
hist(train_X$car_parking_spaces)
table(train_X$car_parking_spaces)
quantile(train_X$car_parking_spaces, na.rm = T, probs = seq(0, 1, 0.01))
summary(train_X$car_parking_spaces)
# Values (0, 1, 2, 3, 8) are observed
# From 0 to 3 (information needed for encoding)
# 93% have value 0, 6% has value 1 and 1% has 2 or 3 
# This will handled when treating outliers
# Integer encoding

# 8 country
barplot(table(train_X$country))
table(train_X$country)
unique(train_X$country)
# Lots of different countries (164)
# Nominal feature: dummy encoding but limited to x most important ones
# We use the expert opinion of the TA and set X = 15 (14 most visited + 1 variable for all other)
cut_off_point <- as.numeric(table(train_X$country))
cut_off_point_perc <- cut_off_point/(sum(cut_off_point))
sort(cut_off_point_perc, decreasing  = T)

#  9 customer type 
train_X$customer_type
barplot(table(train_X$customer_type))
table(train_X$customer_type)
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
# Na: look at average time since last_status_update for the others
# Will be treated in feature engineering

#16 lead_time
train_X$lead_time
boxplot(train_X$lead_time)
hist(train_X$lead_time, breaks = 1000)
# 0 when canceled
# Q3+1.5IQR = 297

#17 market_segment
table(train_X$market_segment) 
# No issues here 

#18 meal_booked
table(train_X$meal_booked)
#No issues here

#19 nr_adults
table(train_X$nr_adults)
boxplot(train_X$nr_adults)
# values 26, 27 -> large groups (canceled)

#20 nr_babies
table(train_X$nr_babies)
# String n/a = 0

#21 nr_booking_changes  
barplot(table(train_X$nr_booking_changes))
table(train_X$nr_booking_changes)
# Frequency decreases drastically, and after 5 it becomes really small. Lots of Na which = 0!

#22 nr_children  
(train_X$nr_children)
table(train_X$nr_children)
# Vast majority has 0 zero children, 1 and 2 also prevalent, 3 (48 times), and 1 observation of 10 children => possible outlier, or group

#23 nr_nights 
barplot(table(train_X$nr_nights))
table(train_X$nr_nights)
# (0 to 10 is well present, > 10 has less observations), lot of different days with 1 observations...

#24 nr_previous_bookings  
barplot(table(train_X$nr_previous_bookings))
table(train_X$nr_previous_bookings)
train_X$nr_previous_bookings
# Lots of high values with extreme low frequencies (1)

#25 previous_bookings_not_canceled  
barplot(table(train_X$previous_bookings_not_canceled))
table(train_X$previous_bookings_not_canceled)
# Lots of high values with extreme low frequencies (1)

#26 previous_cancellations  
barplot(table(train_X$previous_cancellations))
table(train_X$previous_cancellations)

#27 reserved_room_type  
barplot(table(train_X$reserved_room_type))
table(train_X$reserved_room_type)
# I and K were NEVER reserved, but assigned. A lot of A reservations are not assigned to A 

#28 special_requests  
barplot(table(train_X$special_requests))
table(train_X$special_requests)
# Decreases exponentially



# Now a very basic Na overview is given. This is not exhaustive and will be elaborated on in the Data Cleaning section


# Na overview

nrow(train_X) # 83031 rows
sum(is.na((train_X$nr_booking_changes))) #72271, a lot!
sum(is.na((train_X$nr_children))) #16889
sum(is.na((train_X$nr_nights))) #0
sum(is.na((train_X$nr_previous_bookings))) #18393
sum(is.na((train_X$previous_bookings_not_canceled))) #5028
sum(is.na((train_X$previous_cancellations))) #4989
sum(is.na((train_X$reserved_room_type))) #0
sum(is.na((train_X$special_requests))) #0


##############################################################
##############################################################
# 3. End Note
##############################################################
##############################################################

# This is the end of the Data Exploration section of the assignment.
# After this, the authors have gathered a strong data understanding.
# Combining this data understanding with the business understanding, the authors now begin the process of Data Cleaning
# as proposed by the CRISP-DM framework.


