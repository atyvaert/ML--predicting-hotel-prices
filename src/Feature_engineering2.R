##############################################################
##############################################################
# 0. Introduction of File
##############################################################
##############################################################
# The authors now perform Feature Engineering.
# The website TowardsDataScience describes feature engineering as 
# a machine learning technique that leverages data to create new variables that aren’t in the training set.
# Doing so, the authors aim to improve predictive capabilities.

# Feature engineering is the last part of the Data Preparation phase in the CRISP-DM framework.
# After this step, the actual Modeling part will begin.

##############################################################
##############################################################



##############################################################
##############################################################
# 1. Data Import
##############################################################
##############################################################

# The authors follow the Bronze/Silver/Gold data structure.
# In this step, the Silver data is used.
# The final data will be written away to 


# It might be needed to set R language to English for weekdays
# Sys.setlocale("LC_ALL","English")

# The needed packages and libraries are loaded
library(lubridate)
library(readr)
library(dummy)
library(ppcor)
if(!require('ranger')) install.packages('ranger')
if(!require('ppcor')) install.packages('ppcor')
if(!require('corpcor')) install.packages('corpcor')
if(!require('mctest')) install.packages('mctest')
library(corpcor)
library(mctest)
if(!require('caret')) install.packages('caret')
if(!require('rlang')) install.packages('rlang')
library(caret)
library(dplyr)
library(chron)


#The Silver data is read
rm(list = ls())
train <- read.csv('./data/silver_data/train.csv')
val <- read.csv('./data/silver_data/val.csv')
test_X <- read.csv('./data/silver_data/test.csv')

# Separation of dependent and independent variables for training and validation set
train_X <- subset(train, select = -c(average_daily_rate))
train_y <- train$average_daily_rate

val_X <- subset(val, select = -c(average_daily_rate))
val_y <- val$average_daily_rate

# inspect
# str(train_X)
# str(val_X)
# str(test_X)






##############################################################
##############################################################
# 2. Categorical data
##############################################################
##############################################################
train_X_encode <- train_X
val_X_encode <- val_X
test_X_encode <- test_X

##############################################################
# 2.1 Ordinal data: integer encoding
##############################################################
# No ordinal data

##############################################################
# 2.2 Nominal data: one-hot encoding
##############################################################
# We use the dummmy package to treat nominal data as this is a more flexible approach
# and we have a lot of variables with a lot of levels

##############################################################
# 2.3 Special cases
##############################################################

# Before the one-hot encoding, we create a new feature. 
# Explanation can be found in  section 3.5 on correlation
# This column indicates if assigned and reserved room type are the same
# We also delete the column assigned room type due to correlation issues

train_X_encode$room_type_conflict <- ifelse(train_X_encode[, 'assigned_room_type']== train_X_encode[, 'reserved_room_type'], 0, 1)
val_X_encode$room_type_conflict <- ifelse(val_X_encode[, 'assigned_room_type']== val_X_encode[, 'reserved_room_type'], 0, 1)
test_X_encode$room_type_conflict <- ifelse(test_X_encode[, 'assigned_room_type']== test_X_encode[, 'reserved_room_type'], 0, 1)

# drop assigned room type
train_X_encode <- subset(train_X_encode, select = -c(assigned_room_type))
val_X_encode <- subset(val_X_encode, select = -c(assigned_room_type))
test_X_encode <- subset(test_X_encode, select = -c(assigned_room_type))

# First we make a categorical feature arrival_date_weekday
train_X_encode$arrival_date_weekday <- wday(train_X_encode$posix_arrival, label = T)
val_X_encode$arrival_date_weekday <- wday(val_X_encode$posix_arrival, label = T)
test_X_encode$arrival_date_weekday <- wday(test_X_encode$posix_arrival, label = T)

# year_arrival was made categorical
train_X_encode$year_arrival <- as.factor(train_X_encode$year_arrival)
val_X_encode$year_arrival <- as.factor(val_X_encode$year_arrival)
test_X_encode$year_arrival <- as.factor(test_X_encode$year_arrival)


# A feature that indicates what part of the month the customer arrives was also made.
# Months were split in weeks (1-7, 8-14, 15-21, 22-28, 29-...)
train_X_encode$week_of_month <- as.factor(floor(train_X_encode$day_of_month_arrival/7)+1)
val_X_encode$week_of_month <- as.factor(floor(val_X_encode$day_of_month_arrival/7)+1)
test_X_encode$week_of_month <- as.factor(floor(test_X_encode$day_of_month_arrival/7)+1)

##############################################################
# 2.4 Other features
##############################################################
# get categories and dummies
# we only select the top 10 levels with highest frequency so the model does not explode
# For all cases, this includes most of the data
cats <- categories(train_X_encode[, c('booking_distribution_channel',
                                      'canceled', 'customer_type', 'deposit',
                                      'hotel_type', 'is_repeated_guest', 'last_status',
                                      'market_segment', 'meal_booked', 'reserved_room_type',
                                      'arrival_date_weekday', 'year_arrival', 'week_of_month')], p = 10)


# For some variables with high cardinality, we use the specified amount of category levels
# We either use our own knowledge or the expert opinion of the TA
# e.g.: month_arrival separate because we want all 12 categories here

# We make this variable factorial
train_X_encode$day_of_month_arrival <- as.factor(train_X_encode$day_of_month_arrival)


cats <- append(cats, categories(train_X_encode['month_arrival']))
cats <- append(cats, categories(train_X_encode['day_of_month_arrival']))
cats <- append(cats, categories(train_X_encode['country'], p = 15))
cats <- append(cats, categories(train_X_encode['booking_agent'], p = 8)) #7 large agents (>1000) + null
cats <- append(cats, categories(train_X_encode['booking_company'], p = 2))


# Apply on train set (exclude reference categories)
dummies_train <- dummy(train_X_encode[, c('booking_distribution_channel', 
                                          'canceled', 'country', 'customer_type', 'deposit',
                                          'hotel_type', 'is_repeated_guest', 'last_status',
                                          'market_segment', 'meal_booked', 'reserved_room_type',
                                          'month_arrival', 'arrival_date_weekday', 'year_arrival',
                                          'booking_agent', 'booking_company', 'week_of_month', 'day_of_month_arrival')], object = cats)

# Exclude the reference category: take the first one of the variable added
names(dummies_train)
dummies_train <- subset(dummies_train, 
                        select = -c(booking_distribution_channel_Direct,
                                    country_China, canceled_no.cancellation, customer_type_Transient,
                                    deposit_nodeposit, hotel_type_City.Hotel, is_repeated_guest_no,
                                    last_status_Check.Out, market_segment_Online.travel.agent, year_arrival_2015,
                                    meal_booked_meal.package.NOT.booked, reserved_room_type_A, month_arrival_January,
                                    arrival_date_weekday_Mon, booking_company_40, booking_agent_240,
                                    last_status_Canceled, week_of_month_5, day_of_month_arrival_1))

# Apply on val set (exclude reference categories)
# Excluded no.canceled so it becomes one when it was canceled
dummies_val <- dummy(val_X_encode[, c('booking_distribution_channel', 
                                      'canceled', 'country', 'customer_type', 'deposit',
                                      'hotel_type', 'is_repeated_guest', 'last_status',
                                      'market_segment', 'meal_booked', 'reserved_room_type',
                                      'month_arrival', 'arrival_date_weekday', 'year_arrival',
                                      'booking_agent', 'booking_company', 'week_of_month', 'day_of_month_arrival')], object = cats)

dummies_val <- subset(dummies_val, select = -c(booking_distribution_channel_Direct,
                                               country_China, canceled_no.cancellation, customer_type_Transient,
                                               deposit_nodeposit, hotel_type_City.Hotel, is_repeated_guest_no,
                                               last_status_Check.Out, market_segment_Online.travel.agent, year_arrival_2015,
                                               meal_booked_meal.package.NOT.booked, reserved_room_type_A, month_arrival_January,
                                               arrival_date_weekday_Mon, booking_company_40, booking_agent_240,
                                               last_status_Canceled, week_of_month_5, day_of_month_arrival_1))

# Apply on test set (exclude reference categories)
# Excluded no.canceled so it becomes one when it was canceled
dummies_test <- dummy(test_X_encode[, c('booking_distribution_channel', 
                                        'canceled', 'country', 'customer_type', 'deposit',
                                        'hotel_type', 'is_repeated_guest', 'last_status',
                                        'market_segment', 'meal_booked', 'reserved_room_type',
                                        'month_arrival', 'arrival_date_weekday', 'year_arrival',
                                        'booking_agent', 'booking_company', 'week_of_month', 'day_of_month_arrival')], object = cats)

dummies_test <- subset(dummies_test, select = -c(booking_distribution_channel_Direct,
                                                 country_China, canceled_no.cancellation, customer_type_Transient,
                                                 deposit_nodeposit, hotel_type_City.Hotel, is_repeated_guest_no,
                                                 last_status_Check.Out, market_segment_Online.travel.agent, year_arrival_2015,
                                                 meal_booked_meal.package.NOT.booked, reserved_room_type_A, month_arrival_January,
                                                 arrival_date_weekday_Mon, booking_company_40, booking_agent_240,
                                                 last_status_Canceled, week_of_month_5, day_of_month_arrival_1))

# Remove the original predictors and merge them with the other predictors
# Train set
train_X_encode <- subset(train_X_encode, select = -c(booking_distribution_channel,
                                                     canceled, country, customer_type, deposit,
                                                     hotel_type, is_repeated_guest, last_status,
                                                     market_segment, meal_booked, reserved_room_type,
                                                     month_arrival, arrival_date_weekday, booking_agent,
                                                     booking_company, week_of_month, day_of_month_arrival))
train_X_encode <- cbind(train_X_encode, dummies_train)

# Val set
val_X_encode <- subset(val_X_encode, select = -c(booking_distribution_channel,
                                                 canceled, country, customer_type, deposit,
                                                 hotel_type, is_repeated_guest, last_status,
                                                 market_segment, meal_booked, reserved_room_type,
                                                 month_arrival, arrival_date_weekday, booking_agent,
                                                 booking_company, week_of_month, day_of_month_arrival))
val_X_encode <- cbind(val_X_encode, dummies_val)

# Test set
test_X_encode <- subset(test_X_encode, select = -c(booking_distribution_channel,
                                                   canceled, country, customer_type, deposit,
                                                   hotel_type, is_repeated_guest, last_status,
                                                   market_segment, meal_booked, reserved_room_type,
                                                   month_arrival, arrival_date_weekday, booking_agent,
                                                   booking_company, week_of_month, day_of_month_arrival))
test_X_encode <- cbind(test_X_encode, dummies_test)


##############################################################
##############################################################
# 3. Numerical data
##############################################################
##############################################################

##############################################################
# 3.1 time_between_last_status_arrival
##############################################################
# Creation of feature time_between_last_status_arrival
# time_between_arrival_checkout gives a positive difference when last status date is after
# arrival (often when customer checks out)
# time_between_arrival_cancel gives a negative difference when last status date is before
# arrival (often when customer cancels stay)

# train
time_between_last_status_arrival <- as.numeric(round(difftime(train_X_encode$posix_last_status, train_X_encode$posix_arrival)/(60*60*24)))
time_between_arrival_checkout <- time_between_last_status_arrival
time_between_arrival_checkout[time_between_arrival_checkout<0] <- 0
time_between_arrival_cancel <- -time_between_last_status_arrival
time_between_arrival_cancel[time_between_arrival_cancel<0] <- 0

train_X_encode <- cbind(train_X_encode, time_between_arrival_checkout, time_between_arrival_cancel)

# val
time_between_last_status_arrival <- as.numeric(round(difftime(val_X_encode$posix_last_status, val_X_encode$posix_arrival)/(60*60*24)))
time_between_arrival_checkout <- time_between_last_status_arrival
time_between_arrival_checkout[time_between_arrival_checkout<0] <- 0
time_between_arrival_cancel <- -time_between_last_status_arrival
time_between_arrival_cancel[time_between_arrival_cancel<0] <- 0

val_X_encode <- cbind(val_X_encode, time_between_arrival_checkout, time_between_arrival_cancel)

# test
time_between_last_status_arrival <- as.numeric(round(difftime(test_X_encode$posix_last_status, test_X_encode$posix_arrival)/(60*60*24)))
time_between_arrival_checkout <- time_between_last_status_arrival
time_between_arrival_checkout[time_between_arrival_checkout<0] <- 0
time_between_arrival_cancel <- -time_between_last_status_arrival
time_between_arrival_cancel[time_between_arrival_cancel<0] <- 0

test_X_encode <- cbind(test_X_encode, time_between_arrival_checkout, time_between_arrival_cancel)


##############################################################
# 3.2 Indicator nr_weekdays and nr_weekenddays
##############################################################
# The feature nr_nights is split up in nr_weekdays and nr_weekenddays because prices might differ for the weekend.
# One could assume that a beach resort receives more visitors in a weekend day than on an average tuesday ...


features_week_weekend_days <- function(df) {
  # Initiate columns with zero (for the case that nr_nights is 0)
  df$nr_weekdays <- 0
  df$nr_weekenddays <- 0
  
  for(row in 1:nrow(df)){
    if(df$nr_nights[row]!=0){
      weekdays <- 0
      weekenddays <- 0
      for(night in 1:round(df$nr_nights[row])){
        if(is.weekend(as.Date(df$posix_arrival[row])+night)){
          weekenddays <- weekenddays + 1
        }
        else {
          weekdays <- weekdays + 1
        }
      }
      df$nr_weekdays[row] <- weekdays
      df$nr_weekenddays[row] <- weekenddays
    }
  }
  return(df)
}

train_X_encode <- features_week_weekend_days(train_X_encode)
val_X_encode <- features_week_weekend_days(val_X_encode)
test_X_encode <- features_week_weekend_days(test_X_encode)

##############################################################
# 3.3 Numerical data - Indicator variables
##############################################################
# Make indicators variables for several variables as more than 90% is equal to zero and each time,
# there are very few values with 1 or somewhat higher
ind.cols <- c('nr_babies', 'nr_children')

# Apply
train_X_encode[, ind.cols] <- ifelse(train_X_encode[, ind.cols] == 0, 0, 1)
val_X_encode[, ind.cols] <- ifelse(val_X_encode[, ind.cols] == 0, 0, 1)
test_X_encode[, ind.cols] <- ifelse(test_X_encode[, ind.cols] == 0, 0, 1)


##############################################################
# 3.4 Transformations
##############################################################
# for three variables, we perform a log transformation as we want to make the distribution
# less skewed and reduce the range of the variables
train_X_scale <- train_X_encode
val_X_scale <- val_X_encode
test_X_scale <- test_X_encode

trans.cols <- c('lead_time', 'days_in_waiting_list', 'time_between_arrival_checkout')
train_X_encode[, trans.cols] <- log(train_X_encode[, trans.cols])


##############################################################
# 3.4 Scaling
##############################################################
# check if variable is normally distributed or not to see if we need to apply normalization 
# or standardization with the following code
hist(train_X_scale$time_between_arrival_checkout)

# normalization:
norm.cols <- c('nr_adults', 'nr_nights', 'lead_time', 'days_in_waiting_list','previous_bookings_not_canceled',
               'previous_cancellations', 'nr_booking_changes', 'special_requests', 'time_between_arrival_checkout',
               'time_between_arrival_cancel', 'car_parking_spaces', 'nr_weekdays', 'nr_weekenddays')

process <- preProcess(train_X_scale[, norm.cols], method=c("range")) # transformation from training set

train_X_scale[, norm.cols] <- predict(process, train_X_scale[, norm.cols])
val_X_scale[, norm.cols] <- predict(process, val_X_scale[, norm.cols])
test_X_scale[, norm.cols] <- predict(process, test_X_scale[, norm.cols])



##############################################################
# 3.5 Check correlations
##############################################################

# we tested correlations with the following code that we do not repeat all the time
cor(train_X_encode$nr_previous_bookings, train_X_encode$previous_bookings_not_canceled)
cor(train_X_encode$nr_previous_bookings, train_X_encode$previous_cancellations)


#high correlations!
# Some variables had high correlation. How these were handled is explained below:

#nr_booking_changes & nr_booking_changes_FLAG 0.89 -> delete 2nd -> below
#nr_nights & time_between_arrival_checkout 0.58 -> delete 2nd -> below
#assigned_room_type_* and reserved_room_type_* -> delete 1st and add column flag if assigned != reserved
#booking_distribution_channel_Direct & market_segment_Direct -> leave (cor < 0.8)
#booking_distribution_channel_Corporate & market_segment_Corporate -> like above
#booking_distribution_channel_Corporate & booking_company_NULL -> like above
#canceled_stay.cancelled & last_status_canceled -> delete 2nd

# Drop number of previous bookings as this is the sum of previous_cancellations and previous_bookings_not_canceled
# -> This has high correlation
# This happens in the next section

##############################################################
# 3.6 Column deleting
##############################################################

train_X_final <- train_X_scale
val_X_final <- val_X_scale
test_X_final <- test_X_scale

train_X_final <- subset(train_X_scale, select = -c(id, arrival_date, last_status_date,
                                                   nr_previous_bookings, posix_arrival,
                                                   posix_last_status, year_arrival,
                                                   nr_booking_changes, time_between_arrival_checkout, nr_nights))
val_X_final <- subset(val_X_scale, select = -c(id, arrival_date, last_status_date,
                                               nr_previous_bookings, posix_arrival,
                                               posix_last_status, year_arrival,
                                               nr_booking_changes, time_between_arrival_checkout, nr_nights))
test_X_final <- subset(test_X_scale, select = -c(arrival_date, last_status_date,
                                                 nr_previous_bookings, posix_arrival,
                                                 posix_last_status, year_arrival,
                                                 nr_booking_changes, time_between_arrival_checkout, nr_nights))


##############################################################
##############################################################
# 4. Write data away to Gold layer
##############################################################
##############################################################
training_data_after_FE <- train_X_final
training_data_after_FE$average_daily_rate <- train_y

val_data_after_FE <- val_X_final
val_data_after_FE$average_daily_rate <- val_y

test_data_after_FE <- test_X_final

# str(training_data_after_FE)

# Write data away
write.csv(training_data_after_FE,"./data/gold_data/train2.csv", row.names = FALSE)
write.csv(val_data_after_FE,"./data/gold_data/val2.csv", row.names = FALSE)
write.csv(test_data_after_FE,"./data/gold_data/test2.csv", row.names = FALSE)


##############################################################
##############################################################
# 5. End Note
##############################################################
##############################################################

# This is the end of the Feature Engineering section of the assignment.
# After this, the authors now begin the process of modeling
# as proposed by the CRISP-DM framework.

# In "Feature_engineering_PCA" PCA was performed to reduce the number of
# numerical features. The authors didn't end up using this as
# this only reduced the number of features by 2 and 
# didn’t increase predictive performance on the validation set.