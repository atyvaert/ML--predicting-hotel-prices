##############################################################
##############################################################
# Feature engineering
##############################################################
##############################################################
# libraries
library(lubridate)
library(readr)
library(dummy)

# import data
#setwd(dir = '/Users/Artur/Desktop/uni jaar 6 sem 1/machine learning/ml22-team10/data/silver_data')
train <- read.csv('./data/silver_data/train.csv')
test_X <- read.csv('./data/silver_data/test.csv')

#for Viktor
setwd(dir = 'C:/Users/vikto/OneDrive/Documenten/GroepswerkMachineLearning/ml22-team10/data/silver_data')
train <- read.csv('./train.csv')
test_X <- read.csv('./test.csv')

# separate dependent and independent variables
train_X <- subset(train, select = -c(average_daily_rate))
train_y <- train$average_daily_rate

# inspect
str(train_X)
str(test_X)




##############################################################
##############################################################
# 1. Categorical data
##############################################################
##############################################################
train_X_encode <- train_X
test_X_encode <- test_X

##############################################################
# 1.1 Ordinal data: integer encoding
##############################################################


##############################################################
# 1.2 Nominal data: one-hot encoding
##############################################################
# We use the dummmy package to treat nominal data as this is a more flexible approach
# and we have a lot of variables with a lot of levels

# First we make a categorical feature arrival_date_weekday
arrival_date_weekday_train <- wday(train_X_encode$posix_arrival, label = T)
arrival_date_weekday_test <- wday(test_X_encode$posix_arrival, label = T)

train_X_encode <- cbind(train_X_encode, arrival_date_weekday_train)
test_X_encode <- cbind(test_X_encode, arrival_date_weekday_test)

names(train_X_encode)[names(train_X_encode) == "arrival_date_weekday_train"] <- "arrival_date_weekday"
names(test_X_encode)[names(test_X_encode) == "arrival_date_weekday_test"] <- "arrival_date_weekday"


# get categories and dummies
# we only select the top 10 levels with highest frequency so our model does not explode
# For all cases, this includes most of the data: KIJKEN ALS DIT KLOPT BIJ DE REST
cats <- categories(train_X_encode[, c('assigned_room_type', 'booking_distribution_channel',
                                      'canceled', 'country', 'customer_type', 'deposit',
                                      'hotel_type', 'is_repeated_guest', 'last_status',
                                      'market_segment', 'meal_booked', 'reserved_room_type',
                                      'arrival_date_weekday')], p = 10)


# month_arrival seperate because we want all 12 categories here
cats <- append(cats, categories(train_X_encode['month_arrival']))


# apply on train set (exclude reference categories)
dummies_train <- dummy(train_X_encode[,c('assigned_room_type', 'booking_distribution_channel', 
                                         'canceled', 'country', 'customer_type', 'deposit',
                                         'hotel_type', 'is_repeated_guest', 'last_status',
                                         'market_segment', 'meal_booked', 'reserved_room_type',
                                         'month_arrival', 'arrival_date_weekday')], object = cats)

# exclude the reference category: take the first one of the variable you added
names(dummies_train)
dummies_train <- subset(dummies_train, 
                        select = -c(assigned_room_type_A, booking_distribution_channel_TA.TO,
                                    country_Belgium, canceled_no.cancellation, customer_type_Transient,
                                    deposit_nodeposit, hotel_type_City.Hotel, is_repeated_guest_no,
                                    last_status_Check.Out, market_segment_Online.travel.agent,
                                    meal_booked_meal.package.NOT.booked, reserved_room_type_A, month_arrival_January,
                                    arrival_date_weekday_Mon))

# apply on test set (exclude reference categories)
# excluded no.canceled so it becomes one when it was canceled
dummies_test <- dummy(test_X_encode[, c('assigned_room_type', 'booking_distribution_channel', 
                                        'canceled', 'country', 'customer_type', 'deposit',
                                        'hotel_type', 'is_repeated_guest', 'last_status',
                                        'market_segment', 'meal_booked', 'reserved_room_type',
                                        'month_arrival', 'arrival_date_weekday')], object = cats)
dummies_test <- subset(dummies_test, select = -c(assigned_room_type_A, booking_distribution_channel_TA.TO,
                                                 country_Belgium, canceled_no.cancellation, customer_type_Transient,
                                                 deposit_nodeposit, hotel_type_City.Hotel, is_repeated_guest_no,
                                                 last_status_Check.Out, market_segment_Online.travel.agent,
                                                 meal_booked_meal.package.NOT.booked, reserved_room_type_A, month_arrival_January,
                                                 arrival_date_weekday_Mon))

# we remove the original predictors and merge them with the other predictors
## merge with overall training set
train_X_encode <- subset(train_X_encode, select = -c(assigned_room_type, booking_distribution_channel,
                                                     canceled, country, customer_type, deposit,
                                                     hotel_type, is_repeated_guest, last_status,
                                                     market_segment, meal_booked, reserved_room_type,
                                                     month_arrival, arrival_date_weekday))
train_X_encode <- cbind(train_X_encode, dummies_train)
## merge with overall test set
test_X_encode <- subset(test_X_encode, select = -c(assigned_room_type, booking_distribution_channel,
                                                   canceled, country, customer_type, deposit,
                                                   hotel_type, is_repeated_guest, last_status,
                                                   market_segment, meal_booked, reserved_room_type,
                                                   month_arrival, arrival_date_weekday))
test_X_encode <- cbind(test_X_encode, dummies_test)

train_X_encode

##############################################################
# 1.3 Special data: create a indicator variable
##############################################################
# For these data, the are many different values without any meaning, for example
# the number of a booking agency due to anonymity of the data set. Creating dummy variables
# for these values would not be very meaningful, but it is informative to know if the hotel
# was booked through the booking agency, therefore we create an indicator variable

# 1. for columns with null values for training and test set
null.cols <- c('booking_agent', 'booking_company')
new.cols_names <- c('booking_agent_present', 'booking_company_present')
train_X_encode[, new.cols_names] <- ifelse(train_X[, null.cols] == 'NULL', 0, 1)
test_X_encode[, new.cols_names] <- ifelse(test_X[, null.cols] == 'NULL', 0, 1)

# remove original columns
train_X_encode <- subset(train_X_encode, select = -c(booking_agent, booking_company))
test_X_encode <- subset(test_X_encode, select = -c(booking_agent, booking_company))


##############################################################
##############################################################
# 2. Numerical data
##############################################################
##############################################################


# Create feature time_between_last_status_arrival
# time_between_arrival_checkout gives a positive difference when last status date is after
#  arrival (often when customer checks out)
# time_between_arrival_cancel gives a negative difference when last status date is before
#  arrival (often when customer cancels stay)

# train
time_between_last_status_arrival <- as.numeric(round(difftime(train_X_encode$posix_last_status, train_X_encode$posix_arrival)/(60*60*24)))
time_between_arrival_checkout <- time_between_last_status_arrival
time_between_arrival_checkout[time_between_arrival_checkout<0] <- 0
time_between_arrival_cancel <- time_between_last_status_arrival
time_between_arrival_cancel[time_between_arrival_cancel>0] <- 0

train_X_encode <- cbind(train_X_encode, time_between_arrival_checkout, time_between_arrival_cancel)

# test
time_between_last_status_arrival <- as.numeric(round(difftime(test_X_encode$posix_last_status, test_X_encode$posix_arrival)/(60*60*24)))
time_between_arrival_checkout <- time_between_last_status_arrival
time_between_arrival_checkout[time_between_arrival_checkout<0] <- 0
time_between_arrival_cancel <- time_between_last_status_arrival
time_between_arrival_cancel[time_between_arrival_cancel>0] <- 0

test_X_encode <- cbind(test_X_encode, time_between_arrival_checkout, time_between_arrival_cancel)


##DEZE BINNING CODE WEG?##
# binning of the days in waiting list variable 
# write a function to calculate the bin frequency
bin_data_frequency <- function(x_train, x_val, bins = 5) {
  cut(x_val, breaks = quantile(x_train, seq(0, 1, 1 / bins)), include.lowest = TRUE)
}
# apply the function to the days in waiting list variable 
train_X_encode$days_in_waiting_list <- bin_data_frequency(x_train = train_X_encode$days_in_waiting_list, x_val = train_X_encode$days_in_waiting_list, bins = 5)
test_X_encode$days_in_waiting_list <- bin_data_frequency(x_train = train_X_encode$days_in_waiting_list, x_val = test_X_encode$days_in_waiting_list, bins = 5)
# observe the frequencies 
train_X_encode$days_in_waiting_list
test_X_encode$days_in_waiting_list
# perform integer encoding because the levels have a logical order between them 
train_X_encode$days_in_waiting_list <- as.numeric(train_X_encode$days_in_waiting_list)
test_X_encode$days_in_waiting_list <- as.numeric(test_X_encode$days_in_waiting_list)









# create indicators for nr_babies & nr_children
train_X_encode$nr_babies[train_X_encode$nr_babies>=1] <- 1
train_X_encode$nr_children[train_X_encode$nr_children>=1] <- 1
test_X_encode$nr_babies[test_X_encode$nr_babies>=1] <- 1
test_X_encode$nr_children[test_X_encode$nr_children>=1] <- 1
# create indicator variables for days in waiting list
train_X_encode$days_in_waiting_list[train_X_encode$days_in_waiting_list > 0] <- 1
test_X_encode$days_in_waiting_list[test_X_encode$days_in_waiting_list > 0] <- 1


#check multicolinearity 
test_cor <- subset(train_X_encode, select = -c(arrival_date, last_status_date, posix_arrival, year_arrival, posix_last_status, arrival_date_weekday))
# pearson correlation 
pcor(test_cor, method = "pearson")
# correlation matrix 
cor2pcor(cov(test_cor))
# Glauber test for multicollinearity 
omcdiag(test_cor, train_y)

cor(train_X_encode$nr_previous_bookings, train_X_encode$previous_bookings_not_canceled)
cor(train_X_encode$nr_previous_bookings, train_X_encode$previous_cancellations)

# create indicator variables for nr_booking_changes
train_X_encode$nr_booking_changes[train_X_encode$nr_booking_changes > 0] <- 1
test_X_encode$nr_booking_changes[test_X_encode$nr_booking_changes > 0] <- 1
# create indicator variables for car_parking_spaces
train_X_encode$car_parking_spaces[train_X_encode$car_parking_spaces > 0] <- 1
test_X_encode$car_parking_spaces[test_X_encode$car_parking_spaces > 0] <- 1


##############################################################
# 2.2 Scaling
##############################################################
train_X_scale <- train_X_encode
test_X_scale <- test_X_encode


scale_cols <- c("nr_adults", "nr_nights", "lead_time",
                "previous_bookings_not_canceled", "previous_cancellations",
                "special_requests", "time_between_arrival_checkout", "time_between_arrival_cancel")

# apply on training set
mean_train <- colMeans(train_X_scale[, scale_cols])
sd_train <- sapply(train_X_scale[, scale_cols], sd)
train_X_scale[, scale_cols] <- scale(train_X_scale[, scale_cols], center = TRUE, scale = TRUE)

# apply on test set
test_X_scale[, scale_cols] <- scale(test_X_scale[, scale_cols], center = mean_train, scale = sd_train)

##############################################################
# 2.3 Column deleting
##############################################################

train_X_final <- train_X_scale
test_X_final <- test_X_scale

train_X_final <- subset(train_X_scale, select = -c(id, arrival_date, last_status_date,
                                          nr_previous_bookings, posix_arrival,
                                          day_of_month_arrival, posix_last_status))
test_X_final <- subset(train_X_scale, select = -c(id, arrival_date, last_status_date,
                                                   nr_previous_bookings, posix_arrival,
                                                   day_of_month_arrival, posix_last_status))

#wat doen met year_arrival?








