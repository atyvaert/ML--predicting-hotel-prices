##############################################################
##############################################################
# 0. Introduction of File
##############################################################
##############################################################
# Before the actual modeling starts, it is important to clean the data and create features.
# In this file, the focus rests on data cleaning.



##############################################################
##############################################################
# 1. Data Import
##############################################################
##############################################################

# The authors follow the Bronze/Silver/Gold data structure.
# In this step, the raw (Bronze) data is used.


# The needed packages and libraries are loaded
library(readr)
library(dummy)
library(stringr)
library(miscTools)

# It might be needed to set R language to English for weekdays variables.
# Sys.setlocale("LC_ALL","English")

#The Bronze data is read
train <- read_csv('./data/bronze_data/train.csv')
test_X <- read_csv('./data/bronze_data/test.csv')


# Creation of training and validation set
set.seed(100)
valvector <- sample(nrow(train), size=nrow(train)*0.2)
val <- train[valvector, ]
train <- train[-valvector, ]

# separation of dependent and independent variables
train_X <- subset(train, select = -c(average_daily_rate))
train_y <- train$average_daily_rate
train_y <- as.numeric(word(train_y, 1))

val_X <- subset(val, select = -c(average_daily_rate))
val_y <- val$average_daily_rate
val_y <- as.numeric(word(val_y, 1))


##############################################################
##############################################################
# 2. Adjust structure
##############################################################
##############################################################

# lead_time needs to be integer number of days 
# for training set
train_X$lead_time <- substr(train_X$lead_time, start = 1, stop = (nchar(train_X$lead_time)-7))
train_X$lead_time <- as.numeric(train_X$lead_time)

# for val set
val_X$lead_time <- substr(val_X$lead_time, start = 1, stop = (nchar(val_X$lead_time)-7))
val_X$lead_time <- as.numeric(val_X$lead_time)

# for test set
test_X$lead_time <- substr(test_X$lead_time, start = 1, stop = (nchar(test_X$lead_time)-7))
test_X$lead_time <- as.numeric(test_X$lead_time)


##############################################################
##############################################################
# 3. Missing values
##############################################################
##############################################################

# overwrite existing dataframes
train_X_impute <- train_X
val_X_impute <- val_X
test_X_impute <- test_X

##############################################################
# 3.1 Detecting missing values
##############################################################

# training data
colMeans(is.na(train_X_impute))
# val data
colMeans(is.na(val_X_impute))
# test data
colMeans(is.na(test_X_impute))

##############################################################
# 3.2 Missing value imputation and Flagging
##############################################################
# Missing values might increase model performance if they are informative, therefor
# a NA indicator of the missing values is added. This is done by making use of the function naFlag:

naFlag <- function(df, df_val = NULL) {
  if (is.null(df_val)) {
    df_val <- df
  }
  mask <- sapply(df_val, anyNA)
  out <- lapply(df[mask], function(x)as.numeric(is.na(x)))
  if (length(out) > 0) names(out) <- paste0(names(out), "_flag")
  return(as.data.frame(out))
}

# Flag the missing values
train_X_impute <- cbind(train_X_impute, naFlag(df = train_X))
val_X_impute <- cbind(val_X_impute,naFlag(df = val_X, df_val = train_X))
test_X_impute <- cbind(test_X_impute,naFlag(df = test_X, df_val = train_X))

# Inspect
str(train_X_impute)
str(val_X_impute)
str(test_X_impute)

# 3.2.1 Impute NUMERIC variables with information from the training set

# Use the 'impute' function for numeric predictors (imputes with given method or value):
impute <- function(x, method = mean, val = NULL){
  if(is.null(val)){
    val <- method(x, na.rm = T)
  }
  x[is.na(x)] <- val
  return(x)
}


# Impute numerical variables with the median as the authors thought this was most fitting for these variables

median.cols <- c('car_parking_spaces', 'nr_adults', 'nr_children', 'nr_previous_bookings',
                 'previous_bookings_not_canceled', 'previous_cancellations', 'days_in_waiting_list',
                 'lead_time')
train_X_impute[, median.cols] <- lapply(train_X_impute[, median.cols], 
                                        FUN = impute,
                                        method = median)
val_X_impute[, median.cols] <- mapply(val_X_impute[, median.cols],
                                       FUN = impute,
                                       val = colMedians(train_X[, median.cols], na.rm = T))
test_X_impute[, median.cols] <- mapply(test_X_impute[, median.cols],
                                       FUN = impute,
                                       val = colMedians(train_X[, median.cols], na.rm = T))


# 3.2.2 Impute CATEGORICAL variables with information from the training set

# Use the 'modus' function for categorical predictors. This returns the mode of a column.

modus <- function(x, na.rm = FALSE) {
  if (na.rm) x <- x[!is.na(x)]
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

# Handling of missing values
cat.cols <- c('booking_distribution_channel', 'country', 'customer_type', 'deposit', 'hotel_type',
              'is_repeated_guest', 'last_status', 'market_segment')

train_X_impute[, cat.cols] <- lapply(train_X_impute[, cat.cols],
                                     FUN = impute,
                                     method = modus)
val_X_impute[, cat.cols] <- mapply(val_X_impute[, cat.cols],
                                    FUN = impute,
                                    val = sapply(train_X[, cat.cols], modus, na.rm = T))
test_X_impute[, cat.cols] <- mapply(test_X_impute[, cat.cols],
                                    FUN = impute,
                                    val = sapply(train_X[, cat.cols], modus, na.rm = T))


# 3.2.3 Impute 'n/a' or Na values with 0 for nr_babies and nr_booking_changes as 'n/a' suggests
# a value of zero. Next the flag column of nr_booking_changes was deleted as this is not interpreted as a real missing variable 
# due to imputing them with zero.
# Besides, for variables like booking agent and company, NULL is presented as one of the categories
# This should not be considered a NULL value, but rather as not applicable
# for 'n/a':
train_X_impute$nr_babies <- as.numeric(str_replace_all(train_X$nr_babies, "n/a", "0"))
val_X_impute$nr_babies <- as.numeric(str_replace_all(val_X_impute$nr_babies, "n/a", "0"))
test_X_impute$nr_babies <- as.numeric(str_replace_all(test_X_impute$nr_babies, "n/a", "0"))

# for Na values
train_X_impute$nr_booking_changes[is.na(train_X_impute$nr_booking_changes)] <- 0
val_X_impute$nr_booking_changes[is.na(val_X_impute$nr_booking_changes)] <- 0
test_X_impute$nr_booking_changes[is.na(test_X_impute$nr_booking_changes)] <- 0

# for NULL values
train_X_impute$booking_agent <- str_replace_all(train_X$booking_agent, 'NULL', "n/a")
val_X_impute$booking_agent <- str_replace_all(val_X_impute$booking_agent, 'NULL', "n/a")
test_X_impute$booking_agent <- str_replace_all(test_X_impute$booking_agent, 'NULL', "n/a")

train_X_impute$booking_company <- str_replace_all(train_X$booking_company, 'NULL', "n/a")
val_X_impute$booking_company <- str_replace_all(val_X_impute$booking_company, 'NULL', "n/a")
test_X_impute$booking_company <- str_replace_all(test_X_impute$booking_company, 'NULL', "n/a")

# drop the flag variable of nr_booking_changes
train_X_impute = subset(train_X_impute, select = -c(nr_booking_changes_flag))
val_X_impute = subset(val_X_impute, select = -c(nr_booking_changes_flag))
test_X_impute = subset(test_X_impute, select = -c(nr_booking_changes_flag))

# inspect
colMeans(is.na(train_X_impute))
colMeans(is.na(val_X_impute))
colMeans(is.na(test_X_impute))


##############################################################
##############################################################
#3.3 Data inconsistencies
##############################################################
##############################################################

# nr_previous_bookings = previous_cancellations + previous_bookings_not_canceled
# This is not always the case!

# nr_previous_bookings = previous_cancellations + previous_bookings_not_canceled where nr_previous_bookings = 0
train_X_impute$nr_previous_bookings[train_X_impute$nr_previous_bookings==0] <- rowSums(cbind(train_X_impute$previous_bookings_not_canceled, train_X_impute$previous_cancellations))[train_X_impute$nr_previous_bookings==0]
# nr_previous_bookings_not_canceled = nr_previous_bookings - previous_cancellations where previous_bookings_not_canceled = 0
train_X_impute$previous_bookings_not_canceled[train_X_impute$previous_bookings_not_canceled==0] <- (train_X_impute$nr_previous_bookings - train_X_impute$previous_cancellations)[train_X_impute$previous_bookings_not_canceled==0]
# previous_cancellations = nr_previous_bookings - nr_previous_bookings_not_canceled where previous_cancellations = 0
train_X_impute$previous_cancellations[train_X_impute$previous_cancellations==0] <- (train_X_impute$nr_previous_bookings - train_X_impute$previous_bookings_not_canceled)[train_X_impute$previous_cancellations==0]


##############################################################
##############################################################
# 4. Detecting and handling outliers
##############################################################
##############################################################

train_X_outlier <- train_X_impute
val_X_outlier <- val_X_impute # This column is created for uniformity reasons, no outliers are deleted
test_X_outlier <- test_X_impute # This column is created for uniformity reasons, no outliers are deleted

# Creation of vector of all the variables of which valid outliers need to be handled
outlier.cols <- c()

# Inspect outliers with the help of the following function.
# This will not be repeated all times here.
# all the time:
# boxplot(train_X_impute$X)
# quantile(train_X_impute$X, na.rm = T, probs = seq(0, 1, 0.001))
# quantile(scale(train_X_impute$X), na.rm = T, probs = seq(0, 1, 0.001))

# Look at all the numeric variables and detect valid and invalid outliers:
# 1) car_parking_spaces
# All the car parking spaces have a value between 0 and 3, with the majority being 0
# We do not see the a value of 3 car parking spaces as an outlier

# 16) lead_time
outlier.cols <- append(outlier.cols, 'lead_time')

# 19) nr_adults
# Here, a value of more than 10 adults is seen as an outlier. Especially, since no value is higher than 3 except for 
# a one time booking of 40 and 50 adults
train_X_outlier$nr_adults[train_X$nr_adults > 10] <- NA
train_X_outlier$nr_adults <- impute(train_X_outlier$nr_adults, method = median)
outlier.cols <- append(outlier.cols, 'nr_adults')

# 20) nr_babies
# only values of 1 or 2 are present, even these have z-values higher than 3, these are not seen as outliers

# 21) nr_booking_changes  
# >4  are outliers. We see that most values are 0, 1 or 2. The values 4 and 5 are seen as valid outliers.
# However, a value of 21 is seen as an unreasonable outlier.
train_X_outlier$nr_booking_changes[train_X_outlier$nr_booking_changes> 20] <- NA
train_X_outlier$nr_booking_changes <- impute(train_X_outlier$nr_booking_changes, method = median)
outlier.cols <- append(outlier.cols, 'nr_booking_changes')


# 22) nr_children  
#  We see two children as a valid outlier (based on z-scores). However,
# there is also a single observation with 10 children, which we see as an invalid outlier.
train_X_outlier$nr_children[train_X$nr_children > 9] <- NA
train_X_outlier$nr_children <- impute(train_X_outlier$nr_children, method = median)
outlier.cols <- append(outlier.cols, 'nr_children')

# 23) nr_nights
# starting from 98.6% we have outliers based on z-values
# This a value of 10 nights, which is reasonable. However, there is also a value of 69. A stay of longer 
# than one month (= 30 days) is interpreted as an invalid outlier
train_X_outlier$nr_nights[train_X$nr_nights>30] <- NA
train_X_outlier$nr_nights <- impute(train_X_outlier$nr_nights, method = median)
outlier.cols <- append(outlier.cols, 'nr_nights')

# 24) nr_previous_bookings  
# Here, we see a lot of values between 1 and 5. Even some valid outliers up until 26 as well. Even this is already a lot of bookings.
# However, we also have  a value of more than 70. We see more than 30 bookings as an invalid outlier as this means a person booked the hotel
# more than once per month over a timespan of 2 years.
train_X_outlier$nr_previous_bookings[train_X$nr_previous_bookings>30] <- NA
train_X_outlier$nr_previous_bookings <- impute(train_X_outlier$nr_previous_bookings, method = median)
outlier.cols <- append(outlier.cols, 'nr_previous_bookings')


# 25) previous_bookings_not_canceled  
# starting from 99.3% (=21!!), we have outliers. Again, we have a value higher than 70. This could be
# due to solving our data inconsistencies in the previous step. Therefore, we will also see this as an invalid outlier.
# Again we use a value of 30 as cut off point.
train_X_outlier$previous_bookings_not_canceled[train_X$previous_bookings_not_canceled>30] <- NA
train_X_outlier$previous_bookings_not_canceled <- impute(train_X_outlier$previous_bookings_not_canceled, method = median)
outlier.cols <- append(outlier.cols, 'previous_bookings_not_canceled')


# 26) previous_cancellations  
#starting from 99.8% ( = 4!!), we have outliers. However, 4 is still a reasonable value.
# However, when a person cancels his room more than 20 times, we see this as an invalid outlier.
train_X_outlier$previous_cancellations[train_X$previous_cancellations>20] <- NA
train_X_outlier$previous_cancellations <- impute(train_X_outlier$previous_cancellations, method = median)
outlier.cols <- append(outlier.cols, 'previous_cancellations')


# 28) special_requests  
# starting from 97.6% (=3), we have outliers. However, we do not see any unusual values.
outlier.cols <- append(outlier.cols, 'special_requests')

# inspect
str(train_X_outlier)
str(val_X_outlier)
str(test_X_outlier)


# use this function to handle (scale) valid outliers
handle_outlier_z <- function(col){
  col_z <- scale(col)
  ifelse(abs(col_z)>3,
         sign(col_z)*3*attr(col_z,"scaled:scale") + attr(col_z,"scaled:center"), col)
}

# handle all the outliers at once
train_X_outlier[, outlier.cols] <-  sapply(train_X_impute[, outlier.cols], FUN = handle_outlier_z)

# We cannot use the previous method to handle outliers of the number of days in waiting list 
# This is because this variable has a lot of 0 values, and if we would use the z score, a lot of outliers would be identified
#quantile(train_X_impute$days_in_waiting_list, na.rm = T, probs = seq(0, 1, 0.001))
#quantile(scale(train_X_impute$days_in_waiting_list), na.rm = T, probs = seq(0, 1, 0.01))
# When we look at the distribution of the days in waiting list variable, we see that less than 1 % has a value higher than 70 days 
# We arbitrary set the boundary to be an outlier to 70
train_X_outlier$days_in_waiting_list <- ifelse(train_X_impute$days_in_waiting_list >= 70, 70, train_X_impute$days_in_waiting_list)


##############################################################
##############################################################
# 5. Parsing dates
##############################################################
##############################################################

# Parse arrival_date to filter out the month, day of the month and year of arrival
# for training set:
train_X_outlier$posix_arrival <- as.POSIXct(train_X_outlier$arrival_date, format="%B  %d  %Y")
train_X_outlier$day_of_month_arrival <- format(train_X_outlier$posix_arrival, format = '%d')
train_X_outlier$month_arrival <- format(train_X_outlier$posix_arrival, format = '%B')
train_X_outlier$year_arrival <- as.factor(format(train_X_outlier$posix_arrival, format = '%Y'))

# for val set:
val_X_outlier$posix_arrival <- as.POSIXct(val_X_outlier$arrival_date, format="%B  %d  %Y")
val_X_outlier$day_of_month_arrival <- format(val_X_outlier$posix_arrival, format = '%d')
val_X_outlier$month_arrival <- format(val_X_outlier$posix_arrival, format = '%B')
val_X_outlier$year_arrival <- as.factor(format(val_X_outlier$posix_arrival, format = '%Y'))

# for test set:
test_X_outlier$posix_arrival <- as.POSIXct(test_X_outlier$arrival_date, format="%B  %d  %Y")
test_X_outlier$day_of_month_arrival <- format(test_X_outlier$posix_arrival, format = '%d')
test_X_outlier$month_arrival <- format(test_X_outlier$posix_arrival, format = '%B')
test_X_outlier$year_arrival <- as.factor(format(test_X_outlier$posix_arrival, format = '%Y'))

##############################################################
# Special case imputation: 'last_status_date'
##############################################################
# To impute NA we calculate the mean of the differences between arrival_date and last_status_date for each category of last_status
# -> add this value to arrival_date and impute in NA rows of last_status_date
# (this needed to happen after dates parsing)

# train
train_X_outlier$posix_last_status <- as.POSIXlt(train_X_outlier$last_status_date, format='%Y-%m-%dT %H:%M:%S')

mean_diff_canceled <- round(mean((difftime(train_X_outlier$posix_last_status[train_X_outlier$last_status=="Canceled"], train_X_outlier$posix_arrival[train_X_outlier$last_status=="Canceled"], units = "d")), na.rm = T))
mean_diff_check_out <- round(mean((difftime(train_X_outlier$posix_last_status[train_X_outlier$last_status=="Check-Out"], train_X_outlier$posix_arrival[train_X_outlier$last_status=="Check-Out"], units = "d")), na.rm = T))
mean_diff_no_show <- round(mean((difftime(train_X_outlier$posix_arrival[train_X_outlier$last_status=="No-Show"], train_X_outlier$posix_last_status[train_X_outlier$last_status=="No-Show"], units = "d")), na.rm = T))

train_X_outlier$posix_last_status[is.na(train_X_outlier$posix_last_status) & (train_X_outlier$last_status=="Canceled")] <- train_X_outlier$posix_arrival[is.na(train_X_outlier$posix_last_status) & (train_X_outlier$last_status=="Canceled")] + mean_diff_canceled
train_X_outlier$posix_last_status[is.na(train_X_outlier$posix_last_status) & (train_X_outlier$last_status=="Check-Out")] <- train_X_outlier$posix_arrival[is.na(train_X_outlier$posix_last_status) & (train_X_outlier$last_status=="Check-Out")] + mean_diff_check_out
train_X_outlier$posix_last_status[is.na(train_X_outlier$posix_last_status) & (train_X_outlier$last_status=="No-Show")] <- train_X_outlier$posix_arrival[is.na(train_X_outlier$posix_last_status) & (train_X_outlier$last_status=="No-Show")] + mean_diff_canceled

train_X_outlier$posix_last_status <- format(as.POSIXct(train_X_outlier$posix_last_status, format="%Y-%m-%d"), format="%Y-%m-%d")

# val
val_X_outlier$posix_last_status <- as.POSIXlt(val_X_outlier$last_status_date, format='%Y-%m-%dT %H:%M:%S')

val_X_outlier$posix_last_status[is.na(val_X_outlier$posix_last_status) & (val_X_outlier$last_status=="Canceled")] <- val_X_outlier$posix_arrival[is.na(val_X_outlier$posix_last_status) & (val_X_outlier$last_status=="Canceled")] + mean_diff_canceled
val_X_outlier$posix_last_status[is.na(val_X_outlier$posix_last_status) & (val_X_outlier$last_status=="Check-Out")] <- val_X_outlier$posix_arrival[is.na(val_X_outlier$posix_last_status) & (val_X_outlier$last_status=="Check-Out")] + mean_diff_check_out
val_X_outlier$posix_last_status[is.na(val_X_outlier$posix_last_status) & (val_X_outlier$last_status=="No-Show")] <- val_X_outlier$posix_arrival[is.na(val_X_outlier$posix_last_status) & (val_X_outlier$last_status=="No-Show")] + mean_diff_canceled

val_X_outlier$posix_last_status <- format(as.POSIXct(val_X_outlier$posix_last_status, format="%Y-%m-%d"), format="%Y-%m-%d")

# test
test_X_outlier$posix_last_status <- as.POSIXlt(test_X_outlier$last_status_date, format='%Y-%m-%dT %H:%M:%S')

test_X_outlier$posix_last_status[is.na(test_X_outlier$posix_last_status) & (test_X_outlier$last_status=="Canceled")] <- test_X_outlier$posix_arrival[is.na(test_X_outlier$posix_last_status) & (test_X_outlier$last_status=="Canceled")] + mean_diff_canceled
test_X_outlier$posix_last_status[is.na(test_X_outlier$posix_last_status) & (test_X_outlier$last_status=="Check-Out")] <- test_X_outlier$posix_arrival[is.na(test_X_outlier$posix_last_status) & (test_X_outlier$last_status=="Check-Out")] + mean_diff_check_out
test_X_outlier$posix_last_status[is.na(test_X_outlier$posix_last_status) & (test_X_outlier$last_status=="No-Show")] <- test_X_outlier$posix_arrival[is.na(test_X_outlier$posix_last_status) & (test_X_outlier$last_status=="No-Show")] + mean_diff_canceled

test_X_outlier$posix_last_status <- format(as.POSIXct(test_X_outlier$posix_last_status, format="%Y-%m-%d"), format="%Y-%m-%d")


##############################################################
##############################################################
# 6. Write data away to silver layer
##############################################################
##############################################################
training_data_after_data_cleaning <- train_X_outlier
training_data_after_data_cleaning$average_daily_rate <- train_y
val_data_after_data_cleaning <- val_X_outlier
val_data_after_data_cleaning$average_daily_rate <- val_y
test_data_after_data_cleaning <- test_X_outlier

# inspection
#str(training_data_after_data_cleaning)
#str(test_data_after_data_cleaning)

write.csv(training_data_after_data_cleaning,"./data/silver_data/train.csv", row.names = FALSE)
write.csv(val_data_after_data_cleaning,"./data/silver_data/val.csv", row.names = FALSE)
write.csv(test_data_after_data_cleaning,"./data/silver_data/test.csv", row.names = FALSE)



##############################################################
##############################################################
# 7. End Note
##############################################################
##############################################################

# This is the end of the Data Cleaning section of the assignment.
# After this, the authors now begin the process of Feature Engineering
# as proposed by the CRISP-DM framework.
