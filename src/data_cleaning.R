##############################################################
##############################################################
# Data Cleaning
##############################################################
##############################################################
# packages and libraries needed
library(readr)
library(dummy)
library(stringr)
library(miscTools)

# import the data
train <- read_csv('./data/bronze_data/train.csv')
test_X <- read_csv('./data/bronze_data/test.csv')

# separate dependent and independent variables
train_X <- subset(train, select = -c(average_daily_rate))
train_y <- train$average_daily_rate
train_y <- as.numeric(word(train_y, 1))


##############################################################
##############################################################
# 1. Adjust structure
##############################################################
##############################################################

#lead_time to integer number of days for training set
train_X$lead_time <- substr(train_X$lead_time, start = 1, stop = (nchar(train_X$lead_time)-7))
train_X$lead_time <- as.numeric(train_X$lead_time)

# for test set
test_X$lead_time <- substr(test_X$lead_time, start = 1, stop = (nchar(test_X$lead_time)-7))
test_X$lead_time <- as.numeric(test_X$lead_time)


##############################################################
##############################################################
# 2. Missing values
##############################################################
##############################################################

# overwrite existing dataframes
train_X_impute <- train_X
test_X_impute <- test_X

##############################################################
# 2.1 Detecting 
##############################################################
# training data
colMeans(is.na(train_X_impute))
# test data
colMeans(is.na(test_X_impute))

##############################################################
# 2.2 Missing value imputation + Flagging
##############################################################
# Missing values might increase model performance if they are informative, therefore
# we add a NA indicator of the missing values. We do this by making use of the function naFlag:

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
test_X_impute <- cbind(test_X_impute,naFlag(df = test_X, df_val = train_X))

# inspect
str(train_X_impute)
str(test_X_impute)

# 2.2.1 Impute NUMERIC variables with information from the training set

# use the 'impute' function for numeric predictors:
impute <- function(x, method = mean, val = NULL){
  if(is.null(val)){
    val <- method(x, na.rm = T)
  }
  x[is.na(x)] <- val
  return(x)
}


# impute numerical variables with the median as we thought this was most fitting for 
# these variables
median.cols <- c('car_parking_spaces', 'nr_adults', 'nr_children', 'nr_previous_bookings',
                 'previous_bookings_not_canceled', 'previous_cancellations', 'days_in_waiting_list',
                 'lead_time')
train_X_impute[, median.cols] <- lapply(train_X_impute[, median.cols], 
                                        FUN = impute,
                                        method = median)

test_X_impute[, median.cols] <- mapply(test_X_impute[, median.cols],
                                       FUN = impute,
                                       val = colMedians(train_X[, median.cols], na.rm = T))


# 2.2.1 Impute CATEGORICAL variables with information from the training set
# use the 'modus' function for categorical predictors. This returns the mode of a column.
modus <- function(x, na.rm = FALSE) {
  if (na.rm) x <- x[!is.na(x)]
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

# handle
cat.cols <- c('booking_distribution_channel', 'country', 'customer_type', 'deposit', 'hotel_type',
              'is_repeated_guest', 'last_status', 'market_segment')

train_X_impute[, cat.cols] <- lapply(train_X_impute[, cat.cols],
                                     FUN = impute,
                                     method = modus)
test_X_impute[, cat.cols] <- mapply(test_X_impute[, cat.cols],
                                    FUN = impute,
                                    val = sapply(train_X[, cat.cols], modus, na.rm = T))


# impute 'n/a' or Na values with 0 for nr_babies and nr_booking_changes as 'n/a' suggests
# a value of zero
# for 'n/a':
train_X_impute$nr_babies <- as.numeric(str_replace_all(train_X$nr_babies, "n/a", "0"))
test_X_impute$nr_babies <- as.numeric(str_replace_all(test_X_impute$nr_babies, "n/a", "0"))

# for Na values
train_X_impute$nr_booking_changes[is.na(train_X_impute$nr_booking_changes)] <- 0
test_X_impute$nr_booking_changes[is.na(test_X_impute$nr_booking_changes)] <- 0


# inspect
colMeans(is.na(train_X_impute))
colMeans(is.na(test_X_impute))

##############################################################
##############################################################
# Data inconsistencies: nr_previous_bookings = 
# previous_cancellations + previous_bookings_not_canceled
# This is not always the case!
##############################################################
##############################################################
train_X_impute$nr_previous_bookings[train_X_impute$nr_previous_bookings==0] <- rowSums(cbind(train_X_impute$previous_bookings_not_canceled, train_X_impute$previous_cancellations))[train_X_impute$nr_previous_bookings==0]
train_X_impute$previous_bookings_not_canceled[train_X_impute$previous_bookings_not_canceled==0] <- (train_X_impute$nr_previous_bookings - train_X_impute$previous_cancellations)[train_X_impute$previous_bookings_not_canceled==0]
train_X_impute$previous_cancellations[train_X_impute$previous_cancellations==0] <- (train_X_impute$nr_previous_bookings - train_X_impute$previous_bookings_not_canceled)[train_X_impute$previous_cancellations==0]



##############################################################
##############################################################
# 3. Detecting and handling outliers
##############################################################
##############################################################
train_X_outlier <- train_X_impute
test_X_outlier <- test_X_impute

# make a vector of all the variables of which valid outliers need to be handled
outlier.cols <- c()
# outlier.cols <- append(outlier.cols, '')

# We inspected the outlier with the help of the following function, that we will not repeat
# all the time:
# boxplot(train_X_impute$X)
# quantile(train_X_impute$X, na.rm = T, probs = seq(0, 1, 0.001))

# look at all the numeric variables and detect valid and invalid outliers:
# 1) car_parking_spaces
# All the car parking spaces have a value between 0 and 3, with the majority being 0
# as 1, 2 and 3 are seen as outliers, we bring back these values to one as these are 
# valid outliers. However, these values are converted to 0.78. Therefore, we create an
# indicator variable for car_parking_spaces in the feature engineering part
# add car_parking_spaces to variables that need to be handled
outlier.cols <- append(outlier.cols, 'car_parking_spaces')


# 16) lead_time
outlier.cols <- append(outlier.cols, 'lead_time')

# 19) nr_adults
outlier.cols <- append(outlier.cols, 'nr_adults')

# 20) nr_babies
# 1 invalid outlier: "9" treat as NA -> "impute with median 0"
train_X_outlier$nr_babies[train_X$nr_babies==9] <- NA
train_X_outlier$nr_babies <- impute(train_X_outlier$nr_babies, method = median)
#FE: dummy variable babies: 0 = No, 1 = Yes


# 21) nr_booking_changes  
# >3  are outliers, however valid? like lots of travel insecurities => change number of rooms depending on people
train_X_outlier$nr_booking_changes[train_X_outlier$nr_booking_changes>3] <- NA
train_X_outlier$nr_booking_changes <- impute(train_X_outlier$nr_booking_changes, method = median)
outlier.cols <- append(outlier.cols, 'nr_booking_changes')



# 22) nr_children  
# starting from 98% outliers => from 2 (=valid) to 10 (= invalid)
train_X_outlier$nr_children[train_X$nr_children==10] <- NA
train_X_outlier$nr_children <- impute(train_X_outlier$nr_children, method = median)
outlier.cols <- append(outlier.cols, 'nr_children')

# 23) nr_nights
train_X_impute$nr_nights
nr_nights_z <- scale(train_X_impute$nr_nights)
quantile(nr_nights_z, na.rm = T, probs = seq(0, 1, 0.01))
quantile(train_X_impute$nr_nights, na.rm = T, probs = seq(0, 1, 0.01))

#########  NOG AAN TE PASSEN
# starting from 99% outliers => until 30 but valid I suppose (= a whole month)?
# ALso 69 days is about two months, perhaps business trips for consultants can last that long?
train_X_outlier$nr_nights[train_X$nr_nights>30] <- NA
train_X_outlier$nr_nights <- impute(train_X_outlier$nr_nights, method = median)
outlier.cols <- append(outlier.cols, 'nr_nights')

# 24) nr_previous_bookings  
#starting from 100% (78!!) trips, we have an outlier.
outlier.cols <- append(outlier.cols, 'nr_previous_bookings')


# 25) previous_bookings_not_canceled  
#starting from 100% (72!!), we have an outlier. Same comment as before
outlier.cols <- append(outlier.cols, 'previous_bookings_not_canceled')


# 26) previous_cancellations  
#starting from 100% (26!!), we have an outlier. Same comment as before
outlier.cols <- append(outlier.cols, 'previous_cancellations')



# 28) special_requests  
#starting from 98% (3), we have outliers
outlier.cols <- append(outlier.cols, 'special_requests')


# use this function to handle valid outliers
handle_outlier_z <- function(col){
  col_z <- scale(col)
  ifelse(abs(col_z)>3,
         sign(col_z)*3*attr(col_z,"scaled:scale") + attr(col_z,"scaled:center"), col)
}

# handle all the outliers at once
train_X_outlier[, outlier.cols] <-  sapply(train_X_impute[, outlier.cols], FUN = handle_outlier_z)

# We cannot use the previous method to handle outliers of the number of days in waiting list 
# This is because this variable has a lot of 0 values, and if we would use the z score, a lot of outliers would be identified
quantile(train_X_impute$days_in_waiting_list, na.rm = T, probs = seq(0, 1, 0.001))
# When we look at the distribution of the days in waiting list variable, we see that less than 1 % has a value higher than 125 days 
# We arbitrary set the boundary to be an outlier to 125
train_X_outlier$days_in_waiting_list <- ifelse(train_X_impute$days_in_waiting_list >= 125, 125, train_X_impute$days_in_waiting_list)


##############################################################
##############################################################
# 4. Parsing dates
##############################################################
##############################################################

# Parse arrival_date to filter out the month, day of the month and year
# for training set:
train_X_outlier$posix_arrival <- as.POSIXct(train_X_outlier$arrival_date, format="%B  %d  %Y")
train_X_outlier$day_of_month_arrival <- format(train_X_outlier$posix_arrival, format = '%d')
train_X_outlier$month_arrival <- format(train_X_outlier$posix_arrival, format = '%B')
train_X_outlier$year_arrival <- as.factor(format(train_X_outlier$posix_arrival, format = '%Y'))

# for test set:
test_X_outlier$posix_arrival <- as.POSIXct(test_X_outlier$arrival_date, format="%B  %d  %Y")
test_X_outlier$day_of_month_arrival <- format(test_X_outlier$posix_arrival, format = '%d')
test_X_outlier$month_arrival <- format(test_X_outlier$posix_arrival, format = '%B')
test_X_outlier$year_arrival <- as.factor(format(test_X_outlier$posix_arrival, format = '%Y'))


# last_status_date
# to impute NA we calculate the mean of the difference between
# arrival_date and last_status_date for each category
# add this value to arrival_date and impute in NA rows

#train
train_X_outlier$posix_last_status <- as.POSIXlt(train_X_outlier$last_status_date, format='%Y-%m-%dT %H:%M:%S')

mean_diff_canceled <- round(mean((difftime(train_X_outlier$posix_last_status[train_X_outlier$last_status=="Canceled"], train_X_outlier$posix_arrival[train_X_outlier$last_status=="Canceled"], units = "d")), na.rm = T))
mean_diff_check_out <- round(mean((difftime(train_X_outlier$posix_last_status[train_X_outlier$last_status=="Check-Out"], train_X_outlier$posix_arrival[train_X_outlier$last_status=="Check-Out"], units = "d")), na.rm = T))
mean_diff_no_show <- round(mean((difftime(train_X_outlier$posix_arrival[train_X_outlier$last_status=="No-Show"], train_X_outlier$posix_last_status[train_X_outlier$last_status=="No-Show"], units = "d")), na.rm = T))

train_X_outlier$posix_last_status[is.na(train_X_outlier$posix_last_status) & (train_X_outlier$last_status=="Canceled")] <- train_X_outlier$posix_arrival[is.na(train_X_outlier$posix_last_status) & (train_X_outlier$last_status=="Canceled")] + mean_diff_canceled
train_X_outlier$posix_last_status[is.na(train_X_outlier$posix_last_status) & (train_X_outlier$last_status=="Check-Out")] <- train_X_outlier$posix_arrival[is.na(train_X_outlier$posix_last_status) & (train_X_outlier$last_status=="Check-Out")] + mean_diff_check_out
train_X_outlier$posix_last_status[is.na(train_X_outlier$posix_last_status) & (train_X_outlier$last_status=="No-Show")] <- train_X_outlier$posix_arrival[is.na(train_X_outlier$posix_last_status) & (train_X_outlier$last_status=="No-Show")] + mean_diff_canceled

train_X_outlier$posix_last_status <- format(as.POSIXct(train_X_outlier$posix_last_status, format="%Y-%m-%d"), format="%Y-%m-%d")

#test
test_X_outlier$posix_last_status <- as.POSIXlt(test_X_outlier$last_status_date, format='%Y-%m-%dT %H:%M:%S')

test_X_outlier$posix_last_status[is.na(test_X_outlier$posix_last_status) & (test_X_outlier$last_status=="Canceled")] <- test_X_outlier$posix_arrival[is.na(test_X_outlier$posix_last_status) & (test_X_outlier$last_status=="Canceled")] + mean_diff_canceled
test_X_outlier$posix_last_status[is.na(test_X_outlier$posix_last_status) & (test_X_outlier$last_status=="Check-Out")] <- test_X_outlier$posix_arrival[is.na(test_X_outlier$posix_last_status) & (test_X_outlier$last_status=="Check-Out")] + mean_diff_check_out
test_X_outlier$posix_last_status[is.na(test_X_outlier$posix_last_status) & (test_X_outlier$last_status=="No-Show")] <- test_X_outlier$posix_arrival[is.na(test_X_outlier$posix_last_status) & (test_X_outlier$last_status=="No-Show")] + mean_diff_canceled

test_X_outlier$posix_last_status <- format(as.POSIXct(test_X_outlier$posix_last_status, format="%Y-%m-%d"), format="%Y-%m-%d")

##############################################################
##############################################################
# 5. Write data away
##############################################################
##############################################################
training_data_after_data_cleaning <- train_X_outlier
training_data_after_data_cleaning$average_daily_rate <- train_y

test_data_after_data_cleaning <- test_X_outlier

# inspect:
str(training_data_after_data_cleaning)
str(test_data_after_data_cleaning)

write.csv(training_data_after_data_cleaning,"./data/silver_data/train.csv", row.names = FALSE)
write.csv(test_data_after_data_cleaning,"./data/silver_data/test.csv", row.names = FALSE)