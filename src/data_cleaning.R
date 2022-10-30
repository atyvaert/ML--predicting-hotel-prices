library(readr)
library(dummy)

library(stringr)

#READ
# train <- read_csv("C:/Users/vikto/OneDrive - UGent/TweedeMaster/MachineLearning/ML_Team10/data/raw_data/train.csv")
train <- read.csv("data/raw_data/train.csv")
train_X <- subset(train, select = -c(average_daily_rate))
train_y <- train$average_daily_rate

#Create getmode function
# Create the function.
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

#1 arrival_date to Date
train_X$arrival_date <- parse_date(train_X$arrival_date, format = "%B  %d  %Y")

#16 lead_time to integer number of days and impute NA
train_X$lead_time <- substr(train_X$lead_time, start = 1, stop = (nchar(train_X$lead_time)-7))
train_X_impute <- train_X
train_X_impute$lead_time <- as.numeric(train_X$lead_time)
train_X_impute$lead_time[is.na(train_X_impute$lead_time)] <- round(mean(as.integer(train_X$lead_time), na.rm = TRUE))
train_X <- train_X_impute




#CUSTOMER TYPE, DAYS IN WAITING LIST, DEPOSIT, HOTEL TYPE, IS_REPEATED_GUEST, LAST_STATUS
#IMPUTING
# create function to impute missing values 
impute <- function(x, method = mean, val = NULL) {
  if (is.null(val)) {
    val <- method(x, na.rm = TRUE)
  }
  x[is.na(x)] <- val
  return(x)
}
# create a function to calculate the mode
modus <- function(x, na.rm = FALSE) {
  if (na.rm) x <- x[!is.na(x)]
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}
# impute missing values with the mode of the customer type column 
train_X_impute$customer_type <- impute(train_X_impute$customer_type, method = modus)
train_X_impute$customer_type
# impute missing values with the mean of the days in waiting lest column 
train_X_impute$days_in_waiting_list <- impute(train_X_impute$days_in_waiting_list, method = mean)
train_X_impute$days_in_waiting_list
# impute missing values with the mode of the deposit column
train_X_impute$deposit <- impute(train_X_impute$deposit, method = modus)
train_X_impute$deposit
# impute missing values with the mode of the hotel type column 
train_X_impute$hotel_type <- impute(train_X_impute$hotel_type, method = modus)
train_X_impute$hotel_type
# impute missing values with the mode of the is repeated guest column 
train_X_impute$is_repeated_guest <- impute(train_X_impute$is_repeated_guest, method = modus)
train_X_impute$is_repeated_guest
# impute missing values with the mode of the last_status column 
train_X_impute$last_status <- impute(train_X_impute$last_status, method = modus)
train_X_impute$last_status

#ENCODING 
# get all columns that are categorical variables 
train_X_encode <- train_X_impute[, c("customer_type", "deposit", "hotel_type", "is_repeated_guest","last_status" )]
# create dataframe
df_merge <- cbind(X = train_X_encode, y = train_y)
# encode the variables
train_X_encode <- model.matrix(y ~ . - 1, data = df_merge)
train_X_encode


# rm(list=ls()) 

#17 impute market_segment with mode
train_X_impute <- train_X
train_X_impute$market_segment[is.na(train_X_impute$market_segment)] <- getmode(train_X$market_segment)
train_X <- train_X_impute

#19 impute nr_adults with mode
train_X_impute <- train_X
train_X_impute$nr_adults[is.na(train_X_impute$nr_adults)] <- getmode(train_X$nr_adults)
train_X <- train_X_impute

#20 impute nr_babies: "n/a" -> 0
train_X_impute <- train_X
train_X_impute$nr_babies <- as.numeric(str_replace_all(train_X$nr_babies, "n/a", "0"))
train_X <- train_X_impute

#---------
#21 impute nr_booking_changes: "n/a" -> 0
train_X_impute <- train_X
train_X_impute$nr_booking_changes[is.na(train_X_impute$nr_booking_changes)] <- 0
train_X <- train_X_impute

#22 impute nr_children with mode (0)       
train_X_impute <- train_X
train_X_impute$nr_children[is.na(train_X_impute$nr_children)] <- getmode(train_X$nr_children)
train_X <- train_X_impute

#23 impute nr_nights
# No Na values!

#24 impute nr_previous_bookings with mode (0)  
train_X_impute <- train_X
train_X_impute$nr_previous_bookings[is.na(train_X_impute$nr_previous_bookings)] <- getmode(train_X$nr_previous_bookings)
train_X <- train_X_impute

#25 impute previous_bookings_not_canceled with mode (0) 
train_X_impute <- train_X
train_X_impute$previous_bookings_not_canceled[is.na(train_X_impute$previous_bookings_not_canceled)] <- getmode(train_X$previous_bookings_not_canceled)
train_X <- train_X_impute

#26 impute previous_cancellations with mode (0)   
train_X_impute <- train_X
train_X_impute$previous_cancellations[is.na(train_X_impute$previous_cancellations)] <- getmode(train_X$previous_cancellations)
train_X <- train_X_impute

#27 reserved_room_type  
# No Na values!

#28 special_requests  
# No Na values!




#--------




#train_y
train_y <- as.double(gsub('.{6}$', '', train_y))

#WRITE
write.csv(train,"data/silver_data/train.csv", row.names = FALSE)
write.csv(train_X,"data/silver_data/train_X.csv", row.names = FALSE)
write.csv(train_y,"data/silver_data/train_y.csv", row.names = FALSE)
