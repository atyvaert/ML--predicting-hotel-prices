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
# setwd(dir = '/Users/Artur/Desktop/uni jaar 6 sem 1/machine learning/ml22-team10/data/bronze_data')
train <- read.csv('./train.csv', )
train <- read_csv('./data/bronze_data/train.csv', )

# for Viktor:
# setwd(dir = 'C:/Users/vikto/OneDrive - UGent/TweedeMaster/MachineLearning/ML_Team10/data/bronze_data')
# train <- read.csv("/train.csv", fileEncoding = 'latin1')
# test_X <- read.csv('./test.csv')

# separate dependent and independent variables
train_X <- subset(train, select = -c(average_daily_rate))
train_y <- train$average_daily_rate
train_y <- as.numeric(word(train_y, 1))


##############################################################
##############################################################
# 1. Adjust structure
##############################################################
##############################################################
# VRAAG: MOET DIT OOK VOOR TEST SET


#lead_time to integer number of days for training set
train_X$lead_time <- substr(train_X$lead_time, start = 1, stop = (nchar(train_X$lead_time)-7))
train_X$lead_time <- as.numeric(train_X$lead_time)

# for test set
test_X$lead_time <- substr(test_X$lead_time, start = 1, stop = (nchar(test_X$lead_time)-7))
test_X$lead_time <- as.numeric(test_X$lead_time)


# convert the necessary characters to numeric



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


# impute numerical variables with the mean
num.cols <- c('days_in_waiting_list', 'lead_time')
train_X_impute[, num.cols] <- lapply(train_X_impute[, num.cols], 
                                     FUN = impute,
                                     method = mean)

test_X_impute[, num.cols] <- mapply(test_X_impute[, num.cols],
                                    FUN = impute,
                                    val = colMeans(train_X[, num.cols], na.rm = T))


# impute categorical variables with the median
median.cols <- c('car_parking_spaces', 'nr_adults', 'nr_children', 'nr_previous_bookings',
                 'previous_bookings_not_canceled', 'previous_cancellations')
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
# 3. Detecting and handling outliers
##############################################################
##############################################################
train_X_outlier <- train_X_impute
test_X_outlier <- test_X_impute

# VRAAG: OOK NODIG VOOR TEST SET?
# Zei in de les van niet

# make a vector of all the variables of which valid outliers need to be handled
outlier.cols <- c()
outlier.cols <- append(outlier.cols, '')

# look at all the numeric variables and detect valid and invalid outliers:
# 1) car_parking_spaces
car_parking_spaces_z <- scale(train_X_impute$car_parking_spaces)
quantile(car_parking_spaces_z, na.rm = T, probs = seq(0, 1, 0.01))

# hier zien we wel wat outliers, met een maximum van 3 parking spaces
# dit lijken valid outliers (3 parking spaces required by customer is possible)
# add car_parking_spaces to variables that need to be handled
outlier.cols <- append(outlier.cols, 'car_parking_spaces')

# All the car parking spaces have a value between 0 and 3, with the majority being 0
# as 1, 2 and 3 are seen as outliers, we bring back these values to one as these are 
# valid outliers and this indicates if a parking place was required
# VRAAG: HIER OF BIJ FEATURE ENGINEERING



# use this function to handle valid outliers
handle_outlier_z <- function(col){
  col_z <- scale(col)
  ifelse(abs(col_z)>3,
         sign(col_z)*3*attr(col_z,"scaled:scale") + attr(col_z,"scaled:center"), col)
}

# handle all the outlier at once
train_X_outlier[, outlier.cols] <-  sapply(train_X_impute[, outlier.cols], FUN = handle_outlier_z)

# We cannot use the previous method to handle outliers of the number of days in waiting list 
# This is because this variable has a lot of 0 values, and if we would use the z score, a lot of outliers would be identified
quantile(train_X_impute$days_in_waiting_list, na.rm = T, probs = seq(0, 1, 0.001))
# When we look at the distribution of the days in waiting list variable, we see that less than 1 % has a value higher than 125 days 
# We arbitrary set the boundary to be an outlier to 125
# We write a function to identify outliers 
handle_outlier_daysInWaitingList <- function(column) {
  ifelse(column>125,
         125 , column)
}
# applying the function 
daysInWaitingList_col <- c('days_in_waiting_list')
train_X_outlier$days_in_waiting_list <-  sapply(train_X_impute[,daysInWaitingList_col], FUN = handle_outlier_daysInWaitingList)
train_X_outlier$days_in_waiting_list
# in the boxplot you can see that all values above 125 are gone
boxplot(train_X_outlier$days_in_waiting_list)


##############################################################
##############################################################
# 4. Parsing dates
##############################################################
##############################################################
# VRAAG: MOET DIT OOK VOOR TEST SET? denk het wel
# HIER WEL AANGEZIEN DEZE DATA NORMAAL OOK ZO BINNEN KOMT


# Parse arrival_date to filter out the month, day of the month and year
# for training set:
train_X_outlier$posix_arrival <- as.POSIXct(train_X_outlier$arrival_date, format="%B  %d  %Y")
train_X_outlier$day_of_month_arrival <- format(train_X_outlier$posix_arrival, format = '%d')
train_X_outlier$month_arrival <- format(train_X_outlier$posix_arrival, format = '%B')
train_X_outlier$year_arrival <- format(train_X_outlier$posix_arrival, format = '%Y')

# parse last_status_date
train_X_outlier$posix_last_status <- as.POSIXct(train_X_outlier$last_status_date, format="%Y-%m-%d")

# for test set:
test_X_outlier$posix_arrival <- as.POSIXct(test_X_outlier$arrival_date, format="%B  %d  %Y")
test_X_outlier$day_of_month_arrival <- format(test_X_outlier$posix_arrival, format = '%d')
test_X_outlier$month_arrival <- format(test_X_outlier$posix_arrival, format = '%B')
test_X_outlier$year_arrival <- format(test_X_outlier$posix_arrival, format = '%Y')

test_X_outlier$posix_last_status <- as.POSIXct(test_X_outlier$last_status_date, format="%Y-%m-%d")




#WRITE
training_data_after_data_cleaning <- train_X_outlier
training_data_after_data_cleaning$average_daily_rate <- train_y

test_data_after_data_cleaning <- test_X_outlier

# inspect:
print(training_data_after_data_cleaning)
str(test_data_after_data_cleaning)

setwd(dir = 'C:/Users/vikto/OneDrive/Documenten/GroepswerkMachineLearning/ml22-team10/data/silver_data')
write.csv(training_data_after_data_cleaning,"/train.csv", row.names = FALSE)
write.csv(test_data_after_data_cleaning,"/test.csv", row.names = FALSE)
#write.csv(train_X,"data/silver_data/train_X.csv", row.names = FALSE)
#write.csv(train_y,"data/silver_data/train_y.csv", row.names = FALSE)


##########@
# HOORT BIJ FE EN NIET HIER
###########


#ENCODING 
# get all columns that are categorical variables 
train_X_encode <- train_X_impute[, c("customer_type", "deposit", "hotel_type", "is_repeated_guest","last_status" )]
# create dataframe
df_merge <- cbind(X = train_X_encode, y = train_y)
# encode the variables
train_X_encode <- model.matrix(y ~ . - 1, data = df_merge)
train_X_encode
