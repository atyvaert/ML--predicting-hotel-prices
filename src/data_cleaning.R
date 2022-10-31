##############################################################
##############################################################
# Data Cleaning
##############################################################
##############################################################

# packages and libraries needed
library(readr)
library(dummy)
library(stringr)

# import the data
setwd(dir = '/Users/Artur/Desktop/uni jaar 6 sem 1/machine learning/ml22-team10/data/bronze_data')
train <- read.csv('./train.csv')
test_X <- read.csv('./test.csv')

# for Viktor:
# train <- read_csv("C:/Users/vikto/OneDrive - UGent/TweedeMaster/MachineLearning/ML_Team10/data/raw_data/train.csv")
# train <- read.csv("data/raw_data/train.csv")

# separate dependent and independent variables
train_X <- subset(train, select = -c(average_daily_rate))
train_y <- train$average_daily_rate
train_y <- as.numeric(word(train_y, 1))


##############################################################
##############################################################
# 1. Adjust structure + Parse dates
##############################################################
##############################################################
# Parse arrival_date to filter out the month, day of the month and year
train_X$arrival_date
train_X$posix <- as.POSIXct(train_X$arrival_date, format="%B  %d  %Y")
train_X$day_of_month <- format(train_X$posix, format = '%d')
train_X$month <- format(train_X$posix, format = '%B')
train_X$year <- format(train_X$posix, format = '%Y')

#16 lead_time to integer number of days
train_X$lead_time <- substr(train_X$lead_time, start = 1, stop = (nchar(train_X$lead_time)-7))
train_X$lead_time <- as.numeric(train_X$lead_time)
train_X$lead_time


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
train_X_impute <- cbind(train_X_impute, naFlag(df = train_X_impute))
test_X_impute <- cbind(test_X_impute,naFlag(df = test_X_impute, df_val = train_X_impute))

# inspect
str(train_X_impute)
# str(test_X_impute)

# 2.2.1 Impute NUMERIC variables with information from the training set

# use the 'impute' function for numeric predictors:
impute <- function(x, method = mean, val = NULL){
  if(is.null(val)){
    val <- method(x, na.rm = T)
  }
  x[is.na(x)] <- val
  return(x)
}

num.cols <- c('car_parking_spaces', 'days_in_waiting_list')
train_X_impute[, num.cols] <- lapply(train_X_impute[, num.cols], 
                                     FUN = impute,
                                     method = mean)

quantile(train_X_impute$car_parking_spaces, na.rm = T, probs = seq(0, 1, 0.01))

test_X_impute[, num.cols] <- mapply(test_X_impute[, num.cols],
                                    FUN = impute,
                                    val = colMeans(train_X[, num.cols], na.rm = T))


# 2.2.1 Impute CATEGORICAL variables with information from the training set
# use the 'modus' function for categorical predictors. This returns the mode of a column.
modus <- function(x, na.rm = FALSE) {
  if (na.rm) x <- x[!is.na(x)]
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

# handle
cat.cols <- c('booking_distribution_channel', 'country', 'customer_type', 'deposit', 'hotel_type',
              'is_repeated_guest', 'last_status', 'market_segment', 'nr_adults', 'nr_children',
              'nr_previous_bookings', 'previous_bookings_not_canceled', 'previous_cancellations')
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

# make a vector of all the variables of which valid outliers need to be handled
outlier.cols <- c()
# look at all the numeric variables and detect valid and invalid outliers:
# 1) car_parking_spaces
car_parking_spaces_z <- scale(train_X_impute$car_parking_spaces)
quantile(car_parking_spaces_z, na.rm = T, probs = seq(0, 1, 0.01))
# hier zien we wel wat outliers, met een maximum van 3 parking spaces
# dit lijken valid outliers (3 parking spaces required by customer is possible)
# add car_parking_spaces to variables that need to be handled
outlier.cols <- append(outlier.cols, 'car_parking_spaces')




# use this function to handle valid outliers
handle_outlier_z <- function(col){
  col_z <- scale(col)
  ifelse(abs(col_z)>3,
         sign(col_z)*3*attr(col_z,"scaled:scale") + attr(col_z,"scaled:center"), col)
}

# handle all the outlier at once
train_X_outlier[, outlier.cols] <-  sapply(train_X_impute[, outlier.cols], FUN = handle_outlier_z)



#WRITE
write.csv(train,"data/silver_data/train.csv", row.names = FALSE)
write.csv(train_X,"data/silver_data/train_X.csv", row.names = FALSE)
write.csv(train_y,"data/silver_data/train_y.csv", row.names = FALSE)


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
