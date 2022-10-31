##############################################################
##############################################################
# Feature engineering
##############################################################
##############################################################


# import data
setwd(dir = '/Users/Artur/Desktop/uni jaar 6 sem 1/machine learning/ml22-team10/data/silver_data')
train <- read.csv('./train.csv')
test_X <- read.csv('./test.csv')

# separate dependent and independent variables
train_X <- subset(train, select = -c(average_daily_rate))
train_y <- train$average_daily_rate

# inspect
str(train_X)
str(test_X)

# libraries


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

# get categories and dummies
# we only select the top 10 levels with highest frequency so our model does not explode
# For all cases, this includes most of the data: KIJKEN ALS DIT KLOPT BIJ DE REST
cats <- categories(train_X_encode[, c('assigned_room_type', 'booking_distribution_channel',
                                      'canceled', 'country')], p = 10)

# apply on train set (exclude reference categories)
dummies_train <- dummy(train_X_encode[,c('assigned_room_type', 'booking_distribution_channel', 
                                         'canceled', 'country')], object = cats)

# exclude the reference category: take the first one of the variable you added
names(dummies_train)
dummies_train <- subset(dummies_train, 
                        select = -c(assigned_room_type_A, booking_distribution_channel_TA.TO,
                                    country_Belgium, canceled_no.cancellation))

# apply on test set (exclude reference categories)
# excluded no.canceled so it becomes one when it was canceled
dummies_test <- dummy(test_X_encode[, c('assigned_room_type', 'booking_distribution_channel', 
                                        'canceled', 'country')], object = cats)
dummies_test <- subset(dummies_test, select = -c(assigned_room_type_A, booking_distribution_channel_TA.TO,
                                                 country_Belgium, canceled_no.cancellation))

# we remove the original predictors and merge them with the other predictors
## merge with overall training set
train_X_encode <- subset(train_X_encode, select = -c(assigned_room_type, booking_distribution_channel,
                                                     canceled, country))
train_X_encode <- cbind(train_X_encode, dummies_train)
## merge with overall test set
test_X_encode <- subset(test_X_encode, select = -c(assigned_room_type, booking_distribution_channel,
                                                   canceled, country))
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



















