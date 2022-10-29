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




































