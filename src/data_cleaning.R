library(readr)

#READ
train <- read_csv("data/raw_data/train.csv")
train_X <- subset(train, select = -c(average_daily_rate))
train_y <- train$average_daily_rate
#arrival_date to Date
train_X$arrival_date <- parse_date(train_X$arrival_date, format = "%B  %d  %Y")

#lead_time to integer number of days and impute NA
train_X$lead_time <- substr(train_X$lead_time, start = 1, stop = (nchar(train_X$lead_time)-7))
train_X_impute <- train_X
train_X_impute$lead_time <- as.integer(train_X$lead_time)
train_X_impute$lead_time[is.na(train_X_impute$lead_time)] <- round(mean(as.integer(train_X$lead_time), na.rm = TRUE))
train_X <- train_X_impute

#WRITE
write.csv(train,"data/silver_data/train.csv", row.names = FALSE)
