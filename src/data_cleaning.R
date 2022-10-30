library(readr)
library(stringr)

#READ
train <- read_csv("data/raw_data/train.csv")
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

# tot hier done

#19 impute nr_adults with mode
train_X_impute <- train_X
train_X_impute$nr_adults[is.na(train_X_impute$nr_adults)] <- getmode(train_X$nr_adults)
train_X <- train_X_impute

#19 impute nr_adults with mode
train_X_impute <- train_X
train_X_impute$nr_adults[is.na(train_X_impute$nr_adults)] <- getmode(train_X$nr_adults)
train_X <- train_X_impute

#--------





#WRITE
write.csv(train,"data/silver_data/train.csv", row.names = FALSE)
