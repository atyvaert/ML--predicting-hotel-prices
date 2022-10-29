library(stringr)
arrival_date_split <- str_split_fixed(train$arrival_date, "  ", 3)
colnames(arrival_date_split) <- c("month","day","year")
train[c('month', 'day', 'year')] <- arrival_date_split

write.csv(train,"data/silver_data/train.csv", row.names = FALSE)
