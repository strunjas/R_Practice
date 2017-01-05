library(plyr)
library(car)
library(rpart)
library(lubridate)
library(date)
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(DiagrammeR)

getDate = function(given_date){
  return (as.Date(given_date))
}

getToBinAlreadyBooked = function(columnVal){
  if(columnVal=="" | is.na(columnVal))
    return(0)
  return(1)
}

getOverallWeight = function(label, label_matrix, overall_count){
  return (1 - label_matrix[label+1,1]/overall_count)
}

is.date <- function(x) inherits(as.Date(x), 'Date')

booking_info_train = read.csv("/Users/svetlana/Documents/Coding/NewUserBookings/train_users_2.csv",sep=",", header=TRUE)
booking_info_test = read.csv("/Users/svetlana/Documents/Coding/NewUserBookings/test_users.csv",sep=",", header=TRUE)

#booking_info_train$date_account_created_dt = sapply(booking_info_train$date_account_created, getString)
booking_info_train$date_account_created_dt = structure(sapply(booking_info_train$date_account_created, as.Date), class="Date")
min_acct_created_dt_train = min(booking_info_train$date_account_created_dt)
max_acct_created_dt_train = max(booking_info_train$date_account_created_dt)

booking_info_test$date_account_created_dt = structure(sapply(booking_info_test$date_account_created, as.Date), class="Date")
min_acct_created_dt_test = min(booking_info_test$date_account_created_dt)
max_acct_created_dt_test = max(booking_info_test$date_account_created_dt)

final_min_train = min(min_acct_created_dt_train, max_acct_created_dt_train)
final_max_train = max(min_acct_created_dt_train, max_acct_created_dt_train)

final_min_test = min(min_acct_created_dt_test, max_acct_created_dt_test)
final_max_test = max(min_acct_created_dt_test, max_acct_created_dt_test)



booking_info_train$num_days_account_exists = as.integer(final_max_train - booking_info_train$date_account_created_dt)

booking_info_test$num_days_account_exists = as.integer(final_max_test - booking_info_test$date_account_created_dt)

booking_info_train$previouslyBooked = as.factor(sapply(booking_info_train$date_first_booking, getToBinAlreadyBooked))
booking_info_test$previouslyBooked = as.factor(sapply(booking_info_test$date_first_booking, getToBinAlreadyBooked))

booking_info_train$signup_flow_factor = as.factor(booking_info_train$signup_flow)
booking_info_test$signup_flow_factor = as.factor(booking_info_test$signup_flow)

#booking_info_train$id = NULL
booking_info_train$date_account_created = NULL
booking_info_train$date_first_booking = NULL
booking_info_train$date_account_created = NULL
booking_info_train$date_account_created_dt = NULL

#booking_info_test$id = NULL
booking_info_test$date_account_created = NULL
booking_info_test$date_first_booking = NULL
booking_info_test$date_account_created = NULL
booking_info_test$date_account_created_dt = NULL

min(booking_info_train$age)
max(booking_info_train$age)



labels = booking_info_train['country_destination']
booking_info_train = booking_info_train[-grep('country_destination', colnames(booking_info_train))]
df_all = rbind(booking_info_train,booking_info_test)

df_all[df_all$age < 10 | df_all$age > 99 | is.na(df_all$age), 'age'] = -1
df_all$age[df_all$age < 0] = mean(df_all$age[df_all$age > 0])

ohe_feats = c('gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
              'signup_app', 'first_device_type', 'first_browser', 'previouslyBooked', 'signup_flow_factor')

dummies = dummyVars(~ gender +  signup_method + signup_flow + language + affiliate_channel + affiliate_provider +
                       first_affiliate_tracked + signup_app + first_device_type + first_browser + previouslyBooked +
                       signup_flow_factor, data = df_all)

df_all_ohe <- as.data.frame(predict(dummies, newdata = df_all))

df_all_combined <- cbind(df_all[,-c(which(colnames(df_all) %in% ohe_feats))],df_all_ohe)

X = df_all_combined[df_all_combined$id %in% booking_info_train$id,]
X_test = df_all_combined[df_all_combined$id %in% booking_info_test$id,]
labels$country_destination
labels$decoded = recode(labels$country_destination,"'NDF'=0; 'US'=1; 'FR'=2; 'CA'=3; 'GB'=4; 'ES'= 5; 'IT'=6; 'PT'=7; 'AU'= 8; 'NL'= 9; 'DE'=10; 'other'=11;")
labels$country_destination[labels$decoded == 11]

#y <- recode(labels$country_destination,"'NDF'=0; 'US'=1; 'FR'=2; 'CA'=3; 'GB'=4; 'ES'= 5; 'IT'=6; 'PT'=7; 'AU'= 8; 'NL'= 9; 'DE'=10; 'other'=11;")
y = as.integer(labels$country_destination)-1
min(as.integer(y))
max(as.integer(y))

xgb <- xgboost(data = data.matrix(X[,-1]), 
               label = y, 
               eta = 0.1,
               max_depth = 5, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "merror",
               objective = "multi:softprob",
               num_class = 12,
               nthread = 3
)



str(xgb)
y_pred <- predict(xgb, data.matrix(X_test[,-1]))
summary(y_pred)
model <- xgb.dump(xgb, with_stats = T)
model[1:10]
model
names <- dimnames(data.matrix(X[,-1]))[[2]]
importance_matrix <- xgb.importance(names, model = xgb)
# Nice graph
xgb.plot.importance(importance_matrix[1:50,])
#In case last step does not work for you because of a version issue, you can try following :
barplot(importance_matrix[,1])
