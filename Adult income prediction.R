



library(tidyr)
library(tidyverse)
library(caret)
library(magrittr)
library(randomForest)
options(digits = 6)


################################
# Get Data
################################
# Census Income data:
# https://archive.ics.uci.edu/ml/datasets/adult
# https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data

dl <- tempfile()
download.file("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", dl)
adult_income <- read.table(dl, sep = ',', fill = F, strip.white = T) %>% set_colnames( 
  c('age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'))

head(adult_income)

mean(adult_income$capital_gain == 0) 
mean(adult_income$capital_loss == 0)
mean(adult_income$native_country == "United-States")
# we can observe that above 90% adult have zero capital_gain and capital_loss, and about 90% adult come from united states. therefore,these three variables are skew. so delete them.
# regards education, it means same as education_num, relationship is same as marital status. Fnlwgt is not related to our goal.so delete 
adult <- adult_income %>% select(-education, -fnlwgt, -relationship, -capital_gain, -capital_loss, native_country )
head(adult)
dim(adult)
################################
# Explore and Clean Data
################################
#Trim workclass column
#combine Never-worked and Without-pay to unknown;
#combine federal-gov, state-gov, and local-gov levels to government. 
#combine self-emp-inc and self-emp-not-inc to self-employed


table(adult$workclass)
levels(adult$workclass)[c(1, 4, 9)] <- "Unknown"
levels(adult$workclass)[c(5, 6)] <- "Self_Employed"
levels(adult$workclass)[c(2, 3, 6)] <- "Government"
table(adult$workclass)

#explore the workclass and education_num
head(adult)
adult %>% group_by(workclass, income) %>% summarize(n = n()) %>% ggplot(aes(workclass, n, fill = income)) + geom_bar(stat = "identity")
#Those who are self employed have the highest tendency of making greater than $50,000 a year.
adult %>% group_by(education_num, income) %>% summarize(n = n()) %>% ggplot(aes(education_num, n, fill = income)) + geom_bar(stat = "identity")
#the in group proportion of making greater than $50,000 a year increase as the years of education increases

#Trim occupation column
#block the occupation into several groups:Blue-Collar, Professional, Sales, Service, and  White-Collar
table(adult$occupation)
levels(adult$occupation)[1] <- "Unknown"
levels(adult$occupation)[c(2, 5)] <- "White_Collar"
levels(adult$occupation)[c(4, 5, 6, 7, 14 )] <- "Blue_Collar"
levels(adult$occupation)[c(5, 6, 8, 10 )] <- "Service"
levels(adult$occupation)[c(6)] <- "Professional"
levels(adult$occupation)[c(1, 3)] <- "Unknown"
table(adult$occupation)

adult %>% group_by(occupation, income) %>% summarize(n = n()) %>% ggplot(aes(occupation, n, fill = income)) + geom_bar(stat = "identity")
#Nearly half of Professional occupation makes greater than $50,000 a year, while that percentage is only 13% for Service occupation.

#Trim marital_status column
#block the marital_status into Divorced, married, seperated, single, and widowed
table(adult$marital_status)
levels(adult$marital_status)[c(2, 3, 4)] <- "Married"
levels(adult$marital_status)[3] <- "Single"
table(adult$marital_status)

adult %>% group_by(marital_status, income) %>% summarize(n = n()) %>% ggplot(aes(marital_status, n, fill = income)) + geom_bar(stat = "identity")
#For those who are married, nearly half of them are making greater than $50,000 a year.

#Explore race
adult %>% group_by(race, income) %>% summarize(n = n()) %>% ggplot(aes(race, n, fill = income)) + geom_bar(stat = "identity")

# White and Asian-Pacific Islander have high earning potentials – over 25% of the observations of these 2 races make above $50,000 annually.


################################
# Modeling 
################################

# create train_set and test_set 

set.seed(1)
test_index <- createDataPartition(y = adult$age, times = 1, p = 0.2, list = FALSE)
train_set <- adult[-test_index,]
test_set <- adult[test_index,]

#############################
#test logistic regression

fit_glm <- train(income ~ ., method = "glm", data = train_set) #1.5 min

y_hat_glm <- predict(fit_glm, test_set)
Accuracy_glm <- confusionMatrix(y_hat_glm, test_set$income)$overall["Accuracy"]
Accuracy_results <- tibble(method = "logistic regression", Accuracy = Accuracy_glm)
print.data.frame(Accuracy_results)

#############################
#Random forest method
#choose a optimal mtry value
set.seed(1)
mtrytune <- tuneRF(adult[,-10], adult[,10], stepFactor=1.5)
plot(mtrytune)

# test the random forest model
set.seed(1)
my_control <- trainControl(method = "cv", number = 5)
fit_rf <- train(y = train_set[,10], x =train_set[,1:9], method = "rf", ntree = 1000, trControl = my_control)
fit_rf$results
y_hat_rf <- predict(fit_rf, test_set)
Accuracy_rf <- confusionMatrix(y_hat_rf, test_set$income)$overall["Accuracy"]
Accuracy_results <- bind_rows(Accuracy_results,
                              tibble(method="Random forest",  
                                     Accuracy = Accuracy_rf))

#########################################################################
#Finisch :show the RMSE results
#########################################################################
                                                        
print.data.frame(Accuracy_results) 

