library(dplyr)
library(stringr)
library(data.table)
library(MASS)
library(randomForest)
library(purrr)
library(caret)
library(visdat)
library(tidyr)


setwd("C:/Edvancer/Data/")
#Read Data
train_df <- fread("titanic_train.csv")
test_df <- fread("titanic_test.csv")

test_df$Survived=NA
train_df$data="train"
test_df$data="test"

final=rbind(train_df,test_df)

#Data info training data
summary(final)

table(final$Survived)

vis_dat(final)

#Cleaning Data


null_cols <- which(colSums(is.na(final))>0)

no_missingval <- sum(is.na(final$Age))

## Handling NA's

final$Age <- replace_na(final$Age, mean(final$Age, na.rm = TRUE))
final$Age <- round(final$Age)


final$Fare <- replace_na(final$Fare, mean(final$Fare, na.rm = TRUE))


vis_dat(final)


#Name Variable  data

final <- final %>%
  mutate(Title_name = sub(".*,", "", final$Name))

final$Title_name=trimws(final$Title_name)
final$Title_name=word(final$Title_name,1)

cnt=as.data.frame(table(final$Title_name))

table(final$Survived,final$Title_name)

for(i in 1:length(final$Title_name)){
  if(final$Title_name[i] %in% c("Mr.", "Mrs.", "Miss.", "Mrs.", "Master."))
    final$Title_name[i] <- str_replace_all(final$Title_name[i]," ","")
  else
    final$Title_name[i] <- "Other"
}


#Ticket variable train data

final$Ticket=NULL

#Variables Dummification train data

cat_variables <- c("Pclass", "Sex", "SibSp", "Embarked",
                   "Title_name")

CreateDummies=function(data,var,freq_cutoff=0){
  t=table(data[,var])
  t=t[t>freq_cutoff]
  t=sort(t)
  categories=names(t)[-1]
  
  for( cat in categories){
    name=paste(var,cat,sep="_")
    name=gsub(" ","",name)
    name=gsub("-","_",name)
    name=gsub("\\?","Q",name)
    name=gsub("<","LT_",name)
    name=gsub("\\+","",name)
    name=gsub("\\/","_",name)
    name=gsub(">","GT_",name)
    name=gsub("=","EQ_",name)
    name=gsub(",","",name)
    data[,name]=as.numeric(data[,var]==cat)
  }
  
  data[,var]=NULL
  return(data)
}


final=as.data.frame(final)


for(cat in cat_variables){
  final=CreateDummies(final,cat,10)
}


train_df=final[final$data=="train",]
test_df=final[final$data=="test",]
train_df$data=NULL
test_df$data=NULL

test_df$Survived=NULL


# pkgs <- c("RCurl","jsonlite")
# for (pkg in pkgs) {
#   if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
# }

#options(timeout = max(1000, getOption("timeout")))

#install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")))


library(h2o)

# Start the H2O cluster (locally)
h2o.init(nthreads = -1)

# Import a sample binary outcome train/test set into H2O
train <- as.h2o(train_df)
test <- as.h2o(test_df)

y <- "Survived"
x <- setdiff(names(train), y)

# For binary classification, response should be a factor
train[, y] <- as.factor(train[, y])


# Run AutoML for 20 base models
aml <- h2o.automl(x = x, y = y,
                  training_frame = train,
                  max_models = NULL,
                  seed = 1,verbosity = ,)

lb <- aml@leaderboard
print(lb, n = nrow(lb)) 

m <- h2o.get_best_model(aml)

m <- h2o.getModel("StackedEnsemble_BestOfFamily_2_AutoML_5_20220626_25002")


x=as.data.frame(h2o.predict(m,test))

x$PassengerId=test_df$PassengerId

colnames(x)[1]="Survived"

x=x[,c("PassengerId","Survived")]


write.csv(x,"C:/Edvancer/submit_3.csv",row.names = F)

