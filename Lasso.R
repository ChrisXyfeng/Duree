#Packages
set.seed(150912)
require(plyr)
require(dplyr)
require(caret)
require(data.table)
require(Matrix)
require(glmnet)
setwd("C:/Users/Mavis Xue Home/Desktop/Bigdata/Duree")
rm(list = ls())
gc()
train <- as.data.frame(fread("base_ano.txt") %>% 
        subset(duree >=1)) %>%
        select(age, sexe, diag, service, duree)
label_train <- train$duree
train$duree <- train$Selected <- train$moisent <- train$moissor <- train$entree <- train$sortie <- train$ID <- NULL

train <- cbind(age=train$age, as.data.frame(lapply(train[,-1], as.factor)))
#Interaction Diag+Service
train$ds <- paste0(as.character(train$diag), as.character(train$service))
train$ds <- as.factor(train$ds)
str(train)

matrix_train <- sparse.model.matrix(~age+sexe+diag+service, data = train)
#Model Lasso in package Glmnet

clf <- cv.glmnet(matrix_train, label_train,
          nfolds = 5,
          type.measure="mse")
plot(clf)
