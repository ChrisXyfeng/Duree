#Packages
set.seed(150912)
require(plyr)
require(dplyr)
require(data.table)
require(Matrix)
require(glmnet)
require(xgboost)
setwd("C:/Documents and Settings/xueyu/Bureau/BD/Duree")

rm(list = ls())
gc()
train <- as.data.frame(fread("base_ano.txt") %>% 
                         subset(duree >=1)) %>%
                         select(age, sexe, diag, service, duree)
label_train <- train$duree
train$duree <- train$Selected <- train$moisent <- train$moissor <- train$entree <- train$sortie <- train$ID <- NULL

#Interaction Diag + Service
train$ds <- paste0(as.character(train$diag), as.character(train$service))

catimpact <- function(xcol, targetcol) {
  x_impact <- rep(0, length(xcol))
  for (i in unique(xcol)) {
    x_impact[xcol == i] <- mean(targetcol[xcol == i]) 
  }
  return(x_impact)
}
train$ds <- catimpact(xcol = train$ds, targetcol = label_train)
train$diag <- catimpact(xcol = train$diag, targetcol = label_train)
train$service <- catimpact(xcol = train$service, targetcol = label_train)

xgtrain <-  xgb.DMatrix(as.matrix(train),label = label_train, missing=NA)

param0 <- list(
  # some generic, non specific params
  "objective"  = "reg:linear",
  "eval_metric" = "rmse",
  "eta" = 0.1,
  "subsample" = 0.9,
  "colsample_bytree" = 0.9,
  "min_child_weight" = 1,
  "max_depth" = 1
)

model_cv <- xgb.cv(
  params = param0,
  nrounds = 500,
  nfold = 2,
  data = xgtrain,
  early.stop.round = 3,
  maximize = FALSE,
  verbose = TRUE
)


best <- min(model_cv$test.rmse.mean)
bestIter <- which(model_cv$test.rmse.mean==best)    

model.xgb <-  xgb.train(
  nrounds = bestIter,
  params = param0,
  data = xgtrain
)

pred <- predict(model.xgb, xgtrain)
qplot(x= label_train, y=pred, ylim = c(0,500))
qplot(x= label_train, y=pred, ylim = c(0,100), xlim = c(0,100))

#Lasso
clf <- cv.glmnet(as.matrix(train), label_train,
                 nfolds = 5,
                 type.measure="mse")
plot(clf)
pred_lasso <- predict(clf, as.matrix(train))
qplot(x= label_train, y=as.numeric(pred_lasso), ylim = c(0,500))
qplot(x= label_train, y=as.numeric(pred_lasso), ylim = c(0,100), xlim = c(0,100))
sqrt(clf$cvm)
