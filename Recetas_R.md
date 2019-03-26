### Código de los principales algoritmos en R

Regresión lineal, regresión logística, árbol de decisión, SVM, Naive Bayes, kNN, k-Means, Random Forest,
GBM, XGBoost, LightGBM, Catboost

#### Regresión lineal
~~~
#Load Train and Test datasets
#Identify feature and response variable(s) and values must be numeric and numpy arrays
x_train <- input_variables_values_training_datasets
y_train <- target_variables_values_training_datasets
x_test <- input_variables_values_test_datasets
x <- cbind(x_train,y_train)
# Train the model using the training sets and check score
linear <- lm(y_train ~ ., data = x)
summary(linear)
#Predict Output
predicted= predict(linear,x_test) 
~~~

#### Regresión logística
~~~
x <- cbind(x_train,y_train)
# Train the model using the training sets and check score
logistic <- glm(y_train ~ ., data = x,family='binomial')
summary(logistic)
#Predict Output
predicted= predict(logistic,x_test)
~~~

#### Árbol de decisión
~~~
library(rpart)
x <- cbind(x_train,y_train)
# grow tree 
fit <- rpart(y_train ~ ., data = x,method="class")
summary(fit)
#Predict Output 
predicted= predict(fit,x_test)
~~~

#### SVM (Support Vector Machine)
~~~
library(e1071)
x <- cbind(x_train,y_train)
# Fitting model
fit <-svm(y_train ~ ., data = x)
summary(fit)
#Predict Output 
predicted= predict(fit,x_test)
~~~

#### Naive Bayes
~~~
library(e1071)
x <- cbind(x_train,y_train)
# Fitting model
fit <-naiveBayes(y_train ~ ., data = x)
summary(fit)
#Predict Output 
predicted= predict(fit,x_test)
~~~

#### kNN (k-Nearest Neighbors)
~~~
library(knn)
x <- cbind(x_train,y_train)
# Fitting model
fit <-knn(y_train ~ ., data = x,k=5)
summary(fit)
#Predict Output 
predicted= predict(fit,x_test)
~~~

#### K-Means
~~~
library(cluster)
fit <- kmeans(X, 3) # 5 cluster solutio
~~~

#### Random Forest
~~~
library(randomForest)
x <- cbind(x_train,y_train)
# Fitting model
fit <- randomForest(Species ~ ., x,ntree=500)
summary(fit)
#Predict Output 
predicted= predict(fit,x_test)
~~~

#### GBM
~~~
library(caret)
x <- cbind(x_train,y_train)
# Fitting model
fitControl <- trainControl( method = "repeatedcv", number = 4, repeats = 4)
fit <- train(y ~ ., data = x, method = "gbm", trControl = fitControl,verbose = FALSE)
predicted= predict(fit,x_test,type= "prob")[,2] 
~~~

#### XGBoost
~~~
require(caret)
x <- cbind(x_train,y_train)
# Fitting model
TrainControl <- trainControl( method = "repeatedcv", number = 10, repeats = 4)
model<- train(y ~ ., data = x, method = "xgbLinear", trControl = TrainControl,verbose = FALSE)
OR 
model<- train(y ~ ., data = x, method = "xgbTree", trControl = TrainControl,verbose = FALSE)
predicted <- predict(model, x_test)
~~~

#### LightGBM
~~~
require(caret)
require(RLightGBM)
data(iris)

model <-caretModel.LGBM()

fit <- train(Species ~ ., data = iris, method=model, verbosity = 0)
print(fit)
y.pred <- predict(fit, iris[,1:4])

library(Matrix)
model.sparse <- caretModel.LGBM.sparse()

#Generate a sparse matrix
mat <- Matrix(as.matrix(iris[,1:4]), sparse = T)
fit <- train(data.frame(idx = 1:nrow(iris)), iris$Species, method = model.sparse, matrix = mat, verbosity = 0)
print(fit)
~~~

#### Catboost
~~~
set.seed(1)
require(titanic)
require(caret)
require(catboost)

tt <- titanic::titanic_train[complete.cases(titanic::titanic_train),]
data <- as.data.frame(as.matrix(tt), stringsAsFactors = TRUE)
drop_columns = c("PassengerId", "Survived", "Name", "Ticket", "Cabin")
x <- data[,!(names(data) %in% drop_columns)]y <- data[,c("Survived")]
fit_control <- trainControl(method = "cv", number = 4,classProbs = TRUE)
grid <- expand.grid(depth = c(4, 6, 8),learning_rate = 0.1,iterations = 100, l2_leaf_reg = 1e-3,            rsm = 0.95, border_count = 64)
report <- train(x, as.factor(make.names(y)),method = catboost.caret,verbose = TRUE, preProc = NULL,tuneGrid = grid, trControl = fit_control)
print(report)
importance <- varImp(report, scale = FALSE)
print(importance)
~~~

**Referencias**: https://www.analyticsvidhya.com/
