Project Writeup for Practical Machine Learning course
========================================================
Here is my writeup for the class project of the "Practical Machine Learning" Coursera online course.  

Step 1 - Data Cleaning
----------------------

This dataset contains many extraneous columns with many "NA" values.  The first task was to eliminate the unnecessary columns from the dataset and therefore, reduce the number of predictor variables.

I noticed that the columns with "NA" and "#DIV/0!" are summary statistics (such as max, min, average, kurtosis, skewness, ...) which are only calculated at the end of each data "window".  I decided to simply remove these columns from the analysis. It's much quicker (for me anyway) to remove the columns in excel (and then "save as" to a new *.csv file) than removing the columns using R.  The resulting new data frame contains only 53 predictor variables (instead of the original 158!)

Step 2 - Data Partitioning into Training and Cross-Validation Sets
------------------------------------------------------------------

The next step involves splitting the data set into training and cross-validation with a 60-40 ratio per Prof. Leek's recommendation.  The code below accomplishes this split:


```r
    library(caret)
    data_frame <- read.csv("./pml_train.csv")
    blind_test <- read.csv("./pml_test.csv")
    set.seed(101) 
    index <- createDataPartition(data_frame$classe, p = 0.60, list = FALSE)
    training <- data_frame[index, ]
    cross_validation <- data_frame[-index, ]
```

Step 3a - Machine Learning using Support Vector Machine method
-------------------------------------------------------------

I tried using Support Vector Machine to predict the "classe" output variable. The cross-validation is quite good.  Please see the Confusion Matrix outputs.



```r
#   use Support Vector Machine (e1071 package)
    library(e1071)
    set.seed(101)
#   tune_svm <- tune.svm(classe ~ ., data = training, gamma = 3^(-6:-1), cost = 10^(-1:2))
#   tune_svm <- tune.svm(classe ~ ., data = training, gamma = 3^(-3:-3), cost = 3^(1:7))
#   tune_svm <- tune.svm(classe ~ ., data = training, gamma = 3^(-6:0), cost = 3^(5:5))
#   the best gamma and cost are approx 0.037 and 100
    model_svm <- svm(classe ~ ., data = training, gamma = 0.037, cost = 100)
    predict_train_svm <- predict(model_svm, newdata = training) 
    predict_cross_validation_svm <- predict(model_svm, newdata = cross_validation) 
    confusionMatrix(predict_train_svm, training$classe) 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1927    0
##          E    0    0    0    3 2165
## 
## Overall Statistics
##                                     
##                Accuracy : 1         
##                  95% CI : (0.999, 1)
##     No Information Rate : 0.284     
##     P-Value [Acc > NIR] : <2e-16    
##                                     
##                   Kappa : 1         
##  Mcnemar's Test P-Value : NA        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    0.998    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    0.999
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    0.999    1.000
```

```r
    confusionMatrix(predict_cross_validation_svm, cross_validation$classe) 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2219   10    0    1    0
##          B    7 1503   10    0    0
##          C    0    4 1354   21    3
##          D    0    0    4 1263    4
##          E    6    1    0    1 1435
## 
## Overall Statistics
##                                         
##                Accuracy : 0.991         
##                  95% CI : (0.988, 0.993)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.988         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.994    0.990    0.990    0.982    0.995
## Specificity             0.998    0.997    0.996    0.999    0.999
## Pos Pred Value          0.995    0.989    0.980    0.994    0.994
## Neg Pred Value          0.998    0.998    0.998    0.997    0.999
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.283    0.192    0.173    0.161    0.183
## Detection Prevalence    0.284    0.194    0.176    0.162    0.184
## Balanced Accuracy       0.996    0.994    0.993    0.990    0.997
```

Step 3b - Machine Learning using Random Forest technique
--------------------------------------------------------

I also tried Random Forest technique to predict the "classe" variable.  I found that Random Forest gave much better cross-validation results.  In addition, Random Forest can generage a "Variable Importance" chart to give us some insight on the relative importance of the predictor variables. Please see Variable Importance charts and Confusion Matrix reports.


```r
# use Random Forest (randomForest package)
    library(randomForest)
    set.seed(101)
#   tune_rf <- tuneRF(training[, -54], training[, 54], ntreeTry = 100, stepFactor = 1.5)
    model_rf <- randomForest(classe ~ ., data = training, mtry = 7, importance = TRUE)
    predict_train_rf <- predict(model_rf, newdata = training) 
    predict_cross_validation_rf <- predict(model_rf, newdata = cross_validation) 

    varImpPlot(model_rf, n.var = 20)
```

![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3.png) 

```r
    confusionMatrix(predict_train_rf, training$classe) 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                 
##                Accuracy : 1     
##                  95% CI : (1, 1)
##     No Information Rate : 0.284 
##     P-Value [Acc > NIR] : <2e-16
##                                 
##                   Kappa : 1     
##  Mcnemar's Test P-Value : NA    
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    1.000    1.000    1.000    1.000
## Specificity             1.000    1.000    1.000    1.000    1.000
## Pos Pred Value          1.000    1.000    1.000    1.000    1.000
## Neg Pred Value          1.000    1.000    1.000    1.000    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.194    0.174    0.164    0.184
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       1.000    1.000    1.000    1.000    1.000
```

```r
    confusionMatrix(predict_cross_validation_rf, cross_validation$classe) 
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    9    0    0    0
##          B    0 1508   13    0    0
##          C    0    1 1352   22    4
##          D    0    0    3 1264    8
##          E    0    0    0    0 1430
## 
## Overall Statistics
##                                        
##                Accuracy : 0.992        
##                  95% CI : (0.99, 0.994)
##     No Information Rate : 0.284        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.99         
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.993    0.988    0.983    0.992
## Specificity             0.998    0.998    0.996    0.998    1.000
## Pos Pred Value          0.996    0.991    0.980    0.991    1.000
## Neg Pred Value          1.000    0.998    0.998    0.997    0.998
## Prevalence              0.284    0.193    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.172    0.161    0.182
## Detection Prevalence    0.286    0.194    0.176    0.163    0.182
## Balanced Accuracy       0.999    0.996    0.992    0.991    0.996
```

Step 4 - Prediction
-------------------

I also tried K-nearest-neighbor and Logistic Regression techniques and their cross-validation results are not as good as Random Forest's and SVM's so I won't show the R-code for these two techniques.

I expect the out-of-sample error to be quite small given the excellent cross-validation errors shown above.

Finally, I used both Random Forest and SVM to predict the "classe" of the 20 cases in the testing set and I got identical predictions.  I'm not sure if it's OK with the honor code to post results of the prediction in this report so I won't post the results here.  I did get 20/20 score on the predictions.
