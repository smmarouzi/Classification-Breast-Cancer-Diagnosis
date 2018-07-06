#### Breast Cancer Classification Project


  ### In this problem we have to use 30 different columns and we have to predict the Stage of Breast Cancer M (Malignant) and B (Bengin)
  #This analysis has been done using Basic Machine Learning Algorithm with detailed explanation
  #Attribute Information:
    #1. ID number
    #2. Diagnosis (M = malignant, B = benign)
    #3-32.Ten real-valued features are computed for each cell nucleus:
      #a) radius (mean of distances from center to points on the perimeter)
      #b) texture (standard deviation of gray-scale values)
      #c) perimeter
      #d) area
      #e) smoothness (local variation in radius lengths)
      #f) compactness (perimeter^2 / area - 1.0)
      #g) concavity (severity of concave portions of the contour)
      #h) concave points (number of concave portions of the contour)
      #i) symmetry
      #j) fractal dimension ("coastline approximation" - 1)

  ##here 3- 32 are divided into three parts first is Mean (3-13), Stranded Error(13-23) and Worst(23-32) and each contain 10 parameter
    #(radius, texture,area, perimeter, smoothness,compactness,concavity,concave points,symmetry and fractal dimension)
    #Here Mean means the means of the all cells, standard Error of all cell and worst means the worst cell


### here we will import the libraries used for plot and machine learning:

  ## Plot
library(package = "lattice")
library(devtools)
library(ggplot2)
library(easyGgplot2)
    # Correlation Plot
library("corrplot")

  ## Melt
library(reshape2)
library(plyr)

  ## Machine learning 
library("class") #Classification
library("gmodels") #Model Fitting
library("caret") #Classification and Regression Training
library("e1071") #SVM
library("glmnet") #glm
library("ranger") # Random Forest
library("caretEnsemble")
library("pROC")
library("ROCR")

### Import data
setwd("C:/Users/mm/Desktop/data science/R programming/Project")
getwd()
mydata <-read.csv("data.csv")  # read csv file 
  
  # have a look at the data
dim(mydata)
str(mydata) #data structure
attributes(mydata)[[1]] #column names
head(mydata) #show first 6 rows
tail(mydata,2) #show last 2 columns

mydata[!complete.cases(mydata),] #to see is there any missing data or not

### Preprocessing
  # removing the column we do not need
mydata<-mydata[-1] #remove ID numbers
  # convert the factor levels from Malignant and Benign to 1 and 0 respectively
mydata$diagnosis<-factor(mydata$diagnosis)

# frequency percentage of diagnosis categories # Benign: 62.7%, Malignant: 37.3% 
PF<-round(prop.table(table(mydata$diagnosis))*100,digits = 1) 
xx<-barplot(PF,main="Diagnosis percentage", col=c("darkgreen","yellow"),ylim=c(0,100))
legend("topright", c("Benign","Malignant"), cex=1.3, bty="n", fill=c("darkgreen","yellow"))
text(x = xx, y = PF, label = paste(PF,"%"), pos = 3, cex = 0.8, col = "red")


  # summary of data, numbers and visualization
summary(mydata) #to see the statistical summary of the data

## Correlation with the target
correlationVector <-cor(as.numeric(mydata$diagnosis),mydata[,2:ncol(mydata)])
notCorrelated <- which(abs(correlationVector)<0.3)

print(notCorrelated)

# remove attributes that are not highly correlated with the target
mydata=data.frame(diagnosis=mydata[,1],mydata[-c(1,notCorrelated+1)])
names(mydata)

## Pairwise features Correlation
correlationMatrix <- cor(mydata[,2:ncol(mydata)])
corrplot(correlationMatrix, method="ellipse")

  # find attributes that are highly corrected (ideally >0.85)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.85)
  # print indexes of highly correlated attributes
print(highlyCorrelated)

  # remove attributes that are highly correlated
mydata=data.frame(diagnosis=mydata[,1],mydata[-c(1,highlyCorrelated+1)])
names(mydata)

## Summary of not correlated data
mydata.m<- melt(mydata, diagnosis = "Label")
# Summary of datas- Box Plot
p <- ggplot(data = mydata.m, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=diagnosis))
p + facet_wrap( ~ variable, scales="free")

mu <- ddply(mydata.m, ~diagnosis+variable, summarise, grp.mean=mean(value))
head(mu)
p <- ggplot(data = mydata.m, aes(x=value,color="black")) + 
  geom_histogram(aes(fill=diagnosis))
p + facet_wrap( ~ variable, scales="free")+
  geom_vline(data=mu,aes(xintercept=grp.mean,color=diagnosis),linetype="dashed",pch=30)
  
  # remove features with close mean for two diagnosis type of data
##diffmean<-ddply(mu, ~variable, summarise, difmean=diff(grp.mean))
##mydata=data.frame(diagnosis=mydata[,1],mydata[which(diffmean$difmean>0.05)+1])

## Summary of removed closed mean data
summary(mydata)

names(mydata)
#-------------------------------------------------------------

## Split data to train and test
splitdata<-function(df,n){ #df: dataframe to split, n: n% train and 100-n% test
  nd=nrow(df)
  l=round(n*nd/100)
  trainind<-sample(seq_len(nd),size = l)
  mydata_train<-df[trainind,]
  mydata_test<-df[-trainind,]
  mydata=list(mydata_train,mydata_test)
  return(mydata)
}
data=splitdata(mydata,75)
datatrain=data[[1]]
datatest=data[[2]]

### ML Algorithm
  # we are going to try different models
  # we have now 11 attributes and fit the model
  # To avoid overfiting we are going to use cross validation:
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 2,
                           savePredictions="final",
                           preProcOptions = list(thresh = 0.99),                           
                           classProbs = TRUE)

  ## Knn Classification Algorithm: (mydata data)
#mydata_test_pred<-knn(train = datatrain[,-1],test = datatest[,-1],cl=datatrainlabel,k=10)
fit_knn <- train(diagnosis ~ ., datatrain, 
                 method = "knn", 
                 trControl = fitControl, 
                 preProc=c("center", "scale"), 
                 tuneLength = 20)
pred_knn <- predict(fit_knn, datatest)                           
cm_knn<-confusionMatrix(pred_knn, datatest$diagnosis, positive = "M")
cm_knn
ct_cm_knn<-CrossTable(x=datatest$diagnosis,y=pred_knn,prop.chisq = FALSE,prop.tbl=FALSE,prop.col=FALSE,prop.row=FALSE)
d<-ifelse(datatest$diagnosis=="M",1,0)
pd_knn<-ifelse(pred_knn=="M",1,0)
roc_obj<-roc(d,pd_knn)
print(paste0("Area under curve: ",auc(d,pd_knn)))


##
  ## Logistic regression multiclass
#model<-glm(diagnosis~.,datatrain,family=binomial(link='logit'))
#mydata_test_pred<-round(predict(model,datatest[,-1],type="response"),0)
fit_glmnet <- train (diagnosis~.,
                     datatrain,
                     method = "glmnet",
                     tuneLength=20,
                     metric="Accuracy",
                     preProc=c("center", "scale"),
                     trControl = fitControl)

pred_glmnet <- predict(fit_glmnet, datatest)                           
confusionMatrix(pred_glmnet, datatest$diagnosis)
CrossTable(x=datatest$diagnosis,y=pred_glmnet,prop.chisq = FALSE,prop.tbl=FALSE,prop.col=FALSE,prop.row=FALSE)
d<-ifelse(datatest$diagnosis=="M",1,0)
pd_glm<-ifelse(pred_glmnet=="M",1,0)
roc_obj<-roc(d,pd_glm)
print(paste0("Area under curve: ",auc(d,pd_glm)))
##

  ## Random forest
fit_rf <- train (diagnosis~.,
                 datatrain,
                 method = "ranger",
                 
                 metric="Mean_F1",
                 preProc=c("center", "scale"),
                 trControl = fitControl)

pred_rf <- predict(fit_rf, datatest)                           
confusionMatrix(pred_rf, datatest$diagnosis) 
CrossTable(x=datatest$diagnosis,y=pred_rf,prop.chisq = FALSE,prop.tbl=FALSE,prop.col=FALSE,prop.row=FALSE)
pd_rf<-ifelse(pred_glmnet=="M",1,0)
roc_obj<-roc(d,pd_knn)
print(paste0("Area under curve: ",auc(d,pd_knn)))
##

## SVM
fit_svmRadial <- train(diagnosis ~ ., datatrain, 
                       method = "svmRadial", 
                       trControl = fitControl, 
                       preProcess = c("center", "scale"), 
                       tuneLength = 20)

pred_svmRadial <- predict(fit_svmRadial, datatest)                           
confusionMatrix(pred_svmRadial, datatest$diagnosis)
CrossTable(x=datatest$diagnosis,y=pred_svmRadial,prop.chisq = FALSE,prop.tbl=FALSE,prop.col=FALSE,prop.row=FALSE)
pd_svm<-ifelse(pred_svmRadial=="M",1,0)
roc_obj<-roc(d,pd_svm)
print(paste0("Area under curve: ", auc(d,pd_svm)))
##

## Conclusion
model_list <- list(GMLNET=fit_glmnet, RF = fit_rf, KNN=fit_knn, SVM=fit_svmRadial)
resamples <- resamples(model_list)

model_confmat <- lapply(model_list, function(x) confusionMatrix(predict(x, datatest), datatest$diagnosis))
model_accuracy <- sapply(model_confmat, function(x) x$overall['Accuracy'])

  # model accuracy plot
xx<-barplot(100*model_accuracy,main="Compare diffrent Model Accuracy", col=3:6,ylim=c(0,150),xlab = "Training Models",
            names.arg =c("GLMNET","RF","KNN","SVM"))
legend("topright", c(names(model_accuracy)), cex=0.9, bty="n", fill=3:6)
text(x = xx, y = 100*model_accuracy, label = paste(100*round(model_accuracy,4),"%"), pos = 3, cex =1, col = "black")


  # ROC plot
plot(roc(d, pd_glm),
     col=3, lwd=3,print.auc.y = .15,print.auc=TRUE, main="ROC")
plot(roc(d, pd_rf),
     col=4, lwd=3,print.auc.y = .10,print.auc=TRUE,add=TRUE)
plot(roc(d, pd_knn),
     col=5, lwd=3,print.auc.y = .05,print.auc=TRUE,add=TRUE)
plot(roc(d, pd_svm),
     col=6, lwd=3,print.auc.y = 0,print.auc=TRUE,add=TRUE)
legend("bottomright",c("GLM","RF","KNN","SVM"),cex=0.9, bty="n", fill=3:6)


