dataset = read.csv('wine.csv')
dim(dataset)
str(dataset)  #Compactly display the internal structure
head(dataset) #Returns the first or last parts
summary(dataset)
unique(is.na(dataset))

data_1=dataset %>% filter(dataset$Customer_Segment == 1)
data_2=dataset %>% filter(dataset$Customer_Segment == 2)
data_3=dataset %>% filter(dataset$Customer_Segment == 3)


a=table(dataset$Alcohol)
barplot(a,main="Using BarPlot to display alcohol Content of all wines",
        ylab="Count",
        xlab="alcohol",
        col='red',
        legend=rownames(a))

hist(dataset$Alcohol,
     col="blue",
     main="Histogram to display range of alcohol Content of all wines",
     xlab="alcohol",
     ylab="Count",
     labels=TRUE)

boxplot(dataset$Alcohol,
        col="pink",
        main="Boxplot for Descriptive Analysis of alcohol Content of all wines")

sd(dataset$Alcohol)    
sd(data_1$Alcohol)
sd(data_2$Alcohol)
sd(data_3$Alcohol)

a=table(dataset$Customer_Segment)
barplot(a,main="Using BarPlot to display frequency of wines",
        ylab="Count",
        xlab="wines",
        col=rainbow(3),
        legend=rownames(a))

pct=round(a/sum(a)*100)
lbs=paste(c("Winery 1","Winery 2","Winery 3")," ",pct,"%",sep=" ")
library(plotrix)
pie(a,labels=lbs,main="Pie Chart Depicting Ratio of wines")

plot(density(dataset$Alcohol),
     main="Density Plot for alcohol content",
     xlab="alcohol content",ylab="Density")
polygon(density(dataset$Alcohol),col="#ccff66")




#pca
library(caTools)
set.seed(123)
split = sample.split(dataset$Customer_Segment, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-14] = scale(training_set[-14])
test_set[-14] = scale(test_set[-14])

library(caret)
library(e1071)

pca = preProcess(x = training_set[-14], method = 'pca', pcaComp = 2)
training_set = predict(pca, training_set)
training_set = training_set[c(2, 3, 1)]
test_set = predict(pca, test_set)
test_set = test_set[c(2, 3, 1)]

# Fitting SVM to the Training set
classifier = svm(formula = Customer_Segment ~ .,
                 data = training_set,
                 type = 'C-classification',
                 kernel = 'linear')

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-3])

# Making the Confusion Matrix
cm = table(test_set[, 3], y_pred)
cm
# Visualising the Training set results
library(ElemStatLearn)
set = training_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3],
     main = 'SVM (Training set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, 
       pch = '.', 
       col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, pch = 21, 
       bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))

# Visualising the Test set results
library(ElemStatLearn)
set = test_set
X1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.01)
X2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.01)
grid_set = expand.grid(X1, X2)
colnames(grid_set) = c('PC1', 'PC2')
y_grid = predict(classifier, newdata = grid_set)
plot(set[, -3], main = 'SVM (Test set)',
     xlab = 'PC1', ylab = 'PC2',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(as.numeric(y_grid), length(X1), length(X2)), add = TRUE)
points(grid_set, 
       pch = '.', 
       col = ifelse(y_grid == 2, 'deepskyblue', ifelse(y_grid == 1, 'springgreen3', 'tomato')))
points(set, 
       pch = 21, 
       bg = ifelse(set[, 3] == 2, 'blue3', ifelse(set[, 3] == 1, 'green4', 'red3')))



