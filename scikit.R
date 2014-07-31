
runML <- function(file = '/home/wijnand/R_workspace_scikit/resources/train.csv')
{
        set.seed(1234)
        
        # read training data
        data <- read.csv(file, header = F)
        
        #data <- predict(preProcess(data, method="scale"), newdata = data)
        #data <- predict(preProcess(data, method="pca", thresh = 0.95), newdata = data)
        
        outcome <- read.csv('/home/wijnand/R_workspace_scikit/resources/trainLabels.csv', header = F)
        colnames(outcome) <- 'outcome'
        outcome$outcome <- factor(outcome$outcome)
        # bind outcome to training data
        data <- cbind(data, outcome)
        
        # randomize the dataset
        data <- data[order(runif(nrow(data))),]
        
        library(caret)
        inTrain <- createDataPartition(data$outcome, p = 0.9, list=F)
        training <- data[inTrain,]
        testing <- data[-inTrain,]

        ### GBM ###
        print("generalized boosted model")
        trainedModel <- train(outcome ~ . , data=training,
                            method="gbm",
                            verbose = F)
        prediction <- predict(trainedModel, testing)
        print(confusionMatrix(prediction, testing$outcome)$overall[1])
        
        ### KNN NEAREST NEIGHBORS ###
        print("k-nearest neighbors")
        prediction <- knn(train = training[,!(names(training) %in% 'outcome')],
                          test = testing[,!(names(testing) %in% 'outcome')], 
                          cl = training$outcome, 
                          k = 5)
        print(confusionMatrix(prediction, testing$outcome)$overall[1])
        
        ### RANDOM FOREST ###
        print("random forest")
        trainedModel <- train(outcome ~ . , training, 
                              method="rf", 
                              trControl = trainControl(method = "cv", number = 2),
                              metric = "Accuracy",
                              do.trace=F,
                              ntree=1500)
        prediction <- predict(trainedModel, testing)
        print(confusionMatrix(prediction, testing$outcome)$overall[1])
        
        ### EXTREME RANDOMIZED TREES ###
        library(extraTrees)
        options( java.parameters = "-Xmx2g" )
        print("extreme randomized tree")
        trainedModel <- extraTrees(training[,!(names(training) %in% 'outcome')], 
                            training$outcome, 
                            mtry = 10,
                            ntree = 1000, 
                            numRandomCuts = 5, 
                            numThreads = 4)
        prediction <- predict(trainedModel, testing[,!(names(testing) %in% 'outcome')])
        print(confusionMatrix(prediction, testing$outcome)$overall[1])
        
        ### NEURAL NETWORK ###
        print("neural network")
        trainedModel <- train(outcome ~ . , training, 
                              method="nnet", 
                              trControl = trainControl(method = "cv", number = 2),
                              metric = "Accuracy",
                              maxit = 1500,
                              trace = F)
        prediction <- predict(trainedModel, testing)
        print(confusionMatrix(prediction, testing$outcome)$overall[1])
        
        ### SUPPORT VECTOR MACHINES ###
        print("support vector machine")
        library(e1071)
        trainedModel <- svm(outcome ~ . , data=training,
                            type='C',
                            kernel='linear',
                            probability = TRUE)
        prediction <- predict(trainedModel, testing)
        print(confusionMatrix(prediction, testing$outcome)$overall[1])
}