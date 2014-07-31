
runTest <- function()
{
        # read training data
        training <- read.csv('/home/wijnand/R_workspace_scikit/resources/train.csv', header = F)
        outcome <- read.csv('/home/wijnand/R_workspace_scikit/resources/trainLabels.csv', header = F)
        colnames(outcome) <- 'outcome'
        outcome$outcome <- factor(outcome$outcome)
        
        testing <- read.csv('/home/wijnand/R_workspace_scikit/resources/test.csv', header = F)
        
        #predictionKNN <- knn(train = data,
        #                     test = testing,
        #                     cl = outcome[,1], 
        #                     k = 8)
        
        library(extraTrees)
        options( java.parameters = "-Xmx2g" )
        trainedModel <- extraTrees(training, 
                                   outcome[,1], 
                                   mtry = 10,
                                   ntree = 1000, 
                                   numRandomCuts = 5, 
                                   numThreads = 4)
        predictionET <- predict(trainedModel, testing)
                
        solution <- as.data.frame(1:9000)
        solution <- cbind(solution, predictionET)
        colnames(solution) <- c("Id", "Solution")
        
        write.csv(x = solution, file = '/home/wijnand/R_workspace_scikit/resources/prediction.csv', 
                  row.names = F, quote = F)
}

runTraining <- function(file = '/home/wijnand/R_workspace_scikit/resources/train.csv')
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
        inTrain <- createDataPartition(data$outcome, p = 0.8, list=F)
        training <- data[inTrain,]
        testing <- data[-inTrain,]

        ### GBM ###
        print("generalized boosted model")
        trainedModel <- train(outcome ~ . , data=training,
                            method="gbm",
                            verbose = F)
        predictionGBM <- predict(trainedModel, testing)
        print(confusionMatrix(predictionGBM, testing$outcome)$overall[1])
        
        ### KNN NEAREST NEIGHBORS ###
        print("k-nearest neighbors")
        library(class)
        predictionKNN <- knn(train = training[,!(names(training) %in% 'outcome')],
                                     test = testing[,!(names(testing) %in% 'outcome')],
                                     cl = training$outcome, 
                                     k = 8)
        print(confusionMatrix(predictionKNN, testing$outcome)$overall[1])
        
        
        ### RANDOM FOREST ###
        print("random forest")
        trainedModel <- train(outcome ~ . , training, 
                              method="rf", 
                              trControl = trainControl(method = "cv", number = 5),
                              metric = "Accuracy",
                              do.trace=F,
                              ntree=1500)
        predictionRF <- predict(trainedModel, testing)
        print(confusionMatrix(predictionRF, testing$outcome)$overall[1])
        
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
        predictionET <- predict(trainedModel, testing[,!(names(testing) %in% 'outcome')])
        print(confusionMatrix(predictionET, testing$outcome)$overall[1])
        
        ### NEURAL NETWORK ###
        print("neural network")
        trainedModel <- train(outcome ~ . , training, 
                              method="nnet", 
                              trControl = trainControl(method = "cv", number = 5),
                              metric = "Accuracy",
                              maxit = 1500,
                              trace = F)
        predictionNN <- predict(trainedModel, testing)
        print(confusionMatrix(predictionNN, testing$outcome)$overall[1])
        
        ### SUPPORT VECTOR MACHINES ###
        print("support vector machine")
        library(e1071)
        trainedModel <- svm(outcome ~ . , data=training,
                            type='C',
                            kernel='linear',
                            probability = TRUE)
        predictionSVM <- predict(trainedModel, testing)
        print(confusionMatrix(predictionSVM, testing$outcome)$overall[1])
        
        
        ## combine the best methods
        print("voting mechanism of best methods")
        comb <- predictionKNN
        comb <- cbind(comb, predictionRF)
        comb <- cbind(comb, predictionET)
        
        predictionComb <- rowSums(comb)
        predictionComb <- ifelse(predictionComb >= 5, 1, 0)
        predictionComb <- as.factor(predictionComb)
        print(confusionMatrix(predictionComb, testing$outcome)$overall[1])
}