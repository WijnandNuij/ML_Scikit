
loadData <- function(file = '/home/wijnand/R_workspace_scikit/resources/train.csv')
{
        set.seed(1234)
        
        # read training data
        data <- read.csv(file, header = F)
        outcome <- read.csv('/home/wijnand/R_workspace_scikit/resources/trainLabels.csv', header = F)
        colnames(outcome) <- 'outcome'
        outcome$outcome <- factor(outcome$outcome)
        # bind outcome to training data
        data <- cbind(data, outcome)
        
        # randomize the dataset
        data <- data[order(runif(nrow(data))),]
        
        library(caret)
        inTrain <- createDataPartition(data$outcome, p = 0.7, list=F)
        training <- data[inTrain,]
        testing <- data[-inTrain,]
        

        
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
        print("extreme randomized trees")
        trainedModel <- extraTrees(training[,!(names(training) %in% 'outcome')], 
                            training$outcome, 
                            ntree = 1500, 
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