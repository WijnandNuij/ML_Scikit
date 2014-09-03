
runTest <- function()
{
        # read training data
        training <- read.csv('/home/wijnand/R_workspace_scikit/resources/train.csv', header = F, colClasses = "numeric")
        outcome <- read.csv('/home/wijnand/R_workspace_scikit/resources/trainLabels.csv', header = F)
        colnames(outcome) <- 'outcome'
        outcome$outcome <- factor(outcome$outcome)
        
        testing <- read.csv('/home/wijnand/R_workspace_scikit/resources/test.csv', header = F, colClasses = "numeric")
        
        
        library(caret)
        #print(str(training))
        training <- predict(preProcess(training, method=c("center", "scale")), newdata = training)
        #train2 <- predict(preProcess(training, method="pca", pcaComp=12),training)
        
        testing <- predict(preProcess(training, method=c("center", "scale")), newdata = testing)
        #test2 <- predict(preProcess(training, method="pca", pcaComp=12),testing)
        
        
        
        library(caret);
        library(doSNOW);
        registerDoSNOW(makeCluster(4, outfile=""))
        grid <- expand.grid(.gamma = c(0.09090909),
                           .lambda = c(0.18))
        
        ctrl <- trainControl(method = "repeatedcv", 
                            number = 10, 
                            repeats = 10,
                            verboseIter = T,
                            allowParallel = T,
                            classProbs = T)
        
        trainedModel <- train(outcome[,1] ~ . , data=training,
                              method = "rda",
                              tuneGrid = grid,
                              trControl = ctrl,
                              metric = "Kappa")
        print(trainedModel)
        prediction <- predict(trainedModel, testing)
        
        solution <- as.data.frame(1:9000)
        solution <- cbind(solution, prediction)
        colnames(solution) <- c("Id", "Solution")
        
        write.csv(x = solution, file = '/home/wijnand/R_workspace_scikit/resources/prediction_rda.csv', 
                  row.names = F, quote = F)
        
        trainedModel
}