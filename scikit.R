# install.packages("caret", dependencies = c("Depends", "Suggests"))

runTest <- function()
{
        # read training data
        training <- read.csv('/home/wijnand/R_workspace_scikit/resources/train.csv', header = F, colClasses = "numeric")
        outcome <- read.csv('/home/wijnand/R_workspace_scikit/resources/trainLabels.csv', header = F)
        colnames(outcome) <- 'outcome'
        outcome$outcome <- factor(outcome$outcome)
        
        testing <- read.csv('/home/wijnand/R_workspace_scikit/resources/test.csv', header = F, colClasses = "numeric")
        
        
        library(caret)
        training <- predict(preProcess(training, method="center"), newdata = training)
        training <- predict(preProcess(training, method="scale"), newdata = training)
        
        testing <- predict(preProcess(training, method="center"), newdata = testing)
        testing <- predict(preProcess(training, method="scale"), newdata = testing)
                
        library(caret)
        #grid = expand.grid(.trials = c(80),
        #                   .model = c("rules"),
        #                   .winnow = c(F)) 
        trainedModel <- train(outcome[,1] ~ . , data=training,
                              method = "rf",
                              #tuneGrid = grid,
                              metric = "Kappa",
                              tuneLength = 2)
        print(trainedModel)
        prediction <- predict(trainedModel, testing)
        
        solution <- as.data.frame(1:9000)
        solution <- cbind(solution, prediction)
        colnames(solution) <- c("Id", "Solution")
        
        write.csv(x = solution, file = '/home/wijnand/R_workspace_scikit/resources/prediction.csv', 
                  row.names = F, quote = F)
}

runTraining <- function(file = '/home/wijnand/R_workspace_scikit/resources/train.csv')
{
        set.seed(1235)
        
        # read training data
        data <- read.csv(file, header = F, stringsAsFactors=FALSE, dec = ".", sep = ",", colClasses = "numeric")

        print(str(data))
        
        library(caret)
        data <- predict(preProcess(data, method="center"), newdata = data)
        data <- predict(preProcess(data, method="scale"), newdata = data)
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

        
        
        
        ### CARET MODEL LOOP ###
        library(caret)
        
        #name <- c("ada", "bagFDA")
        name <- c("ada", "bagFDA", "C5.0", "C5.0Cost", "C5.0Rules", "C5.0Tree", "CSimca", "fda", "FH.GBML", 
                  "FRBCS.CHI", "FRBCS.W", "GFS.GCCL", "gpls", "hda", "hdda", "J48", "JRip", "lda", "lda2", 
                  "Linda", "LMT", "LogitBoost", "lssvmLinear", "lssvmPoly", "lssvmRadial", "lvq", "mda", 
                  "Mlda", "multinom", "nb", "oblique.tree", "OneR", "ORFlog", "ORFpls", "ORFridge", "ORFsvm", 
                  "pam", "PART", "pda", "pda2", "PenalizedLDA", "plr", "protoclass", "qda", "QdaCov", "rbf", 
                  "rda", "rFerns", "RFlda", "rocc", "rpartCost", "rrlda", "RSimca", "sda", "sddaLDA", "sddaQDA", 
                  "SLAVE", "slda", "smda", "sparseLDA", "stepLDA", "stepQDA", "svmRadialWeights", "vbmpRadial")
        
        for(s in name)
        {
                trainedModel <- train(outcome ~ . , data=training,
                                      method = s,
                                      tuneLength = 1, 
                                      metric = "Accuracy")
                predictionSomeModel <- predict(trainedModel, testing)
                #print(trainedModel)
                print(paste0(s, " : ", confusionMatrix(predictionSomeModel, testing$outcome)$overall[1]))
        }
        
        # "C5.0 : 0.855"
        # "C5.0Cost : 0.87"
        # "C5.0Rules : 0.785"
        # "C5.0Tree : 0.755"
        
        
        
        ### GBM ###
        #print("generalized boosted model")
        #trainedModel <- train(outcome ~ . , data=training,
        #                    method="gbm",
        #                    verbose = F)
        #predictionGBM <- predict(trainedModel, testing)
        #print(confusionMatrix(predictionGBM, testing$outcome)$overall[1])
        
        ### KNN NEAREST NEIGHBORS ###
        #print("k-nearest neighbors")
        #library(class)
        #predictionKNN <- knn(train = training[,!(names(training) %in% 'outcome')],
        #                             test = testing[,!(names(testing) %in% 'outcome')],
        #                             cl = training$outcome, 
        #                             k = 8)
        #print(confusionMatrix(predictionKNN, testing$outcome)$overall[1])
        
        ### SUPPORT VECTOR MACHINES with Radial Basis Function Kernel ###
        #print("svm rbf")
        #library(kernlab)
        #trainedModel <- train(outcome ~ . , training, 
        #                      method="svmRadial", 
        #                      trControl = trainControl(method = "LOOCV"),
        #                      #trControl = trainControl(method = "cv", number = 5),
        #                      tuneGrid = data.frame(.C = c(.25, .5, 1), .sigma = .05),
        #                      metric = "Accuracy",
        #                      tuneLength = 32)
        #predictionSVMrbf <- predict(trainedModel, testing)
        #print(confusionMatrix(predictionSVMrbf, testing$outcome)$overall[1])
        
        ### RANDOM FOREST ###
        #print("random forest")
        #trainedModel <- train(outcome ~ . , training, 
        #                      method="rf", 
        #                      trControl = trainControl(method = "cv", number = 5),
        #                      metric = "Accuracy",
        #                      do.trace=F,
        #                      ntree=1500)
        #predictionRF <- predict(trainedModel, testing)
        #print(confusionMatrix(predictionRF, testing$outcome)$overall[1])
        
        ### EXTREME RANDOMIZED TREES ###
        #library(extraTrees)
        #options( java.parameters = "-Xmx2g" )
        #print("extreme randomized tree")
        #trainedModel <- extraTrees(training[,!(names(training) %in% 'outcome')], 
        #                    training$outcome, 
        #                    mtry = 10,
        #                    ntree = 1000, 
        #                    numRandomCuts = 5, 
        #                    numThreads = 4)
        #predictionET <- predict(trainedModel, testing[,!(names(testing) %in% 'outcome')])
        #print(confusionMatrix(predictionET, testing$outcome)$overall[1])
        
        ### NEURAL NETWORK ###
        #print("neural network")
        #trainedModel <- train(outcome ~ . , training, 
        #                      method="nnet", 
        #                      trControl = trainControl(method = "cv", number = 5),
        #                      metric = "Accuracy",
        #                      maxit = 1500,
        #                      trace = F)
        #predictionNN <- predict(trainedModel, testing)
        #print(confusionMatrix(predictionNN, testing$outcome)$overall[1])
        
        ### SUPPORT VECTOR MACHINES ###
        #print("support vector machine")
        #library(e1071)
        #trainedModel <- svm(outcome ~ . , data=training,
        #                    type='C',
        #                    kernel='linear',
        #                    probability = TRUE)
        #predictionSVM <- predict(trainedModel, testing)
        #rint(confusionMatrix(predictionSVM, testing$outcome)$overall[1])
        

        
        ## combine the best methods
        #print("voting mechanism of best methods")
        #comb <- predictionKNN
        #comb <- cbind(comb, predictionRF)
        #comb <- cbind(comb, predictionET)
        
        #predictionComb <- rowSums(comb)
        #predictionComb <- ifelse(predictionComb >= 5, 1, 0)
        #predictionComb <- as.factor(predictionComb)
        #print(confusionMatrix(predictionComb, testing$outcome)$overall[1])
}