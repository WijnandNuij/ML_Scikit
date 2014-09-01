
runAllModels <- function(baseDir = '/home/wijnand/R_workspace_scikit')
{
        # read training data
        data <- read.csv(paste0(baseDir, '/resources/train.csv'), 
                         header = F, 
                         stringsAsFactors=FALSE, 
                         dec = ".", 
                         sep = ",", 
                         colClasses = "numeric")
        
        print(str(data))
        
        library(caret)
        data <- predict(preProcess(data, method="center"), newdata = data)
        data <- predict(preProcess(data, method="scale"), newdata = data)
        #data <- predict(preProcess(data, method="pca", thresh = 0.95), newdata = data)
        
        outcome <- read.csv(paste0(baseDir, '/resources/trainLabels.csv'), header = F)
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
        
        name <- c("ada", "bagFDA", "C5.0", "C5.0Cost", "C5.0Rules", "C5.0Tree", "CSimca", "FH.GBML", 
                  "FRBCS.CHI", "FRBCS.W", "GFS.GCCL", "gpls", "hda", "hdda", "J48", "JRip", "lda", "lda2", 
                  "Linda", "LMT", "LogitBoost", "lssvmLinear", "lssvmPoly", "lssvmRadial", "lvq", "mda", 
                  "Mlda", "multinom", "nb", "oblique.tree", "OneR", "ORFlog", "ORFpls", "ORFridge", "ORFsvm", 
                  "pam", "PART", "pda", "pda2", "PenalizedLDA", "plr", "protoclass", "qda", "QdaCov", "rbf", 
                  "rda", "rFerns", "RFlda", "rocc", "rpartCost", "rrlda", "RSimca", "sda", "sddaLDA", "sddaQDA", 
                  "SLAVE", "slda", "smda", "sparseLDA", "stepLDA", "stepQDA", "svmRadialWeights", "vbmpRadial")
        
        #name <- c("C5.0", "C5.0Cost")
        
        ctrl = trainControl(method = "repeatedcv", 
                                 number = 10, 
                                 repeats = 2)
        
        
        for(s in name)
        {
                startTime <- Sys.time()
                trainedModel <- train(outcome ~ . , data=training,
                                      method = s,
                                      tuneLength = 12, 
                                      trControl = ctrl,
                                      metric = "Kappa")
                predictionSomeModel <- predict(trainedModel, testing)
                
                print(paste0(s, " : ", confusionMatrix(predictionSomeModel, testing$outcome)$overall[1]))
                
                ## WRITE TO MODEL FILE
                sink(file = paste0(baseDir, '/models/', s, '.txt'))
                print(Sys.time() - startTime)
                print('===============')
                print(trainedModel)
                print('===============')
                print(confusionMatrix(predictionSomeModel, testing$outcome))
                sink()
                
                ## WRITE TO OVERALL RESULTS FILE
                sink(file = paste0(baseDir, '/models/1_overall_results.txt'), append = T)
                print(paste0(s, ';',
                             confusionMatrix(predictionSomeModel, testing$outcome)$overall[1], ';',
                             confusionMatrix(predictionSomeModel, testing$outcome)$overall[2]))
                sink()
                ##
        }
}