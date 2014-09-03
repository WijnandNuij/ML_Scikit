
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
        library(caret);
        library(doSNOW);
        registerDoSNOW(makeCluster(4, outfile=""))
        
        # CLASSIFICATION ONLY
        name <- c("hdda", "JRip", "lda", "lda2", "J48", "LMT", "LogitBoost", "lssvmRadial", "lvq",  
                  "rda", "rFerns", "rocc", "rpartCost", "rrlda", "RSimca", "sda",  
                  "slda", "sparseLDA", "stepLDA", "stepQDA", "svmRadialWeights",
                  "ada", "bagFDA", "C5.0", "C5.0Cost", "C5.0Rules", "C5.0Tree", "CSimca", "gpls", "hda", 
                  "multinom", "nb", "oblique.tree", "OneR", "pam", "PART", "pda", "pda2", "PenalizedLDA", 
                  "plr", "protoclass")
        
        ## NOT WORKING
        
        # "FH.GBML", "FRBCS.CHI", "FRBCS.W", "GFS.GCCL",
        # "Linda", "lssvmLinear", "lssvmPoly", "mda", "sddaLDA", 
        # "sddaQDA",  "SLAVE", "smda", "vbmpRadial","Mlda",
        # "ORFlog", "ORFpls", "ORFridge", "ORFsvm", "qda", "QdaCov", "fda", "RDlda", "rbf")
        
        
        # CLASSIFICATION & REGRESSION MODELS
        
        name <- c("avNNet", "bayesglm", "bdk", "blackboost", "Boruta", "bstLs", 
                  "bstSm", "bstTree", "cforest", "ctree", "ctree2", "dnn", "earth", "elm", "evtree", 
                  "extraTrees", "gam", "gamboost", "gamLoess", "gamSpline", "gaussprLinear", "gaussprPoly", 
                  "gaussprRadial", "gbm", "gcvEarth", "glm", "glmboost", "glmnet", "glmStepAIC", "kernelpls", 
                  "kknn", "knn", "logicBag", "logreg", "mlp", "mlpWeightDecay", "nnet", "nodeHarvest", "parRF", 
                  "partDSA", "pcaNNet", "pls", "plsRglm", "rbfDDA", "rf", "rknn", "rknnBel", "rpart", "rpart2", 
                  "RRF", "RRFglobal", "simpls", "spls", "svmBoundrangeString", "svmExpoString", "svmLinear", "svmPoly", 
                  "svmRadial", "svmRadialCost", "svmSpectrumString", "treebag", "widekernelpls", "xyf"
                  
                  , "bag", "bagEarth")
        
        
        ctrl = trainControl(method = "repeatedcv", 
                            number = 10, 
                            repeats = 2)
        
        
        for(s in name)
        {
                resultFile <- paste0(baseDir, '/models/', s, '.txt')
                if(file.exists(resultFile))
                {
                        print(paste0('skipping model: ', s, ' , already done.'))
                }
                else
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
                        sink(file = resultFile)
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
}