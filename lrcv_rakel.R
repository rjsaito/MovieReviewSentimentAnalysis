#load packages
pacman::p_load(dplyr, glmnet, caret)

# set working directory
setwd("C:/Users/rjsai/Dropbox/UMN Courses/CSCI 5525/project/emotion_label/")

#read data
x = as.matrix(read.table("X2_label.txt"))
x_3gram = as.matrix(read.table("X2_3gram_label.txt"))
x_kpca = as.matrix(read.table("X2_KPCA_label.txt"))
x_3gramkpca = as.matrix(read.table("X2_3gram_KPCA_label.txt"))

y = read.table("y_matrix.txt")
colnames(y) = c("Anger","Fear","Interest","Joy","Love","None","Sadness","Surprise")


#remove "none"?
which.none = which(y$None == 1)
x2 = x[-which.none,]
x2_kpca = x_kpca[-which.none,]
y2 = y[-which.none,]

LRCV_RAKEL = function(X, y, k = 3, nfold = 5, a = 1, quiet = T){

  #initialize
  cftrain = cftest = matrix(0, 2, 2)
  classes = names(y)
  n = length(classes)
  g = matrix(c(-1,1,1,-1), 2, 2)

  #RAKEL model design
  subs = unlist(lapply(1:k, function(K) combn(1:n, m=K, simplify = FALSE)), recursive = F)
  #subs = combn(1:n, m=k, simplify = FALSE)
  #newy = sapply(1:length(subs), function(j)
  #  apply(y, 1, function(v) as.numeric(all(which(v == 1) %in% subs[[j]]))))

  #newy1 = sapply(1:length(subs), function(j)
  #  apply(y, 1, function(v) as.numeric(all(subs[[j]] %in% which(v == 1)))))

  newy2 = sapply(1:length(subs), function(j)
    apply(y, 1, function(v) as.numeric(setequal(which(v == 1), subs[[j]]))))

  newy = newy2
  keep = which(colSums(newy) > 15)
  newy = newy[,keep]
  subsub = subs[keep]

  #Randomly shuffle the data
  X.shuffled = X[sample(nrow(X)),]

  #Create nfold almost-equally sized breaks
  folds = cut(seq(1,nrow(X)),breaks=nfold,labels=FALSE)

  #for each nfold
  for(f in 1:nfold){
    if(quiet) print(paste0("Working on fold ",f))
    test.index = which(f == folds)
    trainX = X[-test.index,]
    trainY = newy[-test.index,]
    testX = X[test.index,]
    testY = y[test.index,]

    #model
    predtest = matrix(0, nrow(testY), ncol(newy))
    predtrain = matrix(0, nrow(trainY), ncol(newy))
    for(j in 1:ncol(newy)){
      if(!quiet) print(paste0("Working on fold ",f, ", labelset ", j, " out of ", length(subs)))
      mod = cv.glmnet(trainX, trainY[,j], alpha=a, family='binomial', nfolds = 5)
      predtrain[,j] = predict(mod, newx = trainX, s = "lambda.min", type = "class")
      predtest[,j] = predict(mod, newx = testX, s = "lambda.min", type = "class")
    }

    #tally
    testYhat = matrix(0, nrow(testY), n)
    trainYhat = matrix(0, nrow(trainY), n)
    for(l in 1:nrow(testY)){
      tallytest = t(sapply(1:ncol(newy), function(j) 
        g[as.numeric(predtest[l,j])+1,][1:n %in% subsub[[j]]+1]))
      testYhat[l,] = as.numeric(colSums(tallytest) > 0)
    }
    for(l in 1:nrow(trainY)){
      tallytrain = t(sapply(1:ncol(newy), function(j) 
        g[as.numeric(predtrain[l,j])+1,][1:n %in% subsub[[j]]+1]))
      trainYhat[l,] = as.numeric(colSums(tallytrain) > 0)
    }
    #suppressMessages(
    cftrain = cftrain + suppressMessages(confusionMatrix(unlist(trainYhat), unlist(y[-test.index,]))$table)
    cftest = cftest + suppressMessages(confusionMatrix(unlist(testYhat), unlist(testY))$table)
  }
  return(list(cftrain, cftest))
}


### test on diff data sets
xout.1 = LRCV_RAKEL(x, y, k = 3, nfold = 5, a = 0, quiet = T)
xout.2 = LRCV_RAKEL(x2, y2, k = 3, nfold = 5, a = 0, quiet = T)
x_kpcaout2.1 = LRCV_RAKEL(x_kpca, y, k = 3, nfold = 5, a = 1, quiet = T)
x_kpcaout2.12 = LRCV_RAKEL(x_kpca, y, k = 3, nfold = 5, a = 0, quiet = T)
x_kpcaout2.2 = LRCV_RAKEL(x2_kpca, y2, k = 3, nfold = 5, a = 1, quiet = T)


xout3 = LRCV_RAKEL(x, y, k = 3, nfold = 5, a = 1, quiet = T)
xout2 = LRCV_RAKEL(x, y, k = 2, nfold = 5, a = 1, quiet = T)

x_3gramout2 = LRCV_RAKEL(x_3gram, y, k = 3, nfold = 5, a = 1, quiet = T)

x_3gramkpcaout2 = LRCV_RAKEL(x_3gramkpca, y, k = 3, nfold = 5, a = 1, quiet = T)

x_kpcaout2 = LRCV_RAKEL(x_kpca, y, k = 2, nfold = 5, a = 1, quiet = T)







