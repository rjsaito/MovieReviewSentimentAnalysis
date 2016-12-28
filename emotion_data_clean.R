pacman::p_load(dplyr, glmnet)

write.file = F

# read data
test = readLines("C:/Users/rjsai/Dropbox/UMN Courses/CSCI 5525/project/emotion/test.txt")
train = readLines("C:/Users/rjsai/Dropbox/UMN Courses/CSCI 5525/project/emotion/train.txt")

# split label
test.mat = do.call(rbind, strsplit(test, ' (?=[^ ]+$)', perl=TRUE))
train.mat = do.call(rbind, strsplit(train, ' (?=[^ ]+$)', perl=TRUE))

# merge
merged = rbind(test.mat, train.mat)

#if multi-label, split
splitted <- strsplit(as.character(merged[,2]), "_")
newdata = data.frame(review = rep.int(merged[,1], sapply(splitted, length)), emotion = unlist(splitted),
	stringsAsFactors = F)
newdataX = newdata[,1]
newdataY = newdata[,2]

#write out
if(write.file) write.table(newdataX, "C:/Users/rjsai/Dropbox/UMN Courses/CSCI 5525/project/emotion/emotion_data_X.txt", 
	row.names = F, col.names = F, quote = F)
if(write.file) write.table(newdataY, "C:/Users/rjsai/Dropbox/UMN Courses/CSCI 5525/project/emotion/y.txt", 
	row.names = F, col.names = F, quote = F)


#cast data
library(reshape2)
newdata_cast = dcast(newdata, review ~ emotion)

# "y matrix
newdata_cast2 = newdata_cast
newdata_cast2[,2:9] = as.numeric(!is.na(newdata_cast2[,2:9]))

newdata2X = newdata_cast2[,1]
newdata2Y = newdata_cast2[,2:9]

if(write.file) write.table(newdata2X, "C:/Users/rjsai/Dropbox/UMN Courses/CSCI 5525/project/emotion/emotion_label_data_X.txt", 
	row.names = F, col.names = F, quote = F)
if(write.file) write.table(newdata2Y, "C:/Users/rjsai/Dropbox/UMN Courses/CSCI 5525/project/emotion/y_matrix.txt", 
	row.names = F, col.names = F, quote = F)




