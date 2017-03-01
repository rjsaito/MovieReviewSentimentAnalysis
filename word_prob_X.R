if(!"pacman" %in% installed.packages()) install.packages("pacman")
pacman::p_load(tm, readr, dplyr)

# path to reviews
neg_review_path = ".../neg/"
pos_review_path = ".../pos"

# read all negative reviews
neg_reviews = NULL
neg_review_files = list.files(neg_review_path, full.names = T)
for(f in neg_review_files) neg_reviews = c(neg_reviews, read_file(f))

# read all positive reviews
pos_reviews = NULL
pos_review_files = list.files(pos_review_path, full.names = T)
for(f in pos_review_files) pos_reviews = c(pos_reviews, read_file(f))

# all reviews
reviews = c(neg_reviews, pos_reviews)

# create term document matrix for neg and pos

#neg
doc.vec.neg <- VectorSource(neg_reviews)
doc.corpus.neg <- Corpus(doc.vec.neg)
doc.corpus.neg <- tm_map(doc.corpus.neg, stemDocument)
doc.corpus.neg <- tm_map(doc.corpus.neg, stripWhitespace)
TDM.neg <- TermDocumentMatrix(doc.corpus.neg)
#can remove sparse words
smallTDM.neg <- removeSparseTerms(TDM.neg, sparse= 1-0.005)

#pos
doc.vec.pos <- VectorSource(pos_reviews)
doc.corpus.pos <- Corpus(doc.vec.pos)
doc.corpus.pos <- tm_map(doc.corpus.pos, stemDocument)
doc.corpus.pos <- tm_map(doc.corpus.pos, stripWhitespace)
TDM.pos <- TermDocumentMatrix(doc.corpus.pos)
#can remove sparse words
smallTDM.pos <- removeSparseTerms(TDM.pos, sparse= 1-0.005)

str(smallTDM.pos$dimnames)


negfreq = smallTDM.neg[findFreqTerms(smallTDM.neg),] %>%
      as.matrix() %>%
      rowSums()
posfreq = smallTDM.pos[findFreqTerms(smallTDM.pos),] %>%
      as.matrix() %>%
      rowSums()
freq = merge(
  data.frame(word = names(negfreq), nneg = negfreq, stringsAsFactors = F),
  data.frame(word = names(posfreq), npos = posfreq, stringsAsFactors = F),
  by = "word", all = T
) 
freq$n = rowSums(freq[c("nneg", "npos")], na.rm = T)
freq = filter(freq, n < 2000*.8)
freq$npos[is.na(freq$npos)] = 0
freq$p.pos = freq$npos/freq$n

#tfidf weight
freq$lognorm = 1 + log(freq$n)
freq$p.pos.lognorm = round(freq$p.pos*freq$lognorm, 4)



#create feature set
doc.vec <- VectorSource(reviews)
doc.corpus <- Corpus(doc.vec)
doc.corpus <- tm_map(doc.corpus, stemDocument)
doc.corpus <- tm_map(doc.corpus, stripWhitespace)
TDM <- TermDocumentMatrix(doc.corpus)
#can remove sparse words
smallTDM <- removeSparseTerms(TDM, sparse= 1-0.005)

word.list = data.frame(
  word = smallTDM$dimnames$Terms, 
  i = 1:length(smallTDM$dimnames$Terms), 
  stringsAsFactors = F
)
freq.ind = merge(freq, word.list)

newTDM = smallTDM
keep = which(smallTDM$i %in% freq.ind$i)
i = newTDM$i[keep]
j = newTDM$j[keep]
X1_ppos = matrix(0, nrow = max(j), ncol = max(i))

for(k in 1:length(i)){
  X1_ppos[j[k], i[k]] = freq.ind$p.pos.lognorm[which(freq.ind$i == i[k])]
}

#write
write.table(X1_ppos, ".../X1_ppos.txt", 
	row.names = F, col.names = F, quote = F)






