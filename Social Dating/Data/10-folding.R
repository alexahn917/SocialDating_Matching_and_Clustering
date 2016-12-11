library(caTools)
path <- "/Users/Alex/Documents/GitHub/SocialDating_Matching_and_Clustering/Social Dating/Data/Social_Dating_Data.txt"
conn <- file(path,open="r")
lines <- readLines(conn)

set.seed(276)
sample = sample.split(lines, SplitRatio = .80)
train = subset(lines, sample == TRUE)
test = subset(lines, sample == FALSE)

cat("Sampled train data:", length(train))
cat("Sampled test data:", length(test))

write.table(train, "data_5.train", quote = FALSE, col.names = FALSE, row.names = FALSE)
write.table(test, "data_5.test", quote = FALSE, col.names = FALSE, row.names = FALSE)
