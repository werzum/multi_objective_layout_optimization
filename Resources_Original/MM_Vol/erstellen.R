
path <- "C:/Users/Christoph/Desktop/MM_Vol/"

file.list <- list.files(path)

source("C:/Users/Christoph/Desktop/Biom1/Uebung6/fPoll.R")

i = 4

for(i in 1:length(file.list)){
  tab.i <- read.table(paste0(path, file.list[i]), sep = "", dec = ",", header = T)
  
  plot(tab.i$BHD, tab.i$h)
  
  tab.i$f <- fPoll(1, tab.i$BHD, tab.i$h)
  
  tab.i$v <- (tab.i$BHD/100)^2 * pi / 4 * tab.i$h * tab.i$f
  
  hist(tab.i$v)
  
  write.csv2(tab.i, paste0("C:/Users/Christoph/Desktop/MM_Vol/output/", file.list[i]),
             row.names = F)
  
}
