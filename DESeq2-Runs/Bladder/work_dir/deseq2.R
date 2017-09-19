
library('DESeq2'); library('data.table'); library('BiocParallel')
register(MulticoreParam(12))

# Argument parsing
args <- commandArgs(trailingOnly = TRUE)
df_path <- args[1]
tissue_path <- args[2]
disease_path <- args[3]
tissue <- 'Bladder'
output_dir <- '/data/'

# Read in vectors
tissue_vector <- read.table(tissue_path)$V1
disease_vector <- read.table(disease_path)$V1

# Read in table and process
n <- read.table(df_path, sep='\t', header=1, row.names=1)
sub <- n[, colnames(n)%in%tissue_vector]
setcolorder(sub, as.character(tissue_vector))

# Preprocessing
countData <- round(sub)
colData <- data.frame(disease=disease_vector, row.names=colnames(countData))
y <- DESeqDataSetFromMatrix(countData = countData, colData = colData, design = ~ disease)

# Run DESeq2
y <- DESeq(y, parallel=TRUE)
res <- results(y, parallel=TRUE)
summary(res)

# Write out table
resOrdered <- res[order(res$padj),]
res_name <- paste(tissue, '.tsv', sep='')
res_path <- paste(output_dir, res_name, sep='/')
write.table(as.data.frame(resOrdered), file=res_path, col.names=NA, sep='\t',  quote=FALSE)

# MA Plot
ma_name <- paste(tissue, '-MA.pdf', sep='')
ma_path <- paste(output_dir, ma_name, sep='/')
pdf(ma_path, width=7, height=7)
plotMA(res, main='DESeq2')
dev.off()

# Dispersion Plot
disp_name <- paste(tissue, '-dispersion.pdf', sep='')
disp_path <- paste(output_dir, disp_name, sep='/')
pdf(disp_path, width=7, height=7)
plotDispEsts( y, ylim = c(1e-6, 1e1) )
dev.off()

# PVal Hist
hist_name <- paste(tissue, '-pval-hist.pdf', sep='')
hist_path <- paste(output_dir, hist_name, sep='/')
pdf(hist_path, width=7, height=7)
hist( res$pvalue, breaks=20, col="grey" )
dev.off()

# Ratios plots
qs <- c( 0, quantile( res$baseMean[res$baseMean > 0], 0:7/7 ) )
bins <- cut( res$baseMean, qs )
levels(bins) <- paste0("~",round(.5*qs[-1] + .5*qs[-length(qs)]))
ratios <- tapply( res$pvalue, bins, function(p) mean( p < .01, na.rm=TRUE ) )
ratio_name <- paste(tissue, '-ratios.pdf', sep='')
ratio_path <- paste(output_dir, ratio_name, sep='/')
pdf(ratio_path, width=7, height=7)
barplot(ratios, xlab="mean normalized count", ylab="ratio of small $p$ values")
dev.off()                                           
