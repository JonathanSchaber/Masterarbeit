#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

library(tools)
library(ggh4x)
library(ggplot2)
library(tikzDevice)
library(reshape2)

post_process <- function(tex_file) {
  # plots post-processing
  no_ext_name <- gsub("\\.tex", "", tex_file)
  pdf_file <- paste0(no_ext_name, ".pdf")
  texi2pdf(tex_file,
           clean = TRUE,
           texi2dvi = Sys.which("lualatex")
           )
  file.remove(tex_file)
  ## file.rename(pdf_file, paste0("./img/", pdf_file))
  unlink(paste0(no_ext_name, "*.png"))
  unlink("Rplots.pdf")
}

data <- readRDS("ma_data.rds")
data <- data.frame(do.call(rbind, lapply(1:length(data), function(i) cbind(i,data[[i]]))))
names(data) <- c("type","length")
data$type <- factor(data$type)

# plot ggplot object
g <- ggplot(data,aes(x=type,y=length,fill=type)) +
  stat_boxplot(
    geom = "errorbar", width = 0.3,
    position = position_dodge(width = 1)
  ) +
  geom_boxplot(
    position = position_dodge(width = 1), outlier.shape = 1,
    outlier.size = 2.5, outlier.alpha = 0.7
  ) +
  ## theme_bw() +
  theme(legend.position = "none", text=element_text(size=20)) +
  #scale_x_discrete(labels=c("1"="Train","2"="Dev","3"="Test")) +
  scale_fill_brewer(palette = "Blues") +
  xlab("") + ylab("Length")

tex_file <- "test.tex"
tikz(tex_file, width = 6, height = 9, standAlone = TRUE, engine = "luatex")
print(g)
dev.off()
post_process(tex_file)



