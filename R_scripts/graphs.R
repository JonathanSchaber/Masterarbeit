#!/usr/bin/env Rscript
# -*- coding: utf-8 -*-

# synopsis: Rscript --vanilla /<path>/<to>/graphs.R <stats-file.json>

# prints out two PDFs: one for loss over epochs one for accuracy over epochs

library(ggh4x)
library(ggplot2)
library(reshape2)
library(rjson)
library(stringr)
library(tikzDevice)
library(tools)

latex_process = function(g_plot, file_name) {
    tikz(file_name, width = 6, height = 9, standAlone = TRUE, engine = "luatex")
    print(g_plot)
    dev.off()
    post_process(file_name)
}

post_process = function(tex_file) {
    # plots post-processing
    no_ext_name = gsub("\\.tex", "", tex_file)
    pdf_file = paste0(no_ext_name, ".pdf")
    texi2pdf(tex_file,
        clean = TRUE,
        texi2dvi = Sys.which("lualatex")
        )
    file.remove(tex_file)
    ## file.rename(pdf_file, paste0("./img/", pdf_file))
    unlink(paste0(no_ext_name, "*.png"))
    unlink("Rplots.pdf")
}

args = commandArgs(trailingOnly=TRUE)

results = fromJSON(file=args[1])

train_loss = vector()
dev_loss = vector()
test_loss = vector()
train_acc = vector()
dev_acc = vector()
test_acc = vector()

for (i in 3:length(results)) {train_loss = c(train_loss, results[[i]]$"Train Loss")}
for (i in 3:length(results)) {dev_loss = c(dev_loss, results[[i]]$"Dev Loss")}
for (i in 3:length(results)) {test_loss = c(test_loss, results[[i]]$"Test Loss")}

for (i in 3:length(results)) {train_acc = c(train_acc, results[[i]]$"Train Accur.")}
for (i in 3:length(results)) {dev_acc = c(dev_acc, results[[i]]$"Dev Accur.")}
for (i in 3:length(results)) {test_acc = c(test_acc, results[[i]]$"Test Accur.")}

epoch = c(1:length(train_acc))

results_df = data.frame(train_loss, dev_loss, test_loss, train_acc, dev_acc, test_acc, epoch)

#results_df$epoch = epoch

g_loss = ggplot(results_df, aes(x=epoch, y=train_loss)) +
    geom_line(aes(color="Train")) +
    geom_line(aes(y=dev_loss, color="Dev")) +
    geom_line(aes(y=test_loss, color="Test")) +
    ylab("Loss") +
    labs(color = "")

g_acc = ggplot(results_df, aes(x=epoch, y=train_acc)) +
    geom_line(aes(color="Train")) +
    geom_line(aes(y=dev_acc, color="Dev")) +
    geom_line(aes(y=test_acc, color="Test")) +
    ylab("Accuracy") +
    labs(color = "")
                
loss_file = str_replace(args[1], ".json", "_Loss.tex")
#loss_file = gsub(".*?/", "", loss_file)
acc_file = str_replace(args[1], ".json", "_Accuracy.tex")
#acc_file = gsub(".*?/", "", acc_file)

latex_process(g_loss, loss_file)
latex_process(g_acc, acc_file)
