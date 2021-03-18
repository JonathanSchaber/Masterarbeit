# TOTALS

# TOTAL GAIN/LOSS

df_gain_loss_tot <- data.frame(
        effect = c("Gain", "Gain", "Gain", "Loss", "Loss"),
        p = c("neutral", "gain", "gain sign", "loss", "loss sign"),
        value = c(3, 25, 7, 9, 4)
    )

df_gain_loss_tot$effect <- factor(df_gain_loss_tot$effect, levels = c("Loss", "Gain"))
df_gain_loss_tot$p <- factor(df_gain_loss_tot$p, levels=c("neutral", "gain", "gain sign", "loss", "loss sign"))

bp <- ggplot(df_gain_loss_tot, aes(x=effect, y=value, fill=p)) +
    geom_bar(width = 1, stat = "identity") +
    theme(legend.position = "none", text=element_text(size=30)) +
    scale_fill_manual("legend", values = c("neutral" = "#C7E9C0", "gain" = "#A1D99B", "gain sign" ="#005A32", "loss" = "#FC9272", "loss sign" = "#CB181D")) +
    xlab("") +
    ylab("") + ylab("Number Of Experiments") +
    coord_flip()

# DATA_SET GAIN/LOSS

data_set_gain <- function(gain_loss, last_row="") {
    df <- data.frame(
            effect = c("Gain", "Gain", "Gain", "Loss", "Loss"),
            p = c("neutral", "gain", "gain sign", "loss", "loss sign"),
            value = gain_loss
        )
    df$effect <- factor(df$effect, levels = c("Loss", "Gain"))
    df$p <- factor(df$p, levels=c("neutral", "gain", "gain sign", "loss", "loss sign"))
    bp <- ggplot(df, aes(x=effect, y=value, fill=p)) +
        geom_bar(width = 1, stat = "identity") +
        theme(legend.position = "none", text=element_text(size=30)) +
        scale_fill_manual("legend", values = c("neutral" = "#C7E9C0", "gain" = "#A1D99B", "gain sign" ="#005A32", "loss" = "#FC9272", "loss sign" = "#CB181D")) +
        ylim(0,11) +
        xlab("") +
        ylab("") + ylab(last_row) +
        coord_flip()
    return(bp)
}


deisear <- c(2, 4, 2, 3, 1)

scare = c(1, 4, 1, 4, 2)

pawsx = c(0, 7, 3, 1, 1)

xnli = c(0, 10, 1, 1, 0)

# TOTAL DUP/ZER/DRA

data_set_dup_zer <- function(dup_zer_draw, data_set, last_row="") {
    df <- data.frame(
            mode = c("duplicate", "zeros", "draw"),
            value = dup_zer_draw
        )
    df$mode <- factor(df$mode, levels=c( "draw", "zeros", "duplicate"))
    bp <- ggplot(df, aes(x=mode, y=value, fill=mode)) +
        geom_bar(width = 1, stat = "identity") +
        theme(legend.position = "none", text=element_text(size=30)) +
        scale_fill_manual("legend", values = c("duplicate" = "#2171B5", "zeros" = "#9ECAE1", "draw" = "#C6DBEF")) +
        ylim(0,8) +
        xlab("") + xlab(data_set) +
        ylab("")+ ylab(last_row) +
        coord_flip()
    return(bp)
}

deisear <- c(6, 5, 1)

scare <- c(7, 4, 1)

pawsx <- c(8, 3, 1)

xnli <- c(5, 7, 0)

as_ggplot <- function(x){
      cowplot::ggdraw() +
        cowplot::draw_grob(grid::grobTree(x))
    }

all_sets <- grid.arrange(deisear_dup, deisear_gain,
                 scare_dup, scare_gain,
                 pawsx_dup, pawsx_gain,
                 xnli_dup, xnli_gain,
                 nrow=4)

final_plot <- as_ggplot(all_sets)


# ASSESSMENT PER DATA SET

df_data_sets <- data.frame(
    data_set = c("deISEAR", "deISEAR", "deISEAR", "deISEAR", "deISEAR", "deISEAR", "SCARE", "SCARE", "SCARE", "SCARE", "SCARE", "SCARE", "PAWS-X", "PAWS-X", "PAWS-X", "PAWS-X", "PAWS-X", "PAWS-X", "XNLI", "XNLI", "XNLI", "XNLI", "XNLI", "XNLI", "MLQA", "MLQA", "MLQA", "MLQA", "MLQA", "MLQA", "XQuAD", "XQuAD", "XQuAD", "XQuAD", "XQuAD","XQuAD"),
    verdict = c("harmful", "harmful", "neutral", "neutral", "helpful", "helpful"),
    unisono = c("harmful", "harmful unisono", "neutral", "neutral unisono", "helpful", "helpful unisono"),
    values = c(20.0,   0.0, 33.33,  1.67, 41.67, 3.33,
               26.67, 3.33,  50.0, 13.33,  6.67,  0.0,
               41.67, 15.0,  30.0,   0.0, 13.33,  0.0,
               28.33,  5.0, 31.67,   5.0,  20.0, 10.0,
               16.67,  0.0,  50.0,   0.0, 33.33,  0.0,
               33.33,  0.0, 33.33,   0.0, 33.33,  0.0)
)

df_data_sets$verdict <- factor(df_data_sets$verdict, levels=c("harmful", "neutral", "helpful"))

df_data_sets$data_set <- factor(df_data_sets$data_set, levels=c("deISEAR", "SCARE", "PAWS-X", "XNLI", "MLQA", "XQuAD"))

bp <- ggplot(df_data_sets, aes(x=verdict, y=values, fill=unisono)) +
    geom_bar(width = 1, stat = "identity", position = "stack") +
    theme(legend.title = element_blank(),
          legend.position = "top",
          # legend.text = element_blank(),
          text=element_text(size=33),
          strip.background = element_blank()) +
    scale_fill_manual("legend", values = c("harmful" = "#FC9272", "harmful unisono" = "#CB181D",
                                           "neutral" = "#9ECAE1", "neutral unisono" = "#2171B5",
                                           "helpful" = "#A1D99B", "helpful unisono" = "#005A32")) +
    xlab("") +
    ylab("") + ylab("%") +
    theme(axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
    facet_grid(.~data_set,
               switch = "x")

# ASSESSMENT PER PERSON

df <- data.frame(
        person = c("Person A", "Person A", "Person A", "Person B", "Person B", "Person B", "Person C", "Person C", "Person C"),
        verdict = c("harmful", "neutral", "helpful"),
        values = c(18, 34, 12, 33, 17 ,14, 9, 39, 16)
    )

df$verdict <- factor(df$verdict, levels=c("harmful", "neutral", "helpful"))

bp <- ggplot(df, aes(x=person, y=values, fill=verdict)) +
    geom_bar(width = 0.8, stat = "identity", position = "dodge") +
    theme(legend.title = element_blank(), legend.position = "right", text=element_text(size=30)) +
    scale_fill_manual("legend", values = c("harmful" = "#FC9272", "neutral" = "#9ECAE1", "helpful" = "#A1D99B")) +
    xlab("") +
    ylab("") + ylab("Counts")

# SCARE LABEL VOTES

df_lab_votes <- data.frame(
        vote = c("No\nLabels", "No\nMajority", "Close\nMajority", "Clear\nMajority"),
        values = c(2.9, 3.0, 13.8, 80.3)
    )

df_lab_votes$vote <- factor(df_lab_votes$vote, levels=c("Clear\nMajority", "Close\nMajority", "No\nMajority", "No\nLabels"))

bp <- ggplot(df_lab_votes, aes(x=vote, y=values, fill=vote)) +
        geom_bar(width = 1, stat = "identity") +
        theme(legend.position = "none", text=element_text(size=27)) +
        scale_fill_manual("legend", values = c("No\nLabels" = "#9ECAE1", "No\nMajority" = "#6BAED6", "Close\nMajority" = "#4292C6", "Clear\nMajority" = "#2171B5")) +
        xlab("") +
        ylab("") + ylab("%")

# SCARE LABELS

df_lab <- data.frame(
        label = c("Positive", "Neutral", "Negative"),
        values = c(1071, 193, 496)
    )

df_lab$label <- factor(df_lab$label, levels=c("Positive", "Neutral", "Negative"))

bp <- ggplot(df_lab, aes(x=label, y=values, fill=label)) +
            geom_bar(width = 1, stat = 'identity') +
            theme(legend.position = 'none', text=element_text(size=27)) +
            # scale_fill_brewer(palette='Blues') +
            scale_fill_manual('legend', values = c('Positive' = '#74C476', 'Neutral' = '#9ECAE1', 'Negative' = '#FC9272')) +
            xlab('') +
            ylab('') + ylab('Number of Examples')


# SCARE STARS

df_lab_stars <- data.frame(
        stars = c('*', '**', '***', '****', '*****'),
        values = c(18.6, 6.0, 8.3, 15.9, 51.2)
    )

df_lab_stars$stars <- factor(df_lab_stars$stars, levels=c('*****', '****', '***', '**', '*'))

bp <- ggplot(df_lab_stars, aes(x=stars, y=values, fill=stars)) +
            geom_bar(width = 1, stat = 'identity') +
            theme(legend.position = 'none', text=element_text(size=27)) +
            # scale_fill_brewer(palette='Blues') +
            scale_fill_manual('legend', values = c('*' = '#C6DBEF', '**' = '#9ECAE1', '***' = '#6BAED6', '****' = '#4292C6', '*****'
    = '#2171B5')) +
            xlab('') +
            ylab('') + ylab('%')

