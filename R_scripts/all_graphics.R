# TOTALS

# TOTAL GAIN/LOSS

df_gain_loss_tot <- data.frame(
        effect = c("Gain", "Gain", "Gain", "Loss", "Loss"),
        p = c("neutral", "gain", "gain sign", "loss", "loss sign"),
        value = c(3, 32, 5, 15, 3)
    )

df_gain_loss_tot$effect <- factor(df_gain_loss_tot$effect, levels = c("Loss", "Gain"))
df_gain_loss_tot$p <- factor(df_gain_loss_tot$p, levels=c("neutral", "gain", "gain sign", "loss", "loss sign"))

bp <- ggplot(df_gain_loss_tot, aes(x=effect, y=value, fill=p)) +
    geom_bar(width = 1, stat = "identity") +
    theme(legend.position = "none", text=element_text(size=30)) +
    scale_fill_manual("legend", values = c("neutral" = "#C7E9C0", "gain" = "#A1D99B", "gain sign" ="#005A32", "loss" = "#FC9272", "lo
    ss sign" = "#CB181D")) +
    xlab("") +
    ylab("") + ylab("Number Of Experiments") +
    coord_flip()

# TOTAL DUP/ZER/DRA

df_dup_zer_tot <- data.frame(
        mode = c("duplicate", "zeros", "draw"),
        value = c(26, 19, 3)
    )

df_dup_zer_tot$mode <- factor(df_dup_zer_tot$mode, levels=c( "draw", "zeros", "duplicate"))

bp <- ggplot(df_dup_zer_tot, aes(x=mode, y=value, fill=mode)) +
    geom_bar(width = 1, stat = "identity") +
    theme(legend.position = "none", text=element_text(size=30)) +
    scale_fill_manual("legend", values = c("duplicate" = "#2171B5", "zeros" = "#9ECAE1", "draw" = "#C6DBEF")) +
    xlab("") +
    ylab("") + ylab("Number of Experiments") +
    coord_flip()

# ASSESSMENT

df <- data.frame(
        person = c("Person A", "Person A", "Person A", "Person B", "Person B", "Person B"),
        verdict = c("harmful", "neutral", "helpful"),
        values = c(18, 34, 12, 33, 17 ,14)
    )

bp <- ggplot(df, aes(x=person, y=values, fill=verdict)) +
    geom_bar(width = 0.8, stat = "identity", position = "dodge") +
    theme(legend.position = "none", text=element_text(size=30)) +
    scale_fill_manual("legend", values = c("harmful" = "#FC9272", "neutral" = "#9ECAE1", "helpful" = "#74C476")) +
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
