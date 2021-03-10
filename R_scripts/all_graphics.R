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

df2 <- data.frame(
        vote = c("No Labels", "No Majority", "Close Majority", "Clear Majority"),
        values = c(2.9, 3.0, 13.8, 80.3)
    )

df2$vote <- factor(df2$vote, levels=c("Clear Majority", "Close Majority", "No Majority", "No Labels"))

bp <- ggplot(df2, aes(x=vote, y=values, fill=as.factor(values))) +
    geom_bar(width = 1, stat = "identity") +
    theme(legend.position = "none", text=element_text(angle=90, size=20)) +
    scale_fill_brewer(palette = "Blues") +
    xlab("") +
    ylab("") + ylab("Percentage")

