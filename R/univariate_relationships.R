library(tidyverse)
library(lubridate)
source("R/data_reader.R")
source("R/get_take_rate.R")

data_path = c("data/df_rfp_dataset_raw_20181218185047.csv",
              "data/df_rfp_dataset_raw_20190208223442.csv")

# both_dfs = map(data_path, ~data_reader(.x, data_ref = "data_reference.csv"))
# names(both_dfs) = basename(data_path)
df1 = data_reader(data_path[1], "data/data_reference.csv")

df1 = df1 %>%
    right_join(get_take_rate(df1))

# Correlation -------------------------------------------------------
library(GGally)

corr_plot = df1 %>%
    select(-matches("*ID*|*Time*|*Key*", ignore.case = FALSE)) %>%
    ggcorr(label = TRUE, label_size = 3, label_round = 2, label_alpha = TRUE,
           hjust = 0.75, layout.exp = 1, size = 2)

ggsave("plots/correlation.png", corr_plot,
       width = 14, height = 14)

# box Plots ----------------------------------------------------------
get_tr_boxplot = function(data, colname, take_rate_lim = 1) {
    col_sym = sym(colname)

    data %>%
        select(!!col_sym, take_rate) %>%
        drop_na() %>%
        ggplot(aes(fct_reorder(!! col_sym, take_rate), take_rate)) +
        geom_boxplot() +
        lims(y = c(0, take_rate_lim)) +
        labs(title = paste(colname, "vs take-rate"),
             x = colname) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
}


factor_cols = colnames(select_if(df1, is.factor))

factor_tr_boxplots = map(factor_cols, ~get_tr_boxplot(df1, .x))
factor_tr_boxplots_025 = map(factor_cols, ~{
    get_tr_boxplot(df1, .x, take_rate_lim = 0.25)
})


map2(factor_tr_boxplots, factor_cols, {
    ~ggsave(
        paste0(.y, ".png"),
        .x,
        path = "plots/univar_takerate",
        width = 12, height = 8
    )
})

map2(factor_tr_boxplots_025, factor_cols, {
    ~ggsave(
        paste0(.y, "capped.png"),
        .x,
        path = "plots/univar_takerate/capped",
        width = 12, height = 6
    )
})


## datausage
get_datause_boxplot = function(data, colname) {
    col_sym = sym(colname)

    data %>%
        select(UserID, !!col_sym, TotalUsageMB) %>%
        group_by(UserID, !!col_sym) %>%
        summarise(TotalUsageMB = sum(TotalUsageMB)) %>%
        filter(TotalUsageMB < 1000) %>%
        drop_na() %>%
        ggplot(aes(fct_reorder(!! col_sym, TotalUsageMB), TotalUsageMB)) +
        geom_boxplot() +
        labs(title = paste0(colname, " vs Data Usage(per user, MB)"),
             x = colname) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 1))
}

data_cols = c(
    "Orig_Country",
    "Dest_Country"
   # "Orig_Region",
   # "FlightDurationType",
   # "Dest_Region",
   # "NightFlight",
   # "Category",
   # "Airline_Region",
   # "AC Frame",
   # "luxury"
)


for (col in data_cols) {
    p = get_datause_boxplot(df1, col)

    ggsave(
        paste0(col, "_datausage_capped1GB.png"),
        p,
        path = "plots/univar_datausage",
        width = 12, height = 8
    )

    gc()
}
