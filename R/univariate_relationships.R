# Setup -------------------------------------------------------------------
library(tidyverse)
library(lubridate)
source("R/data_reader.R")
source("R/get_take_rate.R")

# data -------------------------------------------------------------------
data_path = c("data/df_rfp_dataset_raw_20181218185047.csv",
              "data/df_rfp_dataset_raw_20190208223442.csv")

# both_dfs = map(data_path, ~data_reader(.x, data_ref = "data_reference.csv"))
# names(both_dfs) = basename(data_path)
df1 = data_reader(data_path[1], "data/data_reference.csv")

df1 = df1 %>%
    right_join(get_take_rate(df1))

flight_stats_df1 = get_flight_stats(df1)

# Correlation -------------------------------------------------------
library(GGally)

corr_plot = df1 %>%
    select(-matches("*ID*|*Time*|*Key*", ignore.case = FALSE)) %>%
    ggcorr(label = TRUE, label_size = 3, label_round = 2, label_alpha = TRUE,
           hjust = 0.75, layout.exp = 1, size = 2)

ggsave("plots/correlation.png", corr_plot,
       width = 14, height = 14)

# box Plots ----------------------------------------------------------
get_tr_boxplot = function(data, colname) {
    col_sym = sym(colname)

    data %>%
        select(!!col_sym, TakeRate) %>%
        drop_na() %>%
        filter(TakeRate < quantile(TakeRate, 0.99)) %>%
        ggplot(aes(fct_reorder(!! col_sym, TakeRate), TakeRate)) +
        geom_boxplot() +
        labs(title = paste(colname, "vs take-rate (lower 99th percentile)"),
             x = colname) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
}


factor_cols = colnames(select_if(df1, is.factor))

factor_tr_boxplots = map(factor_cols, ~get_tr_boxplot(df1, .x))
names(factor_tr_boxplots) = factor_cols

for (col in factor_cols) {
    p = get_tr_boxplot(df1, col)

    ggsave(
        paste0(col, ".png"),
        p,
        path = "plots/univar_takerate",
        width = 12, height = 8
    )

    gc()
}


# datausage and revenue plots ---------------------------------------------
data_cols = c(
    "Orig_Country",
    "Dest_Country",
    "Airline",
    "Orig_Region",
    "FlightDurationType",
    "Dest_Region",
    "NightFlight",
    "FlightType",
    "Airline_Region",
    "luxury"
)

flight_info_df = df1 %>%
    select(FlightID, !!!syms(data_cols)) %>%
    distinct()

lower_percentile = function(x, prob = 0.99) {
    x < quantile(x, probs = prob)
}

data_spend_df = flight_stats_df1 %>%
    select(FlightID, DataUsage_total, Spending_total) %>%
    left_join(flight_info_df)


get_dataspend_boxplot = function(data, colname) {
    col_sym = sym(colname)

    data = select(data, DataUsage_total, Spending_total, !!col_sym) %>%
        drop_na() %>%
        filter(
            lower_percentile(DataUsage_total) &
            lower_percentile(Spending_total)
        )

    data[[colname]] = fct_reorder(data[[colname]], data[['DataUsage_total']])

    data %>%
        gather(var, val, -!!col_sym) %>%
        ggplot(aes(!!col_sym, val)) +
        geom_boxplot() +
        facet_wrap(~var, scale = "free", nrow = 2) +
        labs(title = paste0(colname, " vs Data Usage and Revenue(per flight)"),
             x = colname) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# ----------------------------------------------------------------

for (col in data_cols) {
    p = get_dataspend_boxplot(data_spend_df, col)

    print(paste("rendering", col))

    ggsave(
        paste0(col, "_data_revenue.png"),
        p,
        path = "plots/dataconsumption_revenue",
        width = 12, height = 10
    )

    gc()
}
