source("R/data_reader.R")
library(lubridate)
library(tidyverse)

data_path = c("data/df_rfp_dataset_raw_20181218185047.csv",
              "data/df_rfp_dataset_raw_20190208223442.csv")

# both_dfs = map(data_path, ~data_reader(.x, data_ref = "data_reference.csv"))
# names(both_dfs) = basename(data_path)
df1 = data_reader(data_path[1], "data/data_reference.csv")

df1 = df1 %>%
    mutate(take_rate = TotalPassengers/SeatCount)

# Make take_rate ----------------------------------------------------------
take_rate = df1 %>%
    group_by(FlightID) %>%
    summarise(UniquePassenger = length(unique(UserID)),
              TotalPassengers = mean(TotalPassengers),
              take_rate = UniquePassenger / TotalPassengers) %>%
    filter(take_rate <= 1)

df1 = left_join(df1, take_rate)

take_rate_density = take_rate %>%
    ggplot(aes(take_rate)) +
    geom_density() +
    labs(title = "Take rate")

ggsave("plots/density_plots/take_rate.png", take_rate_density)


# Plots ----------------------------------------------------------
p = list(
    flight_length_vs_take_rate = df1 %>%
            distinct(FlightDurationHrs, take_rate) %>%
            ggplot(aes(FlightDurationHrs, take_rate)) +
            geom_point()


    ,flight_type_vs_take_rate =  df1 %>%
            distinct(FlightType, take_rate) %>%
            ggplot(aes(FlightType, take_rate)) +
            geom_boxplot()


    ,flight_duration_type_vs_take_rate =  df1 %>%
            distinct(FlightDurationType, take_rate) %>%
            ggplot(aes(fct_reorder(FlightDurationType, take_rate), take_rate)) +
            geom_boxplot()


    ,cost_vs_take_rate =  df1 %>%
            sample_n(100000) %>%
            ggplot(aes(`Price/USD`, take_rate)) +
            geom_point()


    # ,flight_length_vs_take_rate =  df %>%
    #         ggplot(aes()) +
    #         geom
    #
    #
    # ,flight_length_vs_take_rate =  df %>%
    #         ggplot(aes()) +
    #         geom
    #
    #
    # ,flight_length_vs_take_rate =  df %>%
    #         ggplot(aes()) +
    #         geom
    #
    #
    # ,flight_length_vs_take_rate =  df %>%
    #         ggplot(aes()) +
    #         geom
    #
    #
    # ,flight_length_vs_take_rate =  df %>%
    #         ggplot(aes()) +
    #         geom
    #

)
p


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
    ~ggsave(paste0(.y, ".png") , .x, path = "plots/univariate_plots")
})

map2(factor_tr_boxplots_025, factor_cols, {
    ~ggsave(paste0(.y, "capped.png") , .x, path = "plots/univariate_plots/takerate_capped")
})