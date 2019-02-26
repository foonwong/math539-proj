source("R/data_reader.R")
library(lubridate)
library(tidyverse)

data_path = c("data/df_rfp_dataset_raw_20181218185047.csv",
              "data/df_rfp_dataset_raw_20190208223442.csv")

# both_dfs = map(data_path, ~data_reader(.x, data_ref = "data_reference.csv"))
# names(both_dfs) = basename(data_path)
df1 = data_reader(data_path[1], "data/data_reference.csv")

df1$UserFlightID = paste(df1$UserID, df1$FlightID)

df1 = df1 %>%
    right_join(get_take_rate(df1))

userflight_df1 = get_userflight_stats(df1)
flight_stats_df1 = get_flight_stats(df1)

## ggplot wrappers ----------------------------------------

get_density_plot = function(data, colname, title_suffix = NULL) {
    col_sym = sym(colname)

    data %>%
        select(!!col_sym) %>%
        filter(!!col_sym < quantile(!!col_sym, probs = 0.99, na.rm = TRUE)) %>%
        ggplot(aes(!!col_sym)) +
        geom_density() +
        labs(title = paste("Density:", colname, title_suffix))
}


## Distribution -------------------------------------------

#### Take rate
take_rate_density = get_density_plot(df1, "TakeRate")

ggsave("plots/density_plots/take_rate.png", take_rate_density,
       width = 12, height = 8)

#### Per userflight
userflight_cols = colnames(userflight_df1[, -1])

userflight_den_plot = map(userflight_cols, ~{
    get_density_plot(userflight_df1, colname = .x,
                     title_suffix = "per User+Flight (lower 99th percentile)")
})

map2(userflight_den_plot, userflight_cols, {
    ~ggsave(
        filename = paste0(str_remove_all(.y, "/"), ".png"),
        plot = .x,
        path = "plots/userflight_density",
        widht = 12, height = 6
    )
})




#### Date and time

get_time_distribution_plot = function(df) {
    list(
        hour = df1 %>%
            select(contains("TimeLocal")) %>%
            mutate_all(hour) %>%
            gather(var, hour) %>%
            ggplot(aes(hour)) +
            geom_histogram(bins=24) +
            facet_wrap(~var) +
            labs(title = "Distribution: Local time of day"),

        month = df1 %>%
            select(DepartureTimeUTC) %>%
            mutate_all(month) %>%
            gather(var, month) %>%
            ggplot(aes(month)) +
            geom_histogram(bins=12) +
            facet_wrap(~var) +
            labs(title = "Distribution: month of year")
    )
}

timing_distribution_plots = get_time_distribution_plot(df1)

map2(timing_distribution_plots, names(timing_distribution_plots), {
    ~ggsave(paste0(.y, ".png") , .x, path = "plots/density_plots")
})

#### Cateogrical --------------------------------------------------------
get_bar_plot = function(data, colname) {
    col_sym = sym(colname)

    data %>%
        count(!! col_sym) %>%
        ggplot(aes(fct_reorder(!! col_sym, n), n)) +
        geom_col() +
        labs(title = paste("Counts:", colname)) +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

factor_cols = colnames(select_if(df1, is.factor))
factor_bar_plots = map(factor_cols, ~get_bar_plot(df1, .x))

map2(factor_bar_plots, factor_cols, {
    ~ggsave(paste0(.y, ".png") , .x, path = "plots/bar_plots")
})

## Using departure time as reference
map(both_dfs, ~(range(.x$DepartureTimeUTC)))

#### Numeric ------------------------------------------------------

numeric_cols = df1 %>%
    select_if(is.numeric) %>%
    select(-RawDataKey, -contains("ID")) %>%
    colnames(.)

get_density_plot(df1, numeric_cols[1])
numeric_density_plots = map(numeric_cols, ~get_density_plot(df1, .x))

map2(numeric_density_plots, numeric_cols, {
    ~ggsave(
        filename = paste0(str_remove_all(.y, "/"), ".png"),
        plot = .x,
        path = "plots/density_plots"
    )
})


df1 %>%
    group_by(UserID) %>%
    summarise(TotalUsageMB = sum(TotalUsageMB)) %>%
    filter(TotalUsageMB < 5000) %>%
    ggplot(aes(TotalUsageMB)) +
    geom_density()

## Univariate_relationships



## Comparing both datasets ------------
get_col_intersect = function(df_list, column_name) {
    data = map(df_list, ~.x[[column_name]])

    intersect(data[[1]], data[[2]])
}

get_col_union = function(df_list, column_name) {
    data = map(df_list, ~.x[[column_name]])

    union(data[[1]], data[[2]])
}

intersect_list = map(colnames(both_dfs[[1]]), ~get_col_intersect(both_dfs, .x))
names(intersect_list) = colnames(both_dfs[[1]])

## Geocoding countries (ran and saved)
# ggmap::register_google(Sys.getenv("GMAP_KEY"))
#
# latlons = ggmap::geocode(country_union)
# country_latlon = tibble(country = country_union) %>%
#     bind_cols(latlons)