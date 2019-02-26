get_data_time_allowance_df = function(df) {
    get_data_cap = function(prod_name) {
        data = str_extract(prod_name, regex("[\\d]+[ ]?MB", ignore_case = TRUE))

        as.numeric(str_extract(data, "[\\d]+"))
    }

    get_time_cap = function(prod_name) {
        time = str_remove(prod_name, regex("[\\d]+[ ]?MB", ignore_case = TRUE))

        time = str_extract(time, "[\\d]+[ ]?[h|H|m|M]|[\\w]+[ ][h|H]")
        time = str_replace(time, "One", "1")

        unit = tolower(str_extract(time, "[\\w]$"))
        time = as.numeric(str_extract(time, "[\\d]+"))

        if_else(unit == "h", time * 60, time)
    }

    df %>%
        distinct(ProductName) %>%
        transmute(
            ProductName,
            DataAllowance = get_data_cap(ProductName),
            TimeAllowance_min = get_time_cap(ProductName)
        )
}


get_take_rate_df = function(df) {
    # Returns dataframe

    df %>%
        distinct(FlightID, UserID, TotalPassengers) %>%
        group_by(FlightID) %>%
        summarise(
            UniquePassengers = n(),
            TakeRate = UniquePassengers / mean(TotalPassengers)
        ) %>%
        filter(TakeRate <= 1)
}


get_userflight_stats = function(df) {
    # Requires unique UserFlightID key
    # Returns dataframe

    df %>%
        left_join(get_data_time_allowance_df(df)) %>%
        group_by(UserFlightID) %>%
        summarise(
            DataUsage_total = sum(TotalUsageMB),
            DataUsage_perc = DataUsage_total / sum(DataAllowance),
            Spending_total = sum(`Price/USD`),
            Session_count = n(),
            UniqueProducts = length(unique(ProductName)),
            SessionDurationMin_total = sum(SessionDurationMinutes),
            SessionDurationMin_mean = mean(SessionDurationMinutes),
            SessionDurationMin_perc = SessionDurationMin_total / sum(TimeAllowance_min)
        )
}

get_flight_stats = function(df) {
    # Requires unique UserFlightID key
    # Returns dataframe

    df %>%
        group_by(FlightID) %>%
        summarise(
            DataUsage_total = sum(TotalUsageMB),
            DataUsage_perc = DataUsage_total / sum(DataAllowance),
            Spending_total = sum(`Price/USD`),
            Session_count = n(),
            SessionDurationMin_perc = SessionDurationMin_total / sum(TimeAllowance_min)
        )
}
