source("R/get_take_rate.R")

data_reader = function(data, data_ref, nrows=Inf, to_factors=TRUE) {
    ref = readr::read_csv(data_ref)

    col_type = as.list(ref$dplyr_col_type)
    names(col_type) = ref$col_name

    if (! "categorical" %in% colnames(ref)) {
        error("data reference updated to have separate categorical column.
                Please use the latest from Google drive.")
    }

    fct_index = which(ref[['categorical']] == 1)
    fct_col = ref[['col_name']][fct_index]

    parsed = readr::read_csv(
        data,
        col_types = col_type,
        n_max = nrows
    )

    if (to_factors) {
        parsed = dplyr::mutate_at(parsed, dplyr::vars(fct_col), as.factor)
    }


    caps = distinct(parsed, ProductName) %>%
        mutate_all(.funs = list(CapData_MB = get_datacap, CapTime_Min = get_timecap))

    message("Calculating take rate...")

    parsed %>%
        left_join(caps) %>%
        left_join(get_take_rate_df(parsed))
}

get_datacap = function(nm) {
    quant = str_extract(nm, regex("([\\d]+) ?MB", ignore_case = TRUE))
    str_extract(quant, "[\\d]+")
}

get_timecap = function(nm) {
    time = str_extract(nm, regex("([\\d]+) ?h|([\\d]+) ?min", ignore_case = TRUE))
    hours = which(str_detect(time, regex("H", ignore_case = TRUE)))

    time = as.numeric(str_extract(time, "[\\d]+"))

    time[hours] = time[hours] * 60
    time
}

