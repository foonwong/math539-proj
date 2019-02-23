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

    parsed
}
