data_reader = function(data, data_ref, nrows=Inf) {
    ref = read_csv(data_ref)
    col_type = as.list(ref$dplyr_type)
    names(col_type) = ref$col_name

    readr::read_csv("df_rfp_dataset_raw_20181218185047.csv",
                    col_types = col_type,
                    n_max = nrows)
}