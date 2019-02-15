library(tidyverse)

pana_df = read_csv("~/Downloads/df_rfp_dataset_raw_20181218185047.csv")

get_fct_unique_count = function(col) {
    length(unique(col))
}

get_fct_unique = function(col) {
    u = unique(col)

    if (length(u) > 100) {
        NA
    } else {
        paste(as.character(sort(u)), collapse = ", ")
    }
}

get_fct_R_class = function(col) {
    paste(class(col), collapse = ", ")
}

pana_ref = pana_df %>%
    rename_all(function(x) paste0(x, "@@@")) %>%
    summarise_all(funs(get_fct_R_class, get_fct_unique, get_fct_unique_count)) %>%
    gather(col, val) %>%
    separate(
        col = col,
        into = c("col_name", "type"),
        sep = "@@@_",
        extra = "merge"
    ) %>%
    spread(type, val) %>%
    left_join(tibble(col_name = colnames(pana_df), col_number = 1:ncol(pana_df))) %>%
    rename_all(str_remove_all, pattern = "get_fct_")

pana_ref %>%
    arrange(unique_count) %>%
    mutate(need_clarification = '', notes = '', pandas_dtype = '') %>%
    select(col_number, col_name, R_class, unique_count,
           need_clarification, notes, pandas_dtype, unique) %>%
    write_csv("data_reference.csv")


pana_df %>%
    select(-RawDataKey, -SessionID) %>%
    sample_n(10000) %>%
    naniar::vis_miss()
