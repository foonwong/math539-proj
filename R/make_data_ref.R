library(tidyverse)

ref = read_csv("data_reference.csv")
col_type = as.list(ref$dplyr_type)
names(col_type) = ref$col_name

pana_df = read_csv(
    "df_rfp_dataset_raw_20181218185047.csv",
    col_types = col_type
)


## Helper funs ---------------------------------------------------------------
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

## ---------------------------------------------------------------------------

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


## adding type for dplyr/pandas ----------------------------------------
dplyr_type = c(
    # from ?readr
    character = "c",          # character
    integer = "i",            # integer
    numeric = "n",            # number
    double = "d",             # double
    logical = "l",            # logical
    factor = "f",             # factor
    noidea = "D",             # date
    `POSIXct, POSIXt` = "T",  # date time
    noidea = "t",             # time
    guess = "?",              # guess
    skip = "_/-"              # to skip the column.
)

pandas_type = c(
    character = "object",
    integer = "int",
    numeric = "float",
    double = "float",
    logical = "bool",
    factor = "category",
    `POSIXct, POSIXt` = "datetime64"
)

pana_ref$dplyr_type = dplyr_type[pana_ref$R_class]
pana_ref$pandas_type = pandas_type[pana_ref$R_class]

## ---------------------------------------------------------------------
pana_ref = pana_ref %>%
    arrange(unique_count) %>%
    select(col_number, col_name, R_class, unique_count,
           dplyr_type, pandas_type, unique)

write_csv(pana_ref, "data_reference.csv")