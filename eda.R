source("R/data_reader.R")

d1 = data_reader("data/df_rfp_dataset_raw_20181218185047.csv",
                 data_ref = "data_reference.csv")


d2 = data_reader("data/df_rfp_dataset_raw_20190208223442.csv",
                 data_ref = "data_reference.csv")
