library(tidyverse)
library(ggraph)
library(tidygraph)

route_df = read_csv("data/df_rfp_dataset_raw_20181218185047.csv",
         col_types = cols_only(Routes = "c"))

airport_info = read_csv("airport.csv")
airport_info_cols = c(
    "ident",
    "name",
    lat = "lat",
    lon = "lon",
    country = "iso_country",
    region = "iso_region"
)

route_edge = routes_df %>%
    mutate_all(as.character) %>%
    group_by(Routes) %>%
    summarise(route_weight = length(Routes))

route_node = routes_df %>%
    separate(Routes, into = c('from', 'to')) %>%
    gather(type, node) %>%
    group_by(node) %>%
    summarise(node_weight = length(node))

route_node %>%
    left_join(airport_info, by = c("node" = "ident")) %>%
    rename(lat = "latitude_deg", lon = "lontitude_deg")

