library(tidyverse)
library(ggraph)
library(tidygraph)

route_df = read_csv("data/df_rfp_dataset_raw_20181218185047.csv",
         col_types = cols_only(Routes = "c"))

airport_info_cols = c(
    ICAO = "ident",
    "name",
    lat = "lat",
    lon = "lon",
    country = "iso_country",
    region = "iso_region"
)

airport_info = read_csv("airport.csv") %>%
    select(airport_info_cols)
    # st_as_sf(coords = c('lat', 'lon'))

# airport_coord = airport_info[c('ICAO', 'lat', 'long')]

route_to_st_linestring = function(route, airport_info) {
    route = stringr::str_split(route, pattern = "-", simplify = TRUE)

    get_coord = function(airport) {
        filter(airport_info, ICAO == airport)[c('lat', 'lon')]
    }

    st_linestring(as.matrix(rbind(get_coord(route[1]), get_coord(route[2]))))
}

get_coord = function(iaco_code, airport_info) {
    filter(airport_info, ICAO == iaco_code)[c('lat', 'lon')]
}


route_edge = route_df %>%
    mutate_all(as.character) %>%
    distinct() %>%
    separate(Routes, into = c("from", "to")) %>%
    left_join(airport_coord, by = c("from" = "ICAO", "to" = "ICAO"))

route_node = routes_df %>%
    separate(Routes, into = c('from', 'to')) %>%
    gather(type, node) %>%
    group_by(node) %>%
    summarise(node_weight = length(node))

route_node %>%
    left_join(airport_info, by = c("node" = "ident")) %>%
    rename(lat = "latitude_deg", lon = "lontitude_deg")

tibble(name = letters[1:26],
       point_x = 1:26,
       point_y = 26:1) %>%
    mutate(point = bind_cols(point_x, point_y))


t(airport_info$lat, airport_info$lon)
