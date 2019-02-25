source("R/data_reader.R")
library(lubridate)
library(tidyverse)
library(leaflet)

data_path = c("data/df_rfp_dataset_raw_20181218185047.csv",
              "data/df_rfp_dataset_raw_20190208223442.csv")

both_dataset = map(data_path, ~data_reader(.x, data_ref = "data_reference.csv"))
names(both_dataset) = basename(data_path)


## Overlap?
get_col_intersect = function(df_list, column_name) {
    data = map(df_list, ~.x[[column_name]])

    intersect(data[[1]], data[[2]])
}

get_col_union = function(df_list, column_name) {
    data = map(df_list, ~.x[[column_name]])

    union(data[[1]], data[[2]])
}

intersect_list = map(colnames(both_dataset[[1]]), ~get_col_intersect(both_dataset, .x))
names(intersect_list) = colnames(both_dataset[[1]])


## Dates
## Using departure time as reference
map(both_dataset, ~(range(.x$DepartureTimeUTC)))


## Countries and routes
orig_country_union = get_col_union(both_dataset, "Orig_Country")
dest_country_union = get_col_union(both_dataset, "Dest_Country")

all_routes = get_col_union(both_dataset, "Routes")
route_df = tibble(route = all_routes) %>%
    separate(route, into = c("orig", "dest"))

unique_airports = unique(c(route_df$orig, route_df$dest))

## Geocoding
ggmap::register_google(Sys.getenv("GMAP_KEY"))

latlons = ggmap::geocode(country_union)
country_latlon = tibble(country = country_union) %>%
    bind_cols(latlons)

airport_info = read_csv("airport.csv")
airport_info_cols = c(
    "ident",
    "name",
    lat = "latitude_deg",
    lon = "longitude_deg",
    country = "iso_country",
    region = "iso_region"
)

airport_info = airport_info %>%
    select(airport_info_cols)


airport_lonlat = ggmap::geocode(unique_airports)
airport_lonlat = tibble(airport = unique_airports) %>%
    left_join(airport_info, by = c("airport" = "ident"))

write_csv(country_latlon, "country_latlon.csv")

leaflet(airport_lonlat) %>%
    addCircles(popup = ~paste(name)) %>%
    addTiles()


route_df1 = both_dataset[[1]] %>%
    distinct(FlightID, Routes)

route_df1 %>%
    group_by(Routes) %>%
    summarise(route_weight = length(Routes))

nodes_df1 = str_split(route_df1$Routes, "-", simplify = TRUE) %>%

