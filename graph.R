library(tidyverse)
library(leaflet)
library(htmlwidgets)

## data ---------------------------------------------------------------------

#### Panasonic data ####
data_paths = c("data/df_rfp_dataset_raw_20181218185047.csv",
               "data/df_rfp_dataset_raw_20190208223442.csv")

route_dfs = map(data_paths, {
    ~read_csv(.x, col_types = cols_only(Routes = "c"))
})

names(route_dfs) = basename(data_paths)

#### Airport info ####
airport_info_cols = c(
    ICAO = "ident",
    "name",
    lat = "lat",
    lon = "lon",
    country = "iso_country",
    region = "iso_region"
)

airport_info = read_csv("data/airport_info.csv") %>%
    select(airport_info_cols)

## leaflet ---------------------------------------------------------------------

make_flight_network = function(route_df, airport_info_df) {
    airport_coord = airport_info_df[c('ICAO', 'lat', 'lon')]

    get_coord = function(iaco_code, airport_info) {
        filter(airport_info, ICAO == iaco_code)[c('lat', 'lon')]
    }

    route_edge = route_df %>%
        mutateall(as.character) %>%
        group_by(Routes) %>%
        summarise(edge_weight = length(Routes)) %>%
        separate(Routes, into = c("from", "to")) %>%
        distinct() %>%
        left_join(airport_coord, by = c('from' = 'ICAO')) %>%
        left_join(airport_coord, by = c('to' = 'ICAO'))

    route_node = route_df %>%
        separate(Routes, into = c('from', 'to')) %>%
        gather(type, ICAO) %>%
        group_by(ICAO) %>%
        summarise(node_weight = length(ICAO)) %>%
        left_join(airport_info)

    route_edge %>%
        ggplot(aes(edge_weight)) +
        geom_histogram()


    ## Palette
    pal = colorNumeric(RColorBrewer::brewer.pal(9, "YlOrRd"), domain = NULL)
    node_pal = cut(route_node$node_weight, 9, labels = F)
    weight_pal = cut(route_edge$edge_weight, 9, labels = F)

    leaflet(route_node) %>%
        addProviderTiles(providers$CartoDB.DarkMatter) %>%
        addPolylines(
            lat = ~c(lat.x, lat.y), lng = ~c(lon.x, lon.y),
            data = route_edge,
            color = ~pal(-weight_pal),
            weight = weight_pal^3.5/250,
            opacity = 0.7,
            label = ~paste0(from, "-", to, "; flights: ", edge_weight)
        ) %>%
        addCircles(
            lat = ~lat, lng = ~lon,
            fillColor = ~pal(-node_pal),
            radius = ~node_weight^0.9*1.2,
            stroke = F,
            fillOpacity = 0.7,
            label = ~paste0(name, "; flights: ", node_weight)
        )
}

network_leaflets = map(route_dfs, ~make_flight_network(.x, airport_info))

network_leaflets[[1]]

map2(network_leaflets, names(network_leaflets), {
    ~saveWidget(
        widget = .x,
        file = paste0("flight_network", gsub(".csv", "", .y), ".html"),
        selfcontained = TRUE
    )
})
