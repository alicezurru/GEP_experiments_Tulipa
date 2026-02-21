# Marker and color mappings for plots
const MARKER_MAP = Dict(
    "per_scenario" => :utriangle,
    "cross_scenario" => :circle,
    #"per_and_cross_scenario" => :star5
)

const COLOR_MAP_weight = Dict(
    "dirac" => :red,
    "convex" => :black,
    "conical" => :green,
    "conical_bounded" => :yellow,
)
const COLOR_MAP_method = Dict(
    "convex_hull" => :black,
    "convex_hull_with_null" => :yellow,
    "conical_hull" => :green
)

const FILLER_MAP = Dict(
    "dirac" => :white,
    "convex" => :black,
    "conical" => :green,
    "conical_bounded" => :yellow
)

const COLOR_MAP_method_weight = Dict(
    "k_means_convex_per_scenario" => :gold,
    "k_means_convex_cross_scenario" => :orange,
    "k_means_dirac_per_scenario" => :red,
    "k_means_dirac_cross_scenario" => :red,
    "k_medoids_convex_per_scenario" => :blue,
    "k_medoids_convex_cross_scenario" => :purple,
    "k_medoids_dirac_per_scenario" => :red,
    "k_medoids_dirac_cross_scenario" => :red,
)