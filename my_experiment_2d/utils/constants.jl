# Marker and color mappings for plots
const MARKER_MAP = Dict(
    "per_scenario" => :utriangle,
    "cross_scenario" => :circle,
    "per_and_cross_scenario" => :star5
)

const PROFILES_TYPES_MAP = Dict(
    0 => :black,
    1 => :darkgray,
    2 => :gray,
    3 => :silver,
    4 => :lightgray,
    5 => :white
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
