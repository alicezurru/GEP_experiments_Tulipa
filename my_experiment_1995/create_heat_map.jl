cd(@__DIR__)
using Pkg: Pkg
Pkg.activate(".")

# Load the required packages
import TulipaEnergyModel as TEM
import TulipaIO as TIO
import TulipaClustering as TC
import DuckDB
import HiGHS
import Gurobi
import Distances
import CSV
import Statistics
import JuMP
import TOML
using Revise
using DataFrames
using Random
using CSV
using Plots

# helper functions
@info "Including helper functions"
include("utils/functions.jl")

distance_map = Dict(
    :Euclidean => Distances.Euclidean(),
    :SqEuclidean => Distances.SqEuclidean(),
    :CosineDist => Distances.CosineDist(),
    :Cityblock => Distances.Cityblock(),
    :Chebyshev => Distances.Chebyshev(),
)

# Read and transform user input files to Tulipa input files
config = TOML.parsefile("config.toml")
input_data_path = config["simulation"]["input_data"]
use_ratio = config["clustering"]["use_ratio"]
heuristic_distance = config["clustering"]["heuristic_distance"]
profiles_type = config["simulation"]["profiles_type"]
n_runs = config["simulation"]["n_runs"]
add_worst_every_hour = config["extreme_periods"]["add_worst_every_hour"]
add_best_every_hour = config["extreme_periods"]["add_best_every_hour"]
add_worst_sum = config["extreme_periods"]["add_worst_sum"]
add_worst_sum_real = config["extreme_periods"]["add_worst_sum_real"]
if add_best_every_hour && !add_worst_every_hour
    error("You can add best only when also worst is added")
end
if use_ratio
    if input_data_path == "input_data/storage/"
        error("Ratio is implemented only with no storage right now")
    end
end
if add_worst_sum_real
    if add_worst_every_hour || add_worst_sum
        error("You cannot add them together")
    end
end

case_studies_info = CSV.read(
    "case-studies-info.csv",
    DataFrame;
    types=Dict(
        :base_name => String,
        :period_duration => Int,
        :method => Symbol,
        :distance => Symbol,
        :weight_type => Symbol,
        :niters => Int,
        :learning_rate => Float64,
        :stochastic_method => Symbol,
        :run_case => Bool,
    ),
)

solvers = [:Gurobi] #[:HiGHS, :Gurobi]
representative_periods = [4, 8, 16]
enable_names = true
direct_model = false

function main()

    n_scenarios = 2

    # optimize the energy system for each case study
    for row in eachrow(case_studies_info)
        base_name = row[:base_name]
        period_duration = row[:period_duration]
        method = row[:method]
        distance = distance_map[row[:distance]]
        weight_type = row[:weight_type]
        niters = row[:niters]
        learning_rate = row[:learning_rate]
        stochastic_method = row[:stochastic_method]
        run_case = row[:run_case]

        weight_fitting_kwargs = Dict(
            :learning_rate => learning_rate,
            :niters => niters
        )
        if method ∉ [:k_means, :k_medoids]
            clustering_kwargs = Dict(
                :learning_rate => learning_rate,
                :niters => niters,
                :heuristic_distance => heuristic_distance,
            )
        else
            clustering_kwargs = Dict{Symbol,Any}()
        end

        if !run_case
            continue
        end

        for rp in representative_periods
            case_name = base_name * "_rp_" * "$rp"

            @info "Processing case study: $case_name"

            for n in 1:n_runs
                Random.seed!(n)

                connection = DuckDB.DBInterface.connect(DuckDB.DB)
                TIO.read_csv_folder(connection, input_data_path)
                # let us add extreme periods 
                if add_worst_every_hour
                    # BE CAREFUL: right now this holds only if ratio is not considered
                    if use_ratio
                        error("You still need to implement artificial rps with ratio!")
                    end
                    profiles = ["solar", "wind_onshore", "wind_offshore"]
                    storage = DuckDB.query(
                        connection,
                        "
                            SELECT EXISTS (
                                    SELECT 1
                                    FROM assets_profiles
                                    WHERE asset = 'hydro_reservoir'
                        ) AS has_hydro
                            "
                    ) |> DataFrame
                    storage = storage.has_hydro[1]
                    if storage
                        push!(profiles, "hydro_inflow")
                    end
                    create_init_rps_hourly(connection, profiles_type, period_duration, profiles) # BE CAREFUL: tested only for 1 year
                    if add_best_every_hour
                        create_init_rps_hourly_best(connection, profiles_type, period_duration, profiles)
                    end
                end

                if add_worst_sum
                    if use_ratio
                        error("You still need to implement artificial rps with ratio!")
                    end
                    profiles = ["solar", "wind_onshore", "wind_offshore"]
                    storage = DuckDB.query(
                        connection,
                        "
                            SELECT EXISTS (
                                    SELECT 1
                                    FROM assets_profiles
                                    WHERE asset = 'hydro_reservoir'
                        ) AS has_hydro
                            "
                    ) |> DataFrame
                    storage = storage.has_hydro[1]
                    if storage
                        push!(profiles, "hydro_inflow")
                    end
                    create_init_rps_daily(connection, profiles_type, period_duration, profiles) # BE CAREFUL: tested only for 1 year
                end

                if add_worst_sum_real
                    if use_ratio
                        error("You still need to implement artificial rps with ratio!")
                    end
                    profiles = ["solar", "wind_onshore", "wind_offshore"]
                    storage = DuckDB.query(
                        connection,
                        "
                            SELECT EXISTS (
                                    SELECT 1
                                    FROM assets_profiles
                                    WHERE asset = 'hydro_reservoir'
                        ) AS has_hydro
                            "
                    ) |> DataFrame
                    storage = storage.has_hydro[1]
                    if storage
                        push!(profiles, "hydro_inflow")
                    end
                    create_init_rps_daily_real(connection, profiles_type, period_duration, profiles) # BE CAREFUL: tested only for 1 year
                end

                # to use the ratio availability/demand
                if use_ratio == true # be careful: this works now that we have only one demand location and one availability, so we divide availability by that only demand
                    DuckDB.query(
                        connection,
                        "
                        UPDATE profiles_wide_$profiles_type
                        SET
                            solar = solar / demand,
                            wind_offshore = wind_offshore / demand,
                            wind_onshore = wind_onshore / demand,
                        "
                    ) # hydro_inflow = hydro_inflow / demand; ADD IF STORAGE
                end

                # transform the profiles data from wide to long
                TC.transform_wide_to_long!(
                    connection,
                    "profiles_wide_$profiles_type",
                    "profiles";
                    exclude_columns=["scenario", "year", "timestep"],
                )

                if stochastic_method == :per_scenario
                    layout = TC.ProfilesTableLayout(; cols_to_groupby=[:year, :scenario])
                    if add_worst_every_hour || add_worst_sum || add_worst_sum_real
                        init_rps_df = TIO.get_table(connection, "init_rps")
                        init_rps_df = init_rps_df[:,
                            [:timestep, :year, :scenario, :period, :profile_name, :value]
                        ] # otherwise it throws an error (I am only reordering the columns)
                        TC.cluster!(
                            connection,
                            period_duration,
                            round(Int, rp / n_scenarios);
                            method=method,
                            distance=distance,
                            initial_representatives=init_rps_df,
                            weight_type=weight_type,
                            layout=layout,
                            clustering_kwargs,
                            weight_fitting_kwargs
                        )
                    else
                        TC.cluster!(
                            connection,
                            period_duration,
                            round(Int, rp / n_scenarios);
                            method=method,
                            distance=distance,
                            weight_type=weight_type,
                            layout=layout,
                            clustering_kwargs,
                            weight_fitting_kwargs
                        )
                    end
                    if use_ratio == true
                        DuckDB.query(connection,
                            "UPDATE profiles_rep_periods AS x
                                SET value =
                                    CASE
                                        WHEN x.profile_name = 'demand' THEN x.value
                                        ELSE x.value * d.value
                                    END
                                FROM profiles_rep_periods AS d
                                WHERE d.timestep   = x.timestep
                                AND d.rep_period       = x.rep_period
                                AND d.year       = x.year
                                AND d.scenario   = x.scenario
                                AND d.profile_name = 'demand';
                                    ")
                    end
                elseif stochastic_method == :cross_scenario
                    layout = TC.ProfilesTableLayout(; cols_to_groupby=[:year], cols_to_crossby=[:scenario])
                    if add_worst_every_hour || add_worst_sum || add_worst_sum_real
                        init_rps_df = TIO.get_table(connection, "init_rps")
                        init_rps_df = init_rps_df[:,
                            [:timestep, :year, :scenario, :period, :profile_name, :value]
                        ] # otherwise it throws an error (I am only reordering the columns)
                        TC.cluster!(
                            connection,
                            period_duration,
                            rp;
                            method=method,
                            distance=distance,
                            initial_representatives=init_rps_df,
                            weight_type=weight_type,
                            layout=layout,
                            clustering_kwargs,
                            weight_fitting_kwargs
                        )
                    else
                        TC.cluster!(
                            connection,
                            period_duration,
                            rp;
                            method=method,
                            distance=distance,
                            weight_type=weight_type,
                            layout=layout,
                            clustering_kwargs,
                            weight_fitting_kwargs
                        )
                    end
                    if use_ratio == true
                        DuckDB.query(connection,
                            "UPDATE profiles_rep_periods AS x
                                SET value =
                                    CASE
                                        WHEN x.profile_name = 'demand' THEN x.value
                                        ELSE x.value * d.value
                                    END
                                FROM profiles_rep_periods AS d
                                WHERE d.timestep   = x.timestep
                                AND d.rep_period       = x.rep_period
                                AND d.year       = x.year
                                AND d.profile_name = 'demand';
                                    ")
                    end

                else
                    error("Unknown stochastic method: $stochastic_method")
                end
                if use_ratio == true
                    DuckDB.query(connection,
                        "UPDATE profiles AS x
                            SET value =
                                CASE
                                    WHEN x.profile_name = 'demand' THEN x.value
                                    ELSE x.value * d.value
                                END
                            FROM profiles AS d
                            WHERE d.timestep   = x.timestep
                            AND d.year       = x.year
                            AND d.scenario   = x.scenario
                            AND d.profile_name = 'demand';
                                ")
                end

                df_profiles_rp = TIO.get_table(connection, "profiles_rep_periods")
                df_rp_mapping = TIO.get_table(connection, "rep_periods_mapping")
                df_rep_periods_data = TIO.get_table(connection, "rep_periods_data")
                df_timeframe_data = TIO.get_table(connection, "timeframe_data")
                if use_ratio == true
                    output_folder = joinpath(@__DIR__, "case_mod_ratio_$profiles_type", case_name)
                elseif input_data_path == "input_data/storage/"
                    output_folder = joinpath(@__DIR__, "case_stor_$profiles_type", case_name)
                else
                    output_folder = joinpath(@__DIR__, "case_mod_$profiles_type", case_name)
                end

                mkpath(output_folder)
                CSV.write(joinpath(output_folder, "profiles_rep_periods"), df_profiles_rp)
                CSV.write(joinpath(output_folder, "rep_periods_mapping"), df_rp_mapping)
                CSV.write(joinpath(output_folder, "rep_periods_data"), df_rep_periods_data)
                CSV.write(joinpath(output_folder, "timeframe_data"), df_timeframe_data)

                # from other case study
                images_dir = joinpath(@__DIR__, "heat_map_$profiles_type", "$stochastic_method")
                mkpath(images_dir)

                rp_mapping_wide = unstack(df_rp_mapping, :rep_period, :weight; fill=0.0)
                M = Matrix(rp_mapping_wide[:, Not([:scenario, :year, :period])])
                hm = heatmap(
                    M;
                    xlabel="Representative periods",
                    ylabel="Base periods",
                    xticks=(0:rp:size(M, 2)),
                    yticks=(0:(8760/period_duration):size(M, 1)),
                    colorbar_title="Weight",
                    title="Representative period mapping",
                    color=:GnBu,
                )

                savefig(hm, joinpath(images_dir, "hm_$rp.png"))

            end
        end
    end
    return nothing
end

main()