# THIS WORKS ONLY NOW THAT I DO NOT HAVE INVESTMENTS IN FLOWS OR IN STORAGE
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
add_worst_sum = config["extreme_periods"]["add_worst_sum"]

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
representative_periods = [4, 8, 12, 16, 20, 30, 60, 90, 120, 180, 240, 360]
representative_periods = [8]
enable_names = true
direct_model = false
results_df = DataFrame(;
    base_name=String[],
    rp=Int[],
    solver=Symbol[],
    time_to_cluster=Float64[],
    time_to_read=Float64[],
    time_to_create=Float64[],
    time_to_solve=Float64[],
    time_to_save=Float64[],
    objective_value=Float64[],
    termination_status=String[],
    num_constraints=Int[],
    num_variables=Int[],
    time_to_resolve_benchmark=Float64[],
    objective_value_resolve_benchmark=Float64[],
    termination_status_resolve_benchmark=String[],
    num_loss_of_load_e_demand=Int[],
    penalty_loss_of_load_e_demand=Float64[],
    investment_cost=Float64[],
    operational_cost=Float64[],)

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
                    if add_worst_every_hour || add_worst_sum
                        init_rps_df = TIO.get_table(connection, "init_rps")
                        init_rps_df = init_rps_df[:,
                            [:timestep, :year, :scenario, :period, :profile_name, :value]
                        ] # otherwise it throws an error (I am only reordering the columns)
                        time_to_cluster = @elapsed TC.cluster!(
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
                        time_to_cluster = @elapsed TC.cluster!(
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
                    if add_worst_every_hour || add_worst_sum
                        init_rps_df = TIO.get_table(connection, "init_rps")
                        init_rps_df = init_rps_df[:,
                            [:timestep, :year, :scenario, :period, :profile_name, :value]
                        ] # otherwise it throws an error (I am only reordering the columns)
                        time_to_cluster = @elapsed TC.cluster!(
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
                        time_to_cluster = @elapsed TC.cluster!(
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
                    output_folder = joinpath(@__DIR__, "case_explore_ratio_$profiles_type", case_name)
                else
                    output_folder = joinpath(@__DIR__, "case_explore_$profiles_type", case_name)
                end
                mkpath(output_folder)
                CSV.write(joinpath(output_folder, "profiles_rep_periods"), df_profiles_rp)
                CSV.write(joinpath(output_folder, "rep_periods_mapping"), df_rp_mapping)
                CSV.write(joinpath(output_folder, "rep_periods_data"), df_rep_periods_data)
                CSV.write(joinpath(output_folder, "timeframe_data"), df_timeframe_data)
                df = TIO.get_table(connection, "profiles_rep_periods")
                rep_periods = unique(df.rep_period)
                plots = []
                for rp in rep_periods
                    df_rp = filter(row -> row.rep_period == rp, df)
                    p = plot(size=(100, 300), title="RP $rp")
                    label_map = Dict(
                        "solar" => "Solar",
                        "wind_onshore" => "Wind Onshore",
                        "wind_offshore" => "Wind Offshore",
                        "demand" => "Demand"
                    )

                    for group in groupby(df_rp, :profile_name)
                        raw_name = group.profile_name[1]
                        name = get(label_map, raw_name, raw_name)
                        plot!(p, group.timestep, group.value, label=name)
                    end

                    show_legend = (rp == rep_periods[1])
                    plot!(p,
                        xlabel="Timestep",
                        ylabel="Value",
                        xticks=0:2:period_duration,
                        xlim=(1, period_duration),
                        ylim=(0, 1.5),
                        legend=show_legend ? :topleft : false,
                        legendfontsize=6
                    )
                    push!(plots, p)
                end
                final_plot = plot(plots..., layout=(2, 4), size=(1200, 800))
                mkpath("outputs/plots")
                savefig(final_plot, "outputs/plots/plot_representative_periods.png")


            end
        end
    end

    return nothing
end

main()