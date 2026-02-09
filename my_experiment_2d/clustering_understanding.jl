
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
import DBInterface
using DataFrames
using Plots
using Revise

# helper functions
@info "Including helper functions"
include("utils/functions.jl")

adding_initial_periods = false

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
use_ratio = config["normalization"]["use_ratio"]
profiles_type = config["simulation"]["profiles_type"]

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
        :profiles_type => Int,
        :run_case => Bool,
    ),
)

solvers = [:Gurobi] #[:HiGHS, :Gurobi]
representative_periods = [2, 4, 6]

enable_names = true
direct_model = false
n_scenarios = 2


function main()

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
        profiles_type = row[:profiles_type]

        weight_fitting_kwargs = Dict(
            :learning_rate => learning_rate,
            :niters => niters
        )

        if !run_case
            continue
        end

        for rp in representative_periods
            case_name = base_name * "_rp_" * "$rp"

            @info "Processing case study: $case_name"

            connection = DuckDB.DBInterface.connect(DuckDB.DB)
            TIO.read_csv_folder(connection, input_data_path)

            # transform the profiles data from wide to long
            TC.transform_wide_to_long!(
                connection,
                "profiles_wide_$profiles_type",
                "profiles";
                exclude_columns=["scenario", "year", "timestep"],
            )

            if adding_initial_periods

                DuckDB.query(
                    connection,
                    "
    CREATE TABLE init_rps AS
    SELECT *
    FROM profiles_wide_$profiles_type
    WHERE timestep = 2
    AND (scenario = 0 OR scenario = 1);
    "
                )
                DuckDB.query(
                    connection,
                    "
    ALTER TABLE init_rps
    ADD COLUMN period INTEGER DEFAULT 1;
    "
                )
                DuckDB.query(
                    connection,
                    "

    UPDATE init_rps
    SET timestep = 1;
    "
                )
            end


            if stochastic_method == :per_scenario
                layout = TC.ProfilesTableLayout(; cols_to_groupby=[:year, :scenario])
                if adding_initial_periods
                    TC.transform_wide_to_long!(
                        connection,
                        "init_rps",
                        "initial_representatives";
                        exclude_columns=["scenario", "year", "timestep", "period"],
                    )
                    initial_representatives_df = TIO.get_table(connection, "initial_representatives")
                    @show names(initial_representatives_df)
                    initial_representatives_df = initial_representatives_df[:,
                        [:timestep, :year, :scenario, :period, :profile_name, :value]
                    ] # otherwise it throws an error (I am only reordering the columns)
                    TC.cluster!(
                        connection,
                        period_duration,
                        round(Int, rp / n_scenarios);
                        method=method,
                        distance=distance,
                        initial_representatives=initial_representatives_df,
                        weight_type=weight_type,
                        layout=layout,
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
                        weight_fitting_kwargs
                    )
                end


            elseif stochastic_method == :cross_scenario
                layout = TC.ProfilesTableLayout(; cols_to_groupby=[:year], cols_to_crossby=[:scenario])
                if adding_initial_periods
                    DuckDB.query(
                        connection,
                        "
        UPDATE init_rps
        SET period = 2
        WHERE scenario = 1
        "
                    )
                    TC.transform_wide_to_long!(
                        connection,
                        "init_rps",
                        "initial_representatives";
                        exclude_columns=["scenario", "year", "timestep", "period"],
                    )
                    initial_representatives_df = TIO.get_table(connection, "initial_representatives")
                    @show names(initial_representatives_df)
                    initial_representatives_df = initial_representatives_df[:,
                        [:timestep, :year, :scenario, :period, :profile_name, :value]
                    ] # otherwise it throws an error (I am only reordering the columns)

                    #select!(initial_representatives_df, Not(:scenario))
                    TC.cluster!(
                        connection,
                        period_duration,
                        rp;
                        method=method,
                        distance=distance,
                        initial_representatives=initial_representatives_df,
                        weight_type=weight_type,
                        layout=layout,
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
                        weight_fitting_kwargs
                    )
                end
            elseif stochastic_method == :per_and_cross_scenario # new method
                # for now without adding possibility of artificial period - maybe add later 
                clustering_kwargs = Dict(
                    :add_cross => true,
                    :n_scenarios => n_scenarios
                )
                layout = TC.ProfilesTableLayout(; cols_to_groupby=[:year, :scenario])

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

            else
                error("Unknown stochastic method: $stochastic_method")
            end

            df_profiles_rp = TIO.get_table(connection, "profiles_rep_periods")
            df_rp_mapping = TIO.get_table(connection, "rep_periods_mapping")
            df_rep_periods_data = TIO.get_table(connection, "rep_periods_data")
            df_timeframe_data = TIO.get_table(connection, "timeframe_data")
            output_folder = joinpath(@__DIR__, "case_$profiles_type", case_name)
            mkpath(output_folder)
            CSV.write(joinpath(output_folder, "profiles_rep_periods"), df_profiles_rp)
            CSV.write(joinpath(output_folder, "rep_periods_mapping"), df_rp_mapping)
            CSV.write(joinpath(output_folder, "rep_periods_data"), df_rep_periods_data)
            CSV.write(joinpath(output_folder, "timeframe_data"), df_timeframe_data)


            df_profiles = CSV.read("input_data/profiles-wide_$profiles_type.csv", DataFrame)
            red_df = filter(row -> row.scenario == 0, df_profiles)
            blue_df = filter(row -> row.scenario == 1, df_profiles)

            # red
            plt = scatter(
                red_df.wind_onshore, red_df.demand;
                color=:red,
                label="Scenario 1",
                markersize=5,
                markerstrokewidth=0.5,
            )

            # add blue points
            scatter!(
                plt,
                blue_df.wind_onshore, blue_df.demand;
                color=:blue,
                label="Scenario 2",
                markersize=5,
                markerstrokewidth=0.5,
            )


            rp_demand = filter(row -> row.profile_name == "demand", df_profiles_rp)
            rp_generation = filter(row -> row.profile_name != "demand", df_profiles_rp)
            if stochastic_method == :per_scenario
                rp_joined = innerjoin(rp_demand, rp_generation,
                    on=[:rep_period, :timestep, :year, :scenario],
                    makeunique=true)
                rename!(rp_joined, Dict(:value => :demand, :value_1 => :availability, :profile_name_1 => :technology))
                rp_joined_red = filter(row -> row.scenario == 0, rp_joined)
                rp_joined_blue = filter(row -> row.scenario == 1, rp_joined)
                scatter!(plt, rp_joined_red.availability, rp_joined_red.demand,
                    color=:black,
                    marker=:circle,
                    label="Representative Periods red scenario"
                )
                scatter!(plt, rp_joined_blue.availability, rp_joined_blue.demand,
                    color=:yellow,
                    marker=:circle,
                    label="Representative Periods blue scenario"
                )

            else
                rp_joined = innerjoin(rp_demand, rp_generation,
                    on=[:rep_period, :timestep, :year],
                    makeunique=true)

                rename!(rp_joined, Dict(:value => :demand, :value_1 => :availability, :profile_name_1 => :technology))
                scatter!(plt, rp_joined.availability, rp_joined.demand,
                    color=:black,
                    marker=:circle,
                    label="Representative Periods"
                )
            end


            plot!(
                plt;
                xlabel="Wind availability",
                ylabel="Demand",
                title="Base periods",
                #legend=:topright,
                grid=true,
                framestyle=:box,
                ratio=1,
                xlims=(0.0, 1.0),
                ylims=(0.0, 1.0),
            )

            savefig(plt, joinpath(output_folder, "scatter.png"))

        end
    end

    return nothing
end

main()