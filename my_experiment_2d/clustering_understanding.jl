
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
using DataFrames

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
input_data_path = "input_data/"


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

            if stochastic_method == :per_scenario
                layout = TC.ProfilesTableLayout(; cols_to_groupby=[:year, :scenario])
                time_to_cluster = @elapsed TC.cluster!(
                    connection,
                    period_duration,
                    round(Int, rp / n_scenarios);
                    method=method,
                    distance=distance,
                    weight_type=weight_type,
                    layout=layout,
                    weight_fitting_kwargs
                )
            elseif stochastic_method == :cross_scenario
                layout = TC.ProfilesTableLayout(; cols_to_groupby=[:year], cols_to_crossby=[:scenario])
                time_to_cluster = @elapsed TC.cluster!(
                    connection,
                    period_duration,
                    rp;
                    method=method,
                    distance=distance,
                    weight_type=weight_type,
                    layout=layout,
                    weight_fitting_kwargs
                )
            else
                error("Unknown stochastic method: $stochastic_method")
            end

            df_profiles_rp = TIO.get_table(connection, "profiles_rep_periods")
            df_rp_mapping = TIO.get_table(connection, "rep_periods_mapping")
            output_folder = joinpath(@__DIR__, "case_$profiles_type", case_name)
            mkpath(output_folder)
            CSV.write(joinpath(output_folder, "profiles_rep_periods"), df_profiles_rp)
            CSV.write(joinpath(output_folder, "rep_periods_mapping"), df_rp_mapping)

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