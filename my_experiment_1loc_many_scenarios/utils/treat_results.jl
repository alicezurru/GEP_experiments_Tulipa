# Run this only after results given from main: this is for algorithms that are not deterministic such as k-medoids and k-means

using DataFrames
using CSV
using Plots
using Statistics

# helper functions
@info "Including constants and helper functions"
include("constants.jl")
include("functions.jl")

script_dir = @__DIR__
output_path = joinpath(script_dir, "..", "outputs")

results_df = CSV.read(joinpath(output_path, "results.csv"), DataFrame)

# extract benchmark rows
benchmark_df = results_df[
    results_df.base_name.=="0_HourlyBenchmark",
    [:n_experiment, :objective_value,
        :num_loss_of_load_e_demand,
        :loss_of_load_e_demand]
]

rename!(benchmark_df, Dict(
    :objective_value => :hourly_obj,
    :num_loss_of_load_e_demand => :hourly_lol_count,
    :loss_of_load_e_demand => :hourly_lol_total
))

# join benchmark info back
results_df = leftjoin(results_df, benchmark_df, on=:n_experiment)

# compute relative regret
results_df.rel_regret = ifelse.(
    results_df.base_name .== "0_HourlyBenchmark",
    0.0,
    (results_df.objective_value_resolve_benchmark .- results_df.hourly_obj) ./ results_df.hourly_obj
)

# compute number of loss of load timesteps difference
results_df.num_loss_of_load_e_demand_diff =
    results_df.num_loss_of_load_e_demand .- results_df.hourly_lol_count

# compute loss of load difference
results_df.loss_of_load_e_demand_diff =
    results_df.loss_of_load_e_demand .- results_df.hourly_lol_total

# compute total time 
results_df.total_time =
    results_df.time_to_cluster .+
    results_df.time_to_read .+
    results_df.time_to_create .+
    results_df.time_to_solve .+
    results_df.time_to_save


# compute statistics for different seeds
results_df = combine(
    groupby(results_df, [:base_name, :rp, :n_experiment]), :time_to_cluster => mean => :time_to_cluster_mean,
    :time_to_read => mean => :time_to_read_mean,
    :time_to_create => mean => :time_to_create_mean,
    :time_to_solve => mean => :time_to_solve_mean,
    :time_to_save => mean => :time_to_save_mean,
    :total_time => mean => :total_time_mean,
    :rel_regret => mean => :rel_regret_mean,
    :num_loss_of_load_e_demand => mean => :num_loss_of_load_e_demand_mean,
    :loss_of_load_e_demand => mean => :loss_of_load_e_demand_mean,
)

#results_df |> CSV.write(joinpath(output_path, "stats.csv"); writeheader=true)

# remove benchmark
df = results_df[results_df.base_name.!="0_HourlyBenchmark", :]

# split base_name into method + variant
df.method = first.(split.(df.base_name, "_"))
df.variant = getindex.(split.(df.base_name, "_"), 2)  # "cross" or "per"

# keep relevant columns
df_small = select(df,
    :method, :variant, :rp, :n_experiment,
    :num_loss_of_load_e_demand_mean,
    :loss_of_load_e_demand_mean
)

# make cross & per columns
keys = [:method, :rp, :n_experiment]

df1 = unstack(df_small, keys, :variant, :num_loss_of_load_e_demand_mean)
rename!(df1, Dict(
    "cross" => :num_lol_cross,
    "per" => :num_lol_per
))

df2 = unstack(df_small, keys, :variant, :loss_of_load_e_demand_mean)
rename!(df2, Dict(
    "cross" => :lol_cross,
    "per" => :lol_per
))
df_wide = innerjoin(df1, df2, on=keys)

# # compute ratios
# df_wide.num_lol_ratio = ifelse.(
#     df_wide.num_lol_per .== 0,
#     ifelse.(df_wide.num_lol_cross .== 0, 0.0, -Inf),
#     (df_wide.num_lol_per .- df_wide.num_lol_cross) ./ df_wide.num_lol_per
# )

# df_wide.lol_ratio = ifelse.(
#     df_wide.lol_per .== 0,
#     ifelse.(df_wide.lol_cross .== 0, 0.0, -Inf),
#     (df_wide.lol_per .- df_wide.lol_cross) ./ df_wide.lol_per
# )

df_wide = combine(
    groupby(df_wide, [:method, :rp]), :num_lol_cross => mean => :num_lol_cross,
    :num_lol_per => mean => :num_lol_per,
    :lol_cross => mean => :lol_cross,
    :lol_per => mean => :lol_per,
    # :num_lol_ratio => mean => :num_lol_ratio,
    # :lol_ratio => mean => :lol_ratio,
)

df_wide.num_lol_ratio_avg = (
    df_wide.num_lol_per .- df_wide.num_lol_cross
) ./ df_wide.num_lol_per

df_wide.lol_ratio_avg = (
    df_wide.lol_per .- df_wide.lol_cross
) ./ df_wide.lol_per

df_wide |> CSV.write(joinpath(output_path, "comparison2.csv"); writeheader=true)