# Run this only after results given from main

using DataFrames
using CSV
using Plots

# helper functions
@info "Including constants and helper functions"
include("constants.jl")
include("functions.jl")

mkpath("outputs/plots")
results = "my_experiment_2d/outputs/results.csv"
results_df = CSV.read(results, DataFrame)

hourly_row = results_df[results_df.base_name.=="0_HourlyBenchmark", :]
hourly_obj = only(hourly_row.objective_value)

# compute relative regret 
results_df.rel_regret = [
    row.base_name == "0_HourlyBenchmark" ? 0.0 :
    (row.objective_value_resolve_benchmark - hourly_obj) / hourly_obj
    for row in eachrow(results_df)
]

case_studies_path = "my_experiment_2d/case-studies-info.csv"
case_studies_df = CSV.read(case_studies_path, DataFrame)
output_folder = "my_experiment_2d/outputs/plots"
mkpath(output_folder)
@info "Plotting relative regret"
plot_values_stocmethod_method(results_df, case_studies_df, "rel_regret"; savepath="my_experiment_2d/outputs/plots/relative_regret.png")

@info "Plotting time to cluster"
plot_values_stocmethod_method(results_df, case_studies_df, "time_to_cluster"; savepath="my_experiment_2d/outputs/plots/time_to_cluster.png")

@info "Plotting time to solve"
plot_values_stocmethod_method(results_df, case_studies_df, "time_to_solve"; savepath="my_experiment_2d/outputs/plots/time_to_solve.png")

@info "Plotting time to create"
plot_values_stocmethod_method(results_df, case_studies_df, "time_to_create"; savepath="my_experiment_2d/outputs/plots/time_to_create.png")


# compute total time 
results_df.total_time = [
    row.time_to_cluster + row.time_to_read + row.time_to_create + row.time_to_solve + row.time_to_save
    for row in eachrow(results_df)
]

@info "Plotting total time"
plot_values_stocmethod_method(results_df, case_studies_df, "total_time"; savepath="my_experiment_2d/outputs/plots/total_time.png")

# compute number of steps with loss of load 
hourly_lol_e = only(hourly_row.num_loss_of_load_e_demand)
hourly_lol_h2 = only(hourly_row.num_loss_of_load_h2_demand)
results_df.num_loss_of_load_e_demand = [
    row.num_loss_of_load_e_demand - hourly_lol_e
    for row in eachrow(results_df)
]
results_df.num_loss_of_load_h2_demand = [
    row.num_loss_of_load_h2_demand - hourly_lol_h2
    for row in eachrow(results_df)
]
results_df.num_loss_of_load_tot = [
    row.num_loss_of_load_e_demand + row.num_loss_of_load_h2_demand
    for row in eachrow(results_df)
]


@info "Plotting number of steps with lol e_demand"
plot_values_stocmethod_method(results_df, case_studies_df, "num_loss_of_load_e_demand"; savepath="my_experiment_2d/outputs/plots/num_loss_of_load_e_demand.png")

# @info "Plotting number of steps with lol h2_demand"
# plot_values_stocmethod_method(results_df, case_studies_df, "num_loss_of_load_h2_demand"; savepath="outputs/plots/num_loss_of_load_h2_demand.png")

# @info "Plotting number of steps with lol e_demand"
# plot_values_stocmethod_method(results_df, case_studies_df, "num_loss_of_load_tot"; savepath="outputs/plots/num_loss_of_load_tot_demand.png")
