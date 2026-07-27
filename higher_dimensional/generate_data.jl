# run only once starting from profiles wide with only year 1995 (in github there will already be the generated data)
using Random
using DataFrames
using CSV
using Plots

function noise_ar(; phi=0.9, sigma=0.02)
    eps = zeros(8760)
    eps[1] = sigma * randn() # N(0,1)
    for t in 2:8760
        eps[t] = phi * eps[t-1] + sigma * randn() # to consider autocorrelation
    end
    return eps
end

function modify_series(x, scale; phi=0.9, sigma=0.02)
    noise = noise_ar(phi=phi, sigma=sigma)
    y = scale .* x .* (1 .+ noise)
    y[y.<0] .= 0
    return y
end

function modify_solar(x, scale; phi=0.9, sigma=0.02)
    noise = noise_ar(phi=phi, sigma=sigma)
    y = scale .* x .* (1 .+ noise)
    y[y.<0] .= 0
    y[x.==0] .= 0  # enforce night = 0
    return y
end

function make_scenario(df, scenario;
    scale_solar=1.0, scale_wind_on=1.0, scale_wind_off=1.0,
    scale_hydro=1.0, scale_demand=1.0,
    phi=0.95, sigma=0.01)

    solar = modify_solar(df.solar, scale_solar; phi=phi, sigma=sigma)
    wind_off = modify_series(df.wind_offshore, scale_wind_off; phi=phi, sigma=sigma)
    wind_on = modify_series(df.wind_onshore, scale_wind_on; phi=phi, sigma=sigma)
    demand = modify_series(df.demand, scale_demand; phi=phi, sigma=sigma)
    hydro = modify_series(df.hydro_inflow, scale_hydro; phi=phi, sigma=sigma)

    out = copy(df)
    out.solar = solar
    out.wind_offshore = wind_off
    out.wind_onshore = wind_on
    out.demand = demand
    out.hydro_inflow = hydro

    out.scenario .= scenario

    return out
end

function plot_scenarios(red_df, blue_df, profiles_type)
    # plot periods: at 12 in the morning
    images_dir = joinpath(@__DIR__, "images_$profiles_type",)
    mkpath(images_dir)
    red_df = filter(row -> row.timestep % 24 == 15, red_df)
    blue_df = filter(row -> row.timestep % 24 == 15, blue_df)

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

    plot!(
        plt;
        xlabel="Onshore wind availability",
        ylabel="Demand",
        title=profiles_type,
        #legend=:topright,
        grid=true,
        framestyle=:box,
        ratio=1,
        xlims=(0.0, 1.1),
        ylims=(0.0, 1.1),
    )
    savefig(plt, joinpath(images_dir, "scenarios_onshore.png"))

    plt = scatter(
        red_df.wind_offshore, red_df.demand;
        color=:red,
        label="Scenario 1",
        markersize=5,
        markerstrokewidth=0.5,
    )

    # add blue points
    scatter!(
        plt,
        blue_df.wind_offshore, blue_df.demand;
        color=:blue,
        label="Scenario 2",
        markersize=5,
        markerstrokewidth=0.5,
    )

    plot!(
        plt;
        xlabel="Offshore wind availability",
        ylabel="Demand",
        title=profiles_type,
        #legend=:topright,
        grid=true,
        framestyle=:box,
        ratio=1,
        xlims=(0.0, 1.1),
        ylims=(0.0, 1.1),
    )
    savefig(plt, joinpath(images_dir, "scenarios_offshore.png"))

    plt = scatter(
        red_df.solar, red_df.demand;
        color=:red,
        label="Scenario 1",
        markersize=5,
        markerstrokewidth=0.5,
    )

    # add blue points
    scatter!(
        plt,
        blue_df.solar, blue_df.demand;
        color=:blue,
        label="Scenario 2",
        markersize=5,
        markerstrokewidth=0.5,
    )

    plot!(
        plt;
        xlabel="Solar availability",
        ylabel="Demand",
        title=profiles_type,
        #legend=:topright,
        grid=true,
        framestyle=:box,
        ratio=1,
        xlims=(0.0, 1.1),
        ylims=(0.0, 1.1),
    )
    savefig(plt, joinpath(images_dir, "scenarios_solar.png"))
end



Random.seed!(1)

output_dir = joinpath(@__DIR__, "input_data/storage/")
df_profiles = CSV.read(output_dir * "profiles-wide.csv", DataFrame)

## DISTANT CASE ##

red_df = make_scenario(df_profiles, 0;
    scale_solar=0.7, scale_wind_on=0.7,
    scale_wind_off=0.7
)
blue_df = make_scenario(df_profiles, 1;
    scale_solar=1.3, scale_wind_on=1.3,
    scale_wind_off=1.3
)

df_profilesd = vcat(red_df, blue_df)

# write to CSV
filename = joinpath(output_dir, "profiles-wide_DISTANT.csv")
CSV.write(filename, df_profilesd)
plot_scenarios(red_df, blue_df, "DISTANT")


## CLOSE CASE ##

red_df = make_scenario(df_profiles, 0;
    scale_solar=0.9, scale_wind_on=0.9,
    scale_wind_off=0.9
)
blue_df = make_scenario(df_profiles, 1;
    scale_solar=1.1, scale_wind_on=1.1,
    scale_wind_off=1.1
)

df_profilesc = vcat(red_df, blue_df)

# write to CSV
filename = joinpath(output_dir, "profiles-wide_CLOSE.csv")
CSV.write(filename, df_profilesc)
plot_scenarios(red_df, blue_df, "CLOSE")

## HALFMIXED CASE ##

red_df = make_scenario(df_profiles, 0;
    scale_solar=0.7, scale_wind_on=1.0,
    scale_wind_off=0.7
)
blue_df = make_scenario(df_profiles, 1;
    scale_solar=1.3, scale_wind_on=1.0,
    scale_wind_off=1.3
)

df_profileshm = vcat(red_df, blue_df)

# write to CSV
filename = joinpath(output_dir, "profiles-wide_HALFMIXED.csv")
CSV.write(filename, df_profileshm)
plot_scenarios(red_df, blue_df, "HALFMIXED")


## MIXED CASE ##

red_df = make_scenario(df_profiles, 0;
    scale_solar=1.0, scale_wind_on=1.0,
    scale_wind_off=1.0
)
blue_df = make_scenario(df_profiles, 1;
    scale_solar=1.0, scale_wind_on=1.0,
    scale_wind_off=1.0
)

df_profilesm = vcat(red_df, blue_df)

# write to CSV
filename = joinpath(output_dir, "profiles-wide_MIXED.csv")
CSV.write(filename, df_profilesm)

plot_scenarios(red_df, blue_df, "MIXED")

## EQUAL CASE ##

red_df = copy(df_profiles)
red_df.scenario .= 0

blue_df = copy(df_profiles)
blue_df.scenario .= 1

df_profilese = vcat(red_df, blue_df)

# write to CSV
filename = joinpath(output_dir, "profiles-wide_EQUAL.csv")
CSV.write(filename, df_profilese)


# now for no-storage
select!(df_profilesc, Not(:hydro_inflow))
select!(df_profilesd, Not(:hydro_inflow))
select!(df_profileshm, Not(:hydro_inflow))
select!(df_profilesm, Not(:hydro_inflow))
select!(df_profilese, Not(:hydro_inflow))

output_dir = joinpath(@__DIR__, "input_data/nostorage/")

filename = joinpath(output_dir, "profiles-wide_DISTANT.csv")
CSV.write(filename, df_profilesd)
filename = joinpath(output_dir, "profiles-wide_CLOSE.csv")
CSV.write(filename, df_profilesc)
filename = joinpath(output_dir, "profiles-wide_HALFMIXED.csv")
CSV.write(filename, df_profileshm)
filename = joinpath(output_dir, "profiles-wide_MIXED.csv")
CSV.write(filename, df_profilesm)
filename = joinpath(output_dir, "profiles-wide_EQUAL.csv")
CSV.write(filename, df_profilese)