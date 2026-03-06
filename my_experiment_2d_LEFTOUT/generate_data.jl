using Random
using DataFrames
using CSV
using Plots

# to make results reproducible
Random.seed!(42)

function make_square(scenario, xmin, xmax, ymin, ymax) # this creates 100 points in the square with indicated vertices: 4 fixed corner points and 96 randomly generated inside
    n = 100
    year = fill(2030, n)
    scenario_col = fill(scenario, n)
    timestep = collect(1:n)

    wind = Vector{Float64}(undef, n)
    demand = Vector{Float64}(undef, n)


    wind[1:4] = [xmin, xmin, xmax, xmax]
    demand[1:4] = [ymax, ymin, ymax, ymin]

    for i in 5:n
        wind[i] = xmin + rand() * (xmax - xmin)
        demand[i] = ymin + rand() * (ymax - ymin)
    end

    return DataFrame(
        year=year,
        scenario=scenario_col,
        timestep=timestep,
        wind_onshore=wind,
        demand=demand,
    )
end


output_dir = "input_data/"
for i in 0:5 # closer and closer to each other
    increment = 0.05

    # to build each scenario's squares (adjacent squares)
    red_df = make_square(0, 0.25, 0.50 + increment * i, 0.50, 0.75)
    blue_df = make_square(1, 0.50 - increment * i, 0.75, 0.50, 0.75)

    # combine it in 200 rows (red first, then blue)
    df_profiles = vcat(red_df, blue_df)

    # write to CSV
    filename = joinpath(output_dir, "profiles-wide_$(i).csv")
    print(filename)
    CSV.write(filename, df_profiles)

    # plot periods
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

    savefig(plt, "scenarios_$i.png")
end