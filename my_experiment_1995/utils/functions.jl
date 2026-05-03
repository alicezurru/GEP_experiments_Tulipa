
function get_solver_parameters(optimizer::Symbol)
    if optimizer == :HiGHS
        return HiGHS.Optimizer,
        Dict(
            "output_flag" => true,
            "solver" => "hipo",
            "parallel" => "on",
            "run_crossover" => "off"
        )
    elseif optimizer == :Gurobi
        return Gurobi.Optimizer, Dict("OutputFlag" => 1)
    else
        return HiGHS.Optimizer, Dict()
    end
end


function fix_variables_from_solution!(benchmark_model, reduced_model, var_symbol)

    var_to_fix = benchmark_model.variables[var_symbol].container
    val_to_fix = JuMP.value(reduced_model.variables[var_symbol].container)

    for (var, val) in zip(var_to_fix, val_to_fix)
        JuMP.fix(var, val; force=true)
    end
end

function create_init_rps_hourly(connection, profiles_type, period_duration, profiles)
    profiles_str = join(profiles, ", ")
    DuckDB.query(
        connection,
        "CREATE TABLE init_rps AS
        WITH melted AS (
            SELECT year,
                   scenario,
                   timestep,
                   demand,
                   profile_name,
                   value
            FROM profiles_wide_$profiles_type
            UNPIVOT (
                value FOR profile_name IN ($profiles_str)
            )
        ),

        with_hour AS (
            SELECT
                year,
                scenario,
                timestep,
                ((timestep-1) % $period_duration)+1 AS hour,
                profile_name,
                value,
                demand
            FROM melted
        ),

        max_demand AS (
            SELECT
                year,
                scenario,
                hour AS timestep,
                1 AS period,
                'demand' as profile_name,
                MAX(demand) AS value
            FROM with_hour
            GROUP BY year, scenario, hour
        ),

        with_ratio AS (
            SELECT
                year,
                scenario,
                timestep,
                hour,
                profile_name,
                value,
                value / NULLIF(demand, 0) AS ratio
            FROM with_hour
        ),

        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY year, scenario, hour, profile_name
                    ORDER BY ratio ASC
                ) AS rn
            FROM with_ratio
        )

        SELECT
            year,
            scenario,
            hour AS timestep,
            1 AS period,
            profile_name,
            value
        FROM ranked
        WHERE rn = 1
        UNION ALL 
        SELECT * FROM max_demand
        ORDER BY year, scenario, profile_name, timestep;"
    )
end

function create_init_rps_hourly_best(connection, profiles_type, period_duration, profiles)
    profiles_str = join(profiles, ", ")
    DuckDB.query(
        connection,
        "CREATE TABLE init_rps_best AS
        WITH melted AS (
            SELECT year,
                   scenario,
                   timestep,
                   demand,
                   profile_name,
                   value
            FROM profiles_wide_$profiles_type
            UNPIVOT (
                value FOR profile_name IN ($profiles_str)
            )
        ),

        with_hour AS (
            SELECT
                year,
                scenario,
                timestep,
                ((timestep-1) % $period_duration)+1 AS hour,
                profile_name,
                value,
                demand
            FROM melted
        ),

        min_demand AS (
            SELECT
                year,
                scenario,
                hour AS timestep,
                2 AS period,
                'demand' as profile_name,
                MIN(demand) AS value
            FROM with_hour
            GROUP BY year, scenario, hour
        ),

        with_ratio AS (
            SELECT
                year,
                scenario,
                timestep,
                hour,
                profile_name,
                value,
                value / NULLIF(demand, 0) AS ratio
            FROM with_hour
        ),

        ranked AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY year, scenario, hour, profile_name
                    ORDER BY ratio DESC
                ) AS rn
            FROM with_ratio
        )

        SELECT
            year,
            scenario,
            hour AS timestep,
            2 AS period,
            profile_name,
            value
        FROM ranked
        WHERE rn = 1
        UNION ALL 
        SELECT * FROM min_demand
        ORDER BY year, scenario, profile_name, timestep;"
    )
    DuckDB.query(connection,
        "INSERT INTO init_rps
        SELECT * FROM init_rps_best;")
end

function create_init_rps_daily(connection, profiles_type, period_duration, profiles)
    profiles_str = join(profiles, ", ")
    DuckDB.query(
        connection,
        "CREATE TABLE init_rps AS
        WITH base AS (
            SELECT
                year,
                scenario,
                timestep,
                CAST(((timestep - 1) % $period_duration) + 1 AS INTEGER) AS hour,
                CAST(FLOOR((timestep - 1) / $period_duration) + 1 AS INTEGER) AS day,
                demand,
                $profiles_str
            FROM profiles_wide_$profiles_type
        ),
        av_profiles AS (
            SELECT
                year,
                scenario,
                day,
                hour,
                profile_name,
                value
            FROM base
            UNPIVOT (
                value FOR profile_name IN ($profiles_str)
            )
        ),

        renewable_day_ratio AS (
            SELECT
                p.year,
                p.scenario,
                p.day,
                SUM(p.value) AS renewable_sum,
                SUM(b.demand) AS demand_sum,
                SUM(p.value) / NULLIF(SUM(b.demand),0) AS ratio
            FROM av_profiles p
            JOIN base b
                USING (year, scenario, day, hour)
            GROUP BY p.year, p.scenario, p.day
        ),

        worst_renewable_day AS (
            SELECT year, scenario, day
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY year, scenario
                        ORDER BY ratio ASC
                    ) rn
                FROM renewable_day_ratio
            )
            WHERE rn = 1
        ),

        peak_demand_day AS (
            SELECT year, scenario, day
            FROM (
                SELECT
                    year,
                    scenario,
                    day,
                    SUM(demand) AS total_demand,
                    ROW_NUMBER() OVER (
                        PARTITION BY year, scenario
                        ORDER BY SUM(demand) DESC
                    ) rn
                FROM base
                GROUP BY year, scenario, day
            )
            WHERE rn = 1
        ),

        renewable_profiles AS (
            SELECT
                p.year,
                p.scenario,
                p.hour AS timestep,
                1 AS period,
                p.profile_name,
                p.value
            FROM av_profiles p
            JOIN worst_renewable_day d
                USING (year, scenario, day)
        ),

        demand_profile AS (
            SELECT
                year,
                scenario,
                hour AS timestep,
                1 AS period,
                'demand' AS profile_name,
                demand AS value
            FROM base
            JOIN peak_demand_day
                USING (year, scenario, day)
        )

        SELECT * FROM renewable_profiles
        UNION ALL
        SELECT * FROM demand_profile
        ORDER BY year, scenario, profile_name, timestep;"
    )
end


function create_init_rps_daily_real(connection, profiles_type, period_duration, profiles) # this is not artificial period
    profiles_str = join(profiles, ", ")
    DuckDB.query(
        connection,
        "CREATE TABLE init_rps AS
        WITH base AS (
            SELECT
                year,
                scenario,
                timestep,
                CAST(((timestep - 1) % $period_duration) + 1 AS INTEGER) AS hour,
                CAST(FLOOR((timestep - 1) / $period_duration) + 1 AS INTEGER) AS day,
                demand,
                $profiles_str
            FROM profiles_wide_$profiles_type
        ),
        av_profiles AS (
            SELECT
                year,
                scenario,
                day,
                hour,
                profile_name,
                value
            FROM base
            UNPIVOT (
                value FOR profile_name IN ($profiles_str)
            )
        ),

        renewable_day_ratio AS (
            SELECT
                p.year,
                p.scenario,
                p.day,
                SUM(p.value) AS renewable_sum,
                SUM(b.demand) AS demand_sum,
                SUM(p.value) / NULLIF(SUM(b.demand),0) AS ratio
            FROM av_profiles p
            JOIN base b
                USING (year, scenario, day, hour)
            GROUP BY p.year, p.scenario, p.day
        ),

        worst_day AS (
            SELECT year, scenario, day
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY year, scenario
                        ORDER BY ratio ASC
                    ) rn
                FROM renewable_day_ratio
            )
            WHERE rn = 1
        ),

        renewable_profiles AS (
            SELECT
                p.year,
                p.scenario,
                p.hour AS timestep,
                1 AS period,
                p.profile_name,
                p.value
            FROM av_profiles p
            JOIN worst_day d
                USING (year, scenario, day)
        ),

        demand_profile AS (
            SELECT
                year,
                scenario,
                hour AS timestep,
                1 AS period,
                'demand' AS profile_name,
                demand AS value
            FROM base
            JOIN worst_day
                USING (year, scenario, day)
        )

        SELECT * FROM renewable_profiles
        UNION ALL
        SELECT * FROM demand_profile
        ORDER BY year, scenario, profile_name, timestep;"
    )
end


function plot_values_stocmethod_weight( #considering different options: stochastic_method, weight_type
    results_df::DataFrame,
    case_studies_df::DataFrame,
    values::String;
    savepath="relative_regret.png"
)
    results_with_options = outerjoin(case_studies_df, results_df, on="base_name", makeunique=true)
    results_with_options = filter(row -> row.base_name != "0_HourlyBenchmark", results_with_options)

    rp_vals = sort(unique(results_with_options.rp))
    rp_labels = string.(rp_vals)
    rp_index = Dict(rp => i for (i, rp) in enumerate(rp_vals))

    p = plot(
        xlabel="Number of representative_periods",
        ylabel=get(VALUE_MAP, values) do
            error("Unknown values: $values")
        end,
        title="",
        legend=:topright,
        size=(800, 500),
        xticks=(1:length(rp_vals), rp_labels)
    )

    for g in groupby(results_with_options, :base_name)
        name = g.base_name[1]
        if name == "0_HourlyBenchmark"
            continue
        end
        g_sorted = sort(g, :rp)

        stochastic_method = g.stochastic_method[1]
        mk = get(MARKER_MAP, stochastic_method) do
            error("Unknown stochastic_method: $stochastic_method")
        end

        weight_type = g.weight_type[1]
        mcol = get(COLOR_MAP_weight, weight_type) do
            error("Unknown weight_type: $weight_type")
        end

        column = Symbol(values)
        xidx = [rp_index[rp] for rp in g_sorted.rp]

        scatter!(
            p,
            xidx,
            g_sorted[!, column],
            markershape=mk,
            markersize=8,
            markercolor=mcol,
            label=""
        )
    end

    # Legend for shapes (stochastic methods)
    for (label, marker) in MARKER_MAP
        short_label = replace(label, "_scenario" => "-scenario")
        scatter!(p, [NaN], [NaN];
            markershape=marker, markersize=8, markercolor=:gray30,
            label=short_label)
    end

    # Legend for colors (weight types)
    for (label, color) in COLOR_MAP_weight
        scatter!(p, [NaN], [NaN];
            markershape=:rect, markersize=8, markercolor=color,
            label=get(LEGEND_METHOD_MAP, label) do
                error("Unknown method: $label")
            end)
    end

    savefig(p, savepath)
    @info "Plot saved in: $savepath"
end

# function plot_values_stocmethod_method( # considering options: method, stochastic_method (possible to add weight type dirac)
#     results_df::DataFrame,
#     case_studies_df::DataFrame,
#     values::String;
#     savepath="relative_regret.png",
#     include_dirac=false,
#     logscale=false
# )
#     results_with_options = outerjoin(case_studies_df, results_df, on="base_name", makeunique=true)

#     results_with_options = filter(row -> row.base_name != "0_HourlyBenchmark", results_with_options)
#     results_with_options = filter(row -> row.rp >= 4, results_with_options)

#     results_with_options = filter(row -> row.method != "conical_hull", results_with_options)
#     #results_with_options = filter(row -> row.method != "convex_hull_with_null", results_with_options)
#     #results_with_options = filter(row -> row.method != "convex_hull", results_with_options)

#     rp_vals = sort(unique(results_with_options.rp))
#     rp_labels = string.(rp_vals)
#     rp_index = Dict(rp => i for (i, rp) in enumerate(rp_vals))

#     p = plot(
#         xlabel="Number of representative periods",
#         ylabel=get(VALUE_MAP, values) do
#             error("Unknown values: $values")
#         end,
#         title="",
#         legend=:topright,
#         size=(800, 500),
#         xticks=(1:length(rp_vals), rp_labels)
#     )

#     if logscale
#         yaxis!(p, :log10)
#     end

#     if !include_dirac
#         results_with_options = filter(row -> row.weight_type != "dirac", results_with_options)
#     end
#     for g in groupby(results_with_options, :base_name)
#         name = g.base_name[1]
#         if name == "0_HourlyBenchmark"
#             continue
#         end
#         g_sorted = sort(g, :rp)


#         stochastic_method = g.stochastic_method[1]
#         mk = get(MARKER_MAP, stochastic_method) do
#             error("Unknown stochastic_method: $stochastic_method")
#         end

#         method = g.method[1]
#         mcolout = get(COLOR_MAP_method, method) do
#             error("Unknown method: $method")
#         end

#         weight_type = g.weight_type[1]
#         mcolin = get(FILLER_MAP, weight_type) do
#             error("Unknown weight_type: $weight_type")
#         end

#         column = Symbol(values)
#         xidx = [rp_index[rp] for rp in g_sorted.rp]

#         if !include_dirac
#             mcolout = :black
#         end

#         scatter!(
#             p,
#             xidx,
#             g_sorted[!, column],
#             markershape=mk,
#             markersize=8,
#             markercolor=mcolin,
#             markerstrokecolor=mcolout,
#             label="",
#         )
#     end

#     # # Legend for shapes (stochastic methods)
#     # for (label, marker) in MARKER_MAP
#     #     short_label = replace(label, "_scenario" => "-scenario")
#     #     scatter!(p, [NaN], [NaN];
#     #         markershape=marker, markersize=8, markercolor=:gray30,
#     #         label=short_label)
#     # end

#     # # Legend for colors (method types)
#     # for (label, color) in COLOR_MAP_method
#     #     scatter!(p, [NaN], [NaN];
#     #         markershape=:rect, markersize=8, markercolor=color,
#     #         label=get(LEGEND_METHOD_MAP, label) do
#     #             error("Unknown method: $label")
#     #         end
#     #     )
#     # end
#     if include_dirac
#         # Legend for filler colors (weights type)
#         scatter!(p, [NaN], [NaN];
#             markershape=:rect, markersize=8, markercolor=:white,
#             label="dirac weights")
#     end
#     ylims!(p, 1e-4, 100)

#     savefig(p, savepath)
#     @info "Plot saved in: $savepath"
# end


function plot_values_stocmethod_method( # considering options: method, stochastic_method (possible to add weight type dirac)
    results_df::DataFrame,
    case_studies_df::DataFrame,
    values::String;
    savepath="relative_regret.png",
    include_dirac=false,
    logscale=false
)
    results_with_options = outerjoin(case_studies_df, results_df, on="base_name", makeunique=true)

    results_with_options = filter(row -> row.base_name != "0_HourlyBenchmark", results_with_options)
    results_with_options = filter(row -> row.rp >= 4, results_with_options)

    results_with_options = filter(row -> row.method != "conical_hull", results_with_options)
    #results_with_options = filter(row -> row.method != "convex_hull_with_null", results_with_options)
    #results_with_options = filter(row -> row.method != "convex_hull", results_with_options)

    rp_vals = sort(unique(results_with_options.rp))
    rp_labels = string.(rp_vals)
    rp_index = Dict(rp => i for (i, rp) in enumerate(rp_vals))

    p = plot(
        xlabel="Number of representative periods",
        ylabel=get(VALUE_MAP, values) do
            error("Unknown values: $values")
        end,
        title="",
        legend=:topright,
        legendfont=font(12),
        size=(500, 650),
        xticks=(1:length(rp_vals), rp_labels),
        framestyle=:box,
        xguidefont=font(14),
        yguidefont=font(14),
        tickfont=font(12),
        grid=true,
        minorgrid=true,)

    if logscale
        yaxis!(p, :log10)
    end

    if !include_dirac
        results_with_options = filter(row -> row.weight_type != "dirac", results_with_options)
    end
    for g in groupby(results_with_options, :base_name)
        name = g.base_name[1]
        if name == "0_HourlyBenchmark"
            continue
        end
        g_sorted = sort(g, :rp)


        stochastic_method = g.stochastic_method[1]
        mk = get(MARKER_MAP, stochastic_method) do
            error("Unknown stochastic_method: $stochastic_method")
        end

        method = g.method[1]
        mcolout = get(COLOR_MAP_method, method) do
            error("Unknown method: $method")
        end

        weight_type = g.weight_type[1]
        mcolin = get(FILLER_MAP, weight_type) do
            error("Unknown weight_type: $weight_type")
        end

        column = Symbol(values)
        xidx = [rp_index[rp] for rp in g_sorted.rp]

        if !include_dirac
            mcolout = :black
        end

        scatter!(
            p,
            xidx,
            g_sorted[!, column],
            markershape=mk,
            markersize=8,
            markercolor=mcolin,
            markerstrokecolor=mcolout,
            label="",
        )
    end

    # Legend for shapes (stochastic methods)
    for (label, marker) in MARKER_MAP
        short_label = replace(label, "_scenario" => "-scenario")
        scatter!(p, [NaN], [NaN];
            markershape=marker, markersize=8, markercolor=:gray30,
            label=short_label)
    end

    # Legend for colors (method types)
    for (label, color) in COLOR_MAP_method
        scatter!(p, [NaN], [NaN];
            markershape=:rect, markersize=8, markercolor=color,
            label=get(LEGEND_METHOD_MAP, label) do
                error("Unknown method: $label")
            end
        )
    end
    if include_dirac
        # Legend for filler colors (weights type)
        scatter!(p, [NaN], [NaN];
            markershape=:rect, markersize=8, markercolor=:white,
            label="dirac weights")
    end
    # ylims!(p, -10, 150)
    #ylims!(p, 0, 0.15)

    savefig(p, savepath)
    @info "Plot saved in: $savepath"
end


function plot_values_quantiles(
    stats_df::DataFrame,
    case_studies_df::DataFrame,
    values::String;
    savepath="stats_plot.png",
    logscale=false
)
    results_with_options = outerjoin(case_studies_df, stats_df, on="base_name", makeunique=true)
    results_with_options = filter(row -> row.base_name != "0_HourlyBenchmark", results_with_options)

    #results_with_options = filter(row -> row.rp >= 60, results_with_options)


    col_mean = Symbol(values * "_mean")
    col_q25 = Symbol(values * "_q25")
    col_q75 = Symbol(values * "_q75")

    stats_df = filter(row -> row.base_name != "0_HourlyBenchmark", stats_df)
    rp_vals = sort(unique(stats_df.rp))
    rp_labels = string.(rp_vals)
    rp_index = Dict(rp => i for (i, rp) in enumerate(rp_vals))


    p = plot(
        xlabel="Number of representative periods",
        ylabel=get(VALUE_MAP, values) do
            error("Unknown values: $values")
        end,
        title="",
        legend=:topright,
        size=(800, 500),
        xticks=(1:length(rp_vals), rp_labels)
    )
    if logscale
        yaxis!(p, :log10)
    end


    for g in groupby(results_with_options, :base_name)
        name = g.base_name[1]
        if name == "0_HourlyBenchmark"
            continue
        end

        g_sorted = sort(g, :rp)

        stochastic_method = g.stochastic_method[1]
        mk = get(MARKER_MAP, stochastic_method) do
            error("Unknown stochastic_method: $stochastic_method")
        end
        method = g.method[1]
        weight_type = g.weight_type[1]
        mcolin = get(COLOR_MAP_method_weight, method * "_" * weight_type) do
            error("Unknown method and weight: $(method * "_" * weight_type)")
        end

        mean_vals = g_sorted[!, col_mean]
        q25_vals = g_sorted[!, col_q25]
        q75_vals = g_sorted[!, col_q75]

        lower = mean_vals .- q25_vals
        upper = q75_vals .- mean_vals
        xidx = [rp_index[rp] for rp in g_sorted.rp]

        plot!(
            p,
            xidx,
            mean_vals,
            ribbon=(lower, upper),
            label=name,
            lw=2,
            color=mcolin,
            fillalpha=0.20,
            markershape=mk,
            markersize=6,
            markercolor=mcolin
        )
    end
    #ylims!(p, 0.0, 0.15)

    savefig(p, savepath)
    @info "Plot saved in: $savepath"
end


function plot_values_quantiles_panel(
    stats_df::DataFrame,
    case_studies_df::DataFrame,
    values::AbstractString;
    method::AbstractString,
    panel_title::AbstractString="$values by rp",
    plot_legend::Bool=false
)
    results_with_options = outerjoin(case_studies_df, stats_df, on="base_name", makeunique=true)
    results_with_options = filter(row -> row.base_name != "0_HourlyBenchmark", results_with_options)
    results_with_options = filter(row -> row.method == method, results_with_options)

    rp_vals = sort(unique(results_with_options.rp))
    rp_labels = string.(rp_vals)
    rp_index = Dict(rp => i for (i, rp) in enumerate(rp_vals))

    # Value columns
    col_mean = Symbol(values * "_mean")
    col_q25 = Symbol(values * "_q25")
    col_q75 = Symbol(values * "_q75")

    p = plot(
        xlabel="Number of representative periods",
        ylabel=get(VALUE_MAP, values) do
            error("Unknown value: $values")
        end,
        title=panel_title,
        legend=plot_legend ? :topright : false,
        legendfont=font(10),
        xticks=(1:length(rp_vals), rp_labels),
        size=(200, 100),
        theme=:ggplot2,
        framestyle=:box,
        grid=:y,
        minorgrid=true,
        tick_direction=:out,
        guidefont=font(10),
        tickfont=font(9),
    )


    for g in groupby(results_with_options, :base_name)
        name = g.base_name[1]
        if name == "0_HourlyBenchmark"
            continue
        end

        g_sorted = sort(g, :rp)

        stochastic_method = g_sorted.stochastic_method[1]
        mk = get(MARKER_MAP, stochastic_method) do
            error("Unknown stochastic_method: $stochastic_method")
        end

        weight_type = g_sorted.weight_type[1]
        together = method * "_" * weight_type * "_" * stochastic_method
        mcolin = get(COLOR_MAP_method_weight_stmethod, together) do
            error("Unknown method and weight: $together")
        end

        mean_vals = g_sorted[!, col_mean]
        q25_vals = g_sorted[!, col_q25]
        q75_vals = g_sorted[!, col_q75]

        lower = mean_vals .- q25_vals
        upper = q75_vals .- mean_vals
        xidx = [rp_index[rp] for rp in g_sorted.rp]

        plot!(
            p,
            xidx,
            mean_vals,
            ribbon=(lower, upper),
            label=get(LEGEND_MAP, together) do
                error("Unknown legend value: $together")
            end,
            lw=2,
            linealpha=0.9,
            color=mcolin,
            fillalpha=0.25,
            markershape=mk,
            markersize=7,
            markercolor=mcolin,
            markerstrokecolor=:black,
            markerstrokewidth=0.5,
        )

    end
    yaxis!(p, :log10)
    ylims!(p, 1e-3, 10000)

    return p
end


function plot_values_quantiles_grid(
    stats_dfs::NamedTuple,
    case_studies_df::DataFrame,
    values::AbstractString;
    savepath::Union{Nothing,AbstractString}=nothing,
    size::Tuple{Int,Int}=(1500, 900),
    titles::Union{Nothing,NamedTuple}=nothing,
)

    keys_order = [:DISTANT, :HALFMIXED, :CLOSE, :MIXED]
    available = [k for k in keys(stats_dfs)]
    cols = [k for k in keys_order if k in available]
    if isempty(cols)
        cols = collect(keys(stats_dfs))
    end

    panels = Plots.Plot[]

    # k_means 
    for col in cols
        label = string(col)
        title_txt = isnothing(titles) ? label : get(titles, col, label)
        push!(panels,
            plot_values_quantiles_panel(
                stats_dfs[col], case_studies_df, values;
                method="k_means",
                panel_title=title_txt,
                plot_legend=string(col) == "MIXED" ? true : false
            )
        )
    end

    # k_medoids 
    for col in cols
        #label = string(col)
        #title_txt = isnothing(titles) ? "$label — k_medoids" : get(titles, col, "$label — k_medoids")
        push!(panels,
            plot_values_quantiles_panel(
                stats_dfs[col], case_studies_df, values;
                method="k_medoids",
                panel_title=" ",
                plot_legend=string(col) == "MIXED" ? true : false
            )
        )
    end



    grid = plot(
        panels...,
        layout=(2, length(cols)),
        size=size,
        link=:both, # link x and y across panels
        left_margin=5mm,
        right_margin=12mm,
        top_margin=5mm,
        bottom_margin=5mm
    )


    savefig(grid, savepath)

    return nothing
end


function plot_cost_columns(df::DataFrame, savepath::AbstractString)

    # Create label column: base_name_rp
    df.label = string.(df.base_name, "_", df.rp)


    # Keep only relevant columns
    labels = df.label
    investment = df.investment_cost
    operational = df.operational_cost + df.investment_cost - df.penalty_loss_of_load_e_demand
    penalty = df.investment_cost + df.operational_cost

    # Create stacked bar plot

    data_matrix = hcat(penalty, operational, investment)
    label_order = ["Penalty Loss" "Operational Cost" "Investment Cost"]

    # Create stacked bar plot
    b = bar(
        labels,
        data_matrix,
        label=label_order,
        legend=:topright,
        xlabel="Scenario",
        ylabel="Cost",
        title="Cost Breakdown per Setting",
        bar_position=:stack,
        xticks=(1:length(labels), labels),
        xrotation=45,
        size=(800, 600)
    )

    savefig(b, savepath)

end
