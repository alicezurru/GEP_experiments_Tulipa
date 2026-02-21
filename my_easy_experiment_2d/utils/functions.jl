
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

function plot_values_stocmethod_weight( #considering different options: stochastic_method, weight_type
    results_df::DataFrame,
    case_studies_df::DataFrame,
    values::String;
    savepath="relative_regret.png"
)
    results_with_options = outerjoin(case_studies_df, results_df, on="base_name", makeunique=true)

    rp_vals = sort(unique(results_df.rp))
    rp_labels = string.(rp_vals)
    rp_index = Dict(rp => i for (i, rp) in enumerate(rp_vals))

    p = plot(
        xlabel="Representative Period (rp)",
        ylabel=values,
        title="$values by rp",
        legend=:topright,
        size=(900, 500),
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
        short_label = replace(label, "_scenario" => "")
        scatter!(p, [NaN], [NaN];
            markershape=marker, markersize=8, markercolor=:gray30,
            label=short_label)
    end

    # Legend for colors (weight types)
    for (label, color) in COLOR_MAP_weight
        scatter!(p, [NaN], [NaN];
            markershape=:rect, markersize=8, markercolor=color,
            label=label)
    end

    savefig(p, savepath)
    @info "Plot saved in: $savepath"
end

function plot_values_stocmethod_method( # considering options: method, stochastic_method (possible to add weight type dirac)
    results_df::DataFrame,
    case_studies_df::DataFrame,
    values::String;
    savepath="relative_regret.png",
    include_dirac=false
)
    results_with_options = outerjoin(case_studies_df, results_df, on="base_name", makeunique=true)

    rp_vals = sort(unique(results_df.rp))
    rp_labels = string.(rp_vals)
    rp_index = Dict(rp => i for (i, rp) in enumerate(rp_vals))

    p = plot(
        xlabel="Representative Period (rp)",
        ylabel=values,
        title="$values by rp",
        legend=:topright,
        size=(900, 500),
        xticks=(1:length(rp_vals), rp_labels)
    )
    if !include_dirac
        results_with_options = filter(row -> row.base_name != "0_HourlyBenchmark", results_with_options)
        results_with_options = filter(row -> row.weight_type != "dirac", results_with_options)
    end
    # results_with_options = filter(row -> row.rp >= 10, results_with_options)
    #results_with_options = filter(row -> row.method == "convex_hull", results_with_options)

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

        # if method == "k_means" || method == "k_medoids"
        #     mcolin = get(COLOR_MAP_method, method) do
        #         error("Unknown method: $method")
        #     end
        # end

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
            label=""
        )
    end

    # Legend for shapes (stochastic methods)
    for (label, marker) in MARKER_MAP
        short_label = replace(label, "_scenario" => "")
        scatter!(p, [NaN], [NaN];
            markershape=marker, markersize=8, markercolor=:gray30,
            label=short_label)
    end

    # Legend for colors (method types)
    for (label, color) in COLOR_MAP_method
        scatter!(p, [NaN], [NaN];
            markershape=:rect, markersize=8, markercolor=color,
            label=label)
    end
    if include_dirac
        # Legend for filler colors (weights type)
        scatter!(p, [NaN], [NaN];
            markershape=:rect, markersize=8, markercolor=:white,
            label="dirac weights")
    end

    savefig(p, savepath)
    @info "Plot saved in: $savepath"
end


function plot_values_quantiles(
    stats_df::DataFrame,
    case_studies_df::DataFrame,
    values::String;
    savepath="stats_plot.png",
    include_dirac=false
)
    results_with_options = outerjoin(case_studies_df, stats_df, on="base_name", makeunique=true)

    if !include_dirac
        results_with_options = filter(row -> row.base_name != "0_HourlyBenchmark", results_with_options)
        results_with_options = filter(row -> row.weight_type != "dirac", results_with_options)
    end


    col_mean = Symbol(values * "_mean")
    col_q25 = Symbol(values * "_q25")
    col_q75 = Symbol(values * "_q75")

    stats_df = filter(row -> row.base_name != "0_HourlyBenchmark", stats_df)
    rp_vals = sort(unique(stats_df.rp))

    p = plot(
        xlabel="Representative Period",
        ylabel=values,
        title="$values by rp",
        legend=:topright,
        size=(900, 500),
        xticks=(rp_vals, string.(rp_vals))
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

        plot!(
            p,
            g_sorted.rp,
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

    # for (label, marker) in MARKER_MAP
    #     short_label = replace(label, "_scenario" => "")
    #     scatter!(p, [NaN], [NaN]; markershape=marker, color=:gray30, label=short_label)
    # end

    # for (label, color) in COLOR_MAP_method
    #     scatter!(p, [NaN], [NaN]; markershape=:rect, color=color, label=label)
    # end

    # if include_dirac
    #     scatter!(p, [NaN], [NaN]; markershape=:rect, color=:white, label="dirac weights")
    # end

    savefig(p, savepath)
    @info "Plot saved in: $savepath"
end


function plot_values_quantiles_panel(
    stats_df::DataFrame,
    case_studies_df::DataFrame,
    values::AbstractString;
    method::AbstractString,
    include_dirac::Bool=false,
    panel_title::AbstractString="$values by rp",
)
    results_with_options = outerjoin(case_studies_df, stats_df, on="base_name", makeunique=true)
    results_with_options = filter(row -> row.base_name != "0_HourlyBenchmark", results_with_options)
    results_with_options = filter(row -> row.method == method, results_with_options)
    if !include_dirac
        results_with_options = filter(row -> row.weight_type != "dirac", results_with_options)
    end

    rp_vals = sort(unique(results_with_options.rp))

    # Value columns
    col_mean = Symbol(values * "_mean")
    col_q25 = Symbol(values * "_q25")
    col_q75 = Symbol(values * "_q75")

    p = plot(
        xlabel="Number of representative periods",
        ylabel=values,
        title=panel_title,
        legend=:topright,
        xticks=(rp_vals, string.(rp_vals)),
        size=(500, 350) # panel size before grid composition
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
        mcolin = get(COLOR_MAP_method_weight, method * "_" * weight_type * "_" * stochastic_method) do
            error("Unknown method and weight: $(method * "_" * weight_type * "_" * stochastic_method)")
        end

        mean_vals = g_sorted[!, col_mean]
        q25_vals = g_sorted[!, col_q25]
        q75_vals = g_sorted[!, col_q75]

        lower = mean_vals .- q25_vals
        upper = q75_vals .- mean_vals

        plot!(
            p,
            g_sorted.rp,
            mean_vals,
            ribbon=(lower, upper),
            label=name,
            lw=2,
            color=mcolin,
            fillalpha=0.20,
            markershape=mk,
            markersize=6,
            markercolor=mcolin,
        )
    end

    return p
end


function plot_values_quantiles_grid(
    stats_dfs::NamedTuple,
    case_studies_df::DataFrame,
    values::AbstractString;
    include_dirac::Bool=false,
    savepath::Union{Nothing,AbstractString}=nothing,
    size::Tuple{Int,Int}=(1500, 900),
    titles::Union{Nothing,NamedTuple}=nothing,
)

    keys_order = [:DISTANT, :ADJACENT, :MIXED]
    available = [k for k in keys(stats_dfs)]
    cols = [k for k in keys_order if k in available]
    if isempty(cols)
        cols = collect(keys(stats_dfs))
    end

    panels = Plots.Plot[]

    # k_means 
    for col in cols
        label = string(col)
        title_txt = isnothing(titles) ? "$label — k_means" : get(titles, col, "$label — k_means")
        push!(panels,
            plot_values_quantiles_panel(
                stats_dfs[col], case_studies_df, values;
                method="k_means",
                include_dirac=include_dirac,
                panel_title=title_txt
            )
        )
    end

    # k_medoids 
    for col in cols
        label = string(col)
        title_txt = isnothing(titles) ? "$label — k_medoids" : get(titles, col, "$label — k_medoids")
        push!(panels,
            plot_values_quantiles_panel(
                stats_dfs[col], case_studies_df, values;
                method="k_medoids",
                include_dirac=include_dirac,
                panel_title=title_txt
            )
        )
    end

    grid = plot(panels..., layout=(2, length(cols)), size=size)

    savefig(grid, savepath)

    return nothing
end
