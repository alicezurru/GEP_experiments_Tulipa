
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
    results_with_options = filter(row -> row.rp >= 10, results_with_options)
    results_with_options = filter(row -> row.method == "convex_hull", results_with_options)

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


