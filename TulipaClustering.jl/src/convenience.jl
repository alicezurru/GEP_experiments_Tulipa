export cluster!, dummy_cluster!, transform_wide_to_long!

"""
    cluster!(
        connection,
        period_duration,
        num_rps;
        input_database_schema = "",
        input_profile_table_name = "profiles",
        database_schema = "",
        drop_incomplete_last_period::Bool = false,
        method::Symbol = :k_means,
        distance::SemiMetric = SqEuclidean(),
        initial_representatives::AbstractDataFrame = DataFrame(),
        layout::ProfilesTableLayout = ProfilesTableLayout(),
        weight_type::Symbol = :convex,
        tol::Float64 = 1e-2,
        clustering_kwargs = Dict(),
        weight_fitting_kwargs = Dict(),
    )

Convenience function to cluster the table named in `input_profile_table_name`
using `period_duration` and `num_rps`. The resulting tables
`profiles_rep_periods`, `rep_periods_mapping`, and
`rep_periods_data` are loaded into `connection` in the `database_schema`, if
given, and enriched with `year` information.

This function extracts the table (expecting columns `year`, `profile_name`, `timestep`, `value`),
then calls [`split_into_periods!`](@ref),
[`find_representative_periods`](@ref), [`fit_rep_period_weights!`](@ref), and
finally [`write_clustering_result_to_tables`](@ref).

## Arguments

**Required**

- `connection`: DuckDB connection
- `period_duration`: Duration of each period, i.e., number of `timestep`s.
- `num_rps`: Number of find_representative_periods

**Keyword arguments**

- `input_database_schema` (default `""`): Schema of the input tables
- `input_profile_table_name` (default `"profiles"`): Default name of the `profiles` table inside the above schemaa
- `database_schema` (default `""`): Schema of the output tables
- `drop_incomplete_last_period` (default `false`): controls how the last period is treated if it
  is not complete: if this parameter is set to `true`, the incomplete period
  is dropped and the weights are rescaled accordingly; otherwise, clustering
  is done for `n_rp - 1` periods, and the last period is added as a special
  shorter representative period
- `method` (default `:k_medoids`): clustering method to use `:k_means`, `:k_medoids`, `:convex_hull`, `:convex_hull_with_null`, or `:conical_hull`.
- `distance` (default `Distances.Euclidean()`): semimetric used to measure distance between data points.
- `initial_representatives` initial representatives that should be
    included in the clustering. The period column in the initial representatives
    should be 1-indexed and the key columns should be the same as in the clustering data.
    For the hull methods it will be added before clustering, for :k_means and :k_medoids
    it will be added after clustering.
- `layout` (default `ProfilesTableLayout()`): describes the column names for `period`,
  `timestep`, and `value` in in-memory DataFrames. It does not change the SQL input
  table schema, which must contain `profile_name`, `timestep`, and `value`. Weight
  fitting operates on matrices and does not use `layout`.
- `weight_type` (default `:dirac`): the type of weights to find; possible values are:
    - `:dirac`: each period is represented by exactly one representative
        period (a one unit weight and the rest are zeros)
    - `:convex`: each period is represented as a convex sum of the
      representative periods (a sum with nonnegative weights adding into one)
    - `:conical`: each period is represented as a conical sum of the
      representative periods (a sum with nonnegative weights)
    - `:conical_bounded`: each period is represented as a conical sum of the
      representative periods (a sum with nonnegative weights) with the total
      weight bounded from above by one.
- `tol` (default `1e-2`): algorithm's tolerance; when the weights are adjusted by a value less
  then or equal to `tol`, they stop being fitted further.
- `clustering_kwargs` (default `Dict()`): Extra keyword arguments passed to [`find_representative_periods`](@ref)
- `weight_fitting_kwargs` (default `Dict()`): Extra keyword arguments passed to [`fit_rep_period_weights!`](@ref)
  (e.g., `niters`, `learning_rate`, `adaptive_grad`).
"""
function cluster!(
    connection,
    period_duration,
    num_rps;
    input_database_schema = "",
    input_profile_table_name = "profiles",
    database_schema = "",
    drop_incomplete_last_period::Bool = false,
    method::Symbol = :k_medoids,
    distance::SemiMetric = Euclidean(),
    initial_representatives::AbstractDataFrame = DataFrame(),
    layout::ProfilesTableLayout = ProfilesTableLayout(),
    weight_type::Symbol = :dirac,
    tol::Float64 = 1e-2,
    clustering_kwargs = Dict(),
    weight_fitting_kwargs = Dict(),
)
    validate_data!(
        connection;
        input_database_schema,
        table_names = Dict("profiles" => input_profile_table_name),
        layout,
        initial_representatives,
    )

    if input_database_schema != ""
        input_profile_table_name = "$input_database_schema.$input_profile_table_name"
    end

    profiles =
        DuckDB.query(
            connection,
            "SELECT * FROM $input_profile_table_name
            ",
        ) |> DataFrame

    split_into_periods!(profiles; period_duration, layout)
    grouped_profiles_data = groupby(profiles, layout.cols_to_groupby)

    grouped_profiles_data, metadata_per_group =
        _update_period_numbers_using_crossby_cols!(grouped_profiles_data, layout)

    if !haskey(clustering_kwargs, :add_cross)
        results_per_group = Dict(
            group_key => find_representative_periods(
                group,
                num_rps;
                drop_incomplete_last_period,
                method,
                distance,
                initial_representatives = _get_initial_representatives_for_group(
                    initial_representatives,
                    group_key,
                ),
                layout,
                clustering_kwargs...,
            ) for (group_key, group) in pairs(grouped_profiles_data)
        )

        for clustering_result in values(results_per_group)
            fit_rep_period_weights!(
                clustering_result;
                weight_type,
                tol,
                weight_fitting_kwargs...,
            )
        end

        write_clustering_result_to_tables(
            connection,
            results_per_group,
            metadata_per_group,
            num_rps;
            database_schema,
            layout,
        )

        return results_per_group
    else # for now this only works if we have year and scenario only
        grouped_profiles_data_year = groupby(profiles, [:year])

        grouped_profiles_data_year, metadata_per_year =
            _update_period_numbers_using_crossby_cols!(grouped_profiles_data_year, layout)

        results_per_group = Dict{
            DataFrames.GroupKey{GroupedDataFrame{DataFrame}},
            TulipaClustering.ClusteringResult,
        }()

        rp_matrices_by_year = Dict{Int, Matrix{Float64}}() # to keep track of the rps for every year (assuming year is an int)

        for (group_key, group) in pairs(grouped_profiles_data)
            init_reps =
                _get_initial_representatives_for_group(initial_representatives, group_key)
            year = group_key[:year] # see how to use a symbol here
            if haskey(rp_matrices_by_year, year)
                cl_result = find_representative_periods(
                    group,
                    num_rps;
                    drop_incomplete_last_period,
                    method,
                    distance,
                    initial_representatives = init_reps,
                    layout,
                    rp_matrix_previous = rp_matrices_by_year[year],
                    clustering_kwargs...,
                )
                rp_matrices_by_year[year] =
                    combine_matrices([rp_matrices_by_year[year], cl_result.rp_matrix])
            else
                cl_result = find_representative_periods(
                    group,
                    num_rps;
                    drop_incomplete_last_period,
                    method,
                    distance,
                    initial_representatives = init_reps,
                    layout,
                    clustering_kwargs...,
                )
                rp_matrices_by_year[year] = cl_result.rp_matrix
            end

            results_per_group[group_key] = cl_result
        end

        results_per_year = Dict{
            DataFrames.GroupKey{GroupedDataFrame{DataFrame}},
            TulipaClustering.ClusteringResult,
        }() # now let us put together for every year so that we can give these results to fit_rep_period_weights
        for (year_key, group) in pairs(grouped_profiles_data_year)
            scenario_results = Vector{TulipaClustering.ClusteringResult}()
            for (k, v) in results_per_group
                if k.year == year_key[1]
                    push!(scenario_results, v)
                end
            end
            results_per_year[year_key] =
                combine_clustering_results(scenario_results, num_rps)
        end
        for clustering_result in values(results_per_year)
            fit_rep_period_weights!(
                clustering_result;
                weight_type,
                tol,
                weight_fitting_kwargs...,
            )
        end

        layout =
            ProfilesTableLayout(; cols_to_groupby = [:year], cols_to_crossby = [:scenario])
        # we need cross_values_list from metadata_cross
        grouped_profiles_data_cross = groupby(profiles, layout.cols_to_groupby)
        grouped_profiles_data_cross, metadata_cross =
            _update_period_numbers_using_crossby_cols!(grouped_profiles_data_cross, layout)

        write_clustering_result_to_tables(
            connection,
            results_per_year,
            metadata_cross,
            num_rps * clustering_kwargs[:n_scenarios];
            database_schema,
            layout,
        )

        return results_per_group
    end
end

"""
    dummy_cluster!(connection)

Convenience function to create the necessary columns and tables when clustering
is not required.

This is essentially creating a single representative period with the size of
the whole profile.
See [`cluster!`](@ref) for more details of what is created.
"""
function dummy_cluster!(
    connection;
    input_database_schema = "",
    input_profile_table_name = "profiles",
    layout::ProfilesTableLayout = ProfilesTableLayout(),
    kwargs...,
)
    table_name = if input_database_schema != ""
        "$input_database_schema.$input_profile_table_name"
    else
        input_profile_table_name
    end
    timestep_col = layout.timestep
    period_duration = only([
        row.max_timestep for row in DuckDB.query(
            connection,
            "SELECT MAX($timestep_col) AS max_timestep FROM $table_name",
        )
    ])
    cluster!(connection, period_duration, 1; layout, kwargs...)
end

"""
    transform_wide_to_long!(
        connection,
        wide_table_name,
        long_table_name;
    )

Convenience function to convert a table in wide format to long format using DuckDB.
Originally aimed at converting a profile table like the following:

| year | timestep | name1 | name2 | ⋯  | nameN |
| ---- | -------- | ----- | ----- | -- | ----- |
| 2030 |        1 |   1.0 |   2.5 | ⋯  |   0.0 |
| 2030 |        2 |   1.5 |   2.6 | ⋯  |   0.0 |
| 2030 |        3 |   2.0 |   2.6 | ⋯  |   0.0 |

To a table like the following:

| year | timestep | profile_name | value |
| ---- | -------- | ------------ | ----- |
| 2030 |        1 |        name1 |   1.0 |
| 2030 |        2 |        name1 |   1.5 |
| 2030 |        3 |        name1 |   2.0 |
| 2030 |        1 |        name2 |   2.5 |
| 2030 |        2 |        name2 |   2.6 |
| 2030 |        3 |        name2 |   2.6 |
|    ⋮ |        ⋮ |            ⋮ |     ⋮ |
| 2030 |        1 |        nameN |   0.0 |
| 2030 |        2 |        nameN |   0.0 |
| 2030 |        3 |        nameN |   0.0 |

This conversion is done using the `UNPIVOT` SQL command from DuckDB.

## Keyword arguments

- `exclude_columns = ["year", "timestep"]`: Which tables to exclude from the conversion.
  Note that if you have more columns that you want to exclude from the wide table, e.g., `scenario`,
  you can add them to this list, e.g., `["scenario", "year", "timestep"]`.
- `name_column = "profile_name"`: Name of the new column that contains the names of the old columns
- `value_column = "value"`: Name of the new column that holds the values from the old columns
"""
function transform_wide_to_long!(
    connection,
    wide_table_name,
    long_table_name;
    exclude_columns = ["year", "timestep"],
    name_column = "profile_name",
    value_column = "value",
)
    @assert length(exclude_columns) > 0
    exclude_str = join(exclude_columns, ", ")
    DuckDB.query(
        connection,
        "CREATE OR REPLACE TABLE $long_table_name AS
        UNPIVOT $wide_table_name
        ON COLUMNS(* EXCLUDE ($exclude_str))
        INTO
            NAME $name_column
            VALUE $value_column
        ORDER BY $name_column, $exclude_str
        ",
    )

    return
end

"""
    _get_initial_representatives_for_group(
        initial_representatives::AbstractDataFrame,
        group_key::DataFrames.GroupKey{GroupedDataFrame{DataFrame}},
    )

Get the initial representatives for a specific group from a grouped DataFrame.

# Arguments
- `initial_representatives::AbstractDataFrame`: A DataFrame containing the initial representative data points.
- `group_key::DataFrames.GroupKey{GroupedDataFrame{DataFrame}}`: A key identifying a specific group within a grouped DataFrame.

# Returns
The subset of `initial_representatives` that corresponds to the specified `group_key`.

# Description
This is an internal helper function that extracts the initial representative data points
for a particular group identified by its group key. It is typically used during the
initialization phase of clustering operations on grouped data.
"""
function _get_initial_representatives_for_group(
    initial_representatives::AbstractDataFrame,
    group_key::DataFrames.GroupKey{GroupedDataFrame{DataFrame}},
)
    # Return empty DataFrame if no initial representatives provided
    if isempty(initial_representatives)
        return DataFrame()
    end

    # Start with all rows as potential matches
    num_rows = nrow(initial_representatives)
    rows_matching_group = trues(num_rows)

    # For each grouping column, filter to rows that match the group's value
    for column_name in keys(group_key)
        group_value = group_key[column_name]
        column_values = initial_representatives[!, column_name]
        column_matches = column_values .== group_value

        # Keep only rows that match this column AND all previous columns
        rows_matching_group .&= column_matches
    end

    # Return the subset of initial representatives that belong to this group
    return initial_representatives[rows_matching_group, :]
end

"""
    _update_period_numbers_using_crossby_cols!(grouped_profiles_data, layout)

Update period numbers in grouped profile data by creating sequential periods across groups
defined by `cols_to_crossby`.

# Arguments
- `grouped_profiles_data::GroupedDataFrame`: A grouped DataFrame containing profile data
- `layout::ProfilesTableLayout`: Layout specification containing:
  - `period`: Column name for period numbers
  - `cols_to_crossby`: Column names used to cross/combine groups

# Returns
- `grouped_profiles_data`: Modified GroupedDataFrame with updated period numbers and crossby columns removed
- `metadata_per_group`: Dictionary containing metadata for each group with keys:
  - `group_values`: Values identifying the group
  - `num_periods`: Maximum period number in the group
  - `cross_values_list`: List of NamedTuples containing crossby column values for each cross group

# Details
For each group in the grouped data, this function:
1. Creates subgroups based on `cols_to_crossby` columns
2. Adjusts period numbers so that each subgroup has sequential, non-overlapping periods
3. Period numbers for subgroup `i` are offset by `num_periods * (i - 1)`
4. Removes the `cols_to_crossby` columns from the final result

# Modifications in Place
Modifies `grouped_profiles_data` in place by updating period numbers and removing crossby columns.
"""
function _update_period_numbers_using_crossby_cols!(
    grouped_profiles_data::GroupedDataFrame,
    layout::ProfilesTableLayout,
)
    metadata_per_group = Dict(
        group_key => (
            group_values = collect(group_key),
            num_periods = maximum(group[!, layout.period]),
            cross_values_list = [
                NamedTuple(col => cross_group[1, col] for col in layout.cols_to_crossby) for cross_group in groupby(group, layout.cols_to_crossby)
            ],
        ) for (group_key, group) in pairs(grouped_profiles_data)
    )
    for (group_key, group) in pairs(grouped_profiles_data)
        crossed_profiles_data = groupby(group, layout.cols_to_crossby)
        num_periods = metadata_per_group[group_key].num_periods
        for (cross_idx, cross_group) in enumerate(crossed_profiles_data)
            cross_group[!, layout.period] .+= num_periods * (cross_idx - 1)
        end
    end
    for col in layout.cols_to_crossby
        select!(grouped_profiles_data, Not(col))
    end

    return grouped_profiles_data, metadata_per_group
end

"""
    combine_clustering_results(v) added by me

Combine together clustering results of more scenarios of one year.

# Arguments
- `v`: Vector{TulipaClustering.ClusteringResult} with clustering results for many scenarios
- `n_rp`: Int number of representative periods for each scenario

# Returns
- `TulipaClustering.ClusteringResult`: Modified ClusteringResult that combines for every year all scenarios together

"""
function combine_clustering_results(v::Vector{TulipaClustering.ClusteringResult}, n_rp::Int)
    new_profiles = combine_scenario_profiles(v, n_rp)
    new_weight_matrix = combine_scenario_weight_matrices(v, n_rp)
    new_clustering_matrix = combine_matrices([r.clustering_matrix for r in v])
    new_rp_matrix = combine_matrices([r.rp_matrix for r in v])
    new_key_columns = v[1].auxiliary_data.key_columns # SEE IF I NEED TO REMOVE SCENARIO
    # if :scenario in new_key_columns
    #     new_key_columns = filter(!=(:scenario), new_key_columns)
    # end
    new_auxiliary_data = AuxiliaryClusteringData(
        new_key_columns,
        v[1].auxiliary_data.period_duration,
        v[1].auxiliary_data.last_period_duration,
        v[1].auxiliary_data.n_periods * length(v),
        v[1].auxiliary_data.medoids,
    )
    return ClusteringResult(
        new_profiles,
        new_weight_matrix,
        new_clustering_matrix,
        new_rp_matrix,
        new_auxiliary_data,
    )
end

function combine_scenario_weight_matrices(
    results::Vector{TulipaClustering.ClusteringResult},
    n_rp::Int,
)
    isempty(results) && return zeros(0, 0)

    # determine matrix type from first matrix (if it is sparse, we return it sparse)
    first_mat = first(r.weight_matrix for r in results)
    is_sparse = first_mat isa SparseMatrixCSC

    # compute dimensions
    row_sizes = [size(r.weight_matrix, 1) for r in results]
    total_rows = sum(row_sizes)
    total_cols = n_rp * length(results)

    combined = if is_sparse
        spzeros(Float64, total_rows, total_cols)
    else
        zeros(Float64, total_rows, total_cols)
    end

    # fill block by block
    row_offset = 0
    for (idx, result) in enumerate(results)
        W = result.weight_matrix

        n_rows = size(W, 1)
        col_offset = _rep_period_offset(n_rp, idx)

        combined[
            (row_offset + 1):(row_offset + n_rows),
            (col_offset + 1):(col_offset + size(W, 2)),
        ] .= W

        row_offset += n_rows
    end

    return combined
end

function combine_scenario_profiles(
    results::Vector{TulipaClustering.ClusteringResult},
    n_rp::Int,
)
    profile_dfs = DataFrame[]

    for (idx, result) in enumerate(results)
        df = result.profiles === nothing ? DataFrame() : copy(result.profiles)

        if !isempty(df)
            # if "scenario" in names(df)
            #     select!(df, Not(:scenario))
            # end

            df.rep_period .+= _rep_period_offset(n_rp, idx)
        end

        push!(profile_dfs, df)
    end

    return isempty(profile_dfs) ? DataFrame() : vcat(profile_dfs...; cols = :union)
end

function combine_matrices(matrices::Vector{Matrix{Float64}})
    # compute total rows
    n_rows = size(matrices[1], 1)
    # assume all matrices have same rows
    tot_cols = 0
    for M in matrices
        tot_cols = tot_cols + size(M, 2)
    end

    combined = zeros(Float64, n_rows, tot_cols)

    # fill block by block
    col_offset = 0
    for M in matrices
        n_cols = size(M, 2)
        combined[:, (col_offset + 1):(col_offset + n_cols)] .= M
        col_offset += n_cols
    end
    return combined
end
