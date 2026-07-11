"""
  find_representative_periods(
    clustering_data,
    n_rp;
    drop_incomplete_last_period = false,
    method = :k_means,
    distance = SqEuclidean(),
    initial_representatives = DataFrame(),
    layout = ProfilesTableLayout(),
    kwargs...,
  )

Finds representative periods via data clustering. Honors custom column names via
`layout` (defaults to `(:period, :timestep, :value)`).

Arguments
  - `clustering_data`: long-format data to cluster.
  - `n_rp`: number of representative periods to find.
  - `drop_incomplete_last_period`: controls how the last period is treated if it
    is not complete: if this parameter is set to `true`, the incomplete period
    is dropped and the weights are rescaled accordingly; otherwise, clustering
    is done for `n_rp - 1` periods, and the last period is added as a special
    shorter representative period.
  - `method`: clustering method to use `:k_means`, `:k_medoids`, `:convex_hull`, `:convex_hull_with_null`, or `:conical_hull`.
  - `distance`: semimetric used to measure distance between data points.
  - `initial_representatives`: dataframe of initial RPs. It must use the same key
    columns and follow the same `layout` as `clustering_data`. For hull methods the
    RPs are prepended before clustering; for `:k_means`/`:k_medoids` they are appended
    after clustering.
  - `layout`: `ProfilesTableLayout` describing the column names.
  - other named arguments are forwarded to the clustering method.

# Returns

Returns a `ClusteringResult` with:
  - `profiles::DataFrame`: Long-format representative profiles with columns
    `:rep_period`, `layout.timestep`, all key columns (`auxiliary_data.key_columns`),
    and `layout.value`.
  - `weight_matrix::SparseMatrixCSC{Float64,Int}` (or dense `Matrix{Float64}`):
    rows correspond to source periods and columns to representative periods; entry
    `(p, r)` is the weight of period `p` assigned to representative `r`.
    If the last period is incomplete and `drop_incomplete_last_period` is false,
    it maps to its own representative column with its specific weight; if dropped,
    it is excluded from the rows.
  - `clustering_matrix::Matrix{Float64}`: The feature-by-period matrix used for
    clustering (features are derived from `layout.timestep` crossed with key columns).
  - `rp_matrix::Matrix{Float64}`: The representative profiles in matrix form
    (same feature layout as `clustering_matrix`).
  - `auxiliary_data::AuxiliaryClusteringData`: Auxiliary metadata such as
    `key_columns`, `period_duration`, `last_period_duration`, `n_periods`, and
    (for applicable methods) `medoids` indices.

# Examples

Finding two representatives using default values:
```
julia> df = DataFrame(
           period = kron(1:4, ones(Int, 2)),
           timestep = repeat(1:2, 4),
           profile = "A",
           value = 1:8,
         )

julia> res = TulipaClustering.find_representative_periods(df, 2)
```

Finding two representatives using k-medoids and a custom layout:
```
julia> layout = ProfilesTableLayout(; period = :p, timestep = :ts, value = :val)

julia> df = DataFrame(
           p = kron(1:4, ones(Int, 2)),
           ts = repeat(1:2, 4),
           profile = "A",
           val = 1:8,
         )

julia> res = TulipaClustering.find_representative_periods(df, 2; method = :k_medoids, layout)
```
"""
function find_representative_periods(
    clustering_data::AbstractDataFrame,
    n_rp::Int;
    drop_incomplete_last_period::Bool = false,
    method::Symbol = :k_means,
    distance::SemiMetric = SqEuclidean(),
    initial_representatives::AbstractDataFrame = DataFrame(),
    layout::ProfilesTableLayout = ProfilesTableLayout(),
    weight_type::Symbol = :dirac,
    tol::Float64 = 1e-2,
    weight_fitting_kwargs = Dict(),
    kwargs...,
)
    # 1. Check that the number of RPs makes sense. The first check can be done immediately,
    # The second check is done after we compute the auxiliary data
    if n_rp < 1
        throw(
            ArgumentError(
                "The number of representative periods is $n_rp but has to be at least 1.",
            ),
        )
    end

    # Find auxiliary data and pre-compute additional constants that are used multiple times alter
    aux = find_auxiliary_data(clustering_data; layout)
    n_periods = aux.n_periods

    if n_rp > n_periods
        throw(
            ArgumentError(
                "The number of representative periods exceeds the total number of periods, $n_rp > $n_periods.",
            ),
        )
    end

    if drop_incomplete_last_period && n_rp > n_periods - 1
        throw(
            ArgumentError(
                "The number of representative periods exceeds the total number of complete periods when dropping the last incomplete period, $n_rp > $(n_periods - 1).",
            ),
        )
    end

    has_incomplete_last_period = aux.last_period_duration ≠ aux.period_duration
    is_last_period_excluded = has_incomplete_last_period && !drop_incomplete_last_period
    n_complete_periods = has_incomplete_last_period ? n_periods - 1 : n_periods

    use_apgs = get(kwargs, :use_apgs, false) # check whether we are using apgs or not
    perc_initial_clustering = get(kwargs, :perc_initial_clustering, 3/4) # how many rps we compute before adding real worst cases

    # Check that the initial representatives are compatible with the clustering data
    if !isempty(initial_representatives)
        validate_initial_representatives(
            initial_representatives,
            clustering_data,
            aux,
            is_last_period_excluded,
            n_rp,
            layout,
        )
        i_rp = maximum(initial_representatives.period) # number of provided representative periods
    else
        i_rp = 0
    end

    # 2. Find the weights of the two types of periods and pre-build the weight matrix.
    # We assume that the only period that can be incomplete (i.e., has a duration
    # that is less than aux.period_duration) is the very last one. All other periods
    # are complete periods.
    complete_period_weight, incomplete_period_weight = find_period_weights(
        aux.period_duration,
        aux.last_period_duration,
        n_periods,
        drop_incomplete_last_period,
    )

    if use_apgs
        if method in [:k_means, :conical_hull]
            error("APGS is not implemented for conical hull and kmeans")
        end
        # divide the number of RPs in 2: first are the ones that are used in the clustering at the beginning, second the ones that are added as extreme periods later
        n_rp_first = round(Int, n_rp * perc_initial_clustering)
        n_rp_first = max(1, n_rp_first)
        n_rp_second = n_rp - n_rp_first

        # In both cases, the weights of the complete periods will be found after clustering.
        if is_last_period_excluded
            error(
                "Implementation of this method does not consider possibility of last period incomplete",
            )
            weight_matrix = sparse([n_periods], [n_rp], [incomplete_period_weight])
            n_rp -= 1  # incomplete last period becomes its own representative, exclude it from clustering
        else
            weight_matrix = spzeros(n_complete_periods, n_rp_first)
        end
    else
        if is_last_period_excluded
            weight_matrix = sparse([n_periods], [n_rp], [incomplete_period_weight])
            n_rp -= 1  # incomplete last period becomes its own representative, exclude it from clustering
        else
            weight_matrix = spzeros(n_complete_periods, n_rp)
        end
        n_rp_first = n_rp
    end

    # 3. Build the clustering matrix
    clustering_matrix, keys, n_rp_first = _build_clustering_matrix( # for kmedoids here it modifies n_rp_first
        clustering_data,
        n_rp_first,
        initial_representatives,
        i_rp,
        method,
        aux,
        n_complete_periods,
        layout,
    )

    # 4.1 Do the clustering, now that the data is transformed into a matrix
    clustering_matrix, rp_matrix, assignments, hull_indices =
        _compute_representatives_from_matrix(
            clustering_matrix,
            n_rp_first,
            initial_representatives,
            i_rp,
            method,
            aux,
            n_complete_periods,
            distance;
            kwargs...,
        )

    if use_apgs
        # 4.2 Reinterpret so we are sure to have right results to insert in fit_rep_period_weights
        rp_df, weight_matrix, rp_matrix = _reinterpret_clustering_results(
            clustering_data,
            clustering_matrix,
            keys,
            rp_matrix,
            n_rp_first,
            initial_representatives,
            i_rp,
            method,
            aux,
            n_complete_periods,
            n_periods,
            complete_period_weight,
            weight_matrix,
            is_last_period_excluded,
            distance,
            layout,
        )

        weight_type_to_add_rps = weight_type != :dirac ? weight_type : :convex

        # 4.3 Now let's compute the weights to choose which rps to add on top of the others
        fit_rep_period_weights!(
            weight_matrix,
            clustering_matrix,
            rp_matrix,
            Int64(i_rp);
            weight_type = weight_type_to_add_rps,
            tol,
            weight_fitting_kwargs...,
        )
        art_pos = (1:min(i_rp, size(weight_matrix, 2)))

        row_sums = vec(sum(weight_matrix[:, art_pos]; dims = 2)) # vector with the weights of each base rp to the artificial rps
        positive_rows = findall(>(1e-6), row_sums)
        n_rp_second_eff = min(n_rp_second, length(positive_rows))
        top_rows = positive_rows[partialsortperm(
            row_sums[positive_rows],
            1:n_rp_second_eff;
            rev = true,
        )]

        if !isempty(intersect(hull_indices, top_rows))
            error(
                "We should not select a vector that was already selected before",
                hull_indices,
                top_rows,
            )
        end

        if method in [:k_medoids]
            n_rp_first = n_rp_first + i_rp # it seemed that this was done inside reinterpret but apparently it is not
        end

        weight_matrix = spzeros(n_complete_periods, n_rp_first + n_rp_second_eff)
        rp_matrix = _combine_matrices([rp_matrix, clustering_matrix[:, top_rows]])

        if size(rp_matrix, 2) != n_rp_first + n_rp_second_eff
            error(
                "wrong size of rp_matrix: ",
                size(rp_matrix, 2),
                " instead of ",
                n_rp_first + n_rp_second_eff,
            )
        end

        if n_rp_second != n_rp_second_eff
            @warn "Wanted to add more extreme periods than necessary: $n_rp_second_eff vs $n_rp_second"
            # 4.4 Now let's cluster more to gain optimality if we have representative periods available
            n_rp_final = n_rp_second - n_rp_second_eff
            cols_to_exclude = union(hull_indices, top_rows)
            clustering_matrix_remaining =
                clustering_matrix[:, setdiff(axes(clustering_matrix, 2), cols_to_exclude)] # remove columns with period already represented
            distance_matrix = pairwise(distance, clustering_matrix_remaining; dims = 2)
            kmedoids_result = kmedoids(distance_matrix, n_rp_final)
            rp_matrix_end = clustering_matrix_remaining[:, kmedoids_result.medoids]
            rp_matrix = _combine_matrices([rp_matrix, rp_matrix_end])
            if n_rp_first + n_rp_second_eff + n_rp_final != n_rp
                error("Error with number of RPs")
            end
            weight_matrix = spzeros(n_complete_periods, n_rp)
        end

        if method in [:k_means, :conical_hull] && size(rp_matrix, 2) != 1 # because in reinterpret.. it modifies other stuff with those methods, and we have not implemented it with conical (size to exclude benchmark)
            error("This method is not implemented with ", method)
        end

        rp_df = if rp_matrix ≡ nothing
            nothing
        else
            matrix_and_keys_to_df(rp_matrix, keys; layout)
        end

        assignments = [
            argmin([
                distance(clustering_matrix[:, p], rp_matrix[:, r]) for
                r in axes(rp_matrix, 2)
            ]) for p in 1:n_complete_periods
        ]

        for (p, rp) in enumerate(assignments)
            weight_matrix[p, rp] = complete_period_weight
        end
    else
        # 5. Reinterpret the clustering results into a format we need (here the weight_matrix is also initialized to the closest RP)
        rp_df, weight_matrix, rp_matrix = _reinterpret_clustering_results( # if with k-medoids we did this again we would add artificial periods another time, so we only do the assignments
            clustering_data,
            clustering_matrix,
            keys,
            rp_matrix,
            n_rp_first,
            initial_representatives,
            i_rp,
            method,
            aux,
            n_complete_periods,
            n_periods,
            complete_period_weight,
            weight_matrix,
            is_last_period_excluded,
            distance,
            layout,
        )
    end

    return ClusteringResult(rp_df, weight_matrix, clustering_matrix, rp_matrix, aux, i_rp)
end

function _build_clustering_matrix(
    clustering_data,
    n_rp,
    initial_representatives,
    i_rp,
    method,
    aux,
    n_complete_periods,
    layout,
)
    period_col = layout.period
    if method in [:k_means, :k_medoids] && !isempty(initial_representatives)
        # If clustering is k-means or k-medoids we remove amount of initial representatives from n_rp
        n_rp -= i_rp
        clustering_matrix, keys = df_to_matrix_and_keys(
            clustering_data[clustering_data[!, period_col] .≤ n_complete_periods, :],
            aux.key_columns;
            layout,
        )

    elseif method in [:convex_hull, :convex_hull_with_null, :conical_hull] &&
           !isempty(initial_representatives)
        # If clustering is one of the hull methods, we add initial representatives to the clustering matrix in front
        updated_clustering_data = deepcopy(clustering_data)
        updated_clustering_data[!, period_col] =
            updated_clustering_data[!, period_col] .+ i_rp
        clustering_data = vcat(initial_representatives, updated_clustering_data)

        clustering_matrix, keys = df_to_matrix_and_keys(
            clustering_data[
                clustering_data[
                    !,
                    period_col,
                ] .≤ (n_complete_periods + maximum(
                    initial_representatives[!, period_col],
                )),
                :,
            ],
            aux.key_columns;
            layout,
        )
    else
        clustering_matrix, keys = df_to_matrix_and_keys(
            clustering_data[clustering_data[!, period_col] .≤ n_complete_periods, :],
            aux.key_columns;
            layout,
        )
    end
    return clustering_matrix, keys, n_rp
end

function _compute_representatives_from_matrix(
    clustering_matrix,
    n_rp,
    initial_representatives,
    i_rp,
    method,
    aux,
    n_complete_periods,
    distance;
    kwargs...,
)
    hull_indices = Int[]
    if n_rp == 0 # If due to the additional representatives we have no clustering, create an empty placeholder
        rp_matrix = nothing
        assignments = Int[]
    elseif method ≡ :k_means
        # Do the clustering

        # unpack kwargs
        allowed_kwargs = (:init, :tol, :maxiter, :display, :distance, :rng)
        filtered_kwargs =
            (; (k => kwargs[k] for k in allowed_kwargs if haskey(kwargs, k))...)

        kmeans_result = kmeans(clustering_matrix, n_rp; distance, filtered_kwargs...)

        # Reinterpret the results
        rp_matrix = kmeans_result.centers
        assignments = kmeans_result.assignments
    elseif method ≡ :k_medoids
        # Do the clustering
        # k-medoids uses distance matrix instead of clustering matrix
        distance_matrix = pairwise(distance, clustering_matrix; dims = 2)

        # unpack kwargs
        allowed_kwargs = (:init, :tol, :maxiter, :display)
        filtered_kwargs =
            (; (k => kwargs[k] for k in allowed_kwargs if haskey(kwargs, k))...)

        kmedoids_result = kmedoids(distance_matrix, n_rp; filtered_kwargs...)

        # Reinterpret the results
        rp_matrix = clustering_matrix[:, kmedoids_result.medoids]
        assignments = kmedoids_result.assignments
        aux.medoids = kmedoids_result.medoids

        return clustering_matrix, rp_matrix, assignments, aux.medoids

    elseif method ≡ :convex_hull
        # Do the clustering, with initial indices if provided
        initial_indices = if !isempty(initial_representatives)
            collect(1:i_rp)
        else
            nothing
        end
        clustering_matrix_no_art = clustering_matrix[:, (i_rp + 1):end] # to not start with the artificial ones
        hull_indices = greedy_convex_hull(
            clustering_matrix_no_art;
            # initial_indices = initial_indices,
            n_points = n_rp - i_rp, # so that only the others are computed
            distance,
            kwargs...,
        )
        hull_indices = hull_indices .+ i_rp
        if !isempty(initial_representatives)
            hull_indices = vcat(initial_indices, hull_indices)
        end

        # Reinterpret the results
        rp_matrix = clustering_matrix[:, hull_indices]
        assignments = [
            argmin([
                distance(clustering_matrix[:, h], clustering_matrix[:, p + i_rp]) for
                h in hull_indices
            ]) for p in 1:n_complete_periods
        ]
        clustering_matrix = clustering_matrix[:, (i_rp + 1):end]
        aux.medoids = hull_indices
    elseif method ≡ :convex_hull_with_null
        # Check if we can add null to the clustering matrix. The distance to null can
        # be undefined, e.g., for the cosine distance.
        is_distance_to_zero_undefined =
            isnan(distance(zeros(size(clustering_matrix, 1), 1), clustering_matrix[:, 1]))

        if is_distance_to_zero_undefined
            throw(
                ArgumentError(
                    "cannot add null to the clustering data because distance to it is undefined",
                ),
            )
        end

        # Add null to the clustering matrix
        clustering_matrix_no_art = clustering_matrix[:, (i_rp + 1):end]
        matrix = [zeros(size(clustering_matrix, 1), 1) clustering_matrix_no_art]

        # Do the clustering
        hull_indices = greedy_convex_hull(
            matrix;
            n_points = n_rp + 1 - i_rp,
            distance,
            initial_indices = [1], # only 0
            kwargs...,
        )

        # Remove null from the beginning and shift all indices by one
        popfirst!(hull_indices)
        hull_indices .-= 1
        hull_indices = hull_indices .+ i_rp
        if !isempty(initial_representatives)
            hull_indices = vcat(collect(1:i_rp), hull_indices)
        end

        # Reinterpret the results
        rp_matrix = clustering_matrix[:, hull_indices]
        assignments = [
            argmin([
                distance(clustering_matrix[:, h], clustering_matrix[:, p + i_rp]) for
                h in hull_indices
            ]) for p in 1:n_complete_periods
        ]
        clustering_matrix = clustering_matrix[:, (i_rp + 1):end]

        aux.medoids = hull_indices
    elseif method ≡ :conical_hull
        # Do a gnomonic projection (normalization) of the data
        normal_vector = vec(mean(clustering_matrix; dims = 2))
        normalize!(normal_vector)
        projection_coefficients = [
            1.0 / dot(normal_vector, clustering_matrix[:, j]) for
            j in axes(clustering_matrix, 2)
        ]
        projected_matrix = [
            clustering_matrix[i, j] * projection_coefficients[j] for
            i in axes(clustering_matrix, 1), j in axes(clustering_matrix, 2)
        ]

        initial_indices = if !isempty(initial_representatives)
            collect(1:i_rp)
        else
            nothing
        end

        hull_indices = greedy_convex_hull(
            projected_matrix;
            n_points = n_rp,
            distance,
            mean_vector = normal_vector,
            initial_indices = initial_indices,
            kwargs...,
        )

        # Reinterpret the results
        rp_matrix = clustering_matrix[:, hull_indices]

        assignments = [
            argmin([
                distance(clustering_matrix[:, h], clustering_matrix[:, p + i_rp]) for
                h in hull_indices
            ]) for p in 1:n_complete_periods
        ]
        clustering_matrix = clustering_matrix[:, (i_rp + 1):end]
    else
        throw(ArgumentError("Clustering method is not supported"))
    end
    hull_indices = hull_indices[(i_rp + 1):end]
    return clustering_matrix, rp_matrix, assignments, hull_indices .- i_rp
end

function _reinterpret_clustering_results(
    clustering_data,
    clustering_matrix,
    keys,
    rp_matrix,
    n_rp,
    initial_representatives,
    i_rp,
    method,
    aux,
    n_complete_periods,
    n_periods,
    complete_period_weight,
    weight_matrix,
    is_last_period_excluded,
    distance,
    layout,
)
    period_col = layout.period
    # First, convert the matrix data back to dataframes using the previously saved key columns
    rp_df = if rp_matrix ≡ nothing
        nothing
    else
        matrix_and_keys_to_df(rp_matrix, keys; layout)
    end

    # In case of initial representatives and a non hull method, we add them now
    if !isempty(initial_representatives) && method in [:k_means, :k_medoids]
        # representatives_to_add = select!(
        #     initial_representatives,
        #     period_col => :rep_period,
        #     aux.key_columns...,
        #     layout.value,
        # )
        # representatives_to_add.rep_period .= representatives_to_add.rep_period .+ n_rp
        # rp_df = if rp_df === nothing
        #     representatives_to_add
        # else
        #     vcat(rp_df, representatives_to_add)
        # end
        # rename!(rp_df, :rep_period => period_col)
        # rp_matrix, keys = df_to_matrix_and_keys(rp_df, aux.key_columns; layout)
        # rename!(rp_df, period_col => :rep_period)
        # n_rp += i_rp

        # to add them at the beginning
        representatives_to_add = select!(
            initial_representatives,
            period_col => :rep_period,
            aux.key_columns...,
            layout.value,
        )
        rp_df = if rp_df === nothing
            representatives_to_add
        else
            rp_df.rep_period .= rp_df.rep_period .+ i_rp
            vcat(representatives_to_add, rp_df)
        end
        rename!(rp_df, :rep_period => period_col)
        rp_matrix, keys = df_to_matrix_and_keys(rp_df, aux.key_columns; layout)
        rename!(rp_df, period_col => :rep_period)
        n_rp += i_rp
    end

    # TODO: Verify with Greg if we need this inconditional replacement of assignments or not (it seems like a missing if here)
    assignments = [
        argmin([
            distance(clustering_matrix[:, p], rp_matrix[:, r]) for r in axes(rp_matrix, 2)
        ]) for p in 1:n_complete_periods
    ]

    for (p, rp) in enumerate(assignments)
        weight_matrix[p, rp] = complete_period_weight
    end

    # Next, re-append the last period if it was excluded from clustering
    if is_last_period_excluded
        n_rp += 1
        append_period_from_source_df_as_rp!(
            rp_df;
            source_df = clustering_data,
            period = n_periods,
            rp = n_rp,
            key_columns = aux.key_columns,
            layout = layout,
        )
        if method ≡ :k_medoids
            append!(aux.medoids, n_complete_periods + 1)
        end
    end

    return rp_df, weight_matrix, rp_matrix
end

function _combine_matrices(matrices::Vector{Matrix{Float64}})
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
