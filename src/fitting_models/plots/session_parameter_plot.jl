########################
### RIGTH PANEL PLOT ###
########################
struct SessionParameterPlotData
    median::Any
    interval::Any
    distribution::Any
    color::Any
    direction::Any
end

function get_session_parameter_plot_data(session, confidence, direction, color)

    kernel_density = kde(vec(parent(session)))

    SessionParameterPlotData(
        median(session),
        percentile(session, [100-confidence, confidence]),
        (density = kernel_density.density, x_range = kernel_density.x),
        color,
        direction,
    )
end

function distribution_polygon(data::SessionParameterPlotData, y, distribution_height)

    density = data.distribution.density
    scaled_density = density ./ maximum(density) .* distribution_height
    y_edge = y .+ data.direction .* scaled_density

    x = data.distribution.x_range
    x_poly = vcat(first(x), x, last(x))
    y_poly = vcat(y, y_edge, y)

    return x_poly, y_poly
end

@recipe function f(data::SessionParameterPlotData, y, row_type)

    subplot := 2
    ygrid := false
    ymirror := true

    y_offset = 0.05 * data.direction
    y += y_offset
    distribution_height = 0.3

    if data.direction < 0
        alpha = 0.4
    else
        alpha = 1
    end

    if row_type == "wide"
        @series begin
            seriestype := :shape
            color := data.color
            label := ""
            seriesalpha := 0.3
            distribution_polygon(data, y, distribution_height)
        end
    end

    @series begin
        seriestype := :line
        color := data.color
        label := ""
        linewidth := 2
        seriesalpha := alpha
        [data.interval[1], data.interval[2]], [y, y]
    end

    @series begin
        seriestype := :scatter
        label := ""
        markercolor := data.color
        markershape := :vline
        markersize := 4
        markerstrokewidth := 2
        seriesalpha := alpha
        [data.median], [y]
    end


end


########################
### LEFT PANEL TABLE ###
########################
struct SessionParameterTableData
    table::Any
    column_names::Any
    row_order::Any
end

function get_session_table_data(model, column_names, row_order)
    n=length(row_order)# DELETE THIS (and 1:n below) AFTER TESTING
    SessionParameterTableData(
        model.population_data[1:n, column_names],
        column_names,
        row_order,
    )
end

@recipe function f(data::SessionParameterTableData, row_type, group_by, group_sizes)

    n_rows, n_cols = size(data.table)
    x_offset = n_cols+0.2

    subplot := 1
    title := ""
    xlims := [-x_offset, 0]
    markeralpha := 0 # make marker invisible
    framestyle := :none


    # Frame around table
    @series begin
        seriestype := :path
        subplot := 1
        color := :black
        linewidth := 1
        [0, -x_offset, -x_offset, 0], [n_rows+0.5, n_rows+0.5, 0.5, 0.5]
    end

    # Table content
    for col_i = 1:n_cols
        x = col_i - x_offset - 0.5

        if isnothing(group_by)
            column_name_y = n_rows+0.8
            @series begin
                seriestype := :scatter
                series_annotations := text(
                    string(data.column_names[col_i]), #  the label
                    12,           #  font-size (pt)
                    :black;      #  color
                    halign = :center,
                    valign = :center,
                )
                [x], [column_name_y]
            end

            for row_i = 1:n_rows
                string = data.table[data.row_order[row_i], col_i]
                y = row_i
                @series begin
                    seriestype := :scatter
                    series_annotations := text(
                        string, #  the label
                        9,           #  font-size (pt)
                        :black;      #  color
                        halign = :center,
                        valign = :center,
                    )
                    [x], [y]
                end
            end
        else
            n_groups = length(group_sizes)
            edge_coord = [0, accumulate(+, group_sizes)...]

            group_by_categories = unique(data.table[!, col_i])

            for group_i = 1:n_groups
                y = (edge_coord[group_i]+edge_coord[group_i+1]+1)/2
                string = group_by_categories[group_i]

                @series begin
                    seriestype := :scatter
                    series_annotations := text(
                        string, #  the label
                        9,           #  font-size (pt)
                        :black;      #  color
                        halign = :center,
                        valign = :center,
                    )
                    [x], [y]
                end
            end

        end
    end
end



#################
### FULL PLOT ###
#################
@recipe function f(
    model::ActionModels.ModelFit,
    parameter;
    confidence = 95,
    row_height_adjust = 1,
    n = 6, # DELETE THIS AFTER TESTING AND UNCOMMENT BELOW
    ordered_by_median = false,
    reverse_order = false,
    id_column_names = nothing, # Maybe better naming?
    show_prior = nothing,
    row_type = nothing,
    group_by = nothing,
    color_by = group_by,
)

    #### Pasted in here because they where not in scope for some reason
    id_separator = "."
    id_column_separator = ":"
    ####

    # Color settings
    colors = (
        (:darkgreen, :green),
        (:darkorange, :orange),
        (:purple, :purple),
        (:red, :red),
        (:blue, :blue),
        (:pink, :pink)
    )

    # Data
    posterior_sessions_parameters = get_session_parameters!(model, :posterior)
    prior_sessions_parameters = get_session_parameters!(model, :prior)

    posterior_sessions = posterior_sessions_parameters.value[parameter]#[1:n] #REMOVE 1:n AFTER TESTING
    prior_sessions = prior_sessions_parameters.value[parameter]#[1:n] #REMOVE 1:n AFTER TESTING

    # Constants
    #n = length(posterior_sessions)

    # Row type
    if isnothing(row_type)
        row_type = n > 15 ? "narrow" : "wide"
    end
    if row_type == "narrow"
        row_height = 8 * row_height_adjust
        if isnothing(show_prior)
            show_prior = false
        end
    end

    if row_type == "wide"
        row_height = 60 * row_height_adjust
        if isnothing(show_prior)
            show_prior = true
        end
    end


    # Row Order
    function get_row_order(sessions, indicies)
        if ordered_by_median
            medians = [median(sessions[i]) for i in indicies]
            return indicies[sortperm(medians, rev = reverse_order)]
        else
            return indicies
        end
    end

    group_sizes = []
    if isnothing(group_by)
        row_order = get_row_order(posterior_sessions, collect(1:n))
    else
        row_order = []
        group_by_column = model.population_data[1:n, group_by]
        group_by_categories = unique(group_by_column)
        for category in group_by_categories
            group_row_order = findall(x -> x==category, group_by_column)
            group_row_order = get_row_order(posterior_sessions, group_row_order)
            push!(row_order, group_row_order...)
            push!(group_sizes, length(group_row_order))
        end
    end



    # Session names and table width # THIS SHOULD BE DONE MORE ELEGANT
    if isnothing(group_by)
        if isnothing(id_column_names)
            if row_type == "wide"
                ids = posterior_sessions_parameters.session_ids
                id_column_names = [
                    split(i, id_column_separator)[1] for
                    i in split(string(ids[1]), id_separator)
                ]
            else
                id_column_names = []
            end
        end

        if row_type == "wide"
            table_width = length(id_column_names) * 0.2 + 0.001
        else
            table_width = 0.001
        end
    else
        id_column_names = [group_by]
        table_width = 0.2
    end

    # Plot attributes
    layout := grid(1, 2, widths = [table_width, 1-table_width])
    left_margin := -2.1mm
    right_margin := -2.1mm
    yaxis := false
    ylims := (0.5, n + 1.5)
    tickdirection := :out
    widen := false
    label := ""

    title --> string(parameter)#, "   n=", n)
    size --> (700, row_height * n + 150)


    if isnothing(group_by)
        row_background_sequence = collect(0.5:1:(n+0.5))
    else
        row_background_sequence = [0.5]
        for i in eachindex(group_sizes)
            push!(row_background_sequence, accumulate(+, group_sizes)[i] + 0.5)
        end
    end

    @series begin
        subplot := 1
        seriestype := :hspan
        color := :lightgrey
        alpha := 0.3
        row_background_sequence
    end

    @series begin
        subplot := 2
        seriestype := :hspan
        color := :lightgrey
        alpha := 0.3
        row_background_sequence
    end

    # Left panel table
    table_data = get_session_table_data(model, id_column_names, row_order)
    @series table_data, row_type, group_by, group_sizes

    if !isnothing(color_by)
        color_by_strings = string.(unique(model.population_data[!, color_by]))
    end

    # Rigt panel plot
    for i = 1:n
        # i = the position in the plot (from bottom to top)
        # row_order(i) = the index in the original data frame

        if !isnothing(color_by)
            color_by_string = string(model.population_data[row_order[i], color_by])
            color_number = findfirst(x -> x==color_by_string, color_by_strings)
        else
            color_number = 1
        end

        if show_prior
            prior_data = get_session_parameter_plot_data(
                prior_sessions[row_order[i]],
                confidence,
                -1,
                colors[color_number][2],
            )
            @series prior_data, i, row_type
        end

        posterior_data = get_session_parameter_plot_data(
            posterior_sessions[row_order[i]],
            confidence,
            1,
            colors[color_number][1],
        )
        @series posterior_data, i, row_type
    end
end

