"""
    f(agent::Agent, target_state::Symbol, index::Union{Nothing,Vector{Int}}=nothing)

Plot the trajectory (history) of a state variable for an agent over time.

This plotting recipe visualizes the time course of a given state variable from an agent's history. If the state is multivariate (e.g., a vector or array), you can specify an index to plot a particular dimension. If the state is univariate, the index should be left as `nothing` (the default).

# Arguments
- `agent::Agent`: The agent whose state trajectory will be plotted.
- `target_state::Symbol`: The name of the state variable to plot (e.g., `:expected_value`).
- `index::Union{Nothing,Vector{Int}}`: (Optional) Index for multivariate states (e.g., `[1]` for the first element).

# Example
```jldoctest; setup = :(using ActionModels, StatsPlots; agent = init_agent(ActionModel(RescorlaWagner()), save_history=true); simulate!(agent, [1.0, 0.5, 0.2]))
julia> plot(agent, :expected_value)
Plot{Plots.GRBackend() n=1}
```
For a multivariate state (e.g., a vector):
```jldoctest; setup = :(using ActionModels, StatsPlots; agent = init_agent(ActionModel(RescorlaWagner(type=:categorical, n_categories = 3)), save_history=true); simulate!(agent, [1,2,1]))
julia> plot(agent, :expected_value, [1])
Plot{Plots.GRBackend() n=1}
```
"""
@recipe function f(
    agent::Agent,
    target_state::Symbol,
    index::Union{Nothing,Vector{Int}} = nothing;
)
    #Get the history of the state
    state_history = get_history(agent, target_state)

    if !isnothing(index) && !(first(state_history) isa AbstractArray)
        @error "An index was provided, but the state is not an array."
    end

    if isnothing(index) && first(state_history) isa AbstractArray
        @error "The state is an array. Provide an index to select which value to plot."
    end

    #If the state is multivariate
    if first(state_history) isa AbstractArray && !isnothing(index)

        #Extract the state history for the given index
        index = CartesianIndex(index...)
        state_history = map(i -> state_history[i][index], 1:length(state_history))
    end

    #Replace missings with NaNs for plotting
    state_history = replace(state_history, missing => NaN)

    #The x-axis starts at 0 and goes to the number of timesteps
    x_axis = collect(0:agent.n_timesteps)

    xlabel --> "Timestep"
    yguide --> "$target_state"

    #Plot the history
    @series begin
        label --> nothing
        seriestype --> :path
        markersize --> 5
        title --> "$target_state"
        x_axis, state_history
    end
end
