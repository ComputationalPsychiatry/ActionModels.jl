"""
    PVLDelta

A premade model implementing the Prospect Valence Learning (PVL-Delta) algorithm, commonly used for modeling learning and decision-making in tasks like the Iowa Gambling Task (IGT).

The PVL-Delta model updates expected values for each option based on observed rewards, using a prospect-theoretic transformation and a delta learning rule. Action probabilities are computed via a softmax over expected values.

# Fields
- `n_options::Int64`: Number of available options (e.g., decks in IGT).
- `learning_rate::Float64`: Learning rate (α), controls the speed of value updating.
- `action_noise::Float64`: (Inverse temperature β = 1/action_noise), controls choice stochasticity.
- `reward_sensitivity::Float64`: Prospect theory power parameter (A), controls reward sensitivity.
- `loss_aversion::Float64`: Loss aversion parameter (w), scales negative rewards.
- `initial_value::Array{Float64}`: Initial expected values (V) for each option.
- `act_before_update::Bool`: If true, action is determined before reward is observed (for tasks where action and reward happen on the same timestep).

# Examples
```jldoctest
julia> config = PVLDelta(n_options=4, learning_rate=0.1, action_noise=1.0, reward_sensitivity=0.5, loss_aversion=1.0, initial_value=zeros(4), act_before_update=true)
PVLDelta(4, 0.1, 1.0, 0.5, 1.0, [0.0, 0.0, 0.0, 0.0], true)

julia> action_model = ActionModel(config)
-- ActionModel --
Action model function: pvl_delta_act_before_update
Number of parameters: 5
Number of states: 1
Number of observations: 2
Number of actions: 1
```
"""

export PVLDelta

Base.@kwdef struct PVLDelta <: AbstractPremadeModel
    n_options::Int64
    learning_rate::Float64 = 0.1
    action_noise::Float64 = 1
    reward_sensitivity::Float64 = 0.5
    loss_aversion::Float64 = 1
    initial_value::Array{Float64} = zeros(Float64, n_options)
    act_before_update::Bool = false
end

function ActionModel(config::PVLDelta)

    if !config.act_before_update

        am_function = function pvl_delta(
            attributes::ModelAttributes,
            chosen_option::Int64,
            reward::Float64,
        )

            ## Read in parameters and states ##
            parameters = load_parameters(attributes)
            states = load_states(attributes)

            α = parameters.learning_rate
            A = parameters.reward_sensitivity
            w = parameters.loss_aversion
            β⁻¹ = 1 / parameters.action_noise

            Ev = states.expected_value


            ## Update expected values ##
            #Calculate prediction error
            if reward >= 0
                prediction_error = (reward^A) - Ev[chosen_option]
            else
                prediction_error = -w * (abs(reward)^A) - Ev[chosen_option]
            end

            #Calculate new expected value for the chosen option
            # Ev[chosen_option] = Ev[chosen_option] + α * prediction_error
            Ev = [
                Ev[option_idx] + α * prediction_error * (chosen_option == option_idx) for
                option_idx = 1:config.n_options
            ]


            #Update the expected value for next timestep
            update_state!(attributes, :expected_value, Ev)


            ## Get action probabilities ##
            #Softmax over expected values for each option
            action_probabilities = softmax(Ev * β⁻¹)

            #Avoid underflow and overflow
            action_probabilities = clamp.(action_probabilities, 0.001, 0.999)
            action_probabilities = action_probabilities / sum(action_probabilities)

            return Categorical(action_probabilities)
        end

    else

        am_function = function pvl_delta_act_before_update(
            attributes::ModelAttributes,
            chosen_option::Int64,
            reward::Float64,
        )

            ## Read in parameters and states ##
            parameters = load_parameters(attributes)
            states = load_states(attributes)

            α = parameters.learning_rate
            A = parameters.reward_sensitivity
            w = parameters.loss_aversion
            β⁻¹ = 1 / parameters.action_noise

            Ev = states.expected_value


            ## Get action probabilities ##
            #Softmax over expected values for each option
            action_probabilities = softmax(Ev * β⁻¹)

            #Avoid underflow and overflow
            action_probabilities = clamp.(action_probabilities, 0.001, 0.999)
            action_probabilities = action_probabilities / sum(action_probabilities)


            ## Update expected values ##
            #Calculate prediction error
            if reward >= 0
                prediction_error = (reward^A) - Ev[chosen_option]
            else
                prediction_error = -w * (abs(reward)^A) - Ev[chosen_option]
            end

            #Calculate new expected value for the chosen option
            #Ev[chosen_option] = Ev[chosen_option] + α * prediction_error
            Ev = [
                Ev[option_idx] + α * prediction_error * (chosen_option == option_idx) for
                option_idx = 1:config.n_options
            ]

            #Update the expected value for next timestep
            update_state!(attributes, :expected_value, Ev)

            return Categorical(action_probabilities)
        end
    end

    parameters = (
        learning_rate = Parameter(config.learning_rate),
        reward_sensitivity = Parameter(config.reward_sensitivity),
        action_noise = Parameter(config.action_noise),
        loss_aversion = Parameter(config.loss_aversion),
        initial_value = InitialStateParameter(config.initial_value, :expected_value),
    )

    states = (; expected_value = State(Array{Float64}))

    observations = (; chosen_option = Observation(Int64), reward = Observation())

    actions = (; choice = Action(Categorical),)

    return ActionModel(
        am_function,
        parameters = parameters,
        states = states,
        observations = observations,
        actions = actions,
    )
end
