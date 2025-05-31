##########################################
### TYPES FOR SPECIFYING TURING MODELS ###
##########################################

#Structs for setting missing actions to be either skipped or inferred
abstract type AbstractMissingActions end
struct NoMissingActions <: AbstractMissingActions end
struct SkipMissingActions <: AbstractMissingActions end
struct InferMissingActions <: AbstractMissingActions end


abstract type AbstractPopulationModel end

struct CustomPopulationModel <: AbstractPopulationModel end
struct RegressionPopulationModel <: AbstractPopulationModel end
struct IndependentPopulationModel <: AbstractPopulationModel end
struct SingleSessionPopulationModel <: AbstractPopulationModel end


## Types for the GLM population model ##
Base.@kwdef mutable struct RegPrior
    β::Distribution
    σ::Union{Nothing,Vector{Distribution}}
end

"""
RegressionPrior{Dβ,Dσ}

Type for specifying priors for regression coefficients and random effects in regression population models.

# Type Parameters
- `Dβ`: Distribution type for β regression coefficients.
- `Dσ`: Distribution type for σ random effect deviations.

# Fields
- `β`: Prior or vector of priors for regression coefficients (default: `TDist(3)`). If only one prior is given, it is used for all coefficients. If a vector is given, it should match the number of coefficients in the regression formula.
- `σ`: Prior or vector of priors for random effect deviations (default: truncated `TDist(3)` at 0). If only one prior is given, it is used for all random effects. If a vector is given, it should match the number of random effects in the regression formula.

# Examples
```jldoctest
julia> RegressionPrior()
RegressionPrior{TDist{Float64}, Truncated{TDist{Float64}, Continuous, Float64, Float64, Nothing}}(TDist{Float64}(ν=3.0), Truncated(TDist{Float64}(ν=3.0); lower=0.0))

julia> RegressionPrior(β = TDist(4), σ = truncated(TDist(4), lower = 0))
RegressionPrior{TDist{Float64}, Truncated{TDist{Float64}, Continuous, Float64, Float64, Nothing}}(TDist{Float64}(ν=4.0), Truncated(TDist{Float64}(ν=4.0); lower=0.0))

julia> RegressionPrior(β = [TDist(4), TDist(2)], σ = truncated(TDist(4), lower = 0)) #For setting multiple coefficients separately
RegressionPrior{TDist{Float64}, Truncated{TDist{Float64}, Continuous, Float64, Float64, Nothing}}(TDist{Float64}[TDist{Float64}(ν=4.0), TDist{Float64}(ν=2.0)], Truncated(TDist{Float64}(ν=4.0); lower=0.0))
```
"""
Base.@kwdef struct RegressionPrior{D1<:Distribution,D2<:Distribution}
    β::Union{D1,Vector{D1}} = TDist(3)
    σ::Union{D2,Vector{Vector{D2}}} = truncated(TDist(3), lower = 0)
end

"""
Regression

Type for specifying a regression model in a population model. Contains the formula, prior, and inverse link function.

# Fields
- `formula`: The regression formula (as a `FormulaTerm`).
- `prior`: The prior for regression coefficients and error terms (as a `RegressionPrior`). Default is `RegressionPrior()`.
- `inv_link`: The inverse link function (default: `identity`).

# Examples
```jldoctest
julia> Regression(@formula(y ~ x))
Regression(y ~ x, RegressionPrior{TDist{Float64}, Truncated{TDist{Float64}, Continuous, Float64, Float64, Nothing}}(TDist{Float64}(ν=3.0), Truncated(TDist{Float64}(ν=3.0); lower=0.0)), identity)

julia> Regression(@formula(y ~ x + (1|ID)), exp) # With a random effect and a exponential inverse link function
Regression(y ~ x + :(1 | ID), RegressionPrior{TDist{Float64}, Truncated{TDist{Float64}, Continuous, Float64, Float64, Nothing}}(TDist{Float64}(ν=3.0), Truncated(TDist{Float64}(ν=3.0); lower=0.0)), exp)

julia> Regression(@formula(y ~ x), RegressionPrior(β = TDist(4), σ = truncated(TDist(4), lower = 0)), logistic) #With a custom prior and logistic inverse link function
Regression(y ~ x, RegressionPrior{TDist{Float64}, Truncated{TDist{Float64}, Continuous, Float64, Float64, Nothing}}(TDist{Float64}(ν=4.0), Truncated(TDist{Float64}(ν=4.0); lower=0.0)), logistic)
```
"""
struct Regression
    formula::FormulaTerm
    prior::RegressionPrior
    inv_link::Function

    function Regression(
        formula::FormulaTerm,
        prior::RegressionPrior = RegressionPrior(),
        inv_link::Function = identity,
    )
        new(formula, prior, inv_link)
    end
    function Regression(
        formula::FormulaTerm,
        inv_link::Function,
        prior::RegressionPrior = RegressionPrior(),
    )
        new(formula, prior, inv_link)
    end
end

#########################################
### TYPES FOR MODEL FITTING AND TOOLS ###
#########################################
## Abstract type for containing fitting results for storing in the following ##
abstract type AbstractFittingResult end

### Structs for storing results of model fitting ###
"""
ModelFitInfo

Container for metadata about a model fit, including session IDs and estimated parameter names.

# Fields
- `session_ids::Vector{String}`: List of session identifiers.
- `estimated_parameter_names::Tuple{Vararg{Symbol}}`: Names of estimated parameters.
"""
Base.@kwdef struct ModelFitInfo
    session_ids::Vector{String}
    estimated_parameter_names::Tuple{Vararg{Symbol}}
end

"""
ModelFitResult

Container for the results of a model fit (either prior or posterior), including MCMC chains and (optionally) session-level parameters.

# Fields
- `chains::Chains`: MCMC chains from Turing.jl.
- `session_parameters::Union{Nothing,AbstractFittingResult}`: Session-level parameter results (optional).
"""
Base.@kwdef mutable struct ModelFitResult
    chains::Chains
    session_parameters::Union{Nothing,AbstractFittingResult} = nothing
end

"""
ModelFit{T}

Container for a fitted model, including the Turing model, population model type, data, and results (both posterior and prior).

# Type Parameters
- `T`: The population model type (subtype of `AbstractPopulationModel`).

# Fields
- `model`: The Turing model object.
- `population_model_type`: The population model type.
- `population_data`: The data describing each session (i.e., after calling unique(data, session_cols), so no observations and actions).
- `info`: Metadata about the fit (as a `ModelFitInfo`).
- `prior`: Prior fit results (empty until `sample_posterior!` is called).
- `posterior`: Posterior fit results (empty until `sample_prior!` is called).
"""
Base.@kwdef mutable struct ModelFit{T<:AbstractPopulationModel}
    model::DynamicPPL.Model
    population_model_type::T
    population_data::DataFrame
    info::ModelFitInfo
    prior::Union{ModelFitResult,Nothing} = nothing
    posterior::Union{ModelFitResult,Nothing} = nothing
end

### Structs for containing outputted session parameters and state trajectories ###
"""
SessionParameters

Container for session-level parameter estimates from model fitting.

# Fields
- `value`: NamedTuple for each parameter, containing a NamedTuple for each session. Within is an AxisArray of samples.
- `modelfit`: The associated `ModelFit` object.
- `estimated_parameter_names`: Tuple of estimated parameter names.
- `session_ids`: Vector of session identifiers.
- `parameter_types`: NamedTuple of parameter types.
- `n_samples`: Number of posterior samples.
- `n_chains`: Number of MCMC chains.
"""
struct SessionParameters <: AbstractFittingResult
    value::NamedTuple{names,<:Tuple{Vararg{NamedTuple}}} where {names}
    modelfit::ModelFit
    estimated_parameter_names::Tuple{Vararg{Symbol}}
    session_ids::Vector{String}
    parameter_types::NamedTuple{
        parameter_names,
        <:Tuple{Vararg{Type}},
    } where {parameter_names}
    n_samples::Int
    n_chains::Int
end

"""
StateTrajectories

Container for state trajectory estimates from model fitting.

# Fields
- `value`: NamedTuple for each state, containing a NamedTuple for each session. Within is an AxisArray of samples per timestep.
- `modelfit`: The associated `ModelFit` object.
- `state_names`: Vector of state names.
- `session_ids`: Vector of session identifiers.
- `state_types`: NamedTuple of state types.
- `n_samples`: Number of posterior samples.
- `n_chains`: Number of MCMC chains.
"""
struct StateTrajectories <: AbstractFittingResult
    value::NamedTuple{names,<:Tuple{Vararg{NamedTuple}}} where {names}
    modelfit::ModelFit
    state_names::Vector{Symbol}
    session_ids::Vector{String}
    state_types::NamedTuple{state_names,<:Tuple{Vararg{Type}}} where {state_names}
    n_samples::Int
    n_chains::Int
end


### Type for the save-resume functionality ###
Base.@kwdef struct SampleSaveResume
    save_every::Int = 100
    path::String = "./.samplingstate"
    chain_prefix::String = "ActionModels_chain_segment"
end
