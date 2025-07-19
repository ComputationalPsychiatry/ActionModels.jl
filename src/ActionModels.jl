module ActionModels

#Load packages
using Reexport
@reexport using Distributions #Make distributions available to the user
@reexport using Turing #For fitting models
using Turing: DynamicPPL, AbstractMCMC, LogDensityProblems

using RecipesBase #For defining plots
using KernelDensity, StatsBase, Plots, Measures #For the session parameters plot (get rid of Plots! and Measures if possible)
using DataFrames #For the input format to create_model
using AxisArrays #For storing session parameters and state trajectories
using StatsModels, MixedModels, LogExpFunctions #For the GLM population Model

using JLD2, Suppressor #For the save_resume functionality
using ProgressLogging #For progress bars


#ADType functionality
using ADTypes: AutoForwardDiff
import ForwardDiff
## AD types to be compatible with  ##
#using ADTypes: AutoReverseDiff, AutoMooncake, AutoEnzyme, AutoFiniteDifferences
#import ReverseDiff
#import Mooncake
#import FiniteDifferences: central_fdm
#import Enzyme: set_runtime_activity, Forward, Reverse 

## For defining action models ##
export ActionModel, Parameter, InitialStateParameter, State, Observation, Action
export ModelAttributes,
    load_parameters, load_states, load_actions, update_state!, RejectParameters, AttributeError

## For simulation ##
export init_agent
export simulate!, observe!
export get_parameters, get_states, get_actions, get_history
export set_parameters!, set_states!, set_actions!, reset!
#export plot_trajectory, plot_trajectory!

## For fitting models ##
export create_model
export Regression, RegressionPrior, @formula
export exp, logistic, softmax #export commonly used LogExpFunctions functions
export sample_prior!, sample_posterior!, SampleSaveResume
export NoMissingActions, SkipMissingActions, InferMissingActions
export ParameterChecking, NoParameterChecking
export get_session_parameters!, get_state_trajectories!, summarize  
# export parameter_recovery
# export plot_parameters, plot_trajectories



## Constants for creating ids and names consistently ##
const id_separator = "."
const id_column_separator = ":"


### Types ###
include(joinpath("defining_models", "structs.jl"))
include(joinpath("simulation", "structs.jl"))
include(joinpath("fitting_models", "structs.jl"))


### Functions for model definition ###
include(joinpath("defining_models", "prints.jl"))
include(joinpath("defining_models", "model_attributes.jl"))
include(joinpath("defining_models", "manipulate_attributes.jl"))
include(joinpath("defining_models", "no_submodel_dispatches.jl"))

### Functions for simulation ###
include(joinpath("simulation", "prints.jl"))
include(joinpath("simulation", "simulate.jl"))
include(joinpath("simulation", "manipulate_attributes.jl"))
include(joinpath("simulation", "plots", "plot_trajectory.jl"))

### Functions for fitting models ###
include(joinpath("fitting_models", "prints.jl"))
include(joinpath("fitting_models", "turing_model", "create_model.jl"))
include(joinpath("fitting_models", "turing_model", "helper_functions.jl"))
include(joinpath("fitting_models", "turing_model", "sessions_models", "parameter_checking.jl"))
include(joinpath("fitting_models", "turing_model", "sessions_models", "pointwise_sessions_model.jl"))
include(joinpath("fitting_models", "turing_model", "sessions_models", "fast_sessions_model.jl"))
include(
    joinpath(
        "fitting_models",
        "turing_model",
        "population_models",
        "independent_population_model.jl",
    ),
)
include(
    joinpath(
        "fitting_models",
        "turing_model",
        "population_models",
        "glm_population_model.jl",
    ),
)
include(
    joinpath(
        "fitting_models",
        "turing_model",
        "population_models",
        "single_session_population_model.jl",
    ),
)
include(joinpath("fitting_models", "inference", "sampling.jl"))
include(joinpath("fitting_models", "inference", "save_resume.jl"))
include(joinpath("fitting_models", "inference", "get_session_parameters!.jl"))
include(joinpath("fitting_models", "inference", "get_state_trajectories.jl"))
include(joinpath("fitting_models", "inference", "summarize.jl"))
include(joinpath("fitting_models", "tools", "parameter_recovery.jl"))
include(joinpath("fitting_models", "plots", "session_parameter_plot.jl"))
# include(joinpath("fitting_models", "plots", "plotting_infrastructure.jl"))
# include(joinpath("fitting_models", "plots", "plot_parameters_single_session.jl"))
# include(joinpath("fitting_models", "plots", "plot_parameters_all_sessions.jl"))
# include(joinpath("fitting_models", "plots", "plot_trajectories_single_session.jl"))
# include(joinpath("fitting_models", "plots", "plot_trajectories_all_sessions.jl"))

### Premade model library ###
include(joinpath("premade_models", "rescorla_wagner.jl"))
include(joinpath("premade_models", "pvl_delta.jl"))

end
