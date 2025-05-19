module ActionModels

#Load packages
using Reexport 
@reexport using Distributions #Make distributions available to the user
@reexport using Turing #For fitting models
using Turing: DynamicPPL, AbstractMCMC, LogDensityProblems

using RecipesBase #For defining plots
using DataFrames #For the input format to create_model
using AxisArrays #For storing session parameters and state trajectories
using StatsModels, MixedModels, LogExpFunctions #For the GLM population Model

using MCMCChainsStorage, HDF5 #For the save_resume functionality
using MCMCChainsStorage: MCMCChains
using ProgressLogging #For progress bars


#ADType functionality
using ADTypes: AutoForwardDiff#, AutoReverseDiff, AutoMooncake, AutoEnzyme, AutoFiniteDifferences
import ForwardDiff
#import ReverseDiff
#import Mooncake
#import FiniteDifferences: central_fdm
#import Enzyme: set_runtime_activity, Forward, Reverse #AD types to be compatible with

## For defining action models ##
export ActionModel, Parameter, InitialStateParameter, State, Observation, Action
export ModelAttributes, load_parameters, load_states, load_actions, update_state!, RejectParameters

## For simulation ##
export init_agent
export simulate!, observe!
export get_parameters, get_states, get_actions, get_history
export set_parameters!, set_states!, set_actions!, reset!
#export plot_trajectory, plot_trajectory!

## For fitting models ##
export create_model, RegressionPrior
export sample_prior!, sample_posterior!, SampleSaveResume
export get_session_parameters!, get_state_trajectories!, summarize
# export parameter_recovery
# export plot_parameters, plot_trajectories
export @formula



## Constants for creating ids and names consistently ##
const id_separator = "."
const id_column_separator = ":"
const tuple_separator = "."


### Types ###
include(joinpath("defining_models", "structs.jl"))
include(joinpath("simulation", "structs.jl"))
include(joinpath("fitting_models", "structs.jl"))


### Functions for model definition ###
include(joinpath("defining_models", "prints.jl"))
include(joinpath("defining_models", "model_attributes.jl"))
include(joinpath("defining_models", "manipulate_attributes.jl"))
include(joinpath("defining_models", "no_submodel_dispatches.jl"))
#Read in all premade models
# for premade_model_file in readdir(joinpath("src", "defining_models", "premade_models"))
#     if endswith(premade_model_file, ".jl")
#         include(joinpath("defining_models", "premade_models", premade_model_file))
#     end
# end
include(joinpath("defining_models", "premade_models", "rescorla_wagner.jl"))


### Functions for simulation ###
include(joinpath("simulation", "prints.jl"))
include(joinpath("simulation", "simulate.jl"))
include(joinpath("simulation", "manipulate_attributes.jl"))
include(joinpath("simulation", "plots", "plot_trajectory.jl"))

### Functions for fitting models ###
include(joinpath("fitting_models", "prints.jl"))
include(joinpath("fitting_models", "turing_model", "create_model.jl"))
include(joinpath("fitting_models", "turing_model", "create_session_model.jl"))
include(joinpath("fitting_models", "turing_model", "helper_functions.jl"))
include(joinpath("fitting_models", "turing_model", "population_models", "independent_population_model.jl"))
include(joinpath("fitting_models", "turing_model", "population_models", "glm_population_model.jl"))
include(joinpath("fitting_models", "turing_model", "population_models", "single_session_population_model.jl"))
include(joinpath("fitting_models", "inference", "sampling.jl"))
include(joinpath("fitting_models", "inference", "save_resume.jl"))
include(joinpath("fitting_models", "inference", "get_session_parameters!.jl"))
include(joinpath("fitting_models", "inference", "get_state_trajectories.jl"))
include(joinpath("fitting_models", "inference", "summarize.jl"))
include(joinpath("fitting_models", "tools", "parameter_recovery.jl"))
include(joinpath("fitting_models", "plots", "plotting_infrastructure.jl"))
include(joinpath("fitting_models", "plots", "plot_parameters_single_session.jl"))
include(joinpath("fitting_models", "plots", "plot_parameters_all_sessions.jl"))
include(joinpath("fitting_models", "plots", "plot_trajectories_single_session.jl"))
include(joinpath("fitting_models", "plots", "plot_trajectories_all_sessions.jl"))

end
