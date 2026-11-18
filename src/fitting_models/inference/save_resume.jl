function prepare_sampler(
    sampler::AbstractMCMC.AbstractSampler,
    chains::Chains,
)
    @error "Save and continue for sampler type $(typeof(sampler)) not implemented"
end

# prepare sampler for nuts
function prepare_sampler(sampler::NUTS, chains::Chains)
    # get the last step size
    sampler_ϵ = chains[:step_size][end]
    # create a new sampler with the last state
    # n_adapts can be zero because we load the warmed-up sampler state
    return NUTS(
        0,
        sampler.δ;
        Δ_max = sampler.Δ_max,
        adtype = sampler.adtype,
        init_ϵ = sampler_ϵ,
        max_depth = sampler.max_depth,
    )
end

function validate_saved_sampling_state!(
    save_resume::SampleSaveResume,
    n_segments::Int,
    n_chains::Int,
)
    # check if the path exists
    if !isdir(save_resume.path)
        @warn "Path $(save_resume.path) does not exist, creating it"
        mkdir(save_resume.path)
    end
    # check if the path is a directory
    if !isdir(save_resume.path)
        @error "Path $(save_resume.path) is not a directory"
    end
    # check if the path is writable
    if !(uperm(save_resume.path) & 0x02 == 0x02)
        @error "Path $(save_resume.path) is not writable"
    end


    # find the last segment (for each chain)
    last_segment = Int[]
    for chain = 1:n_chains
        last_seg = 0
        n_segs = 0
        for cur_seg = 1:n_segments
            if isfile(
                joinpath(
                    save_resume.path,
                    "$(save_resume.chain_prefix)_c$(chain)_s$(cur_seg).jld2",
                ),
            )
                last_seg = cur_seg
                n_segs += 1
            end
        end
        if n_segs < last_seg
            @error "Chain $chain has missing segments, check the path $(save_resume.path)"
        end
        push!(last_segment, last_seg)
    end

    return last_segment
end

function load_segment(save_resume::SampleSaveResume, chain_n::Int, segment::Int)
    # load the chain
    chain = JLD2.load_object(joinpath(save_resume.path, "$(save_resume.chain_prefix)_c$(chain_n)_s$(segment).jld2"))
    # extra validation?
    return chain
end

function save_segment(seg::Chains, save_resume::SampleSaveResume, chain_n::Int, seg_n::Int)
    # save the chain
    # JLD2.save_object(joinpath(save_resume.path, "$(save_resume.chain_prefix)_c$(chain_n)_s$(seg_n).jld2"), seg)
    @suppress jldsave(joinpath(save_resume.path, "$(save_resume.chain_prefix)_c$(chain_n)_s$(seg_n).jld2"); seg)
end

function combine_segments(save_resume::SampleSaveResume, n_segments::Int, n_chains::Int)
    chains::Vector{Union{Nothing,Chains}} = fill(nothing, n_chains)
    for chain = 1:n_chains
        segments::Vector{Union{Nothing,Chains}} = fill(nothing, n_segments)
        seg_start = 1
        for segment = 1:n_segments
            seg = load_segment(save_resume, chain, segment)
            # update the range
            seg_end = seg_start + length(seg) - 1
            seg = setrange(seg, seg_start:seg_end)
            seg_start = seg_end + 1
            segments[segment] = seg
        end
        chains[chain] = cat(segments..., dims = 1)
    end

    return chainscat(chains...)
end



function sample_save_resume(
    model::DynamicPPL.Model,
    save_resume::SampleSaveResume,
    n_samples::Integer,
    n_chains::Integer,
    parallelization::AbstractMCMC.AbstractMCMCEnsemble,
    sampler::AbstractMCMC.AbstractSampler,
    init_params::Union{Nothing,DynamicPPL.AbstractInitStrategy} = nothing,
    sampler_kwargs...,
)

    if save_resume.save_every > n_samples
        throw(
            ArgumentError(
                "save_every ($save_resume.save_every) must be less than n_samples ($n_samples)",
            ),
        )
    end

    final_segment = n_samples % save_resume.save_every
    n_segments = Int(floor(n_samples / save_resume.save_every)) + Int(final_segment > 0)

    resume_from::Vector{Union{Nothing,Chains}} = fill(nothing, n_chains)
    samplers = fill(sampler, n_chains)
    last_complete_segment =
        validate_saved_sampling_state!(save_resume, n_segments, n_chains)

    # this outer loop can be parallelized
    for chain = 1:n_chains
        if last_complete_segment[chain] > 0
            resume_from[chain] =
                load_segment(save_resume, chain, last_complete_segment[chain])
            samplers[chain] = prepare_sampler(sampler, resume_from[chain])
        end

        # the inner loop must run sequentially
        for cur_seg = (last_complete_segment[chain]+1):n_segments
            # use the save_every value unless there are some iterations left over
            n_iter =
                final_segment > 0 && cur_seg == n_segments ? final_segment :
                save_resume.save_every
            # Run the sampler
            #A chain that was saved before is resumed from its last state, and a new one starts at the initial parameters
            start = if isnothing(resume_from[chain])
                initial_params_keyword(init_params)
            else
                (; initial_state = loadstate(resume_from[chain]))
            end
            seg = sample(
                model,
                samplers[chain],
                n_iter;
                progress = false,
                save_state = true,
                chain_type = Chains,
                start...,
                sampler_kwargs...,
            )
            # Save the chain
            save_segment(seg, save_resume, chain, cur_seg)
            # Update the resume_from and sampler
            samplers[chain] = prepare_sampler(sampler, seg)
            resume_from[chain] = seg

        end
    end
    chains = combine_segments(save_resume, n_segments, n_chains)

    return chains
end