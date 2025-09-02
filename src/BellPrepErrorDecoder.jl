#!/usr/licensed/julia/1.11/bin/julia
"""
Module for decoding state preparation of GS multitree, using the Hadamarded representation
with a local independent X/Z noise model.
Samples errors on the fly.
"""
module BellPrepErrorDecoder

# external code
using Base.Threads
using Nemo

# my code
using BellTreeTensors
using BellTreeFaults
using BinaryFields # bool_to_nemo, match_syndrome
using SteanePrep # get_system_idxs

"""
Match the spacetime syndrome with errors on just the system
"""
function match_spacetime_syndrome(bitflips, par_checks)

    # all ancilla blocks - just get their final error
    ancilla_errs = propagate_tree_errors_classical.(bitflips[1:end-1])

    # now match the spacetime syndrome using errors only on system
    matching_errs = Array{Array}(undef, length(bitflips[end][1]))
    
    myerr = propagate_tree_classical_layer([bitflips[end][1][1][2:end], bitflips[end][2][1]], bitflips[end][1][1][1:1])
    prop_err = zeros(Bool, length(myerr))
    for t=2:length(bitflips[end][1])
        fresh_syndrome = par_checks[t-1] * bool_to_nemo(Bool.(mod.(myerr .+ prop_err .+ ancilla_errs[t-1],2))[:,:])
	matching_errs[t-1] = nemo_to_bool(match_syndrome(par_checks[t-1], fresh_syndrome))[:,1]
	# next step:
	myerr = propagate_tree_classical_layer([bitflips[end][1][t], bitflips[end][2][t]], myerr .+ bitflips[end][3][t-1])
	# propagate through the error to this point, to next layer
	prop_err = propagate_tree_classical_layer([zeros(Bool, length(bitflips[end][1][t])), zeros(Bool, length(bitflips[end][2][t]))], prop_err .+ matching_errs[t-1])
    end

    # last level: no ancillas, don't measure logical
    fresh_syndrome = par_checks[end][1:end-1,:] * bool_to_nemo(Bool.(mod.(myerr .+ prop_err, 2))[:,:])
    matching_errs[end] = nemo_to_bool(match_syndrome(par_checks[end][1:end-1,:], fresh_syndrome))[:,1]

    # now also see whether they have the same effect on logical or not
    matched_err = [[zeros(Bool, length(bitflips[end][1][t])) for t=1:length(bitflips[end][1])], matching_errs, [zeros(Bool, length(bitflips[end][3][t])) for t=1:length(bitflips[end][3])]]
    last_syndrome = par_checks[end] * bool_to_nemo(Bool.(mod.(myerr .+ propagate_tree_errors_classical(matched_err),2))[:,:])
    # just a sanity check
    @assert all(iszero, last_syndrome[1:end-1,1])
    matched_err, (last_syndrome[end,1]==1)
end

"""
Update error model on all eventual ancillas, version where we only stack at the end
"""
function bayesian_update_level!(level_params::LevelParams, error_probs, bitflips, t)
    block_lengths = vcat([0], cumsum(length.(bitflips[2:end])))

    # don't need to update the ultimate "system"
    @threads for block_i=2:length(bitflips)-1
        @threads for i=1:length(bitflips[block_i])
	    bayesian_update_pair!(level_params, [error_probs[block_i][i],error_probs[1][block_lengths[block_i-1]+i]], [bitflips[block_i][i], bitflips[1][block_lengths[block_i-1]+i]], t)
	end
    end
end

"""
Update error model on eventual ancillas, stacked version
"""
function bayesian_update_level!(stacked_params::StackedBell, level_params::Array, error_probs, bitflips, t)
    # only update the stuff that will become an ancilla in the next block
    new_anc = length(bitflips[t+1])
    @threads for i=1:new_anc
        bayesian_update_pair!(stacked_params, level_params, [error_probs[tt][i] for tt=1:t+1], [bitflips[tt][i] for tt=1:t+1], t)
    end

    # clear away ancillae that have been used now
    for tt=1:t
        error_probs[tt] = error_probs[tt][new_anc + 1:end]
	bitflips[tt] = bitflips[tt][new_anc + 1:end]
    end
end

#= Master decoding function =#
export run_level_bell_decoder

"""
Run stacked probability passing decoder, sampling errors and decoding according to same error model
    - stacked_params is either empty or an instance of StackedBell
    - final_params::StackedPerfectBell : stacked params for last level decoding
    - level_params::Array{LevelParams} : for each eventual ancilla
    - probs: [p_input, p_encoding, p_gate, p_meas]
Returns
    - Weights of two logical classes relative to canonical error, computed before marginal updates and after
    - Logical syndrome bit for canonical error
"""
function level_bell_decoder(stacked_params::Array, final_params::StackedPerfectBell, level_params::Array{LevelParams}, probs; err_probs = nothing, bitflips = nothing, erasure_f = 0, save_start::Bool=false, measurement_error::Bool = false)
    tmax = length(final_params.par_checks)
    if isnothing(err_probs)
        err_probs, bitflips = initialize_level_errors(vcat([lp.counts for lp in level_params],[final_params.level_params.counts]), probs; erasure_f = erasure_f, measurement_error = measurement_error, func = initialize_error_nodes)
    end

    err_s, comm_log = match_spacetime_syndrome([bitflips[i][end] for i=1:tmax], final_params.par_checks)

    # first see what would happen if I tried decoding last level pretending errors only on system (i.e. not updating the error model at all)
    if save_start
        class_probs = last_level_bell_decode(final_params, level_params, [err_probs[i][end] for i=1:tmax], vcat([bitflips[i][end] for i=1:tmax-1], [err_s]))
    else
        class_probs = zeros(2)
    end
    # now use the syndrome info from earlier levels
    # don't need to apply update in level tmax-1, because there's only one system
    for t=1:tmax-2
        if isempty(stacked_params) # only stack at the end
	    bayesian_update_level!(level_params[t], err_probs[t:end], bitflips[t:end], t)
	else
            bayesian_update_level!(stacked_params[t], level_params, err_probs, bitflips, t)
	end
    end

    # check I updated things properly for order 1 stacking
    if !isempty(stacked_params)
        for t=1:tmax
            @assert length(err_probs[t])==1
        end
    end
    
    # last level: leave logical leg open
    final_probs = last_level_bell_decode(final_params, level_params, [err_probs[i][end] for i=1:tmax], vcat([bitflips[i][end] for i=1:tmax-1], [err_s]))
    vcat(class_probs, final_probs), comm_log
end

function level_bell_decoder_rates!(tmax, rates, class_probs, classes; erasure_f=0, log_state::Int=1, print::Bool = true, save_start::Bool = false, stack::Bool = true, measurement_error::Bool = false)
    level_params = [LevelParams(t; log_state = log_state, measure_first = log_state) for t=1:tmax-1]
    if stack
        stacked_params = [StackedBell(t; measure_first = log_state, alternate = true, log_state = log_state) for t=1:tmax-2]
    else
        stacked_params = []
    end
    final_params = StackedPerfectBell(tmax; measure_first = log_state, log_state = log_state)
    for rate_i=1:length(rates)
        print && println(rates[rate_i]); flush(stdout)
        @threads for i=1:size(class_probs, 2)
	    if i%50==1 && print
	        println(i); flush(stdout)
	    end
	    class_probs[:,i,rate_i], classes[i,rate_i] = level_bell_decoder(stacked_params, final_params, level_params, rates[rate_i]; erasure_f = erasure_f, save_start = save_start, measurement_error = measurement_error)
	end
    end
end

"""
Master function for running stack probability passing decoder
sampling errors at given rates
Each element of rates is a four-component vector
"""
function run_level_bell_decoder(tmax, rates; num_samples::Int=100, log_state::Int=1, erasure_f = 0, decode_data = Dict{String,Any}("start"=>Dict(), "final"=>Dict(), "class"=>Dict()), save_start::Bool = false, stack::Bool = true, print::Bool = true, measurement_error::Bool = false)
    # log state is 1 (X) or 2 (Z)
    if haskey(decode_data, "log-basis")
        @assert decode_data["log-basis"]==log_state
	@assert decode_data["heralded"]==erasure_f
	@assert decode_data["stacked"]==stack
	@assert decode_data["measure-last"]==measurement_error
    else
        decode_data["log-basis"] = log_state
	decode_data["heralded"] = erasure_f
	decode_data["stacked"] = stack
	decode_data["measure-last"]=measurement_error
    end
    class_probs = zeros(4, num_samples, length(rates))
    classes = zeros(Bool, num_samples, length(rates))
    level_bell_decoder_rates!(tmax, rates, class_probs, classes; log_state = log_state, erasure_f = erasure_f, stack = stack, save_start = save_start, print=print, measurement_error = measurement_error)

    # Turn into dictionary, concatenate as needed
    for (rate_i, rate) in enumerate(rates)
        if haskey(decode_data["final"], rate) # append to existing data
	    decode_data["final"][rate] = hcat(decode_data["final"][rate], class_probs[3:4,:,rate_i])
	    if save_start
	        decode_data["start"][rate] = hcat(decode_data["start"][rate], class_probs[1:2,:,rate_i])
	    end
            decode_data["class"][rate] = vcat(decode_data["class"][rate], classes[:,rate_i])
	else
	    decode_data["final"][rate] = class_probs[3:4,:,rate_i]
	    if save_start
	        decode_data["start"][rate] = class_probs[1:2,:,rate_i]
	    end
	    decode_data["class"][rate] = classes[:,rate_i]
	end
    end
    decode_data
end



end # module