#!/usr/licensed/julia/1.7/bin/julia
"""
Module for state preparation of Bell tree multitree, under independent X/Z noise
uses reduced bond dimension since everything is essentially classical
This version reads in syndrome data from elsewhere
"""
module BellPrepSyndrome

using Base.Threads
using BellTreeStatePrep
using ProbabilityPassingDecoder
using BellTreeFaults
using CodingExtensions # match_syndrome
using SteanePrep # get_system_idxs
using FieldConversions
using Nemo

export process_syndrome, prep_syndrome_decoder, level_bell_syndrome_decoder, expand_checks

export BellPrepParams
struct BellPrepParams
    stacked_params::Array
    final_params::StackedPerfectBell
    level_params::Array{LevelParams}
    anc_flips::Array
    check_times::AbstractArray
    log_state::Int # which logical state is being prepared
    function BellPrepParams(tmax; log_state::Int = 1, measure_first::Int = 1, alternate::Bool = true)
        if alternate
	    check_times = 1:tmax-1
	    @assert measure_first==log_state
	else
	    check_times = 2-(measure_first==log_state):2:tmax-1
	end

        level_params = [LevelParams(t; log_state = log_state, measure_first = log_state) for t=1:tmax-1]
        stacked_params = [StackedBell(t; log_state = log_state, measure_first = log_state, alternate = alternate) for t=filter(el->el<=tmax-2, check_times)]

	final_params = StackedPerfectBell(tmax; log_state = log_state, measure_first = measure_first, alternate = alternate)
	anc_flips = [[[zeros(Bool, lp.counts[i][t]) for t=1:length(lp.counts[i])] for i=1:4] for lp in level_params]
	new(stacked_params, final_params, level_params, anc_flips, check_times, log_state)
    end
end

function prepare_check_idxs(tmax)
    idxs = [expand_checks(tmax; nmin = t) for t=1:tmax]
    # we could reverse engineer this from sys_idxs, but let's not bother
    my_idxs=[[findall(isequal(t), idxs[i]) for i=1:t] for t=1:tmax]
    sys_idxs = vcat([[1]], [get_system_idxs(t) for t=0:tmax])
    idxs, my_idxs, sys_idxs
end

function expand_checks(n; nmin = 1)
    checks = [n]
    for i=n:-1:nmin+1
        checks = vcat(replace(checks, i=>vcat([i],nmin:i-1))...)
    end
    checks
end

function convert_outcomes(shot, t)
    outcomes = vcat([shot[2^t+1+(j-1)*2^(t-1):2^t+j*2^(t-1)] for j=1:t-1], [shot[1:2^t]])
end

function process_syndrome(outcomes, par_checks, idxs, check_times)
    # Here's where I will read in the measurement outcomes, compute the syndromes
    tmax = length(outcomes)
    syndromes = [[gfp_mat[] for i=1:t-1] for t=2:tmax]
    for t=check_times
        sys_idxs = get_system_idxs(t)
        for j=1:2^(tmax-t-1)
	    push!(syndromes[idxs[t][j]][t], par_checks[t] * bool_to_nemo((outcomes[t][((j-1) * 2^t + 1):j*2^t])[sys_idxs,:]))
	end
    end
    last_syndrome = par_checks[end] * bool_to_nemo(outcomes[end][get_system_idxs(tmax),:])
    push!(syndromes[end], [last_syndrome[1:end-1,:]])
    syndromes, (last_syndrome[end,end]==1)
end

function prep_syndrome_decoder(counts, probs; erasure_f = 0)
    @assert erasure_f==0 # haven't implemented erasure_f > 0 yet
    err_probs, _ = initialize_level_errors(counts, probs; erasure_f = erasure_f, func = initialize_error_probs)
    err_probs
end

# for use in prep_bitflips
function generate_flips(err_probs)
    flips = zeros(Bool, length(err_probs))
    for i=1:length(err_probs)
        if err_probs[i][1]<1 && rand() < err_probs[i][2]
	    flips[i] = true
	end
    end
    flips
end

function prep_bitflips(err_probs)
    bitflips = [[[Vector{Vector{Bool}}([generate_flips(err_probs[i][j][k][m]) for m=1:length(err_probs[i][j][k])]) for k=1:length(err_probs[i][j])] for j=1:length(err_probs[i])] for i=1:length(err_probs)]
end

function update_leaked_level_probs!(error_probs, t; log_state::Int = 1)
    block_lengths = vcat([0], cumsum(length.(error_probs[2:end])))
    @threads for block_i=2:length(error_probs)
        @threads for i=1:length(error_probs[block_i])
	    b_i = block_lengths[block_i-1] + i
	    update_leaked_probs!(error_probs[block_i][i][3][t], error_probs[1][b_i]; log_state = log_state, parity = log_state)
	end
    end
end

function get_index(arr; i::Int = 1)
    if isempty(arr)
        return []
    else
        return arr[i]
    end
end

function level_bell_syndrome_decoder(bell_params, error_probs, syndromes)

    final_params = bell_params.final_params
    tmax = length(final_params.par_checks)
    
    err_s, log_bit = match_spacetime_syndrome([get_index(syndrome) for syndrome in syndromes[end]], final_params.par_checks, final_params.level_params.counts; logical = true)

    # now use the syndrome info from earlier levels
    # don't need to apply update in level tmax-1, because there's only one system
    for t=1:tmax-2
        check_t = BellPrepSyndrome.bayesian_update_level!(bell_params, error_probs, syndromes[t])
	@assert check_t>=t-1
	if check_t==t-1 # the last layer where these ancillas could have been updated, they (and all deeper blocks) instead got leaked errors
	    update_leaked_level_probs!(error_probs[t:end], t; log_state = bell_params.log_state)
	end
    	# clear away ancillae that have been used now
    	for tt=1:t
            error_probs[tt] = error_probs[tt][length(error_probs[t+1]) + 1:end]
        end
    end

    # check I updated things properly for order 1 stacking
    for t=1:tmax
        @assert length(error_probs[t])==1
    end
    
    # last level: leave logical leg open
    final_probs = last_level_bell_decode(final_params, bell_params.level_params, [error_probs[i][end] for i=1:tmax], vcat(bell_params.anc_flips, [err_s]))
    final_probs, log_bit
end

# BellTreeStatePrep
function track_spacetime_syndrome(bitflips, paulis, anc_flips, par_checks, check_times)
    tmax = length(bitflips[1])

    syndromes = Array{gfp_mat}(undef, length(check_times))
    for (t_i, t)=enumerate(check_times)
        myerr = propagate_tree_errors_classical(bitflips, paulis[t], t; on_right = false)
        syndromes[t_i] = par_checks[t] * bool_to_nemo(Bool.(mod.(myerr .+ anc_flips[t_i],2))[:,:])
    end

    # now propagate to end. and return with and without
    # now tack on gate + measurement error
    myerr = propagate_tree_errors_classical(bitflips, paulis[tmax], tmax; on_right = false)
    if length(bitflips)>3
        return myerr, xor.(myerr, bitflips[4][1]), syndromes
    else
        return myerr, myerr, syndromes
    end
end

function get_spacetime_syndrome(bitflips, paulis, par_checks, check_times)
    tmax = length(par_checks)
    syndromes = [[gfp_mat[] for i=1:t-1] for t=2:tmax]
    prop_errs = [[] for t=1:tmax]
    for t=1:tmax
        use_times = filter(ti->ti<t, check_times)
	prop_errs[t] = [[] for i=1:length(bitflips[t])]
        # get the syndromes for this level
	for i=1:length(bitflips[t])
	    # get the leaked errors
	    for tt=filter(ti->!(ti in use_times), 1:t-1)
	        bitflips[t][i][3][tt] .⊻= prop_errs[tt][i]
	    end
	    # get the syndromes
	    myerr, post_meas_err, my_syndromes = track_spacetime_syndrome(bitflips[t][i], paulis, [prop_errs[tt][i] for tt=use_times], par_checks, use_times)
	    if t>1
	        for (t_i,tt)=enumerate(use_times)
	            push!(syndromes[t-1][tt], my_syndromes[t_i])
	    	end
	    end
	    if t in check_times # will be used in subsequent syndromes
	        prop_errs[t][i] = post_meas_err
	    else # will leak, before, the measurement error
	        prop_errs[t][i] = myerr
	    end
	end
	# now shrink prop_errs
	for tt=1:t-1
	    prop_errs[tt] = prop_errs[tt][length(bitflips[t])+1:end]
	end
    end

    # last layer syndrome
    last_syndrome = par_checks[end] * bool_to_nemo(prop_errs[end][1][:,:])
    push!(syndromes[end], [last_syndrome[1:end-1,:]])
    syndromes, (last_syndrome[end,end]==1)
end

function bayesian_update_level!(bell_params::BellPrepParams, error_probs, syndromes)
    t = length(syndromes)
    # only update the stuff that will become an ancilla in the next block
    new_anc = length(error_probs[t+1])
    idx = searchsortedlast(bell_params.check_times, t)
    if idx==0 # nothing learned from this guy
        return 0
    end
    check_t = bell_params.check_times[idx]
    @threads for i=1:new_anc
        bitflips, _ = match_spacetime_syndrome([get_index(syndrome; i= i) for syndrome in syndromes], bell_params.final_params.par_checks[1:t], bell_params.level_params[t].counts; logical = false)
        BellTreeStatePrep.bayesian_update_pair!(bell_params.stacked_params[idx], bell_params.level_params, [error_probs[tt][i] for tt=1:t+1], vcat(bell_params.anc_flips[1:t],[bitflips]), check_t)
    end
    check_t
end

function match_spacetime_syndrome(syndrome, par_checks, counts; logical::Bool = true)

    # now match the spacetime syndrome using errors only on system
    matching_errs = Vector{Vector{Bool}}(undef, length(par_checks))
    
    prop_err = zeros(Bool, 2)
    for t=1:length(par_checks)-1
        if isempty(syndrome[t]) # no syndrome checked here, so guess there's no error
	    matching_errs[t] = zeros(Bool, size(par_checks[t],2))
	else
	    matching_errs[t] = mod.(nemo_to_bool(match_syndrome(par_checks[t], syndrome[t]))[:,1] .+ prop_err,2)
	end
	# propagate through the error to this point, to next layer
	prop_err = propagate_tree_classical_layer([zeros(Bool, counts[1][t+1]), zeros(Bool, counts[2][t+1])], prop_err .+ matching_errs[t])
    end

    if logical # DON'T match final logical
    	matching_errs[end] = mod.(nemo_to_bool(match_syndrome(par_checks[end][1:end-1,:], syndrome[end]))[:,1] .+ prop_err,2)
    elseif isempty(syndrome[end])
        matching_errs[end] = zeros(Bool, size(par_checks[end],2))
    else
    	matching_errs[end] = mod.(nemo_to_bool(match_syndrome(par_checks[end], syndrome[end]))[:,1] .+ prop_err,2)
    end
   
    # now also see whether they have the same effect on logical or not
    matched_err = [[zeros(Bool, counts[1][t]) for t=1:length(par_checks)], matching_errs, [zeros(Bool, counts[3][t]) for t=1:length(par_checks)-1]]
    if isempty(syndrome[end])
        return matched_err, false
    end
    
    last_syndrome = par_checks[end] * bool_to_nemo(propagate_tree_errors_classical(matched_err)[:,:])

    # just a sanity check
    if logical
        @assert last_syndrome[1:end-1,:]==syndrome[end]
	return matched_err, (last_syndrome[end,1] == 1)
    else
        @assert last_syndrome==syndrome[end]
    	return matched_err, false
    end
end

# Only gate and measurement errors get heralded
function initialize_error_prob_stacks(error_probs, erasures, sub_idxs, idxs)
    tmax = length(error_probs)
    for t=1:tmax
        println(t)
        for i=1:t	    
	    for j=1:length(idxs[t][1])
	        for qb_i=1:2^i
		    if erasures[1][i][(idxs[t][i][j]-1)*2^i+qb_i]
		        error_probs[t][sub_idxs[end-t-1][end-j+1]][2][i][sub_idxs[i+2][qb_i]] = [0.5, 0.5]
		    end
		    
		    if i<t && erasures[2][i][(idxs[t][i+1][j]-1)*2^i+qb_i]
		        error_probs[t][sub_idxs[end-t-1][end-j+1]][3][i][sub_idxs[i+3][2*qb_i-1]] = [0.5,0.5] # also post-check-gate errors
		    end
		    
		end
	    end
	end
	if t==tmax
	    err_i = 2
	else
	    err_i = 4
	end
	for j=1:length(sub_idxs[end-t-1])
	    for qb_i=1:2^t
	        if erasures[3][t][qb_i]
		    error_probs[t][sub_idxs[end-t-1][end-j+1]][err_i][end][sub_idxs[t+2][qb_i]] = [0.5, 0.5] # measurement errors
		end
	    end
	end
    end
    error_probs 
end

function initialize_error_probs(counts, probs; erasure_f = 0, measurement_error::Bool = true)
    @assert erasure_f==0
    # state prep errors
    error_probs = [[Array{Array}(undef, counts[i][t]) for t=1:length(counts[i])] for i=1:3]

    for i=1:3
    	for t=1:length(counts[i])
	    for j=1:counts[i][t]
	        error_probs[i][t][j] = [1-probs[i], probs[i]]
	    end
	end
    end

    # if this is an ancilla block, it will also have measurement errors
    if length(counts)>3
	pp = probs[3] + probs[4] - 2 * probs[3]*probs[4]

	measure_stuff = [[1-pp,pp] for i=1:counts[end][1]]
	error_probs = vcat(error_probs, [[measure_stuff]])
    elseif measurement_error # SPECIAL CASE: measurements at end anyway, so concatenate still
       	pp = probs[2] + probs[4] - 2*probs[2]*probs[4]
	error_probs[2][end] = [[1-pp,pp] for i=1:counts[2][end]]
    end
    return error_probs, []
end

end # module