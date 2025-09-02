#!/usr/licensed/julia/1.11/bin/julia
"""
Module for decoding Bell tree / GS code, Steane error correction, 
under independent X/Z noise.
Probability passing decoder, with stacked or unstacked marginal updates.
"""
module BellMemoryDecoder

using QuantumClifford
using Base.Threads # for @threads macro

using TensorExtensions
using Nemo # for FqMatrix
using BinaryFields # bool_to_nemo, match_syndrome
using PauliNoise
using StackedTree
using BellTreeFaults
using BellTreeTensors

#= Data structures =#
export BellPassingParams, BellDecoderParams, StackedSyndromeBell
"""
Keeps all the immutable data associated with the level decoder
which doesn't depend on noise rate
"""
struct BellPassingParams
    level_params::Array{LevelParams} # idxs, orders for contraction at each level
    parities::Array # which tree to use in each level
    error_basis::Array # error basis for propagated errors
    log_state::Int # which logical state is being prepared

    function BellPassingParams(tmax; log_state::Int=1, measure_first::Int=1, alternate::Bool=false)
        
	# on levels where measure "1", we're catching Z errors from odd layers, X errors from even layers
	if alternate
	    parities = fill(measure_first, tmax)
	else
	    parities = [(measure_first + t)%2+1 for t=1:tmax]
	end
	
    	level_params = [LevelParams(t; log_state = log_state, measure_first = i) for i=1:2, t=1:tmax]

	# last two are for the "verification layers"
	error_basis = vcat([prepare_error_basis(t; basis_i = parities[t]%2+1) for t=1:tmax-1],[prepare_error_basis(tmax; basis_i = parities[end]%2+i) for i=1:2])
	new(level_params, parities, error_basis, log_state)
    end	
end

"""
Using stacked probability passing on the ancillas turns out not to give any benefit in the quantum memory test, but is included here for completeness
"""
struct StackedSyndromeBell
    level_params::LevelParams # idxs, gate nodes, and orders for contraction
    stacked_nodes::NamedTuple # stacked nodes
    times::AbstractArray # times where we connect to ancillas/collect syndrome info
    function StackedSyndromeBell(tmax; log_state::Int=1, measure_first::Int = 1, start_anc::Int = 1)
        times = start_anc:2:tmax-1
		
        noisy_list = make_stacked_syndrome_index_list(tmax, times; log_state = log_state, measure_first = measure_first)
	idxs = vcat(noisy_list.sys_idxs, vcat(noisy_list.state_prep_idxs...)..., noisy_list.anc_idxs, noisy_list.top_idxs)
	stacked_nodes = initialize_stacked_nodes(vcat(times,tmax); log_state = log_state, open_logical = false, dim = 2, measure_first = measure_first)
	gate_nodes = make_stacked_gate_nodes(vcat(times,tmax); measure_first = measure_first, log_state = log_state, perfect_layer = 1)
	
	level_params = LevelParams(idxs, noisy_list.order, [], gate_nodes)
	new(level_params, stacked_nodes, times)
    end
end

"""
Parameters for decoding syndrome extraction
"""
struct BellDecoderParams
    par_checks::Array{FqMatrix}
    level_params::Array
    error_basis::Array
    t::Int
    
    function BellDecoderParams(t::Int; stack::Bool = true, measure_first = [1,1])
    	# first will be used for |0>, second for |+>
        par_checks = [prepare_par_check(t; log_state = i, measure_log = true) for i=1:2]
	error_basis = [prepare_error_basis(t; basis_i = i) for i=1:2]
	index_list = make_index_list(t)
	idxs = vcat(index_list...)
	order = idxs[end][1]:-1:1
	perfect_params = [LevelParams(idxs, order, [], make_perfect_tree(t, i)) for i=1:2]
	if stack
	    level_params = [StackedSyndromeBell(t; log_state = i, measure_first = i%2+1, start_anc = 2-(measure_first[i]!=i)) for i=[2,1]]
	else
	    level_params = [make_unstacked_params(t; log_state = i) for i=[2,1]]
	end
	new(par_checks, vcat(level_params, perfect_params), error_basis, t)
    end
end

function make_perfect_tree(tmax, measure_first)
    nodes = [NOTC_GAUGE_NODE, CNOT_NODE[:,1,:,:]]
    tree_nodes = vcat([fill(nodes[(measure_first+i)%2+1], 2^(i-1)) for i=1:tmax]...)
end

"""
Initialize level_params for ancillas, no stacking
"""
function make_unstacked_params(t; log_state::Int = 1)
    # parity will always be opposite that of log_state
    parity = log_state%2 + 1
    index_list, order = make_bare_index_list(t; log_state = log_state, measure_first = parity)
    gate_nodes = make_gate_nodes(t, [2^i for i=0:t-1]; log_state = log_state, log_type=:gauge, measure_first = parity)

    # probably won't actually use
    counts = vcat([length.(idx_list) for idx_list in index_list[1:3]],[[2^t]])
    @assert counts==get_classical_error_counts(t; log_state = log_state, measure_first = parity, measurement_error = true)
    
    anc_count = index_list[2][end][end][2]
    @assert length(index_list[2][end])==2^t
    top_layer = [[index_list[2][end][i][2], anc_count + i, anc_count + i + 2^t] for i=1:2^t]
    idxs = vcat(top_layer, vcat(vcat(index_list[1:3]...)...),[[anc_count + i + 2^t] for i=1:2^t], vcat(index_list[4]...), [[anc_count + i] for i=1:2^t])
    joined_order = vcat([anc_count + i for i=1:2^(t+1)], [idx[end] for idx in index_list[2][end]], order...)

    LevelParams(idxs, joined_order, counts, gate_nodes)
end

#= functions to prepare ancillas =#
export prepare_ancilla_state, prepare_stacked_ancilla

function verify_ancilla!(bell_params, probs, counts, error_probs, bitflips; n_verify::Int = 2, erasure_f = 0)
    tmax = size(counts, 2)
    for i=1:n_verify
        anc_probs, anc_flips = prepare_ancilla_state(bell_params, stacked_params, probs, counts; erasure_f = erasure_f)
	parity = (bell_params.parities[end] + i)%2 + 1

	# update probabilities using these checks
	bayesian_update_pair!(bell_params.level_params[parity,end], [error_probs[parity], anc_probs[parity]], [bitflips[parity], anc_flips[parity]], tmax)

	# opposite checks leak onto system
	update_leaked_pair!(error_probs[parity%2+1][2][end], bitflips[parity%2+1][2][end], anc_probs[parity%2+1], anc_flips[parity%2+1], bell_params.error_basis[end-2+i]; parity = parity%2+1, log_state = bell_params.log_state)

	# fresh errors associated with the check gates (concatenate with last layer of "encoding gates")
	for j=1:2
	    apply_fresh_errors!(error_probs[j][2][end], bitflips[j][2][end], probs[3]; erasure_f = erasure_f)
	end
    end
end

"""
Prepare ancilla state used to perform X or Z checks
Update marginal error probabilities as it is prepared
"""
function prepare_ancilla_state(bell_params, probs, counts; err_stuff = nothing, erasure_f = 0)
    tmax = size(counts,2)
    if isnothing(err_stuff)
        err_stuff = [initialize_level_errors(counts[i,:], probs, func = initialize_error_nodes, erasure_f = erasure_f) for i=1:2]
    end

    # now do bare probability passing on the ancillas used to check this ancilla (and leak errors)
    for t=1:tmax-1
        update_alternating_level!(bell_params, [err_stuff[i][1][t:end] for i=1:2], [err_stuff[i][2][t:end] for i=1:2], t; update_final = true)
    end

    # all I care about are the errors and marginal probabilities on the final state
    return [err_stuff[i][1][end][1] for i=1:2], [err_stuff[i][2][end][1] for i=1:2]
end

"""
Prepare ancilla state using stacked probability passing to update its marginals
This doesn't actually lead to gains
"""
function prepare_stacked_ancilla(bell_params, probs, counts; err_stuff = nothing, erasure_f = 0)
    tmax = size(counts,2)
    if isnothing(err_stuff)
        err_stuff = [initialize_level_errors(counts[i,:], probs, func = initialize_error_nodes, erasure_f = erasure_f) for i=1:2]
    end

    # now do bare probability passing on the ancillas (and leak)
    for t=1:tmax-1
        # DO update the errors that will leak
	# DON'T update the stuff errors in the next check
        update_alternating_level!(bell_params, [err_stuff[i][1][t:end] for i=1:2], [err_stuff[i][2][t:end] for i=1:2], t; update_final = (bell_params.parities[t]==bell_params.log_state))
    end

    # only need the errors that feed into final state
    return [[err_stuff[i][1][j][end] for j=1:tmax] for i=1:2], [[err_stuff[i][2][j][end] for j=1:tmax] for i=1:2]
end

"""
update marginal probabilities on ancilla and leak errors of non-matching type
"""
function update_alternating_level!(noiseless_params, error_probs, bitflips, t; update_final::Bool = false)
    parity = noiseless_params.parities[t]

    block_lengths = vcat([0], cumsum(length.(bitflips[parity][2:end])))

    if update_final
        max_block = length(bitflips[parity])
    else # don't need to update probabilities on the ultimate "system"
        max_block = length(bitflips[parity]) - 1
    end
    @threads for block_i=2:max_block
        @threads for i=1:length(bitflips[parity][block_i])
	    b_i = block_lengths[block_i-1] + i
	    # update probabilities of matching type
	    bayesian_update_pair!(noiseless_params.level_params[parity,t],
	        [error_probs[parity][block_i][i],error_probs[parity][1][b_i]],
		[bitflips[parity][block_i][i], bitflips[parity][1][b_i]], t)
	    update_leaked_pair!(error_probs[parity%2+1][block_i][i][3][t], bitflips[parity%2+1][block_i][i][3][t],
	        error_probs[parity%2+1][1][b_i], bitflips[parity%2+1][1][b_i],
		noiseless_params.error_basis[t]; parity = parity%2+1, log_state = noiseless_params.log_state)
	end
    end

    if !update_final
        # on ultimate system, I *DO* need to update leaked errors
    	update_leaked_pair!(error_probs[parity%2+1][end][1][3][t],
            bitflips[parity%2+1][end][1][3][t],error_probs[parity%2+1][1][end],
	    bitflips[parity%2+1][1][end], noiseless_params.error_basis[t];
	    parity = parity%2+1, log_state = noiseless_params.log_state)
    end
end

function update_leaked_pair!(post_probs, post_flips, error_probs, bitflips, error_basis; parity::Int = 1, log_state::Int = 1)
    update_leaked_probs!(post_probs, error_probs; parity = parity, log_state = log_state)
    # propagate ancilla errors up to the CNOT/NOTC gates
    ancilla_err = propagate_tree_errors_classical(bitflips, error_basis; on_right = (log_state==1 && parity==2))
    @assert length(ancilla_err)==length(post_flips)
    post_flips .⊻= ancilla_err
    validate_probs(post_probs, post_flips)
end

function validate_probs(probs, bitflips)
    for i=1:length(bitflips)
        if bitflips[i]
	    @assert !isequal([1,0], probs[i])
	else
	    @assert !isequal([0,1], probs[i])
	end
    end
end

#= single rounds of syndrome extraction =#

"""
Single round of syndrome extraction
    - parity: whether we're using parity 1 or parity 2 tree.
    parity 1 tree used for logical 0 state, 2 tree for logical + state
"""
function stacked_passing_round!(decoding_params, state_prep_params, sys_probs, sys_flips, anc_probs, anc_flips; parity::Int = 1, first_round::Bool = false)

    # leak errors from ancilla onto system
    update_leaked_pair!(sys_probs[parity%2+1], sys_flips[parity%2+1], anc_probs[parity%2+1][end], anc_flips[parity%2+1][end], decoding_params.error_basis[parity%2+1]; parity = parity%2+1, log_state = parity%2+1)
    
    if first_round # if this is the first round of syndrome extraction, I already know nothing will be on the system
        return
    end

    stacked_params = decoding_params.level_params[parity]
    n_check_nodes = sum([2^i for i = stacked_params.times]) + 2^decoding_params.t
    anc_tensors = [vcat(initialize_bare_errors(anc_probs[parity][t], anc_flips[parity][t]; error_leg=:closed, flip_bit = true), state_prep_params[parity,t].gate[2]) for t=stacked_params.times]

    final_anc_tensors = initialize_stacked_errors(stacked_params.stacked_nodes, anc_probs[parity][end], anc_flips[parity][end]; level = true, error_leg=:closed, noisy_ancilla = true, measurement_error = true)

    # on the system side, we know all of the nodes will be top errors
    sys_tensors = [evaluate_top_error(eprob, bitflip) for (eprob, bitflip) in zip(sys_probs[parity], sys_flips[parity])]
    open_sys_tensors = [evaluate_error(Val(:open), eprob, bitflip; input = true) for (eprob, bitflip) in zip(sys_probs[parity], sys_flips[parity])]

    # now use the level_params to do this contraction, the "open legs" will now be the check nodes
    level_params = decoding_params.level_params[parity].level_params
    
    nodes = vcat(sys_tensors, final_anc_tensors, stacked_params.level_params.gate,vcat(anc_tensors...), fill(CHECK_NODE, n_check_nodes))

    @threads for i=1:length(sys_tensors)
        if sys_probs[parity][i][1]==0 || sys_probs[parity][i][1]==0 # frozen, don't bother updating
	    continue
	end
	# get marginal probability of error
	sys_probs[parity][i] = marginal_bayesian_update(nodes, open_sys_tensors[i], i, level_params.idxs, level_params.order)
    end

    # check to make sure these errors could happen
    validate_probs(sys_probs[parity], sys_flips[parity])
end

"""
One round of syndrome extraction, unstacked version
    - parity: whether we're using parity 1 or parity 2 tree
    parity 1 tree used for logical 0 state, 2 tree for logical + state
"""
function probability_passing_round!(decoding_params, sys_probs, sys_flips, anc_probs, anc_flips; parity::Int = 1, first_round::Bool = false)

    # leak errors from ancilla onto system
    update_leaked_pair!(sys_probs[parity%2+1], sys_flips[parity%2+1], anc_probs[parity%2+1], anc_flips[parity%2+1], decoding_params.error_basis[parity%2+1]; parity = parity%2+1, log_state = parity%2+1)

    if first_round # I already know nothing will be on the system
        return
    end

    anc_tensors = initialize_bare_errors(anc_probs[parity], anc_flips[parity]; error_leg=:closed, flip_bit = true)

    # on the system side, we know all of the nodes will be top errors
    sys_tensors = [evaluate_top_error(eprob, bitflip) for (eprob, bitflip) in zip(sys_probs[parity], sys_flips[parity])]
    open_sys_tensors = [evaluate_error(Val(:open), eprob, bitflip; input = true) for (eprob, bitflip) in zip(sys_probs[parity], sys_flips[parity])]

    # now use the level_params to do this contraction, the "open legs" will now be the check nodes
    level_params = decoding_params.level_params[parity]
    
    nodes = vcat(fill(CHECK_NODE, 2^decoding_params.t), anc_tensors, level_params.gate, sys_tensors)

    @threads for i=1:length(sys_tensors)
        if sys_probs[parity][i][1]==0 || sys_probs[parity][i][1]==0 # frozen, don't bother updating
	    continue
	end
	# get marginal probability of error
	sys_probs[parity][i] = marginal_bayesian_update(nodes, open_sys_tensors[i], length(nodes)-length(sys_tensors)+i, level_params.idxs, level_params.order)
    end

    # check to make sure these errors could happen
    validate_probs(sys_probs[parity], sys_flips[parity])
end

"""
Sample errors after each layer of gates
"""
function apply_fresh_errors!(err_probs, bitflips, prob; erasure_f = 0)
    @threads for qubit_i = 1:length(bitflips)
        new_prob, bitflip = make_error_node(prob; erasure_f = erasure_f)
	bitflips[qubit_i] ⊻= bitflip
	err_probs[qubit_i] = concatenate_channels(err_probs[qubit_i], new_prob)
    end
end

#= Master functions for n rounds of syndrome extraction =#
export run_probability_passing_bell, run_stacked_passing_bell, prep_syndrome_extraction

"""
prepare the data structures needed for syndrome extraction and decoding
"""
function prep_syndrome_extraction(tmax; measure_first = [1,1], check_first = 1, stack::Bool = true)
    decoding_params = BellDecoderParams(tmax; measure_first = measure_first, stack = stack)
    bell_params = [BellPassingParams(tmax; log_state = i, measure_first = measure_first[i]) for i = [check_first, check_first % 2 + 1]]

    counts = [[get_classical_error_counts(t; log_state = i, measure_first = j, measurement_error = true) for j=1:2, t=1:tmax] for i = [check_first, check_first%2 + 1]]

    return bell_params, decoding_params, counts
end

"""
Do n rounds of syndrome extraction using stacked probability passing on ancillas. Then decode with perfect syndrome measurements at end, unstacked tree.
"""
function stacked_passing_syndrome_extraction(bell_params, decoding_params, probs, counts; erasure_f = 0, n_rounds::Int = 1)
    sys_probs = [[[1.0,0] for i=1:2^decoding_params.t] for j=1:2]
    sys_flips = [zeros(Bool, 2^decoding_params.t) for j=1:2]
    for i=1:2*n_rounds
        anc_probs, anc_flips = prepare_stacked_ancilla(bell_params[(i+1)%2+1], probs, counts[(i+1)%2+1]; erasure_f = erasure_f)
	stacked_passing_round!(decoding_params, bell_params[(i+1)%2+1].level_params, sys_probs, sys_flips, anc_probs, anc_flips; parity = bell_params[(i+1)%2+1].log_state%2+1, first_round = (i==1))
	# introduce fresh errors
	for j=1:2
	    apply_fresh_errors!(sys_probs[j], sys_flips[j], probs[3]; erasure_f = erasure_f)
	end
    end

    perfect_syndrome_decode(decoding_params.par_checks, decoding_params.level_params[3:end], sys_probs, sys_flips)
end

"""
Do n rounds of syndrome extraction using unstacked probability passing on ancillas. Then decode with perfect syndrome measurements at end, unstacked tree.
    - n_verify: numner of extra ancillas used to verify the ancillas that couple to system
"""
function probability_passing_syndrome_extraction(bell_params, decoding_params, probs, counts; erasure_f = 0, n_rounds::Int = 1, n_verify::Int = 2)
    sys_probs = [[[1.0,0] for i=1:2^decoding_params.t] for j=1:2]
    sys_flips = [zeros(Bool, 2^decoding_params.t) for j=1:2]
    for i=1:2*n_rounds
        anc_probs, anc_flips = prepare_ancilla_state(bell_params[(i+1)%2+1],
	    probs, counts[(i+1)%2+1]; erasure_f = erasure_f)
	verify_ancilla!(bell_params[(i+1)%2+1], probs, counts[(i+1)%2+1], anc_probs, anc_flips; erasure_f = erasure_f, n_verify = n_verify)
	probability_passing_round!(decoding_params, sys_probs, sys_flips, anc_probs, anc_flips; parity = bell_params[(i+1)%2+1].log_state%2+1, first_round = (i==1))
	# introduce fresh errors
	for j=1:2
	    apply_fresh_errors!(sys_probs[j], sys_flips[j], probs[3]; erasure_f = erasure_f)
	end
	
    end
    
    # Now do the last level:
    perfect_syndrome_decode(decoding_params.par_checks, decoding_params.level_params[3:end], sys_probs, sys_flips)
end

"""
Decoding on last layer
"""
function perfect_syndrome_decode(par_checks::Array, level_params::Array, sys_probs, sys_flips)
    # Now do the last level:
    classes = zeros(Bool, 2)
    weights = zeros(2, 2)
    @threads for i=1:2
        classes[i], weights[:,i] = perfect_syndrome_decode(par_checks[i], level_params[i], sys_probs[i], sys_flips[i])
    end
    classes, weights
end

function perfect_syndrome_decode(par_check::FqMatrix, level_params::LevelParams, sys_probs, sys_flips)
    # evaluate syndrome and get "canonical" error with that syndrome
    err_s = match_syndrome(par_check[1:end-1,:], sys_flips[:,:]')

    weights = ncon!(vcat(level_params.gate, [evaluate_top_error(eprob, bitflip) for (eprob, bitflip) in zip(sys_probs, nemo_to_bool(err_s)[:,1])]), level_params.idxs; order = level_params.order, op = normalize_mult)

    # correct class is the one associated to err * err_s
    last_syndrome = par_check[end,:] * (err_s + bool_to_nemo(sys_flips[:,:]))
    correct_class = (last_syndrome[1] == 1)
    return correct_class, weights
end

function probability_passing_bell_rates!(tmax, rates, class_probs, classes; erasure_f=0, measure_first = [1,1], check_first::Int = 1, n_rounds::Int = 1, n_verify::Int = 2)
    bell_params, decoding_params, counts = prep_syndrome_extraction(tmax; measure_first = measure_first, check_first = check_first, stack = false)
    for rate_i=1:length(rates)
        println(rates[rate_i]); flush(stdout)
        @threads for i=1:size(classes, 2)
	    if i%50==1
	        println(i); flush(stdout)
	    end
	    classes[:, i,rate_i], class_probs[:,:,i,rate_i]  = probability_passing_syndrome_extraction(bell_params, decoding_params, rates[rate_i], counts; erasure_f = erasure_f, n_rounds = n_rounds, n_verify = n_verify)
	end
    end
end

function stacked_passing_bell_rates!(tmax, rates, class_probs, classes; erasure_f=0, measure_first = [1,1], check_first::Int = 1, n_rounds::Int = 1)
    bell_params, decoding_params, counts = prep_syndrome_extraction(tmax; measure_first = measure_first, check_first = check_first, stack = true)
    for rate_i=1:length(rates)
        println(rates[rate_i]); flush(stdout)
        @threads for i=1:size(classes, 2)
	    if i%50==1
	        println(i); flush(stdout)
	    end
	    classes[:, i,rate_i], class_probs[:,:,i,rate_i]  = stacked_passing_syndrome_extraction(bell_params, decoding_params, rates[rate_i], counts; erasure_f = erasure_f, n_rounds = n_rounds)
	end
    end
end

"""
Master function for n rounds of syndrome extraction, probability passing decoder
"""
function run_probability_passing_bell(tmax, rates; num_samples::Int = 100, measure_first = [1,1], check_first::Int = 1, decode_data = Dict{String, Any}("class"=>Dict(), "weights"=>Dict()), n_rounds::Int = 1, erasure_f = 0, n_verify::Int = 2)
    for (myvar, tag) in zip([n_verify, check_first, n_rounds, measure_first, erasure_f], ["verify", "first-check", "rounds", "first-meas", "heralded"])
        if haskey(decode_data, tag)
	    @assert decode_data[tag]==myvar
	else
	    decode_data[tag] = myvar
	end
    end

    classes = zeros(Bool, 2, num_samples, length(rates))
    weights = zeros(2, 2, num_samples, length(rates))
    probability_passing_bell_rates!(tmax, rates, weights, classes; erasure_f = erasure_f, check_first = check_first, n_rounds = n_rounds, measure_first = measure_first, n_verify = n_verify)

    # Turn into dictionary, concatenate as needed
    for (rate_i, rate) in enumerate(rates)
        if haskey(decode_data["class"], rate) # append to existing data
	    decode_data["class"][rate] = hcat(decode_data["class"][rate],  classes[:,:,rate_i])
            decode_data["weights"][rate] = cat(decode_data["weights"][rate], weights[:,:,:,rate_i], dims = 3)
	else
	    decode_data["class"][rate] = classes[:,:,rate_i]
	    decode_data["weights"][rate] = weights[:,:,:,rate_i]
	end
    end
    decode_data
end

"""
Master function for n rounds of syndrome extraction, probability passing decoder that uses stacked TTN for updating the ancilla error probabilities. Doesn't lead to any improvements
"""
function run_stacked_passing_bell(tmax, rates; num_samples::Int = 100, measure_first = [1,1], check_first::Int = 1, decode_data = Dict{String, Any}("class"=>Dict(), "weights"=>Dict()), n_rounds::Int = 1, erasure_f = 0)
    for (myvar, tag) in zip([check_first, n_rounds, measure_first, erasure_f], ["first-check", "rounds", "first-meas", "heralded"])
        if haskey(decode_data, tag)
	    @assert decode_data[tag]==myvar
	else
	    decode_data[tag] = myvar
	end
    end

    classes = zeros(Bool, 2, num_samples, length(rates))
    weights = zeros(2, 2, num_samples, length(rates))
    stacked_passing_bell_rates!(tmax, rates, weights, classes; erasure_f = erasure_f, check_first = check_first, n_rounds = n_rounds, measure_first = measure_first)

    # Turn into dictionary, concatenate as needed
    for (rate_i, rate) in enumerate(rates)
        if haskey(decode_data["class"], rate) # append to existing data
	    decode_data["class"][rate] = hcat(decode_data["class"][rate],  classes[:,:,rate_i])
            decode_data["weights"][rate] = cat(decode_data["weights"][rate], weights[:,:,:,rate_i], dims = 3)
	else
	    decode_data["class"][rate] = classes[:,:,rate_i]
	    decode_data["weights"][rate] = weights[:,:,:,rate_i]
	end
    end
    decode_data
end

end # module