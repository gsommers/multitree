#!/usr/licensed/julia/1.11/bin/julia
"""
Module that sets up all the tensor network stuff 
for decoding the Bell tree (generalized Shor code)
"""
module BellTreeTensors

using Nemo # for FqMatrix
using Base.Threads
using QuantumClifford

using BinaryFields # bool_to_nemo
using StackedTree
using BellTreeFaults
using TensorExtensions
using StabilizerTree

export LevelParams, StackedBell, StackedPerfectBell
"""
Parameters associated with TN contraction at level t
Note: some of these features are only used in other modules
not published to this repo.
"""
struct LevelParams
    idxs::Array
    order::Array
    counts::Array
    st_idxs::Array
    gate::Array
    
    function LevelParams(t; log_state::Int = 1, measure_first::Int = 1)
        idxs, order = make_bare_index_list(t; log_state = log_state, measure_first = measure_first)
	flattened_list = vcat(vcat(idxs...)...)
	joined_idxs, joined_order = join_index_lists(idxs, flattened_list, order)
	gate_nodes = [make_gate_nodes(t, [2^i for i=0:t-1]; log_state = log_state, log_type = log_type, measure_first = measure_first) for log_type = [:stab, :gauge]]
	counts = vcat([length.(idx_list) for idx_list in idxs[1:3]],[[2^t]])
	@assert counts==get_classical_error_counts(t; log_state = log_state, measure_first = measure_first, measurement_error = true)
	st_idxs = get_spacetime_idxs(counts)
	new(joined_idxs, joined_order, counts, st_idxs, gate_nodes)
    end

    LevelParams(idxs, order, counts, gate; st_idxs = []) = new(idxs, order, counts, st_idxs, gate)
end

# order for tensor network contraction
function order_indices(order, top_layer; offset::Int = 0)
    indices = vcat([idxs[end] for idxs in top_layer], [idxs[1:2] for idxs in top_layer]...)
    for idx_list in order
        push!(indices, idx_list...)
	push!(indices, [idxs .+ offset for idxs in idx_list]...)
    end
    indices
end

function join_index_lists(index_list, flattened_list, order)
    # "system" side first
    sys_count = index_list[2][end][end][2]
    index_list2 = [idxs .+ sys_count for idxs in flattened_list]
    top_layer = [[index_list[2][end][i][2] + sys_count, index_list[2][end][i][2], 2*sys_count+i] for i=1:length(index_list[2][end])]
    num_gate_nodes = length(vcat(index_list[4]...))
    idxs = vcat(flattened_list, index_list2[1:end-num_gate_nodes], [[2*sys_count+i] for i=1:length(top_layer)], index_list2[end-num_gate_nodes+1:end], top_layer)
    joined_order = order_indices(order, top_layer; offset = sys_count)
    return (idxs = idxs, order = joined_order)
end

"""
Parameters associated with stacked probability passing at depth tmax
"""
struct StackedBell
    level_params::LevelParams # idxs, gate nodes, and orders for contraction
    stacked_nodes::NamedTuple # stacked nodes
    times::AbstractArray # times where we connect to ancillas/collect syndrome info
    
    function StackedBell(tmax; log_state::Int=1, measure_first::Int = 1, alternate::Bool = true)
        if alternate # perform matching checks in every layer
	    times = 1:tmax
	else
	    times = 2-tmax%2:2:tmax
	end
	
        noisy_list = make_stacked_noisy_index_list(tmax, times; log_state = log_state, open_logical = false, measure_first = measure_first, level = true)
	idxs = vcat(vcat(noisy_list.sys_idxs...)..., noisy_list.anc_idxs, noisy_list.top_idxs)
	stacked_nodes = initialize_stacked_nodes(times; log_state = log_state, open_logical = false, dim = 2, measure_first = measure_first)
	gate_nodes = make_stacked_gate_nodes(times; measure_first = measure_first, log_state = log_state, perfect_layer = 0)
	counts = [length.(idx_list) for idx_list in noisy_list.sys_idxs[1:3]]
	@assert counts==get_classical_error_counts(tmax; log_state = log_state, measure_first = measure_first, measurement_error = false)
	
	level_params = LevelParams(idxs, noisy_list.order, counts, gate_nodes; st_idxs = get_spacetime_idxs(counts))
	new(level_params, stacked_nodes, times)
    end
end

function get_spacetime_idxs(counts)
    idxs = []
    for i=1:3
        for j=1:length(counts[i])
	    for k=1:counts[i][j]
	        push!(idxs, [i,j,k])
	    end
	end
    end
    idxs
end

"""
Parameters associated with final level of decoding,
after directly measuring stabilizers on system
"""
struct StackedPerfectBell
    level_params::LevelParams # idxs, gate nodes, and orders for contraction at last level
    stacked_nodes::NamedTuple # stacked nodes for last level
    par_checks::Array{FqMatrix} # parity check matrices (including logical) for each level
    times::AbstractArray # times where we connect to ancillas/collect syndrome info

    function StackedPerfectBell(tmax; log_state::Int=1, measure_first::Int=1, alternate::Bool=true)
   
        if alternate
	    @assert log_state==measure_first
	    times = 1:tmax
	else
	    times = vcat(2-(measure_first==log_state):2:tmax-1, [tmax])
	end

	noisy_list = make_stacked_noisy_index_list(tmax, times[1:end-1]; log_state = log_state, measure_first = log_state, level = false, open_logical = true, perfect_layer = 1)
	idxs = vcat(vcat(noisy_list.sys_idxs...)..., noisy_list.anc_idxs, noisy_list.top_idxs)
	counts = [length.(idx_list) for idx_list in noisy_list.sys_idxs[1:3]]
	@assert counts==get_classical_error_counts(tmax; log_state = log_state, measure_first = log_state, measurement_error = false)

	# stacking
	stacked_nodes = initialize_stacked_nodes(times; measure_first = log_state, log_state = log_state, open_logical = true, dim = 2)
	gate_nodes = make_stacked_gate_nodes(times; log_state = log_state, perfect_layer = 1, measure_first = log_state)
	last_level_params = LevelParams(idxs, noisy_list.order, counts, gate_nodes)

	# for syndrome decoding on last level
	par_checks = [prepare_par_check(t, log_state = log_state) for t=1:tmax]
	new(last_level_params, stacked_nodes, par_checks, times[1:end-1])
    end
end

"""
Number of error locations of each type, in each layer
"""
function get_classical_error_counts(t::Int; log_state::Int = 1, measure_first::Int = 1, measurement_error::Bool = true)
    encoding_counts = [2^i for i=1:t]
    check_counts = [2^i for i=1:t-1]
    input_counts = zeros(Int, t)
    for i=measure_first+1:2:t
        input_counts[i] = 2^(i-1)
    end
    if log_state==1
        input_counts[1] = 1
    elseif measure_first == 2 # errors on both input legs
        input_counts[1] = 2
    end
    if measurement_error
        return [input_counts, encoding_counts, check_counts, [2^t]]
    else
        return [input_counts, encoding_counts, check_counts]
    end
end     
        
# Syndrome matching/parity checks
export prepare_par_check
function prepare_par_check(t; log_state::Int=1, measure_log::Bool = true)
    if measure_log
        stab_mat = vcat(track_fresh_stabilizers((tHadamard ⊗ tHadamard)*tCNOT, t, log_state%2+1:2:t)...,track_logical_reps((tHadamard ⊗ tHadamard)*tCNOT, t)[log_state:log_state,:])
    else
        stab_mat = vcat(track_fresh_stabilizers((tHadamard ⊗ tHadamard)*tCNOT, t, log_state%2+1:2:t)...)
    end
    if size(stab_mat,1)==0 # shouldn't happen, unless you do weird check order
        println("ALERT! no syndrome learned in this level")
        return matrix(GF(2), zeros(Bool,0,0))
    end

    if t%2==log_state%2 # will be Z type, so keep last n columns
        return bool_to_nemo(stab_mat[:,end÷2+1:end])
    else # will be X type, so keep first n columns
        return bool_to_nemo(stab_mat[:,1:end÷2])
    end
end

# Gate nodes

function make_gate_nodes(tmax; log_state::Int = 1, measure_first::Int = 1)
    make_gate_nodes(tmax, [2^(t-1) for t=1:tmax]; log_state = log_state, measure_first = measure_first)
end

function make_gate_nodes(tmax, counts; log_state::Int = 1, measure_first::Int = 1, log_type=:logical)
    vcat([fill(get_gate_node(t, measure_first; log_state = log_state, log_type=log_type), counts[t]) for t=1:tmax]...)
end

# bottom of stack is "gauge", rest are "stab" IF perfect_layer is 1
function make_stacked_gate_nodes(tmax::Int; log_state::Int = 1, measure_first::Int = 1, perfect_layer::Int = 1)
    make_stacked_gate_nodes(1:tmax; measure_first = measure_first, log_state = log_state, perfect_layer = perfect_layer)
end

function make_stacked_gate_nodes(times::AbstractArray; log_state::Int = 1, measure_first::Int = 1, perfect_layer::Int = 0)
    gate_nodes = []
    for t=1:times[end]
        gate_node = get_gate_node(t, measure_first; log_state = log_state, log_type=:stab)
        n = count(el->el>=t, times)
	if perfect_layer==0 || (t>1 && (measure_first + t)%2==1)
	    push!(gate_nodes, fill(gate_node, 2^(t-1)*n)...)
	else # also get "gauge node"
	    gauge_node = get_gate_node(t, measure_first; log_state = log_state, log_type=:gauge)
	    push!(gate_nodes, repeat(vcat(fill(gate_node, n - perfect_layer), fill(gauge_node, perfect_layer)),2^(t-1))...)
	end
    end
    gate_nodes
end

function get_gate_node(t, measure_first; log_state::Int=1, log_type=:logical)
    b_i = (measure_first + t)%2 + 1

    if log_type==:logical
        return [NOTC_NODE, CNOT_NODE][b_i]
    elseif t>1 || log_state==measure_first
        if b_i==2
	    return CNOT_NODE
	elseif log_type==:gauge
	    return NOTC_GAUGE_NODE
	else
	    return NOTC_NODE[:,1,:,:]
	end
    elseif log_state==1 # and this means measure_first=2
        if log_type==:stab
	    return CNOT_NODE[1,:,:,:]
	else # sum over left leg instead of right!!! -> ends up as NOTC_GAUGE_NODE
	    return NOTC_GAUGE_NODE # dropdims(sum(CNOT_NODE, dims=1),dims=1)
	end
    else # log_state = 2, measure_first = 1
        if log_type==:stab
	    return NOTC_NODE[1,1,:,:]
	else # sum over both input legs
	    return dropdims(sum(NOTC_GAUGE_NODE, dims=1),dims=1)
	end
    end
end

# Decoding
export bayesian_update_pair!, last_level_bell_decode

"""
Version of bayesian_update_pair! used in Steane EC decoder
"""
function bayesian_update_pair!(level_params, error_probs, bitflips, t)
    anc_tensors = initialize_bare_errors(error_probs[2], bitflips[2]; error_leg=:closed)
    sys_probs = [error_probs[1][1][1:t], error_probs[1][2][1:t], error_probs[1][3][1:t-1]]
    sys_errs = [bitflips[1][1][1:t], bitflips[1][2][1:t], bitflips[1][3][1:t-1]]
    sys_tensors = initialize_bare_errors(sys_probs, sys_errs; error_leg=:closed) 
    open_sys_tensors = initialize_bare_errors(sys_probs, sys_errs; error_leg=:open)

    nodes = vcat(sys_tensors, level_params.gate[1], anc_tensors, level_params.gate[2], fill(CHECK_NODE, 2^t))
    @threads for i=1:length(sys_tensors)
        idx = level_params.st_idxs[i] # spacetime index (error type, time, node index)
	if error_probs[1][idx[1]][idx[2]][idx[3]][1]==0 || error_probs[1][idx[1]][idx[2]][idx[3]][1]==1 # frozen, don't bother updating
	    continue
	end
        # get marginal probability of error
	error_probs[1][idx[1]][idx[2]][idx[3]] = marginal_bayesian_update(nodes, open_sys_tensors[i], i, level_params.idxs, level_params.order)
    end
end

"""
Update error model on the "system side" given the interaction with ancillas whose error models are encoded in level_params.
This is the version used in the stacked probability passing decoder, state prep gadget
"""
function bayesian_update_pair!(stacked_params::StackedBell, level_params::Array, error_probs::Array, bitflips::Array, tmax::Int)
    anc_times = stacked_params.times
    n_check_nodes = sum([2^i for i=anc_times])
    anc_tensors = [vcat(initialize_bare_errors(error_probs[t], bitflips[t]; error_leg = :closed), level_params[t].gate[2]) for t=anc_times]

    sys_tensors = initialize_stacked_errors(stacked_params.stacked_nodes, error_probs[end], bitflips[end]; level = true, error_leg=:closed, noisy_ancilla = true, measurement_error = false)
    open_sys_tensors = initialize_stacked_errors(stacked_params.stacked_nodes, error_probs[end], bitflips[end]; level = true, error_leg=:open, noisy_ancilla = true, measurement_error = false)

    nodes = vcat(sys_tensors, stacked_params.level_params.gate, vcat(anc_tensors...), fill(CHECK_NODE, n_check_nodes))
    @threads for i=1:length(sys_tensors)
        idx = stacked_params.level_params.st_idxs[i] # spacetime index (error type, time, node index)
	if error_probs[end][idx[1]][idx[2]][idx[3]][1]==0 || error_probs[end][idx[1]][idx[2]][idx[3]][1]==1 # frozen, don't bother updating
	    continue
	end

    	# get marginal probability of error
	error_probs[end][idx[1]][idx[2]][idx[3]] = marginal_bayesian_update(nodes, open_sys_tensors[i], i, stacked_params.level_params.idxs, stacked_params.level_params.order)
    end
    return error_probs[end]
end

function marginal_bayesian_update(nodes, open_leg, i, idxs, order)
    probs = ncon!(vcat(nodes[1:i-1],[open_leg], nodes[i+1:end]), vcat(idxs[1:i-1], [vcat(idxs[i], [-1])], idxs[i+1:end]); order = order, op = normalize_mult)
    return probs ./ sum(probs) # normalize to 1
end

function last_level_bell_decode(stacked_params, level_params, error_probs, bitflips; op = normalize_mult)
    tmax = length(stacked_params.par_checks)
    sys_tensors = initialize_stacked_errors(stacked_params.stacked_nodes, error_probs[end], bitflips[end]; level = false, error_leg=:closed, noisy_ancilla = true)

    # don't insert errors on ancillas, because I've already matched syndrome on just system
    anc_nodes = [vcat(initialize_bare_errors(error_probs[t], bitflips[t]; error_leg = :closed, flip_bit = false), level_params[t].gate[2]) for t=stacked_params.times]
    sys_nodes = vcat(sys_tensors, stacked_params.level_params.gate)
    
    ncon!(vcat(sys_nodes, vcat(anc_nodes...), fill(CHECK_NODE, sum(2 .^ stacked_params.times))), stacked_params.level_params.idxs; order = stacked_params.level_params.order, op = op)
end

end # module