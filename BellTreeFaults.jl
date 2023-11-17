#!/usr/licensed/julia/1.7/bin/julia
"""
Module for faults in Bell tree: binary tree where each gate is (H otimes H) CNOT
Treating noise as independent X/Z, we separately decode bit and phase flips
so everything is essentially classical (bond dimension 2)
"""
module BellTreeFaults

using PauliNoise
using StackedTree
using StabilizerTree

# these things are true for all the Bell tree
export CNOT_NODE, NOTC_NODE, CHECK_NODE, NOTC_GAUGE_NODE

const CNOT_NODE = cat(cat([1 0; 0 0], [0 0; 0 1], dims=3),cat([0 1; 0 0], [0 0; 1 0], dims = 3),dims=4)
const CHECK_NODE = dropdims(sum(CNOT_NODE, dims=1), dims=1)
const NOTC_NODE = permutedims(CNOT_NODE,[2,1,4,3])
const NOTC_GAUGE_NODE = dropdims(sum(NOTC_NODE,dims=2),dims=2)

"""
Initialize error probabilities and bitflips
"""
function make_error_node(probs::Array; erasure_f = 0)
    # erasure_f is not used here, probs[1] is unheralded, probs[2] heralded
    if sum(probs)==0
        return [1,0], false
    else
        r = rand()
	if r < probs[2] # heralded (erasure)
	    return [0.5, 0.5], rand(Bool)
	else
	    return [1-probs[1], probs[1]], (r<probs[1])
	end
    end
end

function make_error_node(prob::Number; erasure_f = 0)
    r = rand()
    if r < prob * erasure_f # heralded (erasure)
        return [0.5,0.5], rand(Bool)
    else
        probs = [1-(1-erasure_f)*prob/(1-erasure_f*prob), (1-erasure_f)*prob/(1-erasure_f*prob)]
	if r < prob # unheralded error
	    return probs, true
	else # no error
	    return probs, false
        end
    end
end

# Tree error propagation
export propagate_tree_errors_classical, propagate_tree_classical_layer, prepare_error_basis, track_bell_error_probs, update_leaked_probs!

"""
Get basis of errors starting from logical leg (left) and stabilizer leg (right)
If basis_i=1, only track the errors that start as bit flips entering even layers, Z errors entering odd layers (i.e. affect the logical + state
If basis_i=2, do the opposite
"""
function prepare_error_basis(tmax; basis_i::Int = 1)
    cliff = Bool[0 0 1 1; 0 0 0 1; 1 0 0 0; 1 1 0 0]

    paulis = [Array{Array}(undef, tmax), Array{Array}(undef, tmax)]
    for (fresh_i, fresh)=enumerate([false, true])
        bases = [track_operator_spreading(cliff, pauli, tmax; fresh = fresh) for pauli = [Bool[1,0], Bool[0,1]]]

	matching_bases = [bases[(t+basis_i)%2+1][end-t] for t=0:tmax-1]
	if (basis_i+tmax)%2==0 # should end up all X's
	    @assert all([all(iszero, pauls[end÷2+1:end]) for pauls in matching_bases])
	    paulis[fresh_i] = [pauls[1:end÷2] for pauls in matching_bases]
	
	else # should end up all Z's
	    @assert all([all(iszero, pauls[1:end÷2]) for pauls in matching_bases])
	    paulis[fresh_i] = [pauls[end÷2+1:end] for pauls in matching_bases]
        end
    end
    paulis
end


"""
Propagate error from time t to t+1,
errs[1]: errors on fresh stabilizer legs (only on CNOT layers)
errs[2]: errors after encoding gate
"""
function propagate_tree_classical_layer(errs, incoming_err)
    if isempty(errs[1]) # this is a layer with "gauge" inputs
        # which also means it's a NOTC layer
	gate = Bool[1 1; 0 1]
	err = vcat([gate * [ei, false] for ei=incoming_err]...)
    else
        gate = Bool[1 0; 1 1]
	@assert length(errs[1])==length(incoming_err)
	err = vcat([gate * [incoming_err[i], errs[1][i]] for i=1:length(errs[1])]...)
    end
    err .+ errs[2]
end

function propagate_tree_errors_classical(errs)
    myerr = propagate_tree_classical_layer([errs[1][1][2:end], errs[2][1]], errs[1][1][1:1])
    for t=2:length(errs[1])
        myerr = propagate_tree_classical_layer([errs[1][t], errs[2][t]], myerr .+ errs[3][t-1])
    end
    if length(errs)==4 # also has measurement errors
        return Bool.(mod.(myerr .+ errs[4][end], 2))
    else
        return Bool.(mod.(myerr,2))
    end
end

# this is faster, if I already have the error basis
function propagate_tree_errors_classical(errs, paulis; on_right::Bool = false)
    propagate_tree_errors_classical(errs, paulis, length(errs[1]); on_right = on_right)
end

function propagate_tree_errors_classical(errs, paulis, tmax; on_right::Bool = false)
   myerr = copy(errs[2][tmax])

    if on_right # special case where the first error is on right leg, not left
        if errs[1][1][1]
	    myerr .⊻= paulis[2][1]
	end
    else
        for i=1:length(errs[1][1])
            if errs[1][1][i]
	        myerr .⊻= paulis[i][1]
	    end
        end
    end

    for tp=2:tmax
	for j=1:2^(tp-1)
	    if length(errs[1][tp])>=j && errs[1][tp][j] # state prep error
	        myerr[(2^(tmax-tp+1))*(j-1)+1:(2^(tmax-tp+1)*j)] .⊻= paulis[2][tp]
	    end
	    if xor(errs[2][tp-1][j],errs[3][tp-1][j]) # gate error
            	myerr[(2^(tmax-tp+1))*(j-1)+1:(2^(tmax-tp+1)*j)] .⊻= paulis[1][tp]
	    end
	end
    end
    myerr
end

function track_bell_error_probs(error_probs; parity::Int = 1, log_state::Int = 1)
    tmax = length(error_probs[1])
    prop_probs = [[1.0,0] for i=1:2^tmax]
    trans_mats = [reshape(NOTC_NODE,(4,4)), reshape(CNOT_NODE,(4,4))]
    # fresh qubits
    for t=parity+1:2:tmax
	prop_probs[2^(tmax-t)+1:2^(tmax-t+1):2^tmax] = error_probs[1][t]
    end
    for t=parity:2:tmax
        if t>1
	    @assert isempty(error_probs[1][t])
	end
    end
    # special for the "logical leg" at t=1
    if log_state==1
        if parity == 1 # this error is on the logical leg
            prop_probs[1] = error_probs[1][1][1]
    	else
            prop_probs[2^(tmax-1)+1] = error_probs[1][1][1]
	end
    elseif parity == 2 # errors on both legs
        prop_probs[1:2^(tmax-1):end] = error_probs[1][1]
    else
        @assert isempty(error_probs[1][1])
    end
	
    # now do the gates
    for t=1:tmax
        sites = 1:2^(tmax-t):2^(tmax)
	@assert length(error_probs[2][t])==length(sites)
	for i=1:length(sites)÷2
	    marginal_out = propagate_error_pair(prop_probs[sites[2*(i-1)+1:2*i]], trans_mats[(parity+t)%2+1])
	    # now concatenate on the next layer of encoding gate errors, and check gate errors if t<tmax
	    for j=1:2
	        prop_probs[sites[2*(i-1)+j]] = concatenate_channels(marginal_out[j], error_probs[2][t][2*(i-1)+j])
	    	if t<tmax
	            prop_probs[sites[2*(i-1)+j]] = concatenate_channels(prop_probs[sites[2*(i-1)+j]], error_probs[3][t][2*(i-1)+j])
		end
	    end
	end
    end
    prop_probs
end

# Error channels (classical)
export evaluate_error, concatenate_channels

function concatenate_channels(probs1, probs2)
    return [sum(probs1 .* probs2), probs1[1]*probs2[2]+probs1[2]*probs2[1]]
end

function get_err_perm(i)
    if i==0
        return [1, 0]
    else
        return [0, 1]
    end
end

function evaluate_error(::Val{:open}, probs, bitflip::Bool; input::Bool = false)
    if bitflip
        prob_node = cat([0 probs[1]; probs[1] 0], [probs[2] 0; 0 probs[2]], dims = 3)
    else
        prob_node = cat([probs[1] 0; 0 probs[1]], [0 probs[2]; probs[2] 0], dims = 3)
    end

    if input
        return prob_node[1,:,:]
    else
        return prob_node
    end
end

function evaluate_error(::Val{:closed}, probs, bitflip::Bool; input::Bool = false)
    if input
        return evaluate_top_error(probs, bitflip)
    end
    if bitflip
        return pauli_classical_tensor(probs[end:-1:1])
    else
        return pauli_classical_tensor(probs)
    end
end

# Initializing errors
export initialize_error_nodes, initialize_bare_errors, initialize_level_errors

function initialize_level_errors(error_counts, probs; erasure_f=0, func = initialize_error_nodes, perfect_last::Bool = true)
    tmax = length(error_counts)
    st_probs = [Array{Array}(undef, 2^(tmax-t-1)) for t=1:tmax-1]
    st_errs = [Array{Array}(undef, 2^(tmax-t-1)) for t=1:tmax-1]
    for t=1:tmax-1, i=1:2^(tmax-t-1)
        st_probs[t][i], st_errs[t][i] = func(error_counts[t], probs; erasure_f = erasure_f)
    end
    sys_probs, sys_errs = func(error_counts[end], probs; erasure_f = erasure_f, perfect_last = perfect_last)
    vcat(st_probs, [[sys_probs]]), vcat(st_errs, [[sys_errs]])
end

function initialize_bare_errors(error_probs, bitflips; error_leg=:open, flip_bit::Bool = true)
    tmax = length(error_probs[1])
    errs = [[Array{Array}(undef, length(error_probs[j][i])) for i=1:length(error_probs[j])] for j=1:3]
    for t=1:tmax
    	# input errors
	errs[1][t] = [evaluate_error(Val(error_leg), error_probs[1][t][node_i], bitflips[1][t][node_i] && flip_bit; input = true) for node_i=1:length(errs[1][t])]

	# encoding gate errors

	errs[2][t] = [evaluate_error(Val(error_leg), error_probs[2][t][node_i], bitflips[2][t][node_i] && flip_bit; input = false) for node_i=1:length(errs[2][t])]

	t==tmax && break
	
	# check gate errors

	errs[3][t] = [evaluate_error(Val(error_leg), error_probs[3][t][node_i], bitflips[3][t][node_i] && flip_bit; input = false) for node_i=1:length(errs[3][t])]

    end

    if length(error_probs) > 3 # also have measurement errors
        @assert error_leg==:closed # because this is the ancilla side
	return vcat(vcat(vcat(errs...)...), [evaluate_top_error(eprob, bitflip && flip_bit) for (eprob, bitflip) in zip(error_probs[4][1], bitflips[4][1])])
    else
	return vcat(vcat(errs...)...)
    end
end

"""
Initialize error nodes from scratch, not matching syndrome
"""
function initialize_error_nodes(counts, probs; erasure_f = 0, perfect_last::Bool = false)
    # state prep errors
    error_probs = [[Array{Array}(undef, counts[i][t]) for t=1:length(counts[i])] for i=1:3]
    bitflips = [[zeros(Bool, counts[i][t]) for t=1:length(counts[i])] for i=1:3]
    for i=1:3
    	for t=1:length(counts[i])
	    for j=1:counts[i][t]
	        error_probs[i][t][j], bitflips[i][t][j] = make_error_node(probs[i]; erasure_f = erasure_f)
	    end
	end
    end

    # if this is an ancilla block, it will also have measurement errors
    if length(counts)>3
	last_probs = Array{Array}(undef, counts[end][1])
	last_flips = zeros(Bool, counts[end][1])
	for i=1:counts[end][1]
	    prob1, flip1 = make_error_node(probs[3]; erasure_f = erasure_f)
	    prob2, flip2 = make_error_node(probs[4]; erasure_f = erasure_f)
	    last_probs[i] = concatenate_channels(prob1, prob2)
	    last_flips[i] = xor(flip1, flip2)
	end
	error_probs = vcat(error_probs, [[last_probs]])
	bitflips = vcat(bitflips, [[last_flips]])
    elseif !perfect_last # SPECIAL CASE: measurements at end, so concatenate still
	measure_stuff = [make_error_node(probs[4]; erasure_f = erasure_f) for i=1:counts[2][end]]
	for i=1:counts[2][end]
	    error_probs[2][end][i] = concatenate_channels(measure_stuff[i][1], error_probs[2][end][i])
	    bitflips[2][end][i] = xor(bitflips[2][end][i], measure_stuff[i][2])
	end
    end
    return error_probs, bitflips
end

function update_leaked_probs!(post_probs, error_probs; parity::Int = 1, log_state::Int = 1)
    leaked_probs = track_bell_error_probs(error_probs; parity = parity, log_state = log_state)
    for i=1:length(leaked_probs)
        post_probs[i] = concatenate_channels(post_probs[i], leaked_probs[i])
    end
end

end # module