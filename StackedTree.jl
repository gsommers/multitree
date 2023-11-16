#!/usr/licensed/julia/1.7/bin/julia
"""
stacking nodes propagates error probabilities through layers of syndrome checks
"""
module StackedTree

using TensorExtensions
using Base.Threads
using PauliNoise

export stack_errors, initialize_stacked_nodes, make_stacked_index_list, make_bare_index_list, contract_error

# for the more general case of "quantum errors"
function make_internal()
    mat = zeros(Int, 4, 4, 4, 4)
    for i=1:4
        mat[:,:,i,i] = pauli_tensor(i)
    end
    mat
end

const INTERNAL_NODE = make_internal()
const STAB_NODE = INTERNAL_NODE[1,:,:,:]

function stack_errors(node::Array, n)
    bottom_node = dropdims(sum(node, dims = ndims(node)),dims = ndims(node))
    
    if n==1
        return bottom_node
    else
        index_list = vcat([vcat([-n*j-1 for j=0:ndims(node)-2],[1])], [vcat([-n*j-i for j=0:ndims(node)-3], [i-1, i]) for i=2:n-1], [vcat([-n*j for j=1:ndims(bottom_node)-1], [n-1])])
	return ncon!(vcat(fill(node, n-1), [bottom_node]), index_list, order=n-1:-1:1)
    end
end

function stack_errors(n; input_error::Bool = true, dim::Int = 4)
    # contract "physical qubit" on the top
    stacked_internal = stack_errors(INTERNAL_NODE[1:dim,1:dim,1:dim,1:dim], n)
    if input_error
        return (internal = stacked_internal, input = stack_errors(STAB_NODE[1:dim,1:dim,1:dim], n))
    else
        return (internal = stacked_internal, input = nothing)
    end
end


function initialize_stacked_nodes(t::Int; log_state::Int = -1, open_logical::Bool = false, dim::Int = 4, measure_first::Int = -1)
    initialize_stacked_nodes(1:t; log_state = log_state, measure_first = measure_first, open_logical = open_logical, dim = dim)
end

function initialize_stacked_nodes(times::AbstractArray; log_state::Int = -1, open_logical::Bool = false, dim::Int = 4, measure_first::Int = -1)
    t = times[end]
    stacked_nodes = [stack_errors(count(el->el>=i, times); input_error = (measure_first%2!=i%2), dim = dim) for i=1:t]
    if open_logical
        if length(times)==1 # special case, just use internal node
	    stacked_input = dropdims(sum(INTERNAL_NODE[1:dim,1:dim,1:dim,1:dim],dims=4),dims=4)
	else
            # bottom layer is special because it has an open logical leg
      	    stacked_input = ncon!(Vector{Any}(vcat(fill(STAB_NODE[1:dim,1:dim,1:dim], length(times)-1), [dropdims(sum(INTERNAL_NODE[1:dim,1:dim,1:dim,1:dim], dims = 4),dims = 4)])), vcat([[-2,-length(times)-2,1]],[[-i-1,i-1,i] for i=2:length(times)-1], [[-1,-length(times)-1,length(times)-1]]))
	end
    else
        stacked_input = stack_errors(STAB_NODE[1:dim,1:dim,1:dim], length(times))
    end
    
    return (input = stacked_input, bulk = stacked_nodes)
end

function make_bare_index_list(tmax; measure_first::Int = 1, log_state::Int = 1, i::Int = 0)
    input_idxs, encoding_idxs, check_idxs, gate_idxs = make_system_index_list(tmax, zeros(Bool, tmax); n=1, log_state = log_state, level = true, measure_first = measure_first, i = i)
    order = []
    for t=tmax:-1:1
        push!(order, [arr[end-1:end] for arr in gate_idxs[t]]...)
	push!(order, [arr[1:end-2] for arr in gate_idxs[t]]...)
	if t>1
	    push!(order, [arr[1:1] for arr in check_idxs[t-1]]...)
	end
    end
    
    (idxs = [input_idxs, encoding_idxs, check_idxs, gate_idxs], order = order)
end

# Assuming Z stabilizers fed in on stabilizer leg, and log_state on logical leg
function mark_root_legs(measure_first, log_state; i::Int = 0, n::Int = 1)
    if log_state < 0 # special case where I keep everything
        return [(i + j*n) .+ (1:n) for j=0:1]
    end

    # Four cases to consider
    if log_state==1 # no matter what, only one stabilizer is caught
        return [i .+ (1:n)]
    elseif measure_first==1 # no input idxs at all
	return []
    else # both are caught
        return [(i+j*n) .+ (1:n) for j=0:1]
    end
end

function make_system_index_list(tmax, anc_flags; n::Int = 1, log_state::Int = 1, measure_first::Int = 1, i::Int = 0, level::Bool = true)
    input_idxs = [[] for t=1:tmax]
    gate_idxs = [[] for t=1:tmax]
    encoding_idxs = [[] for t=1:tmax]
    check_idxs = [[] for t=1:tmax-1]
    
    for t=1:tmax
	if t==1
	    input_idxs[t] = mark_root_legs(measure_first, log_state; i = i, n = n)
	elseif measure_first%2 != t%2 
            input_idxs[t] = [(i+j*n) .+ (1:n) for j=0:2^(t-1)-1]
	end
	if !isempty(input_idxs[t])
	    i = input_idxs[t][end][end]
	end
	
	if t < tmax || level # same number of legs leaving as entering
	    encoding_idxs[t] = [(i+j*(2*n)) .+ (1:2*n) for j=0:2^t-1]
	else 
	    encoding_idxs[t] = [(i+j*(2*n-1)) .+ (1:2*n-1) for j=0:2^t-1]
	end
	if t==1
	    gate_idxs[t] = [vcat([input_idxs[t][i][k] for i=1:length(input_idxs[t])],  [encoding_idxs[t][i][k] for i=1:2]) for k=1:n]
	else
	    if t%2==measure_first%2 # only the left leg, no input leg
	        idxs = [[check_idxs[t-1][j][k+n], encoding_idxs[t][2*j-1][k],encoding_idxs[t][2*j][k]] for k=1:n, j=1:2^(t-1)]
	    else
	        idxs = [[check_idxs[t-1][j][k+n],input_idxs[t][j][k], encoding_idxs[t][2*j-1][k],encoding_idxs[t][2*j][k]] for k=1:n, j=1:2^(t-1)]
	    end
	    gate_idxs[t] = reshape(idxs, length(idxs))
	end

	if t<tmax
	    i = encoding_idxs[t][end][end]
            if anc_flags[t] # there will be one fewer layer in stack at t+1
	        n -= 1
	    end

	    check_idxs[t] = [vcat(idxs[end-n+1:end], i+n*(j-1) .+ (1:n)) for (j,idxs) in enumerate(encoding_idxs[t])]
	    i = check_idxs[t][end][end]
	end
    end
    @assert n==1
    input_idxs, encoding_idxs, check_idxs, gate_idxs
end

# n is the width of the stack
function get_top_layer_idxs(sys_idxs, anc_idxs, t, n)
    idx_count = anc_idxs[end][end]
    @assert length(anc_idxs)==2^t
    return [[anc_idxs[i][2], sys_idxs[t][i][n+1], idx_count + i] for i=1:2^t], idx_count
end

export make_stacked_noisy_index_list
# all but top has ancillas
function make_stacked_noisy_index_list(tmax::Int; log_state::Int = 1, open_logical::Bool = true, measure_first::Int = 1, perfect_layer::Int = 0)
    make_stacked_noisy_index_list(tmax, 1:tmax-1; log_state = log_state, open_logical = open_logical, measure_first = measure_first, perfect_layer = perfect_layer, level = false)
end

function make_stacked_noisy_index_list(tmax::Int, anc_times; log_state::Int = 1, open_logical::Bool = true, measure_first::Int = 1, perfect_layer::Int = 0, level::Bool = true)
    anc_flags = [t in anc_times for t=1:tmax]
    input_idxs, encoding_idxs, check_idxs, gate_idxs = make_system_index_list(tmax, anc_flags; n = length(anc_times) + perfect_layer, log_state = log_state, measure_first = measure_first, level = level, i = 0)

    # appendages
    i = encoding_idxs[tmax][end][end]
    if isempty(anc_times)
        ancilla_idxs = []
	top_layer_idxs = []
    else
        ancilla_idxs = [[] for t=1:anc_times[end]]
	top_layer_idxs = [[] for t=1:anc_times[end]]
    end
    for (t_i,t)=enumerate(anc_times)
        idxs, _ = make_bare_index_list(t; log_state = log_state, measure_first = measure_first, i = i)
    	top_idxs, i = get_top_layer_idxs(encoding_idxs, idxs[2][end], t, length(anc_times)-t_i+1 + perfect_layer)
	ancilla_idxs[t] = vcat(idxs, [[[[i+j] for j=1:2^t]]])
	i += 2^t
	top_layer_idxs[t] = top_idxs
    end

    order = []
    for t=tmax:-1:1
    	if anc_flags[t]
    	    # check gates
    	    push!(order, [arr[end:end] for arr in top_layer_idxs[t]]...)
	    push!(order, [arr[1:2] for arr in top_layer_idxs[t]]...)
	end

	# encoding gate errors for entire stack
        push!(order, [arr[end-1:end] for arr in gate_idxs[t]]...)
	for j=anc_times[end:-1:1]
	    if j<t
	        break
	    end
	    push!(order, [arr[end-1:end] for arr in ancilla_idxs[j][4][t]]...)
	end
	push!(order, [arr[1:end-2] for arr in gate_idxs[t]]...)
	for j=anc_times[end:-1:1]
	    if j<t
	        break
	    end
	    push!(order, [arr[1:end-2] for arr in ancilla_idxs[j][4][t]]...)
	end

	# post-check-gate errors from previous level
	if t>1
	    push!(order, [arr[1:end÷2] for arr in check_idxs[t-1]]...)
	    for j=anc_times[end:-1:1]
	        if j < t
		    break
		end
	        push!(order, [arr[1:end÷2] for arr in ancilla_idxs[j][3][t-1]]...)
	    end
	end
    end

    if open_logical # feed in unknown state from bottom
        input_idxs[1][1] = vcat([-1], input_idxs[1][1])
    end
    
    sys_idxs = [input_idxs, encoding_idxs, check_idxs, gate_idxs]
    anc_idxs = vcat([vcat(vcat(ancilla_i[[1,2,3,5,4]]...)...) for ancilla_i=ancilla_idxs[anc_times]]...)
    
    (sys_idxs = sys_idxs, anc_idxs = anc_idxs, top_idxs = vcat(top_layer_idxs...), order = vcat(order...))
end

export make_stacked_syndrome_index_list
function make_stacked_syndrome_index_list(tmax::Int, anc_times; log_state::Int = 1, measure_first::Int = 1)
    noisy_params = make_stacked_noisy_index_list(tmax, anc_times; log_state = log_state, open_logical = false, measure_first = measure_first, perfect_layer = 1, level = true)
    nmax = noisy_params[:top_idxs][end][end]
    
    # the true "system idxs"
    final_sys_idxs = [[i] for i=nmax+1:nmax+2^tmax]
    # now tack on the last layer of gate/measurement errors on what ends up as the "ancilla"
    final_anc_idxs = [[i] for i=nmax+2^tmax+1:nmax+2^(tmax+1)]
    # and another layer of "top idxs"
    top_idxs = [[noisy_params[:sys_idxs][2][end][i][end], final_sys_idxs[i][1], final_anc_idxs[i][1]] for i=1:2^tmax]

    # these come first now
    order = vcat([arr[end] for arr in top_idxs], vcat([arr[1:2] for arr in top_idxs]...))

    push!(noisy_params[:sys_idxs][3], final_anc_idxs)
    (sys_idxs = final_sys_idxs, state_prep_idxs = noisy_params[:sys_idxs], anc_idxs = noisy_params[:anc_idxs], top_idxs = vcat(noisy_params[:top_idxs], top_idxs), order = vcat(order, noisy_params[:order])) 
end

function make_stacked_index_list(tmax; log_state::Int = 1, level::Bool = true)
    input_idxs = [[] for t=1:tmax]
    gate_idxs = [[] for t=1:tmax]
    encoding_idxs = [[] for t=1:tmax]
    check_idxs = [[] for t=1:tmax-1]
    i = 0
    for t=1:tmax
        if t==1 && log_state==2 
	    jmax=1
	else
	    jmax = 2^(t-1)-1
	end
	if t==1 || log_state%2 != t%2 # input errors
            input_idxs[t] = [(i+j*(tmax-t+1)) .+ (1:tmax-t+1) for j=0:jmax]
	    i = input_idxs[t][end][end]
	end
	encoding_idxs[t] = [(i+j*(2*(tmax-t)+1)) .+ (1:2*(tmax-t)+1) for j=0:2^t-1]
	
	if t==1
	    gate_idxs[t] = [vcat([input_idxs[t][i][k] for i=1:length(input_idxs[t])],  [encoding_idxs[t][i][k] for i=1:2]) for k=1:tmax-t+1]
	else
	    if t%2==log_state%2 # only the left leg, no input leg
	        idxs = [[check_idxs[t-1][j][k+tmax-t+1], encoding_idxs[t][2*j-1][k],encoding_idxs[t][2*j][k]] for k=1:tmax-t+1, j=1:2^(t-1)]
	    else
	        idxs = [[check_idxs[t-1][j][k+tmax-t+1],input_idxs[t][j][k], encoding_idxs[t][2*j-1][k],encoding_idxs[t][2*j][k]] for k=1:tmax-t+1, j=1:2^(t-1)]
	    end
	    gate_idxs[t] = reshape(idxs, length(idxs))
	end

	if t<tmax
	    i = encoding_idxs[t][end][end]
	    check_idxs[t] = [vcat(idxs[tmax-t+2:end], i+(tmax-t)*(j-1) .+ (1:(tmax-t))) for (j,idxs) in enumerate(encoding_idxs[t])]
	    i = check_idxs[t][end][end]
	end

    end
    order = []
    for t=tmax:-1:1
        push!(order, [arr[end-1:end] for arr in gate_idxs[t]]...)
	push!(order, [arr[1:end-2] for arr in gate_idxs[t]]...)
	if t>1
	    push!(order, [arr[1:end÷2] for arr in check_idxs[t-1]]...)
	end
    end
    if level
        last_idx = encoding_idxs[end][end][1]
        encoding_idxs[end] = [[encoding_idxs[end][i][1], last_idx + i] for i=1:length(encoding_idxs[end])]
    else # this is the last level, so now there's an open logical
        input_idxs[1][1] = vcat([-1], input_idxs[1][1])
    end
    
    (idxs = [input_idxs, encoding_idxs, check_idxs, gate_idxs], order = order)
end

# for dimension 2 (classical)
function contract_error(::Val{:closed}, probs, bitflip::Bool, node; apply_error::Bool = true)
    if bitflip && apply_error # use opposite order for probabilities
        prob_node = probs[end:-1:1]
    else # use standard order
        prob_node = probs
    end
    ncon!([node, prob_node], [vcat([-i for i=1:ndims(node)-1], [1]), [1]])
end

# for dimension 4 (quantum)
function contract_error(::Val{:closed}, probs, err_ind::Int, node; apply_error::Bool = true)
    ncon!([node, probs[PAULI_TENSOR[err_ind,:]]], [vcat([-i for i=1:ndims(node)-1], [1]), [1]])
end

# dimension 2 (classical)
function contract_error(::Val{:open}, probs, bitflip::Bool, node; apply_error::Bool = true)
    if bitflip && apply_error # use opposite order for probabilities
        prob_node = [0 probs[2]; probs[1] 0]
    else # use standard order
        prob_node = [probs[1] 0; 0 probs[2]]
    end
    ncon!([node, prob_node], [vcat([-i for i=1:ndims(node)-1],[1]),[1, -ndims(node)]])
end

# dimension 4 (quantum)
function contract_error(::Val{:open}, probs, err_ind::Int, node; apply_error::Bool = true)
    prob_node = zeros(4,4)
    for i=1:4
        prob_node[PAULI_TENSOR[err_ind,i],i] = probs[i]
    end
    ncon!([node, prob_node], [vcat([-i for i=1:ndims(node)-1],[1]),[1, -ndims(node)]])
end

export initialize_stacked_errors

export evaluate_top_error
function evaluate_top_error(probs, bitflip)
    if bitflip
        return probs[end:-1:1]
    else
        return probs
    end
end

function initialize_stacked_errors(stacked_nodes, error_probs, bitflips; level::Bool = true, error_leg=:open, apply_error::Bool = true, noisy_ancilla::Bool = true, measurement_error::Bool = true)
    tmax = length(stacked_nodes.bulk)
    # t, t, t-1
    errs = [[Array{Array}(undef, length(error_probs[1][i])) for i=1:tmax],
        [Array{Array}(undef, length(error_probs[2][i])) for i=1:tmax],
    	[Array{Array}(undef, length(error_probs[3][i])) for i=1:tmax-1]]

    for t=1:tmax
    	# input errors
        @threads for node_i=1:length(errs[1][t])
	    if t==1 && node_i==1 # special case
	        errs[1][t][node_i] = contract_error(Val(error_leg), error_probs[1][t][node_i], bitflips[1][t][node_i], stacked_nodes.input; apply_error=apply_error)
	    else
	        errs[1][t][node_i] = contract_error(Val(error_leg), error_probs[1][t][node_i], bitflips[1][t][node_i], stacked_nodes.bulk[t][:input]; apply_error = apply_error)
	    end
	end

	# encoding gate errors
	@threads for node_i=1:length(errs[2][t])
	    if (noisy_ancilla && t < tmax) || (level && t==tmax) # leave uncontracted because "check nodes" come after
	        errs[2][t][node_i] = contract_error(Val(error_leg), error_probs[2][t][node_i], bitflips[2][t][node_i], stacked_nodes.bulk[t][:internal]; apply_error = apply_error)
	    else
	        errs[2][t][node_i] = contract_error(Val(error_leg), error_probs[2][t][node_i], bitflips[2][t][node_i], selectdim(stacked_nodes.bulk[t][:internal],tmax-t+2,1); apply_error = apply_error)
	    end
	end
	t==tmax && break
	
	# check gate errors
	@threads for node_i=1:length(errs[3][t])
	    errs[3][t][node_i] = contract_error(Val(error_leg), error_probs[3][t][node_i], bitflips[3][t][node_i], stacked_nodes.bulk[t+1][:internal]; apply_error = apply_error)
	end

    end

    if length(error_probs) > 3 && measurement_error # also have measurement errors
        @assert error_leg==:closed # because this is the ancilla side
	@assert typeof(bitflips[4][1][1])==Bool # only implemented classical version
	return vcat(vcat(vcat(errs...)...), [evaluate_top_error(eprob, bitflip && apply_error) for (eprob, bitflip) in zip(error_probs[4][1], bitflips[4][1])])
    else
	return vcat(vcat(errs...)...)
    end
end

end # module