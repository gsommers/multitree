#!/usr/licensed/julia/1.11/bin/julia
module SteanePrep

using Base.Threads
using QuantumClifford
using StatsBase

using StabilizerTree # for stabilizer and logical tableau

export sample_sites
# sample out of the region `sample_from`, either fixed number or iid
function sample_sites(num_e, sample_from::Array; fixed_n::Bool = true)
    if fixed_n
        sites = sample(sample_from, Int(round(num_e)), replace = false)
    else
        sites = sample_from[findall(el->el<=num_e/length(sample_from), rand(length(sample_from)))]
    end
    sites
end

# sample sites on a system of length L, either a fixed number or iid
function sample_sites(num_e, L::Int; fixed_n::Bool=true)
    sample_sites(num_e, [1:L;]; fixed_n = fixed_n)
end

"""
Functions to get erasure locations, gate locations, in version with or without resetting
"""
export get_gate_idxs, get_system_idxs, get_erasure_sites, get_check_idxs, get_encoding_idxs, label_qubits


function get_erasure_blocks(rates, tmax)
    encoding_sites = [[sample_sites(rates[2] * 2^t, 2^t; fixed_n = false) for i=1:2^(tmax-t)] for t=1:tmax]
    check_sites = [[[sample_sites(rates[3] * 2^t, 2^t; fixed_n = false) for i=1:2^(tmax-t)] for t=2:tmax] for j=1:2]
    measure_sites = [[[sample_sites(rates[4] * 2^t, 2^t; fixed_n = false) for i=1:2^(tmax-t-1)] for t=1:tmax-1] for j=1:2]
    input_sites = vcat([[sample_sites(rates[1] * 2, 2; fixed_n = false) for i=1:2^(tmax-1)]], [[sample_sites(rates[1] * 2^t, 2^t; fixed_n = false) for i=1:2^(tmax-t-1)] for t=1:tmax-1])
    [input_sites, encoding_sites, check_sites, measure_sites]
end

# turn 1d array of sites to be erased, into 2d array of spacetime locations
function get_erasure_sites(sites::Array, n_per_layer::Int, num_layers::Int)
    spacetime_sites = [Int[] for i=1:num_layers]
    for site in sites
        t = (site-1)÷n_per_layer+1
	push!(spacetime_sites[t], (site-1)%n_per_layer+1)
    end
    spacetime_sites
end

function get_erasure_sites(n_layers::Int, n_per_layer::Int, r; fixed_n::Bool = false)
    sites = sample_sites(r * n_layers * n_per_layer, n_layers * n_per_layer; fixed_n = fixed_n)
    get_erasure_sites(sites, n_per_layer, n_layers)
end

"""
Get erasure locations, for version where I reset qubits
Can be fixed n or not, with (possibly different) rate for each type of error
"""
function get_erasure_sites(rates, tmax; fixed_n::Bool = true, two_qubit::Bool = false)
    if length(rates)==1
        rates = fill(rates[1], 4)
    end
    input_sites = get_erasure_sites(tmax + 1, 2^(tmax-1), rates[1]; fixed_n = fixed_n)

    # now I just need to fix up the first layer - where everything's an input
    input_sites = vcat([vcat(input_sites[1], input_sites[2] .+ 2^(tmax-1))], input_sites[3:end])

    if two_qubit # either erase both qubits involved in a gate, or neither
        encoding_sites = get_erasure_sites(tmax, 2^(tmax-1), rates[2]; fixed_n = fixed_n)
	check_sites = get_erasure_sites(tmax-1, 2^(tmax-1), rates[3]; fixed_n = fixed_n)
	gate_idxs = get_gate_idxs(1, tmax)
	encoding_sites[1] = vcat([gate_idxs[i] for i in encoding_sites[1]]...)
	for t=2:tmax
	    gate_idxs = get_gate_idxs(t, tmax)
	    encoding_sites[t] = vcat([gate_idxs[i] for i in encoding_sites[t]]...)
	    check_sites[t-1] = vcat([gate_idxs[i] for i in check_sites[t-1]]...)
	end
    else
        encoding_sites = get_erasure_sites(tmax, 2^tmax, rates[2]; fixed_n = fixed_n)
        check_sites = get_erasure_sites(tmax-1, 2^tmax, rates[3]; fixed_n = fixed_n)
    end
    measure_sites = get_erasure_sites(tmax -1, 2^(tmax -1), rates[4]; fixed_n = fixed_n)
    
    erasure_sites = [input_sites, encoding_sites, check_sites, measure_sites]
end

function get_gate_idxs(t, tmax)
    return reshape([(j+i) .+ [0, 2^(t-1)] for i=1:2^(t-1), j=0:2^t:2^tmax-1],2^(tmax-1))
end

function get_system_idxs(tmax)
    system_idxs = [1]
    for t=1:tmax
        system_idxs = vcat(2 .* system_idxs .- 1, 2 .* system_idxs)
    end
    system_idxs
end

#= Transversal "check" gates with ancillas =#
export transversal_cnot!, trace_layer!

function trace_layer!(s, idxs, sites)
    if !isempty(sites)
        traceout!(s, idxs[sites])
    end
    s
end

# trace out the qubits at sites
function trace_layer!(s, sites)
    if !isempty(sites)
        traceout!(s, sites)
    end
    s
end

function transversal_cnot!(s, sites, measure_type; had::Bool = true)
    for site_pair in sites
        if measure_type==1 # Z checks
	    apply!(s, sCNOT(site_pair[1], site_pair[2]))
	else # X checks
	    apply!(s, sCNOT(site_pair[2], site_pair[1]))
	    if had
	        apply!(s, sHadamard(site_pair[2]))
	    end
	end
    end
    s
end

function pair_blocks(stabs, erasure_sites, to_measure; check_gate::Bool = true, had::Bool = true, input =:Z, control::Int = 1, measure_type::Int = 1)

    sites = [[i,nqubits(stabs[1])+i] for i=1:nqubits(stabs[1])]
    if !check_gate
        s = stabs[control]⊗MixedDestabilizer(one(Stabilizer,nqubits(stabs[control]), basis =input))
    else
    	s = stabs[control]⊗stabs[control%2+1]

    	transversal_cnot!(s, sites, measure_type; had = had)

    	# errors associated with check gates
    	trace_layer!(s, 1:nqubits(s), erasure_sites[1])

    	# errors associated with measurement
    	trace_layer!(s, nqubits(s)÷2+1:nqubits(s), erasure_sites[2])

    	# measure
    	for i=nqubits(s)÷2+1:nqubits(s)
            projectZ!(s, i; keep_result = false)
        end
    	if input !=:Z
            reset_qubits!(s, one(Stabilizer,nqubits(s)÷2, basis=input), nqubits(s)÷2+1:nqubits(s))
        end
    end
    
    # get entropy of state, following perfect stabilizer measurements
    ent = system_entropy!(copy(s), to_measure,nqubits(s); check=true)

    # reorder qubits
    # s.tab = s.tab[:,vcat(sites...)]
    s, ent
end
    

"""
Prepare logical state 0 or +.
"""
export prep_state_fixed_n, prep_state_blocks, prep_state_blocks2

"""
Version where I reset/feed back in measured qubits to keep a constant number,
and try different versions of pairing blocks
"""
function prep_state_blocks2(gate, tmax, rates, to_measure; input = :Z, log_state = :Z, measure_first::Int=1, erasure_sites = nothing, alternate::Bool = true)
    my_s = MixedDestabilizer(one(Stabilizer,1,basis=log_state)⊗one(Stabilizer,1,basis=input))

    entropies = [zeros(Int,2,2^(tmax-t-1)) for t=1:tmax-1]
    wirings = [zeros(Int,2^(tmax-t-1)) for t=1:tmax-1]
    
    # erasure locations of each type, fixed fraction/layer or iid
    if isnothing(erasure_sites)
        erasure_sites = get_erasure_blocks(rates, tmax)
    end

    big_gate = gate
    stabs = [deepcopy(my_s) for i=1:2^(tmax-1)]
    input_sites = 1:2
    gate_idxs = get_gate_idxs(1,1)
    # up to depth tmax...
    for t=1:tmax

    	# Step 0: erasure errors on input legs
    	for i=1:length(stabs)
	    trace_layer!(stabs[i], input_sites, erasure_sites[1][t][i])
    	end
	
    	# Step 1: encoding gates
    	for i=1:length(stabs)
            apply!(stabs[i], big_gate, vcat(gate_idxs...))
	    trace_layer!(stabs[i], 1:2^t, erasure_sites[2][t][i])
    	end

	if t==tmax
	    break
	end
	
       	if alternate
	    parity = (t+measure_first)%2+1
	else # measure same basis each time (for use in syndrome measurement)
	    parity = measure_first
	end
        stabs, entropies[t], wirings[t] = block_layer(t, to_measure[t], erasure_sites, stabs; input = input, parity = parity)
	big_gate = big_gate⊗big_gate
	gate_idxs = get_gate_idxs(t+1, t+1)
	input_sites = [idx[2] for idx in gate_idxs]
    end

    if length(to_measure)==tmax
        return stabs[end], system_entropy!(copy(stabs[end]), to_measure[end], 2^tmax), entropies, wirings, erasure_sites
    else
        return stabs[end], entropies, wirings, erasure_sites
    end
end

# one layer of optimizing block pairs
function block_layer(t, to_measure, erasure_sites, stabs; input = :Z, parity = 1)

    # Step 2: CNOTs with ancillas
    gate_idxs = get_gate_idxs(t+1,t+1)
    input_sites = [idx[2] for idx in gate_idxs]

    # try four different ways of pairing up
    new_stabs = Array{MixedDestabilizer}(undef, length(stabs)÷2)
    entropies = zeros(Int,2, length(stabs)÷2)
    wirings = ones(Int, length(stabs)÷2)
    
    @threads for i=1:length(stabs)÷2
        success = false
        for j=1:4
	    stab, ent = pair_blocks(stabs[2*i-1:2*i], [erasure_sites[3][1][t][i], erasure_sites[4][1][t][i]], to_measure; check_gate = (j<=2), measure_type = parity, control = (j+1)%2+1, had = true, input = input)
	    if ent==0
	        if j<=2
		    new_stabs[i], entropies[2,i] = pair_blocks(stabs[2*i-1:2*i], [erasure_sites[3][2][t][i], erasure_sites[4][2][t][i]], to_measure; check_gate = true, measure_type = parity, control = (j+1)%2+1, had = true, input = input)
		else
		    new_stabs[i] = stab
		end
		wirings[i] = j
		success = true
		break
	    end
	end

	if !success # all bad, default to first wiring
	    new_stabs[i], entropies[2,i] = pair_blocks(stabs[2*i-1:2*i], [erasure_sites[3][2][t][i], erasure_sites[4][2][t][i]], to_measure; check_gate = true, measure_type = parity, control = 1, had = true, input = input)
	    entropies[1,i] = 1
	end
	
    end

    new_stabs, entropies, wirings
end

"""
Alternate version where I dynamically rewire and try all wirings every time (rather than stopping on success)
"""
function prep_state_blocks(gate, tmax, rates, to_measure; input = :Z, log_state = :Z, measure_first::Int=1, erasure_sites = nothing, alternate::Bool = true)
    my_s = MixedDestabilizer(one(Stabilizer,1,basis=log_state)⊗one(Stabilizer,1,basis=input))

    entropies = [zeros(Int,6,2^(tmax-t-1)) for t=1:tmax-1]
    wirings = [zeros(Int,2^(tmax-t-1)) for t=1:tmax-1]
    # erasure locations of each type, fixed fraction/layer or iid
    if isnothing(erasure_sites)
        erasure_sites = get_erasure_blocks(rates, tmax)
    end

    big_gate = gate
    stabs = [deepcopy(my_s) for i=1:2^(tmax-1)]
    # up to depth tmax...
    input_sites = 1:2
    gate_idxs = get_gate_idxs(1,1)
    for t=1:tmax
    	# Step 0: erasure errors on input legs
	for i=1:length(stabs)
	    trace_layer!(stabs[i], input_sites, erasure_sites[1][t][i])
	end
	
        # Step 1: encoding gates
	for i=1:length(stabs)
	    apply!(stabs[i], big_gate, vcat(gate_idxs...))
	    trace_layer!(stabs[i], 1:2^t, erasure_sites[2][t][i])
	end

	t==tmax && break
	big_gate = big_gate⊗big_gate
	
	# Step 2: CNOTs with ancillas
	gate_idxs = get_gate_idxs(t+1,t+1)
	input_sites = [idx[2] for idx in gate_idxs]
    	if alternate
	    parity = (t+measure_first)%2+1
	else # measure same basis each time (for use in syndrome measurement)
	    parity = measure_first
	end

	# try four different ways of pairing up
	new_stabs = Array{MixedDestabilizer}(undef, 4, length(stabs)÷2)
	for i=1:length(stabs)÷2
	    @threads for j=1:4
	        new_stabs[j,i], entropies[t][j,i] = pair_blocks(stabs[2*i-1:2*i], [erasure_sites[3][1][t][i], erasure_sites[4][1][t][i]], to_measure[t]; check_gate = (j<=2), measure_type = parity, control = (j+1)%2+1, had = true, input = input)
	    end
	    # now redo these to get the actual erasure pattern
	    for j=1:2
	        new_stabs[j,i], entropies[t][j+4,i] = pair_blocks(stabs[2*i-1:2*i], [erasure_sites[3][2][t][i], erasure_sites[4][2][t][i]], to_measure[t]; check_gate = true, measure_type = parity, control = (j+1)%2+1, had = true, input = input)
	    end
	end
	#stabs = [new_stabs[5-argmin(entropies[t][4:-1:1,j]),j] for j=1:size(entropies[t],2)
	wirings[t] = [argmin(entropies[t][1:4,j]) for j=1:size(entropies[t],2)]
	stabs = new_stabs[argmin(entropies[t][1:4,:],dims=1)]
    end

    final_ent = system_entropy!(copy(stabs[end]), to_measure[end], 2^tmax)

    stabs[end], final_ent, entropies, wirings, erasure_sites
end

"""
Default wiring, reset/feed back in measured qubits to keep a constant number
"""
function prep_state_fixed_n(gate, tmax, rates; input = :Z, log_state = :Z, measure_first::Int=1, fixed_n::Bool = true, erasure_sites = nothing, clusters = nothing, input_sites = nothing, ancilla_stabs = nothing, alternate::Bool = true, two_qubit::Bool = false)
    s = MixedDestabilizer(one(Stabilizer, 2^tmax, basis=log_state))
    record = zeros(Int, tmax-1)
    entropies = zeros(Int, tmax+1)

    # erasure locations of each type, fixed fraction/layer or iid
    if !isnothing(clusters) # this is the subtrees version
        subtree = true
        erasure_sites = sample_subtree_sites(clusters, input_sites, tmax, rates; fixed_n = fixed_n)
    else # fixed per layer, or iid
        subtree = false
	if isnothing(erasure_sites)
            erasure_sites = get_erasure_sites(rates, tmax; fixed_n = fixed_n, two_qubit = two_qubit)
        end
    end
    # up to depth tmax...
    gate_idxs = get_gate_idxs(1, tmax)
    big_gate = tensor_pow(gate, 2^(tmax-1))
    s_reset = one(Stabilizer, 2^(tmax-1), basis=input)
    reset_qubits!(s, s_reset, 2:2:2^tmax)
    input_sites = 1:2^tmax
    for t=1:tmax

    	# Step 0: erasure errors on input legs
	trace_layer!(s, input_sites, erasure_sites[1][t]; subtree = subtree)
	
        # Step 1: encoding gates
	apply!(s, big_gate, vcat(gate_idxs...))
	trace_layer!(s, 1:2^tmax, erasure_sites[2][t]; subtree = subtree)

	t==tmax && break
	# Step 2: CNOTs with ancillas
	gate_idxs = get_gate_idxs(t+1, tmax)
    	if alternate
	    parity = (t+measure_first)%2+1
	else # measure same basis each time (for use in syndrome measurement)
	    parity = measure_first
	end
	
	transversal_cnot!(s, gate_idxs, parity; had = isnothing(ancilla_stabs))
	trace_layer!(s, 1:2^tmax, erasure_sites[3][t]; subtree = subtree)
	
	# Step 3: measure ancillas
	input_sites = [site_pair[2] for site_pair in gate_idxs]
	trace_layer!(s, input_sites, erasure_sites[4][t]; subtree = subtree)

	# special case where I pass in the stuff I actually want to measure
	if !isnothing(ancilla_stabs)
	    for p in ancilla_stabs[t]
	        pre_rank = s.rank
		_, anticom, _ = project!(s, p; keep_result = true)
	        if anticom==0
		    record[t] += 1
		end
	    end
	else
	    for site in input_sites
	        pre_rank = s.rank
	    	projectZ!(s, site; keep_result = false)
	    	if s.rank > pre_rank
	            record[t] += 1
	        end
	    end
	end
	entropies[t] = nqubits(s) - s.rank
	# reset inputs
	reset_qubits!(s, s_reset, input_sites)
    end
    entropies[tmax] = nqubits(s) - s.rank
    s, entropies, record, erasure_sites
end

"""
Run the fully butterfly network circuit, including perfect stabilizer measurement at end
"""
function prep_state_fixed_n(gate, tmax, rates, to_measure; input = :Z, log_state = :Z, measure_first::Int=1, fixed_n::Bool = true, erasure_sites = nothing, clusters = nothing, input_sites = nothing, ancilla_stabs = nothing, two_qubit::Bool = false)
    s, entropies, record, erasure_sites = prep_state_fixed_n(gate, tmax, rates; input = input, log_state = log_state, measure_first = measure_first, fixed_n = fixed_n, erasure_sites = erasure_sites, clusters = clusters, input_sites = input_sites, ancilla_stabs = ancilla_stabs, two_qubit = two_qubit)
    entropies[end] = system_entropy!(copy(s), to_measure, 2^tmax; check=false)
    s, entropies, record, erasure_sites
end


#= Functions to set up and perform perfect stabilizer measurements at end =#
export system_entropy!

# set up the measurements to do as well as the gates

function system_entropy!(s, to_measure, L::Int; check::Bool = true)
    system_entropy!(s, to_measure, [1:L;]; check = check)
end

# do a perfect measurement of all the stabilizers (not the logical)
function system_entropy!(s, to_measure, system_qubits::Array; check::Bool = true)
    if check
        nqubits(s)==nqubits(to_measure[1]) ? nothing : println("$(nqubits(to_measure[1])), $(nqubits(s))")
    end
    for p in to_measure
        project!(s, p; keep_result = false)
    end
    # is my logical state pure?
    ent = entanglement_entropy(s, system_qubits, Val(:rref))
    @assert ent <= 1
    ent
end

end # module