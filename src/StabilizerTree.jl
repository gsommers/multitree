#!/usr/licensed/julia/1.11/bin/julia

"""
Track stabilizers and logicals on a binary tree with arbitrary gate
"""
module StabilizerTree

using QuantumClifford

#= Tracking operator spreading from root or fresh leaf =#
export track_operator_spreading, track_fresh_stabilizers, track_stabilizer_generators, track_logical_reps, track_substabilizers, prepare_error_basis

function track_operator_spreading(cliff, pauli, tmax; fresh::Bool = true)
    tabs = Array{Array}(undef, tmax)
    if fresh
        # feed in fresh, on right
	tabs[1] = Bool.(mod.(cliff * [0,pauli[1],0,pauli[2]], 2))
    else
        # feed in on "logical leg"
	tabs[1] = Bool.(mod.(cliff * [pauli[1],0,pauli[2],0], 2))
    end

    image = [tabs[1][[1,3]], tabs[1][[2,4]]]
    for t=2:tmax
        image = vcat([inflate_pauli(cliff, pauli) for pauli in image]...)
	bits = vcat(image...)
	tabs[t] = vcat(bits[1:2:end], bits[2:2:end])
    end

    tabs
end

# track the operators fed in on the right, on fresh branches, in the given levels
function track_fresh_stabilizers(gate, tmax, levels; init_pauli = P"Z")
    paulis = track_operator_spreading(stab_to_gf2(gate.tab)', Bool.(init_pauli.xz), tmax; fresh = true)

    # only get the stabilizers associated with the levels t
    stabs = [zeros(Bool, 2^(t-1), 2^(tmax+1)) for t=levels]
    for (t_i, t)=enumerate(levels)
        tp = tmax -t + 1
	for i=1:2^(t-1)
	    stabs[t_i][i, ((i-1)*2^tp+1):i*2^tp] = paulis[tp][1:end÷2]
	    stabs[t_i][i, (end÷2+(i-1)*2^tp+1):(end÷2+i*2^tp)] = paulis[tp][end÷2+1:end]
	end
    end
    stabs
end

# get the parity check matrix for the tree code with one logical
function track_stabilizer_generators(gate, tmax; init_pauli = P"Z")
    stabs = track_fresh_stabilizers(gate, tmax, 1:tmax; init_pauli = init_pauli)
    vcat(stabs[end:-1:1]...)
end

# get the stabilizer tableaus up to depth t, for t=1:tmax
function track_substabilizers(gate, tmax; init_pauli = P"Z")
    stab = Stabilizer(track_stabilizer_generators(gate, tmax; init_pauli = init_pauli))
    starts = vcat([0], cumsum([2^(tmax-t) for t=1:tmax-1]))
    vcat([stab[vcat([(1:2^(t1-t)) .+ starts[t] for t=1:t1]...),1:2^t1] for t1=1:tmax-1], [stab])
end

# get the logical X and Z reps for tree code with k=1
function track_logical_reps(gate, tmax)
    cliff = stab_to_gf2(gate.tab)'
    logicals = zeros(Bool, 2, 2^(tmax+1))
    for (i, pauli) in enumerate([Bool[1,0], Bool[0,1]])
        logicals[i,:] = track_operator_spreading(cliff, pauli, tmax; fresh = false)[end]
    end
    logicals
end

# apply inflation rule to input pauli, with I on right qubit
function inflate_pauli(cliff, pauli)
    inflated = Bool.(mod.(cliff * [pauli[1],0,pauli[2],0], 2))
    [inflated[[1,3]], inflated[[2,4]]]
end

function prepare_error_basis(gate, tmax::Int)
    cliff = stab_to_gf2(gate.tab)'

    paulis = [Array{Stabilizer}(undef, tmax), Array{Stabilizer}(undef, tmax)]
    for (fresh_i, fresh)=enumerate([false, true])
        bases = [track_operator_spreading(cliff, pauli, tmax; fresh = fresh) for pauli = [Bool[1,0], Bool[0,1]]]
	paulis[fresh_i] = [all_generators(Stabilizer(hcat(bases[1][tp], bases[2][tp])'))[[4,2,1,3]] for tp=tmax:-1:1]
    end
    paulis
end

end # module