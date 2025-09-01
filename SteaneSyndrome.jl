#!/usr/licensed/julia/1.11/bin/julia

"""
Lower level functions for Steane syndrome extraction
"""
module SteaneSyndrome

using Base.Threads
using QuantumClifford

using StabilizerTree # for stabilizer and logical tableau
using SteanePrep # for trace_layer!, sample_sites!,...

PAULIS = Dict("X"=>[P"X", projectX!, :X], "Z"=>[P"Z", projectZ!, :Z])

export prep_logical_purification, syndrome_extraction

function stab_entropy(md::MixedDestabilizer, nqbits)
    return nqbits - md.rank
end

"""
mutual information between sites on idxs_a and idxs_b of the state described by mixed destabilizer tableau md
"""
function get_mutual_AB(md::MixedDestabilizer, idxs_a, idxs_b)
    idxs_not_ab = setdiff(1:nqubits(md), union(idxs_a,idxs_b))
    md_ab = traceout!(copy(md), idxs_not_ab)
    ent_a = stab_entropy(traceout!(copy(md_ab), setdiff(idxs_b, idxs_a)), length(idxs_a))
    ent_b = stab_entropy(traceout!(copy(md_ab), setdiff(idxs_a, idxs_b)), length(idxs_b))
    ent_ab = stab_entropy(md_ab, nqubits(md) - length(idxs_not_ab))
    #println("$(ent_a), $(ent_b)")
    return ent_a + ent_b - ent_ab
end

function get_encoding_tensor(stab, logicals)
    vcat(stab⊗S"I"[1:end-1], Stabilizer([logicals[1]⊗P"X", logicals[2]⊗P"Z"]))
end

"""
Note: gate will always be (H⊗H) * CNOT, but I chose not to hard-code it
"""
function prep_logical_purification(gate, tmax; input = P"Z", check::Bool = false)
    # prepare logical state, purify with reference
    stab = Stabilizer(track_stabilizer_generators(gate, tmax; init_pauli = input))

    L = nqubits(stab)
    
    logicals = track_logical_reps(gate, tmax)
    if all(iszero, logicals[1,1:end÷2]) # X-type logical is actually all Z's
        @assert all(iszero, logicals[2,end÷2+1:end]) # so Z-type logical is all X's
        offset = 1
    else
        @assert all(iszero, logicals[1,end÷2+1:end]) # so X-type logical is all X's
	@assert all(iszero, logicals[2,1:end÷2]) # so Z-type logical is all X's
	offset = 2
    end
	
    md = MixedDestabilizer(get_encoding_tensor(stab, Stabilizer(logicals)))

    md, stab, offset
end

"""
verify the state s using n_verify additional ancillas
"""
function verify_ancilla!(s, gate, tmax, rates; input = :Z, measure_first::Int = 1, fixed_n::Bool = false, two_qubit::Bool = false, log_state = :X, n_verify::Int = 2)
    L = nqubits(s)
    # Do one or both types of checks - technically I could do these in either order...

    measure_sites = [sample_sites(L*rates[4], [i*L+1:(i+1)*L;]; fixed_n = fixed_n) for i=1:n_verify]
    if two_qubit
        gate_sites = [sample_sites(L*rates[3], L; fixed_n = fixed_n) for i=1:n_verify]
   	gate_sites = [vcat([[j, j+i*L] for j in gate_sites[i]]...) for i=1:n_verify]
    else
        gate_sites = [sample_sites(2*L*rates[3], [1:L;i*L+1:(i+1)*L;]; fixed_n=fixed_n) for i=1:n_verify]
    end
    
    for i=1:n_verify
        s_anc,_,_,_ = prep_state_fixed_n(gate, tmax, rates; input = input, log_state =log_state, alternate = false, measure_first = measure_first,
	    two_qubit = two_qubit, fixed_n = fixed_n)
	s = s⊗s_anc
	transversal_cnot!(s, [[j,j+i*L] for j=1:L], (measure_first + i)%2+1; had = true)
	trace_layer!(s, gate_sites[i])
	trace_layer!(s, measure_sites[i])

	# measure every ancilla
	for j=i*L+1:(i+1)*L
	    projectZ!(s, j; keep_result = false)
	end
    end
    s
end

"""
n rounds of syndrome extraction
rates: 
    - rates[1]: input error
    - rates[2]: encoding gate error
    - rates[3]: check gate error
    - rates[4]: measurement error
If two_qubit = true, gate errors are 2 qubit errors
"""
function syndrome_extraction(md::MixedDestabilizer, stab::Stabilizer, gate::CliffordOperator, tmax::Int, rates; input = :Z, measure_first = [1,1], fixed_n::Bool = false, n_rounds::Int = 1, offset::Int = 1, break_on_mixed::Bool = false, check_first::Int = 1, two_qubit::Bool = false, n_verify::Int = 0)
    mutuals = zeros(Int, 2 * n_rounds + 1)
    L = nqubits(stab)

    # prepare + ancilla
    log_states = [:X, :Z]
    sys_idxs = get_system_idxs(tmax)
    gate_idxs = [[j,L+1+sys_idxs[j]] for j=1:L]

    if two_qubit
        gate_sites = [sample_sites(L*rates[3], L; fixed_n = fixed_n) for i=1:2*n_rounds]
	gate_sites = [vcat([gate_idxs[i] for i in gate_s]...) for gate_s in gate_sites]
    else
        gate_sites = [sample_sites(2*L*rates[3], [1:L;L+2:2*L+1;]; fixed_n = fixed_n)  for i=1:2*n_rounds]
    end
    erasure_sites = [gate_sites, [sample_sites(L*rates[4], [L+2:2*L+1;]; fixed_n = fixed_n) for i=1:2*n_rounds]]
    
    fully_mixed_round = 2*n_rounds + 1
    for i=1:2*n_rounds
    	s,_,_,_ = prep_state_fixed_n(gate, tmax, rates; input = input, log_state =log_states[(i+check_first-2)%2+1], alternate = false, measure_first = measure_first[(i+check_first-2)%2+1], two_qubit = two_qubit, fixed_n = fixed_n)
	if n_verify>0 # also check this ancilla against noisy ancilla(s) of the same depth
	    s = verify_ancilla!(s, gate, tmax, rates; input = input, log_state = log_states[(i+check_first-2)%2+1], measure_first = measure_first[(i+check_first-2)%2+1], two_qubit = two_qubit, fixed_n = fixed_n, n_verify = n_verify)
	end    
	md = md⊗s
	
	transversal_cnot!(md, gate_idxs, (i+check_first+offset-2)%2+1; had = true)
	
	trace_layer!(md, erasure_sites[1][i])

	# now measurement errors on ancillae
	trace_layer!(md, erasure_sites[2][i])

	# measure every ancilla
	for j=L+2:2*L+1
	    projectZ!(md, j; keep_result = false)
	end

	md, mutuals[i], fully_mixed = remove_mutual(md, L; n_copies = n_verify + 1)
	if fully_mixed && i < fully_mixed_round
	    fully_mixed_round = i
	    if break_on_mixed # already fully mixed, no point in continuing
	        break
	    end
	end
    end

    if fully_mixed_round <= 2*n_rounds && break_on_mixed
        return md, mutuals, fully_mixed_round
    else
        # perfect stabilizer measurements on system
    	for p=stab
            project!(md, p⊗P"I"; keep_result = false)
        end

    	# mutual info
    	mutuals[end] = get_mutual_AB(md, 1:L, [L+1])
	if fully_mixed_round <= 2*n_rounds
	    @assert mutuals[end]==0
	end
    	return md, mutuals, fully_mixed_round
    end
end

"""
syndrome extraction gadget
rates: 
    - rates[1]: input error
    - rates[2]: encoding gate error
    - rates[3]: check gate error
    - rates[4]: measurement error
If two_qubit = true, gate errors are 2 qubit errors
"""
function syndrome_extraction(gate, tmax, rates; input = "Z", measure_first = [1,1], fixed_n::Bool = false, check::Bool = false, break_on_mixed::Bool = false, n_rounds::Int = 1, check_first::Int = 1, two_qubit::Bool = false, n_verify::Int = 1)
    md, stab, offset = prep_logical_purification(gate, tmax; input = PAULIS[input][1], check = check)
    syndrome_extraction(md, stab, gate, tmax, rates; input = PAULIS[input][3], measure_first = measure_first, fixed_n = fixed_n, n_rounds = n_rounds, offset = offset, break_on_mixed = break_on_mixed, check_first = check_first, two_qubit = two_qubit, n_verify = n_verify)
end

"""
Get mutual info between system and reference, then remove ancillas
"""
function remove_mutual(md, L; n_copies::Int = 1)
    bg = bigram(md; clip = true) # put in clipped gauge
    sys_stabs = findall(r->r[2]<=L+1, eachrow(bg))
    @assert size(bg,1) - length(sys_stabs)==L * n_copies # since ancillae should be pure now
    if isempty(sys_stabs)
        # no way to ever get back the mutual information
        return one(MixedDestabilizer, 0, L+1), 0, true
    else
        mutual = count(r->r[1]<=L && r[2]==L+1, eachrow(bg[sys_stabs,:]))
    	md = MixedDestabilizer(stabilizerview(md)[sys_stabs,1:L+1])
    	return md, mutual, false
    end
end

end # module