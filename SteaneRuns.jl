#!/usr/licensed/julia/1.11/bin/julia
"""
Module containing master functions for determining failure probability
of state preparation gadget and syndrome extraction, fully heralded errors
"""
module SteaneRuns

using Base.Threads
using QuantumClifford
using StatsBase
using StabilizerTree # for stabilizer and logical tableau (but I always just do GS code)
using SteanePrep # lower-level state preparation functions
using KnillPrep

PAULIS = Dict("X"=>[P"X", projectX!, :X], "Z"=>[P"Z", projectZ!, :Z])

#= Master functions for preparing states at different erasure rates =#
export steane_erasures_reset, steane_erasures_blocks, knill_erasures_blocks, knill_erasures_reset

function steane_erasures_reset!(gate, tmax, rates, entropies, records; log_state =:Z, init_pauli = "Z", measure_first::Int=1, fixed_n::Bool = false)
    to_measure = Stabilizer(track_stabilizer_generators(gate, tmax,init_pauli=PAULIS[init_pauli][1]))[:, get_system_idxs(tmax)]
    
    for r_i=1:length(rates)
        @threads for num_i=1:size(entropies,2)
	    _, entropies[:,num_i,r_i], records[:,num_i, r_i] = prep_state_fixed_n(gate, tmax, rates[r_i], to_measure; log_state = log_state, input = PAULIS[init_pauli][3], measure_first = measure_first)
	end
    end
    entropies, records
end

"""
master function for version of state prep where I reset and feed back in.
For matching checks, measure_first should be 1 for |+>, and 2 for |0>
"""
function steane_erasures_reset(gates, tmax, idxs, rates; log_state = "X", init_pauli = "Z", num_samples::Int=1000, entropy_data = Dict{String,Any}("entropies"=>Dict(), "record"=>Dict()), measure_first::Int=1, fixed_n::Bool = true)
    entropies = zeros(Int, tmax+1,num_samples, length(rates), length(idxs))
    records = zeros(Int, tmax-1,num_samples, length(rates), length(idxs))
    if haskey(entropy_data, "stab-basis")
        @assert entropy_data["stab-basis"]==init_pauli
	@assert entropy_data["log-basis"]==log_state
	@assert entropy_data["first-meas"]==measure_first
	@assert entropy_data["fixed-n"]==fixed_n
    else
	entropy_data["stab-basis"]=init_pauli
	entropy_data["log-basis"]=log_state
	entropy_data["first-meas"]=measure_first
	entropy_data["fixed-n"] = fixed_n
    end

    for idx_i=1:length(idxs)
        steane_erasures_reset!(gates[idxs[idx_i]], tmax, rates, @view(entropies[:,:,:,idx_i]), @view(records[:,:,:,idx_i]); init_pauli = init_pauli, log_state = PAULIS[log_state][3], fixed_n = fixed_n, measure_first = measure_first)
    end

    # now turn into dictionary
    for (idx_i, idx) in enumerate(idxs)
        if !haskey(entropy_data["entropies"], idx)
	    entropy_data["entropies"][idx] = Dict()
	    entropy_data["record"][idx] = Dict()
	end
	for (r_i, r) in enumerate(rates)
	    # append to existing data
	    if haskey(entropy_data["entropies"][idx], r)
	        entropy_data["entropies"][idx][r] = hcat(entropy_data["entropies"][idx][r], entropies[:,:,r_i, idx_i])
		entropy_data["record"][idx][r] = hcat(entropy_data["record"][idx][r], records[:,:,r_i, idx_i])
	    else # new rate or idx
	        entropy_data["entropies"][idx][r] = entropies[:,:,r_i,idx_i]
	        entropy_data["record"][idx][r] = records[:,:,r_i,idx_i]
	    end
	end
    end
    entropy_data
end

function steane_erasures_blocks!(gate, tmax, rates, entropies, records, wirings; log_state =:Z, init_pauli = "Z", measure_first::Int=1)
    to_measure = [s[:,get_system_idxs(t)] for (t,s)=enumerate(track_substabilizers(gate, tmax,init_pauli=PAULIS[init_pauli][1]))]
    for t=1:tmax-1
        id_string = PauliOperator(zeros(Bool, 2*nqubits(to_measure[t])))
	to_measure[t] = Stabilizer([p⊗id_string for p in to_measure[t]])
    end
    
    for r_i=1:length(rates)
        @threads for num_i=1:size(entropies,1)
	    _, entropies[num_i,r_i], records[num_i, r_i], wirings[num_i, r_i],_ = prep_state_blocks2(gate, tmax, rates[r_i], to_measure; log_state = log_state, input = PAULIS[init_pauli][3], measure_first = measure_first, alternate = true)
	end
    end
    entropies, records
end

"""
Master function for preparing logical state of GS code, erasure errors, with dynamical rewiring.
For matching checks, measure_first should be 1 for |+>, and 2 for |0>
"""
function steane_erasures_blocks(gates, tmax, idxs, rates; log_state = "X", init_pauli = "Z", num_samples::Int=1000, entropy_data = Dict{String,Any}("entropies"=>Dict(), "record"=>Dict(), "wirings"=>Dict()), measure_first::Int=1)
    if haskey(entropy_data, "stab-basis")
        @assert entropy_data["stab-basis"]==init_pauli
	@assert entropy_data["log-basis"]==log_state
	@assert entropy_data["first-meas"]==measure_first
    else
	entropy_data["stab-basis"]=init_pauli
	entropy_data["log-basis"]=log_state
	entropy_data["first-meas"]=measure_first
    end
    master_blocks(gates, tmax, idxs, rates; entropy_data = entropy_data, f = steane_erasures_blocks!, num_samples = num_samples, init_pauli = init_pauli, extra_args = (log_state = PAULIS[log_state][3], measure_first = measure_first))
end

"""
This function is overly general because I also used it for preparing logical Bell states, Knill-style and Steane-style. Not included in this repo.
"""
function master_blocks(gates, tmax, idxs, rates; entropy_data = Dict{String,Any}("entropies"=>Dict(), "record"=>Dict(), "wirings"=>Dict()), f = steane_erasures_blocks!, extra_args = (), num_samples = 1000, init_pauli = "Z", reshape_dat::Bool = false, max_j::Int=4)
    entropies = zeros(Int, num_samples, length(rates), length(idxs))
    records = Array{Array}(undef,num_samples, length(rates), length(idxs))
    wirings = Array{Array}(undef,num_samples, length(rates), length(idxs))

    for idx_i=1:length(idxs)
        f(gates[idxs[idx_i]], tmax, rates, @view(entropies[:,:,idx_i]), @view(records[:,:,idx_i]), @view(wirings[:,:,idx_i]); init_pauli = init_pauli, extra_args...)
    end

    n = ndims(wirings[1,1,1][1])+1
    println(n)
    # now turn into dictionary
    for (idx_i, idx) in enumerate(idxs)
        if !haskey(entropy_data["entropies"], idx)
	    entropy_data["entropies"][idx] = Dict()
	    entropy_data["record"][idx] = Dict()
    	    entropy_data["wirings"][idx] = Dict()
	end
	for (r_i, r) in enumerate(rates)

	    # append to existing data
	    rec_dat = [cat([records[num_i,r_i,idx_i][t] for num_i=1:num_samples]...,dims=3) for t=1:tmax-1]
	    wiring_dat = [cat([wirings[num_i,r_i,idx_i][t] for num_i=1:num_samples]...,dims=n) for t=1:tmax-1]
	    if haskey(entropy_data["entropies"][idx], r)
	        entropy_data["entropies"][idx][r] = vcat(entropy_data["entropies"][idx][r], entropies[:,r_i, idx_i])
		if reshape_dat
		   entropy_data["record"][idx][r] = cat(entropy_data["record"][idx][r],reshape_record(rec_dat; max_j = max_j),dims=3)
		   entropy_data["wirings"][idx][r] = cat(entropy_data["wirings"][idx][r], reshape_wirings(wiring_dat; max_j = max_j), dims=3)
		else
		    entropy_data["record"][idx][r] = [cat(entropy_data["record"][idx][r][t], cat([records[num_i,r_i,idx_i][t] for num_i=1:num_samples]...,dims=3),dims=3) for t=1:tmax-1]
		    entropy_data["wirings"][idx][r] = [cat(entropy_data["wirings"][idx][r][t], cat([wirings[num_i,r_i,idx_i][t] for num_i=1:num_samples]...,dims=n),dims=n) for t=1:tmax-1]
		end
	    else # new rate or idx
	        entropy_data["entropies"][idx][r] = entropies[:,r_i,idx_i]
		if reshape_dat
		    entropy_data["record"][idx][r] = reshape_record(rec_dat; max_j = max_j)
		    entropy_data["wirings"][idx][r] = reshape_wirings(wiring_dat; max_j = max_j)
		else
	            entropy_data["record"][idx][r] = rec_dat
		    entropy_data["wirings"][idx][r] = wiring_dat
		end
	    end
	end
    end
    entropy_data
end

end # module