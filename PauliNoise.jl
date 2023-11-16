#!/usr/licensed/julia/1.7/bin/julia
"""
Pauli channels, mostly classical
"""
module PauliNoise

export propagate_error_pair, pauli_tensor, pauli_classical_tensor

export PAULI_TENSOR
const PAULI_TENSOR = [1 2 3 4; 2 1 4 3; 3 4 1 2; 4 3 2 1]

# tensor associated with inserting a Pauli (permutes I, X, Y, Z)
function pauli_tensor(probs::Array)
    [probs[ten_i] for ten_i in PAULI_TENSOR]
end

function pauli_tensor(i::Int)
    probs = zeros(Int,4)
    probs[i] = 1
    pauli_tensor(probs)
end

pauli_classical_tensor(probs) = [probs[1] probs[2]; probs[2] probs[1]]

function propagate_error_pair(error_probs, trans_mat)
    n = length(error_probs[1])
    input_vec = reshape([error_probs[1][i]*error_probs[2][j] for i=1:n,j=1:n], n^2)
    # propagate under gate
    output_vec = reshape(trans_mat * input_vec,(n,n))

    # marginalize probabilities: sum over columns or rows
    marginal_out = [dropdims(sum(output_vec, dims=2),dims=2), dropdims(sum(output_vec, dims=1),dims=1)]
end

end