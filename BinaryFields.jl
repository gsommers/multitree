#!/usr/licensed/julia/1.7/bin/julia
module BinaryFields

using Nemo

export nemo_to_bool, bool_to_nemo, match_syndrome

#= Conversion between nemo GF(2) matrix and Matrix{Bool} =#
function bool_to_nemo(V)
    @assert typeof(V[1,1])==Bool
    matrix(Nemo.GF(2), V)
end

function nemo_to_bool(V::gfp_mat)
    A = zeros(Bool, size(V)...)
    for i=1:size(V,2), j=1:size(V,1)
        A[j,i] = (V[j,i]==1)
    end
    A
end

# given a syndrome, find a Pauli string with that syndrome
function match_syndrome(par_check::gfp_mat, syndrome::gfp_mat)
    check, err = can_solve_with_solution(par_check, syndrome)
    @assert check
    err
end

function match_syndrome(par_check::gfp_mat, pauli_tab::AbstractMatrix)
    syndrome = par_check * bool_to_nemo(pauli_tab')
    match_syndrome(par_check, syndrome)
end

end # module