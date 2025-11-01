# multitree
Self-contained repository for multitree state preparation of the generalized Shor (GS) code, also called the Bell tree. Additional code and data available upon request; please contact <gsommers@princeton.edu>. 

To use the modules included in this repo (in the subdirectory `src/`), execute the following from the command line:

```export JULIA_LOAD_PATH="~/path/to/repo:"```

or, inside a Julia session, do:

```push!(LOAD_PATH, "~/path/to/repo")```

Then you can access the functions in these modules through the command `using <module name>`. 

External dependencies
---------------------
 - `QuantumClifford` (for stabilizer formalism)
 - `Nemo` (for finite fields)

The module `TensorExtensions.jl` is mostly copied from the code [here](https://www.tensors.net/code), a Julia implementation of the MATLAB function [NCON](https://arxiv.org/abs/1402.0939), modified to allow for alternative definitions of "multiplication" (see the functions `tropical_matrix_mult`, `normalize_mult`).

Code
-------------------
The modules contained in `src/` fall into three broad groups:

 - Functions for (stacked) probability passing of the GS code:
   * Higher-level functions: `BellPrepErrorDecoder.jl` for classical simulations of state prep (generating error patterns on the fly, `BellPrepSyndromeDecoder.jl` for decoding state prep given a syndrome (used on experimental data), `BellMemoryDecoder.jl` for decoding the Steane EC gadget. `QuantinuumRuns.jl` calls functions in `BellPrepSyndromeDecoder.jl` to process raw data for Quantinuum System H2.
   * Lower-level functions: `BellTreeFaults.jl` (propagating faults, initializing error nodes in the tensor network), `BellTreeTensors.jl` (basic data structures for the stacked decoder of state prep and probability passing decoder of EC; marginal updates of error propabilities), `StackedTree.jl` (tensor and index lists for stacked decoder), `PauliNoise.jl` (error tensors for Pauli channels, not specific to Bell tree)
 - Functions for optimal decoding of fully heralded errors:
   * Higher-level functions: `SteaneRuns.jl`.
   * Lower-level functions: `SteanePrep.jl` (for state preparation) and `SteaneSyndrome.jl` (for syndrome extraction/Steane EC)
 - General utilities
   - `StabilizerTree.jl` (track operators in binary tree with arbitrary gates)
   - `BinaryFields.jl` (boolean linear algebra)
   - `TensorExtensions.jl` (tensor network contraction)
   
Data
-------------------
The folder `dat/` contains data used to produce the figures in the main text:
 - `dat/heralded/` contains state preparation data for fully heralded errors (Fig. 3a)
 - `dat/unheralded/` contains state preparation data for unheralded errors, stacked decoder (Fig. 3b)
 - `dat/H2/erasures/` contains data from Quantinuum System H2, with added erasure errors (Fig. 4a)
 - `dat/H2/coherent/` contains data from Quantinuum System H2, with added heralded coherent errors (Fig. 4b)
 - `dat/memory/` contains data for one round and d rounds of syndrome measurement (Fig. 5)
