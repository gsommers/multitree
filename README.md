# multitree
Self-contained repository for multitree state preparation of the "Bell tree code." Additional code (generalizing to other gates, optimal decoder for fully heralded noise, syndrome extraction gadget) available upon request; please contact <gsommers@princeton.edu>. 

To use the modules included in this repo, execute the following from the command line:

```export JULIA_LOAD_PATH="~/path/to/repo:"```

or, inside a Julia session, do:

```push!(LOAD_PATH, "~/path/to/repo")```

Then you can access the functions in these modules through the command `using <module name>`.

External dependencies
---------------------
 - `QuantumClifford` (for stabilizer formalism)
  - `Nemo` (for finite fields)

The module `TensorExtensions.jl` is mostly copied from the code [here](https://www.tensors.net/code), a Julia implementation of the MATLAB function [NCON](https://arxiv.org/abs/1402.0939), modified to allow for alternative definitions of "multiplication" (see the functions `tropical_matrix_mult`, `normalize_mult`).
