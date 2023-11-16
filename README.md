To use the modules included in this repo, execute the following from the command line:

```export JULIA_LOAD_PATH="~/path/to/repo:"```

or, inside a Julia session, do:

```push!(LOAD_PATH, "~/path/to/repo")```

For example, inside the demo notebook, you would do
```push!(LOAD_PATH, "../")```

Then you can access the functions in these modules through the command `using <module name>`.