# 1. Start Julia
julia

# 2. Include the files
julia> include("au_fukaya_engine.jl")
julia> include("au_fukaya_ir.jl")
julia> include("au_fukaya_ir_to_julia.jl")
julia> include("run_compiler.jl")

# 3. Run the compiler
julia> main()

# Or run directly:
julia> arch = Dict(:D=>256, :V=>675, :n_layers=>24, :n_heads=>4)
julia> weights = compile_transformer("corpus.txt", arch, "teacher.bson")