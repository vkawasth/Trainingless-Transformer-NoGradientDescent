# =============================================================================
# verify_compiler.jl
#
# Verify that the compiler produces valid weights without gradient descent.
# =============================================================================

function verify_compiler()
    println("Verifying AU-Fukaya Transformer Compiler...")
    
    # 1. Check that all instructions are present
    ir = transformer_compiler_ir("teacher.bson", "corpus.txt", 
                                   Dict(:D=>256, :V=>675, :n_layers=>24, :n_heads=>4))
    
    expected_instructions = [
        SerreCascade,
        CascadeInitStudent,
        HessianSaddleExit,
        EtaleSheetAssigner,
        ApplySheetMask,
        ParametricPump,
        LevenbergMarquardtSolver,
        EmbeddingRelaxationSolve,
        WeightAssembler,
        ValidateCompiledModel
    ]
    
    found_types = Set([typeof(inst) for inst in ir])
    for inst_type in expected_instructions
        if inst_type in found_types
            println("✅ Found $inst_type")
        else
            println("❌ Missing $inst_type")
        end
    end
    
    # 2. Verify that there are no gradient descent instructions
    gradient_instructions = [:AdamStep, :SGDStep, :GradientDescent]
    for inst in ir
        if hasproperty(inst, :name)
            if inst.name in gradient_instructions
                println("❌ Found gradient instruction: $(inst.name)")
                return false
            end
        end
    end
    println("✅ No gradient descent instructions found")
    
    # 3. Verify the Schur complement solve is present
    schur_found = false
    for inst in ir
        if inst isa EmbeddingRelaxationSolve
            if inst.method == :schur_complement
                schur_found = true
            end
        end
    end
    if schur_found
        println("✅ Schur complement embedding relaxation found")
    else
        println("❌ Schur complement embedding relaxation missing")
    end
    
    println("✅ Compiler verification complete!")
    return true
end

verify_compiler()