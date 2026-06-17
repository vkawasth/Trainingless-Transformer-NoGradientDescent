# =============================================================================
# PART 6: TRANSFORMER-SPECIFIC INSTRUCTION LOWERING
# =============================================================================

function compile_instruction!(ctx::FkCompilerContext,
                               inst::SerreCascade)::Expr
    v = fresh!(ctx, "serre")
    bind!(ctx, inst.result, v)
    quote
        $(v) = compute_serre_cascade($(inst.teacher_checkpoint),
                                      $(inst.l_start),
                                      $(inst.n_operators))
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::CascadeInitStudent)::Expr
    cascade = fetch(ctx, inst.cascade)
    v = fresh!(ctx, "student")
    bind!(ctx, inst.result, v)
    quote
        $(v) = init_student_from_cascade($(cascade),
                                          $(inst.proj_subspace),
                                          $(inst.epsilon))
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::HessianSaddleExit)::Expr
    model = fetch(ctx, inst.model)
    v = fresh!(ctx, "saddle")
    bind!(ctx, inst.result, v)
    quote
        $(v) = compute_saddle_exit($(model),
                                    $(inst.n_iter),
                                    $(inst.n_batches),
                                    $(inst.line_points))
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::EtaleSheetAssigner)::Expr
    model = fetch(ctx, inst.model)
    jac = fetch(ctx, inst.jacobian_chain)
    v = fresh!(ctx, "sheet")
    bind!(ctx, inst.result, v)
    quote
        $(v) = assign_etale_sheet($(model), $(jac))
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::ApplySheetMask)::Expr
    model = fetch(ctx, inst.model)
    mask = fetch(ctx, inst.mask)
    v = fresh!(ctx, "masked")
    bind!(ctx, inst.result, v)
    quote
        $(v) = apply_sheet_mask($(model), $(mask))
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::ParametricPump)::Expr
    model = fetch(ctx, inst.model)
    v = fresh!(ctx, "pumped")
    bind!(ctx, inst.result, v)
    quote
        $(v) = parametric_pump($(model),
                                $(inst.k_iterations),
                                $(inst.eta_MF))
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::LevenbergMarquardtSolver)::Expr
    model = fetch(ctx, inst.model)
    v = fresh!(ctx, "lm")
    bind!(ctx, inst.result, v)
    quote
        $(v) = levenberg_marquardt($(model),
                                    $(inst.n_iters),
                                    $(inst.n_hvp),
                                    $(inst.μ_start),
                                    $(inst.μ_min))
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::EmbeddingRelaxationSolve)::Expr
    model = fetch(ctx, inst.model)
    v = fresh!(ctx, "relaxed")
    bind!(ctx, inst.result, v)
    quote
        $(v) = solve_embedding_relaxation($(model),
                                           $(QuoteNode(inst.method)),
                                           $(inst.tolerance))
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::WeightAssembler)::Expr
    cascade = fetch(ctx, inst.cascade)
    saddle = fetch(ctx, inst.saddle)
    sheet = fetch(ctx, inst.sheet)
    pumped = fetch(ctx, inst.pumped)
    lm = fetch(ctx, inst.lm)
    relaxed = fetch(ctx, inst.relaxed)
    v = fresh!(ctx, "weights")
    bind!(ctx, inst.result, v)
    quote
        $(v) = assemble_weights($(cascade), $(saddle), $(sheet),
                                 $(pumped), $(lm), $(relaxed))
    end
end

function compile_instruction!(ctx::FkCompilerContext,
                               inst::ValidateCompiledModel)::Expr
    weights = fetch(ctx, inst.compiled)
    v = fresh!(ctx, "validation")
    bind!(ctx, inst.result, v)
    quote
        $(v) = validate_model($(weights),
                               $(inst.validation_corpus),
                               $(inst.teacher_loss))
        @assert $(v).loss < $(inst.teacher_loss) / 10
    end
end