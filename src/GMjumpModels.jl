# ==============================================================================
# Modele JuMP pour calculer la relaxation linéaire du 2SPA sur un objectif donne, activant eventuellement une ϵ-contrainte

function computeLinearRelax2SPA(  nbvar::Int,
                                  nbctr::Int,
                                  A::Array{Int,2},
                                  c1::Array{Int,1},
                                  c2::Array{Int,1},
                                  epsilon,
                                  obj::Int,
                                  instance::String,
)
  model = Model(GLPK.Optimizer)
  @variable(model, 0.0 <= x[1:nbvar] <= 1.0 )
  @constraint(model, [i=1:nbctr],(sum((x[j]*A[i,j]) for j in 1:nbvar)) == 1)
  if obj == 1
    @objective(model, Min, sum((c1[i])*x[i] for i in 1:nbvar))
    @constraint(model, sum((c2[i])*x[i] for i in 1:nbvar) <= epsilon)
  else
    @objective(model, Min, sum((c2[i])*x[i] for i in 1:nbvar))
    @constraint(model, sum((c1[i])*x[i] for i in 1:nbvar) <= epsilon)
  end
  optimize!(model)
  println("--->Termination status: ", termination_status(model))
  if termination_status(model) != OPTIMAL
    write_to_file(model,"model_with_bug_"*string(termination_status(model))*"_"*instance*".mps")
  end
  return objective_value(model), value.(x)
end