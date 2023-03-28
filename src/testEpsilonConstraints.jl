using JuMP, GLPK, Printf
struct NonFeasible <: Exception end 

include("GMdatastructures.jl")
include("GMjumpModels.jl")
include("GMgenerators.jl")
include("GMparsers.jl")

const instanceDir = "../SPA/instances/"
const verbose = true

function evaluerSolution(x::Vector{Float64}, c1::Array{Int,1}, c2::Array{Int,1})

    z1 = 0.0; z2 = 0.0
    for i in 1:length(x)
        z1 += x[i] * c1[i]
        z2 += x[i] * c2[i]
    end
    return round(z1, digits=2), round(z2, digits=2)
end

function nettoyageSolution!(x::Vector{Float64})

    nbvar=length(x)
    for i in 1:nbvar
        if     round(x[i], digits=3) == 0.0
                   x[i] = 0.0
        elseif round(x[i], digits=3) == 1.0
                   x[i] = 1.0
        else
                   x[i] = round(x[i], digits=3)
        end
    end
end

function main()
    formattedInstances::Vector{String} = [instance[4:end] for instance in readdir(instanceDir) if instance[end-3:end]==".txt"] 

    for fname in formattedInstances
        instanceWithBug = fname
        println(fname)
        tailleSampling = 6

        c1, c2, A = loadInstance2SPA(fname) # instance numerique de SPA
        nbctr = size(A,1)
        nbvar = size(A,2)
        nbobj = 2

        # structure pour les points qui apparaitront dans l'affichage graphique
        d = tListDisplay([],[], [],[], [],[], [],[], [],[], [],[], [],[])

        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        @printf("1) calcule les etendues de valeurs sur les 2 objectifs\n\n")

        # calcule la valeur optimale relachee de f1 seule et le point (z1,z2) correspondant
        f1RL, xf1RL = computeLinearRelax2SPA(nbvar, nbctr, A, c1, c2, typemax(Int), 1, fname) # opt fct 1
        minf1RL, maxf2RL = evaluerSolution(xf1RL, c1, c2)

        # calcule la valeur optimale relachee de f2 seule et le point (z1,z2) correspondant
        f2RL, xf2RL = computeLinearRelax2SPA(nbvar, nbctr, A, c1, c2, typemax(Int), 2, fname) # opt fct 2
        maxf1RL, minf2RL = evaluerSolution(xf2RL, c1, c2)

        verbose ? @printf("  f1_min=%8.2f ↔ f1_max=%8.2f (Δ=%.2f) \n",minf1RL, maxf1RL, maxf1RL-minf1RL) : nothing
        verbose ? @printf("  f2_min=%8.2f ↔ f2_max=%8.2f (Δ=%.2f) \n\n",minf2RL, maxf2RL, maxf2RL-minf2RL) : nothing


        # --------------------------------------------------------------------------
        # --------------------------------------------------------------------------
        @printf("2) calcule les generateurs par e-contrainte alternant minimiser z1 et z2\n\n")

        nbgen, L = calculGenerateurs(A, c1, c2, tailleSampling, minf1RL, maxf2RL, maxf1RL, minf2RL, d, fname)
    end
end
