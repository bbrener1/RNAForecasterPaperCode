using DiffEqFlux, DifferentialEquations
using JLD2
include("trainRNAForecaster.jl");
include("splicedDataPerturbationEffectPredictions.jl");

#read in the expression data for reference
using DelimitedFiles
spliced = readdlm("pancSplicedCountsHVGs.csv", ',')
unspliced = readdlm("pancUnsplicedCountsHVGs.csv", ',')

zeroPropSplicedGenes = Array{Float64}(undef, size(spliced)[1])
for i=1:size(spliced)[1]
    zeroPropSplicedGenes[i] = length(findall(x->x ==0, spliced[i,:]))/size(spliced)[2]
end
zeroPropUnsplicedGenes = Array{Float64}(undef, size(unspliced)[1])
for i=1:size(unspliced)[1]
    zeroPropUnsplicedGenes[i] = length(findall(x->x ==0, unspliced[i,:]))/size(unspliced)[2]
end

splicedSub = spliced[intersect(findall(x->x < 0.98, zeroPropSplicedGenes), findall(x->x < 0.98, zeroPropUnsplicedGenes)),:]
unsplicedSub = unspliced[intersect(findall(x->x < 0.98, zeroPropSplicedGenes), findall(x->x < 0.98, zeroPropUnsplicedGenes)),:]

#log and set to Float32
splicedSub = Float32.(log1p.(splicedSub))
unsplicedSub = Float32.(log1p.(unsplicedSub))

using Flux: loadmodel!
function loadForecaster(fileName::String, inputNodes::Int, hiddenLayerNodes::Int)
    #recreate neural network structure
    nn = Chain(Dense(inputNodes, hiddenLayerNodes, relu),
               Dense(hiddenLayerNodes, inputNodes))
    model = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(),
                       save_everystep = false,
                       reltol = 1e-3, abstol = 1e-3,
                       save_start = false)
    #load parameters into the model
    model = loadmodel!(model, load_object(fileName))
    return model
end

outModel = loadForecaster("pancNeuralODEResult.jld2", size(splicedSub)[1], 6000)

include("trainRNAForecaster.jl");
include("splicedDataPerturbationEffectPredictions.jl");

geneNames = readdlm("pancHVGNames.csv", ',')
geneNames = geneNames[intersect(findall(x->x < 0.98, zeroPropSplicedGenes), findall(x->x < 0.98, zeroPropUnsplicedGenes))]
geneNames = string.(geneNames)

#look at impact of high expression of pancreas development genes
pancDevGenes = ["Cpa1", "Hes1", "Pdx1", "Ptf1a", "Rbpj", "Sox9", "Ptf1a", "Insm1",
 "Neurog3", "Neurod1", "Onecut1", "Arx", "Foxa1", "Foxa2", "Nkx2-2", "Pax6", "Mafa",
 "Mnx1", "Nkx6-1", "Pax4", "Hnf6", "Gata4", "Gata6", "Mafb"]

pancDevGenes = pancDevGenes[findall(in(geneNames), pancDevGenes)]
#find each gene's max expression level
geneMaxes = Vector{Float32}(undef, length(pancDevGenes))
for i=1:length(pancDevGenes)
    geneMaxes[i] = maximum(splicedSub[findall(x->x==pancDevGenes[i], geneNames),:])
end

pancDevTFResult = perturbEffectPredictions(outModel, splicedSub, 500,
    perturbGenes = pancDevGenes, geneNames = geneNames, perturbLevels = geneMaxes)

save_object("pResults_DevTFs_PancData_500cells.jld2", pancDevTFResult)
