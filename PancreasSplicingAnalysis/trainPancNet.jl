using Pkg
Pkg.activate("../")

using DiffEqFlux, DifferentialEquations
using Flux.Data: DataLoader
using JLD2
include("../trainRNAForecasterV2.jl");

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

trainedModel = trainRNAForecaster(splicedSub, unsplicedSub, hiddenLayerNodes = 6000,
 batchsize = 200, learningRate = 0.0001, checkStability = false, useGPU = false, nEpochs = 10)

outModel = cpu(trainedModel[1])

# Changed to state from params here
save_object("pancNeuralODEResult.jld2", Flux.state(outModel))

writedlm("pancGPU_Losses.csv", trainedModel[2], ',')
