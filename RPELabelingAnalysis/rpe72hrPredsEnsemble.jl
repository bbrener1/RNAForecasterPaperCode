using DelimitedFiles
labeledData = readdlm("rpe_LabeledTranscripts.csv", ',')
unlabeledData = readdlm("rpe_UnlabeledTranscripts.csv", ',')
totalData = readdlm("rpe_TotalTranscripts.csv", ',')
geneNames = readdlm("rpe_geneNames.csv", ',')
labelingTime = readdlm("rpe_cellLabelingTime.csv", ',')

include("estimateT1.jl");

#filtering by zeroes and variance
zeroPropGenes = Array{Float64}(undef, size(totalData)[1])
for i=1:size(totalData)[1]
    zeroPropGenes[i] = length(findall(x->x ==0, totalData[i,:]))/size(totalData)[2]
end

#remove genes with very high zero portion
labeledData = labeledData[setdiff(1:end, findall(x->x > 0.98, zeroPropGenes)),:]
unlabeledData = unlabeledData[setdiff(1:end, findall(x->x > 0.98, zeroPropGenes)),:]
totalData = totalData[setdiff(1:end, findall(x->x > 0.98, zeroPropGenes)),:]
geneNames = geneNames[setdiff(1:end, findall(x->x > 0.98, zeroPropGenes))]

#gene variances
using Statistics
geneVars = var(totalData, dims = 2)
#remove low variance genes
highVar = findall(x->x > quantile(vec(geneVars), 0.75), vec(geneVars))
labeledData = Float32.(labeledData[highVar,:])
unlabeledData = Float32.(unlabeledData[highVar,:])
totalData = Float32.(totalData[highVar,:])
geneNames = geneNames[highVar]


#calculate gene wise degradation rate via the slope between new labeled and total
#transcripts
#degradation rate: gamma = -ln(1-k)/t
t1Estimate = estimateT1LabelingData(labeledData, totalData,
    unlabeledData, vec(labelingTime))

#start with testing just the 1hr labeling cells; might be some way to train with
#each and combine at the end
hourCells = findall(x->x == 60.0, vec(labelingTime))
t1_60min = t1Estimate[:,hourCells]
t2_60min = totalData[:,hourCells]

#log and set to Float32
t1_60min = Float32.(log1p.(t1_60min))
t2_60min = Float32.(log1p.(t2_60min))

#load rnaForecaster
using DiffEqFlux, DifferentialEquations
using JLD2
trainedForecaster = load_object("RPEEnsembleForecaster.jld2")

#get predictions for 72 hours into the future
include("makeRecursivePredictions.jl");
futureExpressionPredictions = ensembleExpressionPredictions(trainedForecaster, t2_60min, 72,
enforceMaxPred = true, maxPredictionMult = 1.2f0, useGPU = true)

save_object("rpe_72hour_Predictions_Ensemble.jld2", futureExpressionPredictions)
