using DelimitedFiles
totalData = readdlm("rpe_TotalTranscripts.csv", ',')
geneNames = readdlm("rpe_geneNames.csv", ',')
labelingTime = readdlm("rpe_cellLabelingTime.csv", ',')

#filtering by zeroes and variance
zeroPropGenes = Array{Float64}(undef, size(totalData)[1])
for i=1:size(totalData)[1]
    zeroPropGenes[i] = length(findall(x->x ==0, totalData[i,:]))/size(totalData)[2]
end

#remove genes with very high zero portion
totalData = totalData[setdiff(1:end, findall(x->x > 0.98, zeroPropGenes)),:]
geneNames = geneNames[setdiff(1:end, findall(x->x > 0.98, zeroPropGenes))]

#gene variances
using Statistics
geneVars = var(totalData, dims = 2)
#remove low variance genes
highVar = findall(x->x > quantile(vec(geneVars), 0.75), vec(geneVars))
totalData = totalData[highVar,:]
geneNames = geneNames[highVar]
geneNames = string.(geneNames)

subCells = findall(x->x == 60.0, vec(labelingTime))
t2_60min = log1p.(totalData[:,subCells])

using JLD2
futureExpressionPredictions = load_object("rpe_72hour_Predictions_Ensemble.jld2")

using RCall

@rput futureExpressionPredictions t2_60min geneNames

R"""
save.image("ccPreds_Ensemble_72hr.RData")
"""

R"""
library(tricycle)
load("ccPreds_Ensemble_72hr.RData")

#initial states
rownames(t2_60min) = geneNames
iniProj = estimate_cycle_position(t2_60min, gname.type = "ENSEMBL", species = "human")

#calculate for each prediction
ccData = matrix(0.0, dim(futureExpressionPredictions)[2], dim(futureExpressionPredictions)[3])
for(i in 1:dim(futureExpressionPredictions)[3]){
    tmpCounts = futureExpressionPredictions[,,i]
    rownames(tmpCounts) = geneNames
    ccData[,i] = estimate_cycle_position(tmpCounts, gname.type = "ENSEMBL", species = "human")
}

write.csv(iniProj, "initialDataTricycleScores_Ensemble_72hr.csv")
write.csv(ccData, "predExpTricycleScores_Ensemble_72hr.csv")
"""
