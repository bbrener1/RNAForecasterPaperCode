using Pkg
Pkg.activate("VAE")
using DelimitedFiles

biData = Array{Float32}(undef, 7, 800, 2000)
for i=0:1999
    cellExpr = readdlm("contTimeSim/Bifurcating" * "/simulations/E" * string(i) * ".csv", ',')
    cellExpr = cellExpr[2:end, 3:end]
    biData[:,:, (i+1)] = cellExpr
end

#randomly select 50 cells
using StatsBase
using Random
Random.seed!(123)
cellsToUse = sample(1:2000, 50, replace = false)
biData1 = Matrix{Float32}(undef, 7, 40000)
function createMat(cellsToUse, biData)
    j =1
    for i in cellsToUse
        biData1[:,1+(j-1)*800:(800*j)] = biData[:,:,i]
        j+=1
    end
    return biData1
end
biData1 = createMat(cellsToUse, biData)

#time 1 should be 250 and predict through 450

t1 = Array{Float32}(undef, 7, 2000)
t2 = Array{Float32}(undef, 7, 2000)
for i=1:2000
    t1[:,i] = biData[:,250,i]
    t2[:,i] = biData[:,251,i]
end


#training the neural network, predicting future expression
#states and returning loss data
predictionErrors = Vector{Float32}(undef, 200)

include("trainRNAForecaster.jl")
include("makeRecursivePredictions.jl")

#train network and record training and validation error
trainingResults = createEnsembleForecaster(t1, t2, nNetworks = 10,
 checkStability = false, hiddenLayerNodes = 50)

#now see what the loss is over simulated time as we try to recursively
#predict expression into the future states on which the model was not
#trained on
exprPreds = ensembleExpressionPredictions(trainingResults, t1, 200)


#compare to simulated benchmark
for j=1:size(exprPreds)[3]
    actual = biData[:, j+250,1:2000]
    #calculate average cell-wise mse
    predictionErrors[j] = mse(exprPreds[:,:,j], actual, agg= sum)/(size(actual)[1] * size(actual)[2])

end

#save data
using JLD2
save_object("exprPreds_BifurcatingSim_EarlierStart.jld2", exprPreds)
save_object("bifurcatingSimulationEnsemblePredictionErrors_EarlierStart.jld2", predictionErrors)


#use same randomly selected 50 cells to UMAP across the 200 time points
umapDataPred = Matrix{Float32}(undef, 7, 10000)
for i=1:200
    umapDataPred[:,1+(i-1)*50:(50*i)] = exprPreds[:,cellsToUse,i]
end

umapData = hcat(biData1, umapDataPred)

#UMAP in R
using RCall

@rput umapData

R"""
library(Seurat)

colnames(umapData) = c(1:dim(umapData)[2])
rownames(umapData) = c(1:dim(umapData)[1])
seu = CreateSeuratObject(umapData)

seu@meta.data$time = c(rep(c(1:800), 50), rep(c(251:450),50))
seu@meta.data$type = c(rep("Simulated", 40000), rep("Predicted", 10000))
seu = ScaleData(seu)
seu <- RunPCA(seu, verbose = FALSE, npcs = 5, features = c(1:dim(umapData)[1]))
seu <- FindNeighbors(seu, dims = 1:4)
seu <- FindClusters(seu)
seu <- RunUMAP(seu, dims = 1:4, verbose = FALSE)
saveRDS(seu, "bifurcatingDataWithPreds_EarlierStart.rds")
pdf("bifurcatingSimUMAPwithPreds_EarlierStart.pdf")
FeaturePlot(seu, features = 'time', cols = rev(heat.colors(6)))
UMAPPlot(seu, group.by = 'type')
dev.off()
"""
