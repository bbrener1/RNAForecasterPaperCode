using Pkg
Pkg.activate("VAE")
using DelimitedFiles

biData = Array{Float32}(undef, 7, 800, 2000)
for i=0:1999
    cellExpr = readdlm("contTimeSim/Bifurcating" * "/simulations/E" * string(i) * ".csv", ',')
    cellExpr = cellExpr[2:end, 3:end]
    biData[:,:, (i+1)] = cellExpr
end

#UMAP in R to determine the branch points
using RCall

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

@rput biData1

R"""
library(Seurat)

colnames(biData1) = c(1:dim(biData1)[2])
rownames(biData1) = c(1:dim(biData1)[1])
seu = CreateSeuratObject(biData1)

seu@meta.data$time = rep(c(1:800), 50)
seu = ScaleData(seu)
seu <- RunPCA(seu, verbose = FALSE, npcs = 5, features = c(1:dim(biData1)[1]))
seu <- FindNeighbors(seu, dims = 1:3)
seu <- FindClusters(seu)
seu <- RunUMAP(seu, dims = 1:3, verbose = FALSE)
saveRDS(seu, "bifurcatingDataSeu1.rds")
pdf("bifurcatingSimUMAP.pdf")
FeaturePlot(seu, features = 'time', cols = rev(heat.colors(6)))
dev.off()
"""

#time 1 should be 365 and predict through 465

t1 = Array{Float32}(undef, 7, 2000)
t2 = Array{Float32}(undef, 7, 2000)
for i=1:2000
    t1[:,i] = biData[:,365,i]
    t2[:,i] = biData[:,366,i]
end


#training the neural network, predicting future expression
#states and returning loss data
predictionErrors = Vector{Float32}(undef, 100)

include("trainRNAForecaster.jl")
include("makeRecursivePredictions.jl")

#train network and record training and validation error
trainingResults = createEnsembleForecaster(t1, t2, nNetworks = 10,
 checkStability = false, hiddenLayerNodes = 50)

#now see what the loss is over simulated time as we try to recursively
#predict expression into the future states on which the model was not
#trained on
exprPreds = ensembleExpressionPredictions(trainingResults, t1, 100)


#compare to simulated benchmark
for j=1:size(exprPreds)[3]
    actual = biData[:, j+365,1:2000]
    #calculate average cell-wise mse
    predictionErrors[j] = mse(exprPreds[:,:,j], actual, agg= sum)/(size(actual)[1] * size(actual)[2])

end

#save data
using JLD2
save_object("exprPreds_BifurcatingSim.jld2", exprPreds)
save_object("bifurcatingSimulationEnsemblePredictionErrors.jld2", predictionErrors)


#use same randomly selected 50 cells to UMAP across the 100 time points
umapDataPred = Matrix{Float32}(undef, 7, 5000)
for i=1:100
    umapDataPred[:,1+(i-1)*50:(50*i)] = exprPreds[:,cellsToUse,i]
end

umapData = hcat(biData1, umapDataPred)

@rput umapData

R"""
library(Seurat)

colnames(umapData) = c(1:dim(umapData)[2])
rownames(umapData) = c(1:dim(umapData)[1])
seu = CreateSeuratObject(umapData)

seu@meta.data$time = c(rep(c(1:800), 50), rep(c(366:465),50))
seu@meta.data$type = c(rep("Simulated", 40000), rep("Predicted", 5000))
seu = ScaleData(seu)
seu <- RunPCA(seu, verbose = FALSE, npcs = 5, features = c(1:dim(umapData)[1]))
seu <- FindNeighbors(seu, dims = 1:4)
seu <- FindClusters(seu)
seu <- RunUMAP(seu, dims = 1:4, verbose = FALSE)
saveRDS(seu, "bifurcatingDataWithPreds.rds")
pdf("bifurcatingSimUMAPwithPreds.pdf")
FeaturePlot(seu, features = 'time', cols = rev(heat.colors(6)))
UMAPPlot(seu, group.by = 'type')
dev.off()
"""
