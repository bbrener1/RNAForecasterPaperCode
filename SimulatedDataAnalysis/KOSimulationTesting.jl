using JLD2
data = load_object("contTimeSim/KOSimulatedDataSets.jld2")
#get one simulation - used because it was simulated first
data = data[:,:,:,8]

#subset data for umap
beforeAfterKO = data[:,97:107,:]
#reformat as matrix
umapDataMat = Matrix{Float32}(undef, size(data)[1], size(data)[3]*size(beforeAfterKO)[2])
for i=1:size(beforeAfterKO)[2]
    umapDataMat[:,(1+ (size(data)[3]*(i-1))):(i*size(data)[3])] = beforeAfterKO[:,i,:]
end

timeKOLabels = repeat(["Time1_PreKO","Time2_PreKO","Time3_PreKO","Time4_PreKO",
"Time5_PreKO", "TimeKOInduced", "Time1_PostKO","Time2_PostKO","Time3_PostKO","Time4_PostKO","Time5_PostKO"], inner = size(data)[3])

#use R for seurat
using RCall

@rput umapDataMat timeKOLabels

R"""
library(Seurat)

colnames(umapDataMat) = c(1:dim(umapDataMat)[2])
rownames(umapDataMat) = c(1:dim(umapDataMat)[1])
seu = CreateSeuratObject(umapDataMat)

seu@meta.data$timeKOLabels = timeKOLabels
seu = ScaleData(seu)
seu <- RunPCA(seu, verbose = FALSE, npcs = 10, features = c(1:dim(umapDataMat)[1]))
ElbowPlot(seu, ndims = 10)
seu <- FindNeighbors(seu, dims = 1:7)
seu <- FindClusters(seu)
seu <- RunUMAP(seu, dims = 1:7, verbose = FALSE)
pdf("KOSimDataUMAP_RandRef10.pdf")
UMAPPlot(seu, group.by = 'timeKOLabels', order = c("Time1_PreKO","Time2_PreKO","Time3_PreKO","Time4_PreKO",
"Time5_PreKO", "TimeKOInduced", "Time1_PostKO","Time2_PostKO","Time3_PostKO","Time4_PostKO","Time5_PostKO"))
dev.off()
saveRDS(seu, "KOSimDataSeu1.rds")

"""


#RNAForecaster
t1 = Array{Float32}(undef, 10, 2000)
t2 = Array{Float32}(undef, 10, 2000)
for i=1:2000
    t1[:,i] = data[:,100,i]
    t2[:,i] = data[:,101,i]
end

predictionErrors = Vector{Float32}(undef, 50)
#using RNAForecaster
include("trainRNAForecaster.jl");
include("makeRecursivePredictions.jl");
trainingResults = createEnsembleForecaster(t1, t2,
 nNetworks = 10, checkStability = false,
  hiddenLayerNodes = 100, trainingProp = 1.0)

#now see what the loss is over simulated time as we try to recursively
#predict expression into the future states on which the model was not
#trained on
exprPreds = ensembleExpressionPredictions(trainingResults, t2, 50,
 perturbGenes = ["10"], geneNames = string.(collect(1:10)), perturbationLevels = [0.0f0])


#compare to simulated benchmark
for j=101:(size(exprPreds)[3]+100)
    actual = data[:, j+2,1:2000]
    #calculate average cell-wise mse
    predictionErrors[j-100] = mse(exprPreds[:,:,(j-100)], actual, agg= sum)/(size(actual)[1] * size(actual)[2])
end


using JLD2
save_object("ensemblePredictionErrors_KOSim_RandRef10.jld2", predictionErrors)
save_object("exprPredsKOSim_RandRef10.jld2", exprPreds)

#load performance on unperturbed data
unpErrors = load_object("multiNetEnsemblePredictionErrors.jld2")
unpErrors_RR10 = unpErrors[:,8]


#create umap with predicted results
umapDataMat2 = Matrix{Float32}(undef, size(data)[1], size(data)[3]*(size(beforeAfterKO)[2]+5))
umapDataMat2[:,1:size(data)[3]*size(beforeAfterKO)[2]] = umapDataMat
for i=1:5
    umapDataMat2[:,((size(data)[3]*size(beforeAfterKO)[2] + 1) + (size(data)[3]*(i-1))):((i*size(data)[3]) + (size(data)[3]*size(beforeAfterKO)[2]))] = exprPreds[:,:,i]
end

timeKOLabels2 = repeat(["Time1_PreKO","Time2_PreKO","Time3_PreKO","Time4_PreKO",
"Time5_PreKO", "TimeKOInduced", "Time1_PostKO","Time2_PostKO","Time3_PostKO",
"Time4_PostKO","Time5_PostKO", "Pred_T1_PostKO", "Pred_T2_PostKO",
 "Pred_T3_PostKO", "Pred_T4_PostKO", "Pred_T5_PostKO"], inner = size(data)[3])

 @rput umapDataMat2 timeKOLabels2

 R"""
 library(Seurat)

 colnames(umapDataMat2) = c(1:dim(umapDataMat2)[2])
 rownames(umapDataMat2) = c(1:dim(umapDataMat2)[1])
 seu = CreateSeuratObject(umapDataMat2)

 seu@meta.data$timeKOLabels = timeKOLabels2
 seu = ScaleData(seu)
 seu <- RunPCA(seu, verbose = FALSE, npcs = 10, features = c(1:dim(umapDataMat)[1]))
 ElbowPlot(seu, ndims = 10)
 seu <- FindNeighbors(seu, dims = 1:8)
 seu <- FindClusters(seu)
 seu <- RunUMAP(seu, dims = 1:8, verbose = FALSE)
 pdf("KOSimDataUMAP_WithPreds_RR10.pdf", width = 8, height = 8)
 UMAPPlot(seu, group.by = 'timeKOLabels', order = c("Time1_PreKO","Time2_PreKO","Time3_PreKO","Time4_PreKO",
 "Time5_PreKO", "TimeKOInduced", "Time1_PostKO","Time2_PostKO","Time3_PostKO","Time4_PostKO","Time5_PostKO",
 "Pred_T1_PostKO", "Pred_T2_PostKO", "Pred_T3_PostKO", "Pred_T4_PostKO", "Pred_T5_PostKO"))
 dev.off()
 saveRDS(seu, "KOSimDataSeu_WithPreds.rds")


#alternate UMAP
seuSub = seu[,c(6001:10000,12001:32000)]
seuSub@meta.data$timeKOLabels2 = seuSub@meta.data$timeKOLabels
seuSub@meta.data$timeKOLabels2[which(seuSub@meta.data$timeKOLabels2 %in% c("Time4_PreKO","Time5_PreKO"))] = "Training Data"
seuSub@meta.data$timeKOLabels2[which(seuSub@meta.data$timeKOLabels2 %in% c("Time1_PostKO","Time2_PostKO",
"Time3_PostKO","Time4_PostKO","Time5_PostKO"))] = "Post KO Test Data"
seuSub@meta.data$timeKOLabels2[which(seuSub@meta.data$timeKOLabels2 %in% c("Pred_T1_PostKO","Pred_T2_PostKO",
 "Pred_T3_PostKO", "Pred_T4_PostKO", "Pred_T5_PostKO"))] = "Predictions Post KO"
 pdf("KOSimDataUMAP_V2_RR10.pdf", width = 8, height = 8)
 UMAPPlot(seuSub, group.by = 'timeKOLabels2')
 dev.off()
 """
