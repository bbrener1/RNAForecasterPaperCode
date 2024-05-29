include_prefix = "/Users/bbrener1/haxx/RNAForecasterPaperCode/"

using Pkg
Pkg.activate(include_prefix)

include(include_prefix * "trainRNAForecaster.jl");
include(include_prefix * "splicedDataPerturbationEffectPredictions.jl");

using DiffEqFlux, DifferentialEquations
using JLD2
using Flux: loadmodel!

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

function loadForecaster(fileName::String, inputNodes::Int, hiddenLayerNodes::Int)
    model = defaultNetwork(inputNodes,hiddenLayerNodes)
    model = loadmodel!(model, load_object(fileName))
    return model
end

print(size(splicedSub))

outModel = loadForecaster("pancNeuralODEResult.jld2", size(splicedSub)[1], 6000)

KOResult = JLD2.load("KO_test_predictions_25.jld2")["single_stored_object"]

#functions to convert to interpretable output
geneNames = readdlm("pancHVGNames.csv", ',')
geneNames = geneNames[intersect(findall(x->x < 0.98, zeroPropSplicedGenes), findall(x->x < 0.98, zeroPropUnsplicedGenes))]
geneNames = string.(geneNames)

using CSV
koImpact = totalPerturbImpact(KOResult, geneNames)
CSV.write("KOImpact_PancData_25cells.csv", koImpact)
changesFromKO = genePerturbExpressionChanges(KOResult, geneNames, "Ctrb1")
CSV.write("Ctrb1_KOImpact_PancData_25cells.csv", changesFromKO)
changesFromOtherKOs = geneResponseToPerturb(KOResult, geneNames, "Ctrb1")
CSV.write("KOImpactOnCtrb1_PancData_25cells.csv", changesFromOtherKOs)


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

pancDevTFResult = perturbEffectPredictions(outModel, splicedSub, 25,
    perturbGenes = pancDevGenes, geneNames = geneNames, perturbLevels = geneMaxes)

save_object("pResults_DevTFs_PancData_25cells.jld2", pancDevTFResult)

pancDevGenesOrdered = geneNames[findall(in(pancDevGenes), geneNames)]

pImpact = totalPerturbImpact(pancDevTFResult, pancDevGenesOrdered)
CSV.write("pImpact_DevTFs_PancData_25cells.csv", pImpact)

pGeneExpressionChanges = Vector{DataFrame}(undef, length(pancDevGenesOrdered))
pGeneExpressionChangesAll = Vector{DataFrame}(undef, length(pancDevGenesOrdered))

for i=1:length(pancDevGenesOrdered)
    tmp = genePerturbExpressionChanges(pancDevTFResult, geneNames, pancDevGenesOrdered[i],
    genesPerturbed = pancDevGenesOrdered)
    pGeneExpressionChangesAll[i] = tmp
    pGeneExpressionChanges[i] = tmp[findall(in(pancDevGenesOrdered), tmp.Genes),:]
end

#calculate percentile scores for each
percentiles = Matrix{Float32}(undef, length(pancDevGenesOrdered), length(pancDevGenesOrdered))
for j =1:length(pancDevGenesOrdered)
    d1 = pGeneExpressionChanges[j]
    d2 = pGeneExpressionChangesAll[j]
    for i=1:length(pancDevGenesOrdered)

        ind = findall(x->x == pancDevGenesOrdered[i], d1.Genes)[1]
        mec = d1[ind,3]
        if mec > 0
            perc = length(findall(x->x > mec, d2[:,3])) / length(findall(x->x > 0, d2[:,3]))
            percentiles[i,j] = round(((1 - perc)*100)) - 1
        elseif mec == 0
            percentiles[i,j] = 0
        else
            perc = length(findall(x->x < mec, d2[:,3])) / length(findall(x->x < 0, d2[:,3]))
            percentiles[i,j] = round(((1 - perc)*100)) - 1
        end
    end
end

writedlm("PancTFPercentiles_ExpPert.csv", percentiles)


KOGeneExpressionChanges = Vector{DataFrame}(undef, length(pancDevGenesOrdered))
KOGeneExpressionChangesAll = Vector{DataFrame}(undef, length(pancDevGenesOrdered))

for i=1:length(pancDevGenesOrdered)
    tmp = genePerturbExpressionChanges(KOResult, geneNames, pancDevGenesOrdered[i])
    KOGeneExpressionChangesAll[i] = tmp
    KOGeneExpressionChanges[i] = tmp[findall(in(pancDevGenesOrdered), tmp.Genes),:]
end

#calculate percentile scores for each
percentiles = Matrix{Float32}(undef, length(pancDevGenesOrdered), length(pancDevGenesOrdered))
for j =1:length(pancDevGenesOrdered)
    d1 = KOGeneExpressionChanges[j]
    d2 = KOGeneExpressionChangesAll[j]
    for i=1:length(pancDevGenesOrdered)

        ind = findall(x->x == pancDevGenesOrdered[i], d1.Genes)[1]
        mec = d1[ind,2]
        if mec > 0
            perc = length(findall(x->x > mec, d2[:,2])) / length(findall(x->x > 0, d2[:,2]))
            percentiles[i,j] = round(((1 - perc)*100)) - 1
        elseif mec == 0
            percentiles[i,j] = 0
        else
            perc = length(findall(x->x < mec, d2[:,2])) / length(findall(x->x < 0, d2[:,2]))
            percentiles[i,j] = round(((1 - perc)*100)) - 1
        end
    end
end

writedlm("PancTFPercentiles_ExpKO.csv", percentiles)
writedlm("PancDevGenesOrdered.csv", pancDevGenesOrdered)

## plotting
using Plots
using DelimitedFiles

geneNames = readdlm("PancDevGenesOrdered.csv")
geneNames = string.(vec(geneNames))
pertPercentile = readdlm("PancTFPercentiles_ExpPert.csv")
KOPercentile = readdlm("PancTFPercentiles_ExpKO.csv")

p1 = heatmap(geneNames, geneNames, KOPercentile, xlabel = "KO'd Gene",interpolate=false)
p2 = heatmap(geneNames, geneNames, pertPercentile, xlabel = "Perturbed Gene",interpolate=false)

savefig(p1, "KOPancGenePercentileHeatmap.png")
savefig(p2, "PertPancGenePercentileHeatmap.png")

# KOPvals = readdlm("KOPancGenePvalMat.csv", ',')
include("testingSigRegulation.jl")
KOPvals = findSigRegulation(KOResult,geneNames)

KOPvals = 
p3 = heatmap(geneNames, geneNames, KOPvals, xlabel = "KO'd Gene",
 color = reverse(cgrad(:balance, scale = :log)))
savefig(p3, "KOPancGenePvalHeatmap.pdf")


#barplot of top distance genes
using CSV, DataFrames
geneDists = CSV.read("KOImpact_PancData_25cells.csv", DataFrame)

p4 = plot(geneDists[:,1], geneDists[:,2], kind="bar")
savefig(p4, "KOPancGeneTopEuclideanDist.pdf")
