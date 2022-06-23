using DelimitedFiles
using DiffEqFlux, DifferentialEquations
using JLD2
using StatsBase, Random

#select a random simulation
sim= sample(3:119, 1)[1] #102
data = load_object("contTimeSim/randomNetworkTimeSeriesSimData.jld2")
data = data[:,101:152,:,sim]

t1 = Array{Float32}(undef, 10, 2000)
t2 = Array{Float32}(undef, 10, 2000)
for i=1:2000
    t1[:,i] = data[:,1,i]
    t2[:,i] = data[:,2,i]
end

#train set of ten networks
using Distributed
addprocs(7)
@everywhere include("trainRNAForecaster.jl");
@everywhere include("makeRecursivePredictions.jl");
trainedNetworks = createEnsembleForecaster(t1, t2, nNetworks = 10, hiddenLayerNodes = 100,
    stabilityChecksBeforeFail = 10)

nodePreds = ensembleExpressionPredictions(trainedNetworks, t1, 50)

Random.seed!(123)
randCells = sample(1:2000, 12, replace = false)

simData = data[:,2:51, randCells]
predDataNODE = Array{Float32}(undef, 10, 12, 50, 10)
for i=1:length(nodePreds)
    predDataNODE[:,:,:,i] = nodePreds[i][:,randCells,:]
end

save_object("singleGenePredictions50TPs_10Networks.jld2", predDataNODE)

##this part is done separately for plotting
using Plots
using PDFmerger
using DelimitedFiles
using JLD2
simData = load_object("singleGeneSimData.jld2")
predDataNODE = load_object("singleGenePredictions50TPs_10Networks.jld2")

x=1:50
labels = Matrix{String}(undef, 1,11)
labels[1,1] = "Actual"
for i=2:11
    labels[1,i] = "Network " * string(i-1)
end

for i=1:12
    for j=1:10
        pData = predDataNODE[j,i,:,1]
        for k=2:10
            pData = hcat(pData, predDataNODE[j,i,:,k])
        end
        y=hcat(simData[j,:,i], pData)
        p = plot(x,y, xlabel = "Time", ylabel = "Expression", label = labels)
        savefig(p, "temp.pdf")
        append_pdf!("singleGeneExpressionPredsSimData_multipleNetworks.pdf",
         "temp.pdf", cleanup=true)
     end
end

#plot #13 with same y axis as the single network plot for easy recognition in figure
pData = predDataNODE[3,2,:,1]
for k=2:10
    pData = hcat(pData, predDataNODE[3,2,:,k])
end
y=hcat(simData[3,:,2], pData)
p = plot(x,y, xlabel = "Time", ylabel = "Expression", label = labels, ylims = (1.45, 2.55))
savefig(p, "figureExampleMultipleNetworks.pdf")

#mean and median predictions
using Statistics
for i=1:12
    for j=1:10
        pData = predDataNODE[j,i,:,1]
        for k=2:10
            pData = hcat(pData, predDataNODE[j,i,:,k])
        end
        means = mean(pData, dims=2)
        medians = median(pData, dims=2)
        y=hcat(simData[j,:,i], means, medians)
        p = plot(x,y, xlabel = "Time", ylabel = "Expression", label = ["Actual" "Mean Prediction" "Median Prediction"])
        savefig(p, "temp.pdf")
        append_pdf!("singleGeneExpressionPredsSimData_multipleNetAverages.pdf",
         "temp.pdf", cleanup=true)
     end
end
