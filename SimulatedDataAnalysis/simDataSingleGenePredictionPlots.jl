using DelimitedFiles
using DiffEqFlux, DifferentialEquations
using JLD2
using StatsBase, Random

#select a random simulation
sim= sample(3:119, 1)[1] #102
data = Array{Float32}(undef, 10, 52, 2000)
for i=0:1999
    cellExpr = readdlm("contTimeSim/RandomRef" * string(sim) * "/simulations/E" * string(i) * ".csv", ',')
    cellExpr = cellExpr[2:end, 2:end]
    data[:,:, (i+1)] = cellExpr[:,101:152]
end

t1 = Array{Float32}(undef, 10, 2000)
t2 = Array{Float32}(undef, 10, 2000)
for i=1:2000
    t1[:,i] = data[:,1,i]
    t2[:,i] = data[:,2,i]
end

include("trainRNAForecaster.jl")
trainedNetwork = trainRNAForecaster(t1, t2, hiddenLayerNodes = 100)

##train MLP
t1Temp = copy(t1)
t2Temp = copy(t2)
#randomly shuffle the input data cells
Random.seed!(123)
shuffling = shuffle(1:size(t1Temp)[2])
t1Temp = t1Temp[:,shuffling]
t2Temp = t2Temp[:,shuffling]
trainX = t1Temp[:,1:1800]
trainY = t2Temp[:,1:1800]
testX = t1Temp[:,1801:2000]
testY = t2Temp[:,1801:2000]

using Flux: DataLoader
trainData = DataLoader((trainX, trainY), batchsize=100)
testData = DataLoader((testX, testY), batchsize=100)

model = Chain(Dense(10,32), Dense(32,64), Dense(64, 100), Dense(100,100),
 Dense(100,64), Dense(64, 32), Dense(32,10))

loss(x,y) = mse(model(x), y)

#optimizer with learning rate of 0.005
opt = ADAM(0.005)

ps = Flux.params(model)
for epoch in 1:10
    for (x, y) in trainData
        gs = gradient(() -> loss(x,y), ps) # compute gradient
        Flux.Optimise.update!(opt, ps, gs) # update parameters
    end
end


function predictCellFuturesMLP(trainedNetwork, expressionData::Matrix{Float32}, tSteps::Int;
     perturbGenes::Vector{String} = Vector{String}(undef,0), geneNames::Vector{String} = Vector{String}(undef,0),
     perturbationLevels::Vector{Float32} = Vector{Float32}(undef,0),
     enforceMaxPred::Bool = true, maxPrediction::Float32 = 2*maximum(expressionData))


    inputData = copy(expressionData)
    predictions = Array{Float32}(undef, size(expressionData)[1], size(expressionData)[2], tSteps)

    for i=1:tSteps
        for j=1:size(expressionData)[2]
            predictions[:,j,i] = trainedNetwork(inputData[:,j])
        end
        #set negative predictions to zero
        predictions[findall(x->x < 0, predictions)] .= 0

        if enforceMaxPred
            predictions[findall(x->x > maxPrediction, predictions)] .= maxPrediction
        end

        inputData = predictions[:,:,i]

    end

    return predictions
end

mlpPreds = predictCellFuturesMLP(model, t1, 50)
##

include("makeRecursivePredictions.jl");
nodePreds = predictCellFutures(trainedNetwork[1], t1, 50)

Random.seed!(123)
randCells = sample(1:2000, 12, replace = false)

simData = data[:,2:51, randCells]
predDataNODE = nodePreds[:, randCells,:]
predDataMLP = mlpPreds[:, randCells,:]

save_object("singleGeneSimData.jld2", simData)
save_object("singleGenePredictions50TPs_NODE.jld2", predDataNODE)
save_object("singleGenePredictions50TPs_MLP.jld2", predDataMLP)

#also save one cells simulated data for an illustrative plot
simDataExample = data[:,1:50,1]
writedlm("simDataExample.csv", simDataExample, ',')


##this part is done separately for plotting
using Plots
using PDFmerger
using DelimitedFiles
using JLD2
simData = load_object("singleGeneSimData.jld2")
predDataNODE = load_object("singleGenePredictions50TPs_NODE.jld2")
predDataMLP = load_object("singleGenePredictions50TPs_MLP.jld2")

x=1:50
for i=1:12
    for j=1:10
        y=hcat(predDataNODE[j,i,:], predDataMLP[j,i,:], simData[j,:,i], )
        p = plot(x,y, xlabel = "Time", ylabel = "Expression", label = ["Neural ODE" "MLP" "Actual"])
        savefig(p, "temp.pdf")
        append_pdf!("singleGeneExpressionPredsSimData_withMLP.pdf",
         "temp.pdf", cleanup=true)
     end
end

#plot example of simulated data over 50 tps in one cell
simDataExample = readdlm("simDataExample.csv", ',')

p2 = plot(x, transpose(simDataExample), xlabel = "Simulated Time", ylabel = "Expression",
 label = ["Gene 1" "Gene 2" "Gene 3" "Gene 4" "Gene 5" "Gene 6" "Gene 7" "Gene 8" "Gene 9" "Gene 10"])

savefig(p2, "simDataExample.pdf")
