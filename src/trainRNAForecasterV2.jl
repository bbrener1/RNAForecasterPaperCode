using DiffEqFlux, DifferentialEquations
using Flux
# using CUDA
using Flux: mse
using Random, Statistics, StatsBase, LinearRegression
using Distributed
using Base.Iterators: partition

function minZero(num::AbstractArray{<:Number})
    num[findall(x->x < 0, num)] .= 0
    num
end

function meanLoss(data_loader, model)
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        ŷ = model(x)[1]
        ls += mse(ŷ, y)
        num +=  size(x)[end]
    end
    return ls / num
end

function trainNetwork(trainData, hiddenLayerNodes::Int; 
    learningRate::Float64 = 1e-4, nEpochs::Int = 10, useGPU::Bool = false, valData = nothing)

    width = size(trainData[1][1])[1]

    nn,model = defaultNetwork(width,hiddenLayerNodes)

    if useGPU
        trainData = gpu(trainData)
        valData = gpu(valData)
        nn = gpu(nn)
        model = gpu(model)
    end

    loss(x,y) = mse(model(x)[1], y)
    opt = ADAM(learningRate)

    losses = zeros(Float32, nEpochs, 2)

    for epoch = 1:nEpochs
        println("Epoch $epoch")
        for d in trainData
            gs = gradient(()->loss(d...),Flux.params(model))
            Flux.Optimise.update!(opt, Flux.params(model), gs)
        end
        losses[epoch,1]= meanLoss(trainData, model)
        if valData != nothing
            losses[epoch,2] = meanLoss(valData, model)
        end
    end

    return (model, losses)
end


"""
`trainRNAForecaster(expressionDataT1::Matrix{Float32}, expressionDataT2::Matrix{Float32};
     trainingProp::Float64 = 0.8, hiddenLayerNodes::Int = 2*size(expressionDataT1)[1],
     shuffleData::Bool = true, seed::Int = 123, learningRate::Float64 = 0.005,
     nEpochs::Int = 10, batchsize::Int = 100, checkStability::Bool = false, iterToCheck::Int = 50,
     stabilityThreshold::Float32 = 2*maximum(expressionDataT1), stabilityChecksBeforeFail::Int = 5,
     useGPU::Bool = false)`

Function to train RNAForecaster based on expression data. Main input is two
matrices representing expression data from two different time points in the same cell.
This can be either based on splicing or metabolic labeling currently.
Each should be log normalized and have genes as rows and cells as columns.

# Required Arguments
* expressionDataT1 - Float32 Matrix of log-normalized expression counts in the format of genes x cells
* expressionDataT2 - Float32 Matrix of log-normalized expression counts in the format
 of genes x cells from a time after expressionDataT1
# Keyword Arguments

## Subsetting
These are passed on to subsetData
* trainingProp - proportion of the data to use for training the model, the rest will be
 used for a validation set. If you don't want a validation set, this value can be set to 1.0
* shuffleData - should the cells be randomly shuffled before training
* batchsize - batch size for training

## Network parameters
* hiddenLayerNodes - number of nodes in the hidden layer of the neural network
* learningRate - learning rate for the neural network during training
* nEpochs - how many times should the neural network be trained on the data.
 Generally yields small gains in performance, can be lowered to speed up the training process
 * seed - random seed
 * useGPU - use a GPU to train the neural network? highly recommended for large data sets, if available
 
 ## Retraining
 * checkStability - should the stability of the networks future time predictions be checked,
 retraining the network if unstable?
* iterToCheck - when checking stability, how many future time steps should be predicted?
* stabilityThreshold - when checking stability, what is the maximum gene variance allowable across predictions?
* stabilityChecksBeforeFail - when checking stability, how many times should the network
 be allowed to retrain before an error is thrown? Used to prevent an infinite loop.

"""

function trainRNAForecaster(expressionDataT1::Matrix{Float32}, expressionDataT2::Matrix{Float32};
    trainingProp::Float64 = 0.8, shuffleData::Bool = true, hiddenLayerNodes::Int = 2*size(expressionDataT1)[1], seed::Int = 123, 
    learningRate::Float64 = 0.005, nEpochs::Int = 10, batchsize::Int = 100, checkStability::Bool = false, iterToCheck::Int = 50,
    stabilityThreshold::Float32 = 2*maximum(expressionDataT1), stabilityChecksBeforeFail::Int = 5, useGPU::Bool = false)

    println("Loading Data...")

    (trainData,valData) = subsetData(expressionDataT1,expressionDataT2,trainingProp=trainingProp,shuffleData=shuffleData,seed=seed)

    println("Training model...")

    trainedNet = trainNetwork(trainData, hiddenLayerNodes, learningRate=learningRate, 
        nEpochs=nEpochs, useGPU=useGPU, valData=valData)

    # optionally re-train if we are checking stability
    iter = 1
    while checkStability
        println("Re-training unstable $iter")
        if iter > stabilityChecksBeforeFail
            error("Failed to find a stable solution after " * string(iter-1) * " attempts.
            Try increasing the stabilityChecksBeforeFail variable or, if a slightly less
            stable solution is acceptable, increase the stabilityThreshold variable.")
        end

        trainedNet = trainNetwork(trainData, hiddenLayerNodes, learningRate=learningRate, 
            nEpochs=nEpochs, useGPU=useGPU, valData=valData)

        checkStability = checkModelStability(trainedNet[1], trainX, iterToCheck, stabilityThreshold, useGPU, batchsize)
        Random.seed!(seed+iter)
        iter +=1
    end

    return trainedNet
end


"""
`subsetData(expressionDataT1::Matrix{Float32}, expressionDataT2::Matrix{Float32};
trainingProp::Float64 = 0.8, shuffleData::Bool = true,batchsize::Int=100)`

Subsets data into training and validation sets with batching, in the format expected by the rest of the package 
(eg trainNetwork and the like)

# Required Arguments
* expressionDataT1 - Float32 Matrix of log-normalized expression counts in the format of genes x cells
* expressionDataT2 - Float32 Matrix of log-normalized expression counts in the format
 of genes x cells from a time after expressionDataT1
# Keyword Arguments
* trainingProp - proportion of the data to use for training the model, the rest will be
 used for a validation set. If you don't want a validation set, this value can be set to 1.0
* shuffleData - should the cells be randomly shuffled before training
* batchsize - batch size for training

"""
function subsetData(expressionDataT1::Matrix{Float32}, expressionDataT2::Matrix{Float32};
    trainingProp::Float64 = 0.8, shuffleData::Bool = true,batchsize::Int=100,seed::Int = 123) 

    trainData = nothing
    valData = nothing

    if shuffleData
        Random.seed!(seed)
        shuffling = shuffle(1:size(expressionDataT1)[2])
        expressionDataT1 = expressionDataT1[:,shuffling]
        expressionDataT2 = expressionDataT2[:,shuffling]
    end
    
    if trainingProp < 1.0
        #subset the data into training and validation sets
        #determine how many cells should be in training set
        cellsInTraining = Int(round(size(expressionDataT1)[2]*trainingProp))
        
        trainX = expressionDataT1[:,1:cellsInTraining]
        trainY = expressionDataT2[:,1:cellsInTraining]
        valX = expressionDataT1[:,cellsInTraining+1:size(expressionDataT1)[2]]
        valY = expressionDataT2[:,cellsInTraining+1:size(expressionDataT1)[2]]
        
        valData = ([(valX[:,i], valY[:,i]) for i in partition(1:size(valX)[2], batchsize)])

    else
        trainX = expressionDataT1
        trainY = expressionDataT2

    end

    trainData = ([(trainX[:,i], trainY[:,i]) for i in partition(1:size(trainX)[2], batchsize)])

    return (trainData,valData)
end

function checkModelStability(model, initialConditions::Matrix{Float32},
    iterToCheck::Int, expressionThreshold::Float32, useGPU::Bool, batchsize::Int)
    
    println("Checking Model Stability...")
    
    inputData = copy(initialConditions)
    recordedVals = Array{Float32}(undef, size(inputData)[1], size(inputData)[2], iterToCheck)

    if useGPU
        inputData = inputData |> gpu
    end

    for i=1:iterToCheck
        recordedVals[:,:,i] = inputData    
        exprPreds = Matrix{Float32}(undef, size(inputData)[1], size(inputData)[2])
    
        #calculate model predictions
        inputData = ([inputData[:,k] for k in partition(1:size(inputData)[2], batchsize)])
        
        for j=1:length(inputData)
            exprPreds[:,(1+((j-1)*batchsize)):(j*batchsize)] = cpu(trainedNetwork(inputData[[j]]...)[1])
        end
        
        exprPreds = minZero(exprPreds)
        inputData = exprPreds
    end
    
    #check expression levels
    stable = length(findall(x->x > expressionThreshold, recordedVals)) == 0
    println("Model is stable: $stable")
    return stable
end

function defaultNetwork(width,hiddenLayerNodes) 

    nn = Chain(Dense(width, hiddenLayerNodes, relu),
        Dense(hiddenLayerNodes, width))
    
    model = NeuralODE(nn, (0.0f0, 1.0f0), Tsit5(),
                    save_everystep = false,
                    reltol = 1e-3, abstol = 1e-3,
                    save_start = false)
    
    return nn,model

end
