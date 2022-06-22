include("trainRNAForecaster.jl");
include("makeRecursivePredictions.jl");

using DelimitedFiles
using JLD2

data = Array{Float32}(undef, 10, 52, 2000, 117)
for j=3:119
    for i=0:1999
        cellExpr = readdlm("contTimeSim/RandomRef" * string(j) * "/simulations/E" * string(i) * ".csv", ',')
        cellExpr = cellExpr[2:end, 2:end]
        data[:,:, (i+1), (j-2)] = cellExpr[:,101:152]
    end
end

t1 = Array{Float32}(undef, 10, 2000, 117)
t2 = Array{Float32}(undef, 10, 2000, 117)
for j=1:117
    for i=1:2000
        t1[:,i, j] = data[:,1,i,j]
        t2[:,i, j] = data[:,2,i,j]
    end
end

#loop across each data set, training the neural network, predicting future expression
#states and returning loss data
trainLosses = Matrix{Float32}(undef, 10, 117)
validationLosses = Matrix{Float32}(undef, 10, 117)
predictionErrors = Matrix{Float32}(undef, 50, 117)

for k=1:117
    #train network and record training and validation error
    trainingResults = trainRNAForecaster(t1[:,:,k], t2[:,:,k], hiddenLayerNodes = 100,
    checkStability = false)

    trainLosses[:,k] = trainingResults[2][:,1]
    validationLosses[:,k] = trainingResults[2][:,2]

    #now see what the loss is over simulated time as we try to recursively
    #predict expression into the future states on which the model was not
    #trained on
    exprPreds = predictCellFutures(trainingResults[1], t1[:,:,k], 50,
    maxPrediction = 2.5f0*maximum(t1[:,:,k]))

    #compare to simulated benchmark
    for j=1:size(exprPreds)[3]
        actual = data[:, j+1,1:2000,k]
        #calculate average cell-wise mse
        predictionErrors[j,k] = mse(exprPreds[:,:,j], actual, agg= sum)/(size(actual)[1] * size(actual)[2])

    end
end

#save data
writedlm("Losses/SimDataRandRefTrainLosses_NSC.csv", trainLosses, ',')
writedlm("Losses/SimDataRandRefValidationLosses_NSC.csv", validationLosses, ',')
writedlm("Losses/SimDataPredictionErrorSimTime_NSC.csv", predictionErrors, ',')
