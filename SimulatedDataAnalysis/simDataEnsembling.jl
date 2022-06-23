using JLD2
data = load_object("contTimeSim/randomNetworkTimeSeriesSimData.jld2")
data = data[:,101:152,:,:]

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
predictionErrors = Matrix{Float32}(undef, 50, 117)

using Distributed

addprocs(9)
@everywhere include("trainRNAForecaster.jl")
@everywhere include("makeRecursivePredictions.jl")

for k=1:117
    try
        #train network and record training and validation error
        trainingResults = createEnsembleForecaster(t1[:,:,k], t2[:,:,k],
         nNetworks = 10, stabilityChecksBeforeFail = 10, hiddenLayerNodes = 100)

        #now see what the loss is over simulated time as we try to recursively
        #predict expression into the future states on which the model was not
        #trained on
        exprPreds = ensembleExpressionPredictions(trainingResults, t1[:,:,k], 50)


        #compare to simulated benchmark
        for j=1:size(exprPreds)[3]
            actual = data[:, j+1,1:2000,k]
            #calculate average cell-wise mse
            predictionErrors[j,k] = mse(exprPreds[:,:,j], actual, agg= sum)/(size(actual)[1] * size(actual)[2])

        end

    catch
        println("Network training on iteration " * string(k) * " failed to
        find a stable solution.")
    end
end

#save data
using JLD2
save_object("multiNetEnsemblePredictionErrors.jld2", predictionErrors)
