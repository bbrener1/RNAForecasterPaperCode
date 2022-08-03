using JLD2
data = load_object("contTimeSim/KOSimulatedDataSets.jld2")
#get data just before and after KO
data = data[:,100:154,:,:]

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

addprocs(24)
@everywhere include("trainRNAForecaster.jl")
@everywhere include("makeRecursivePredictions.jl")

for k=1:117
    #train network and record training and validation error
    trainingResults = createEnsembleForecaster(t1[:,:,k], t2[:,:,k],
     nNetworks = 10, checkStability = false, hiddenLayerNodes = 100,
      trainingProp = 1.0)

    #now see what the loss is over simulated time as we try to recursively
    #predict expression into the future states on which the model was not
    #trained on
    exprPreds = ensembleExpressionPredictions(trainingResults, t2[:,:,k], 50,
        perturbGenes = ["10"], geneNames = string.(collect(1:10)),
        perturbationLevels = [0.0f0])


    #compare to simulated benchmark
    for j=1:size(exprPreds)[3]
        actual = data[:, j+3,1:2000,k]
        #calculate average cell-wise mse
        predictionErrors[j,k] = mse(exprPreds[:,:,j], actual, agg= sum)/(size(actual)[1] * size(actual)[2])

    end
end

#save data
using JLD2
save_object("ensemblePredictionErrors_KO_10Networks.jld2", predictionErrors)
