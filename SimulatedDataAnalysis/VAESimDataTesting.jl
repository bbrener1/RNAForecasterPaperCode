using RNAForecaster
using JLD2
include("trainVAE.jl")

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
for k=1:117
        #train VAE on first two time points
        trainData = hcat(t1[:,:,k], t2[:,:,k])
        trainData = Flux.Data.DataLoader(trainData, batchsize =200, shuffle=true)
        encoder, decoder = trainVAE(trainData)

        t1Encoded = encodeData(encoder, t1[:,:,k], 4)
        t2Encoded = encodeData(encoder, t2[:,:,k], 4)
        #train network and record training and validation error
        trainingResults = createEnsembleForecaster(t1Encoded, t2Encoded,
         nNetworks = 10, hiddenLayerNodes = 20, checkStability = false)

        #now see what the loss is over simulated time as we try to recursively
        #predict expression into the future states on which the model was not
        #trained on
        exprPreds = ensembleExpressionPredictions(trainingResults, t1Encoded, 50)

        #decode predictions
        exprPreds2 = Array{Float32}(undef, 10, 2000, 50)
        for i=1:50
            exprPreds2[:,:,i] = decodeData(exprPreds[:,:,i], decoder, 10)
        end


        #compare to simulated benchmark
        for j=1:size(exprPreds2)[3]
            actual = data[:, j+1,1:2000,k]
            #calculate average cell-wise mse
            predictionErrors[j,k] = mse(exprPreds2[:,:,j], actual, agg= sum)/(size(actual)[1] * size(actual)[2])

        end

end

#save data
save_object("VAETransformedEnsemblePredictionErrors.jld2", predictionErrors)
