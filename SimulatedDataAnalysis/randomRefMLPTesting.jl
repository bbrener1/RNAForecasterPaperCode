using DelimitedFiles
using Flux
using Flux.Data: DataLoader
using Flux: mse
using JLD2
using Random


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
    t1Temp = copy(t1[:,:,k])
    t2Temp = copy(t2[:,:,k])
    #randomly shuffle the input data cells
   Random.seed!(123)
   shuffling = shuffle(1:size(t1Temp)[2])
   t1Temp = t1Temp[:,shuffling]
   t2Temp = t2Temp[:,shuffling]
    trainX = t1Temp[:,1:1800]
    trainY = t2Temp[:,1:1800]
    testX = t1Temp[:,1801:2000]
    testY = t2Temp[:,1801:2000]

    trainData = DataLoader((trainX, trainY), batchsize=10)
    testData = DataLoader((testX, testY), batchsize=10)

    model = Chain(Dense(10,32), Dense(32,64), Dense(64, 100), Dense(100,100),
     Dense(100,64), Dense(64, 32), Dense(32,10))

    loss(x,y) = mse(model(x), y)

    #optimizer with learning rate of 0.005
    opt = ADAM(0.005)

    function meanLoss(data_loader, model)
        ls = 0.0f0
        num = 0
        for (x, y) in data_loader
            ŷ = model(x)
            ls += mse(ŷ, y)
            num +=  size(x)[end]
        end
        return ls / num
    end


    ps = Flux.params(model)
    losses = Matrix{Float32}(undef, 10, 2)
    for epoch in 1:10
        for (x, y) in trainData
            gs = gradient(() -> loss(x,y), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end

        # Report on train and test
        losses[epoch,1]= meanLoss(trainData, model)
        losses[epoch,2] = meanLoss(testData, model)
    end

    #save the trained neural network
    save_object("Networks/MLPExpressionForecaster_RandRef" * string(k+2) * ".jld2", model)

    trainLosses[:,k] = losses[:,1]
    validationLosses[:,k] = losses[:,2]


    #now see what the loss is over simulated time as we try to recursively
    #predict expression into the future states on which the model was not
    #trained on
    function minZero(num)
        num[findall(x->x < 0, num)] .= 0
        num
    end

    trainData = DataLoader(t1[:,:,k], batchsize=10)
    predictionError = Vector{Float32}(undef, 50)
    for j=1:50
        #calculate model predictions
        preds = []
        for x in trainData
            pred = model(x)
            push!(preds, pred)
        end

        exprPreds = Matrix{Float32}(undef, 10,2000)
        for i=1:length(preds)
            tmpPreds = minZero(preds[i])
            tmpPreds[findall(x->x > 2.5f0*maximum(t1[:,:,k]), tmpPreds)] .= 2.5f0*maximum(t1[:,:,k])
            exprPreds[:,((10*(i-1))+1):(10*i)] = tmpPreds
        end

        actual = Array{Float32}(undef, 10, 2000)
        for i=1:2000
            actual[:,i] = data[:,j+1,i, k]
        end

        #cell and gene-wise average mse
        predictionError[j] = mse(exprPreds, actual, agg=sum)/(size(exprPreds)[1] * size(exprPreds)[2])

        trainData = DataLoader(exprPreds, batchsize=10)

    end

    #save prediction error results over simulated time
    predictionErrors[:,k] = predictionError

end

#save data
writedlm("Losses/SimDataRandRefTrainLosses_MLP.csv", trainLosses, ',')
writedlm("Losses/SimDataRandRefValidationLosses_MLP.csv", validationLosses, ',')
writedlm("Losses/SimDataPredictionErrorSimTime_MLP.csv", predictionErrors, ',')
