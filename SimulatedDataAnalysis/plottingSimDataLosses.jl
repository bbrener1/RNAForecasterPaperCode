using Plots
using StatsPlots
using DelimitedFiles

nodeValidationLossR = readdlm("SimDataRandRefValidationLosses.csv", ',')
nodePredLossR = readdlm("SimDataPredictionErrorSimTime.csv", ',')

mlpValidationLossR = readdlm("SimDataRandRefValidationLosses_MLP.csv", ',')
mlpPredLossR = readdlm("SimDataPredictionErrorSimTime_MLP.csv", ',')

unstableNNs = [15, 20, 26, 83, 106, 107]
toUse = setdiff(1:117, unstableNNs)

#plot the validation losses as a boxplot
vlDataR = hcat(nodeValidationLossR[end,toUse],
 mlpValidationLossR[end,toUse])

using HypothesisTests
UnequalVarianceTTest(vlDataR[:,1], vlDataR[:,2])



b1 = boxplot(["Neural ODE" "Multilayer Perceptron"], vlDataR, legend = false, ylabel = "MSE")

savefig(b1, "simDataValidationError__RR.pdf")

#boxplot of the MSE after 50 predictions into the future simulated expression states
sfDataR = hcat(nodePredLossR[end,toUse], mlpPredLossR[end,toUse])

b2 = boxplot(["Neural ODE" "Multilayer Perceptron"], log1p.(sfDataR), legend = false, ylabel = "log MSE")

savefig(b2, "simData50TPError__RR.pdf")


#plot of the prediction accuracy of each model on each data set over
#50 simulated time points
sfDataR2 = hcat(nodePredLossR[:,toUse], mlpPredLossR[:,toUse])

#create group variable and labels
group = repeat(1:2, inner=111)

 labels = append!(["Neural ODE"], repeat([""], 110), ["Multilayer Perceptron"],
  repeat([""], 110))

for i=1:222
    if i==1
        display(plot(1:50, log1p.(sfDataR2[:,i]), legend = :topleft, ylabel = "log MSE", xlabel = "Simulated Time", color = group[i], label = labels[i]))
    else
        display(plot!(1:50, log1p.(sfDataR2[:,i]), legend = :topleft, ylabel = "log MSE", color = group[i], label = labels[i]))
    end
end

savefig("simDataTimeCourseError_RR.pdf")

#boxplot at 1,10,20,30,40,50
labels = Matrix{String}(undef, 1, 12)
iter = 1
for i in [1,10,20,30,40,50]
    labels[1,iter] = "t" * string(i) * " - Neural ODE"
    labels[1,(iter+1)] = "t" * string(i) * " - MLP"
    iter+=2
end

sfDataR2Sub = sfDataR2[[1,10,20,30,40,50],:]
boxplotData = hcat(transpose(sfDataR2Sub[:,1:111]), transpose(sfDataR2Sub[:,112:222]))
boxplotData = boxplotData[:,[1,7,2,8,3,9,4,10,5,11,6,12]]

defaultCols = get_color_palette(:auto, 2)
pal = repeat([defaultCols[1], defaultCols[2]], outer = 6)
boxplot(labels, log1p.(boxplotData), legend = false, ylabel = "log MSE", palette = pal)
savefig("simDataMLP_NODE_TP_Boxplots.pdf")

#getting statistical significance of loss level comparisons
testResults = Vector{Any}(undef, 6)
iter = 1
for i=1:6
    testResults[i] = UnequalVarianceTTest(boxplotData[:,iter], boxplotData[:,iter+1])
    iter+=2
end


##compare to results without stability check
nodeValidationLossNSC = readdlm("SimDataRandRefValidationLosses_NSC.csv", ',')
nodePredLossNSC = readdlm("SimDataPredictionErrorSimTime_NSC.csv", ',')

vlDataNSC = hcat(nodeValidationLossNSC[end,toUse], nodeValidationLossR[end,toUse])


b1 = boxplot(["Stability Not Checked" "Stability Checked"], vlDataNSC,
 legend = false, ylabel = "MSE", palette = :Dark2_5)

savefig(b1, "simDataStabilityImpactValError.pdf")


sfDataNSC = hcat(nodePredLossNSC[:,toUse], nodePredLossR[:,toUse])

#create group variable and labels
group = repeat(1:2, inner=111)

 labels = append!(["Stability Not Checked"], repeat([""], 110),
  ["Stability Checked"], repeat([""], 110))

for i=1:222
    if i==1
        display(plot(1:50, log1p.(sfDataNSC[:,i]), legend = :topleft,
         ylabel = "log MSE", xlabel = "Simulated Time", color = group[i],
          label = labels[i], palette = :Dark2_5))
    else
        display(plot!(1:50, log1p.(sfDataNSC[:,i]), legend = :topleft,
         ylabel = "log MSE", color = group[i], label = labels[i], palette = :Dark2_5))
    end
end

savefig("simDataStabilityImpact50TPs.pdf")

#boxplot at 1,10,20,30,40,50
labels = Matrix{String}(undef, 1, 12)
iter = 1
for i in [1,10,20,30,40,50]
    labels[1,iter] = "t" * string(i) * " - Stability Not Checked"
    labels[1,(iter+1)] = "t" * string(i) * " - Stability Checked"
    iter+=2
end

sfDataNSCSub = sfDataNSC[[1,10,20,30,40,50],:]
boxplotData = hcat(transpose(sfDataNSCSub[:,1:111]), transpose(sfDataNSCSub[:,112:222]))
boxplotData = boxplotData[:,[1,7,2,8,3,9,4,10,5,11,6,12]]

defaultCols = get_color_palette(:auto, 4)
pal = repeat([defaultCols[3], defaultCols[4]], outer = 6)
boxplot(labels, log1p.(boxplotData), legend = false, ylabel = "log MSE", palette = pal)
savefig("simDataStabilityCheck_TP_Boxplots.pdf")

#getting statistical significance of loss level comparisons
testResults = Vector{Any}(undef, 6)
iter = 1
for i=1:6
    testResults[i] = UnequalVarianceTTest(boxplotData[:,iter], boxplotData[:,iter+1])
    iter+=2
end


##compare to ensembled median prediction error
using JLD2
ensemblePreds = load_object("multiNetEnsemblePredictionErrors_200TPs.jld2")
ensemblePreds25 = load_object("multiNetEnsemblePredictionErrors_25Networks_200TPs.jld2")

unstableSims = []
for i=1:size(ensemblePreds)[2]
    if length(findall(x->x > 0.00001, ensemblePreds[:,i])) == 0
        push!(unstableSims, i)
    elseif length(findall(x->x > 1e3, ensemblePreds[:,i])) > 0
        push!(unstableSims, i)
    end
end


unstableSims25 = []
for i=1:size(ensemblePreds25)[2]
    if length(findall(x->x > 0.00001, ensemblePreds25[:,i])) == 0
        push!(unstableSims25, i)
    elseif length(findall(x->x > 1e3, ensemblePreds25[:,i])) > 0
        push!(unstableSims25, i)
    end
end

unstableSims = append!(unstableSims, unstableSims25, unstableNNs)
toUse = setdiff(1:117, unique(unstableSims))


ensCompData = hcat(nodePredLossR[:,toUse], ensemblePreds[1:50,toUse], ensemblePreds25[1:50, toUse])

#create group variable and labels
group = repeat(1:3, inner=89)

labels = append!(["Single Network"], repeat([""], 88),
  ["10 Network Ensemble"], repeat([""], 88), ["25 Network Ensemble"], repeat([""], 88))

for i=1:267
    if i==1
        display(plot(1:50, ensCompData[:,i], legend = :topleft,
         ylabel = "MSE", xlabel = "Simulated Time", color = group[i],
          label = labels[i], palette = :Dark2_5))
    else
        display(plot!(1:50, ensCompData[:,i], legend = :topleft,
         ylabel = "MSE", color = group[i], label = labels[i], palette = :Dark2_5))
    end
end

savefig("simDataEnsembleImpact50TPs.pdf")


#boxplot at 1,10,20,30,40,50
labels = Matrix{String}(undef, 1, 18)
iter = 1
for i in [1,10,20,30,40,50]
    labels[1,iter] = "t" * string(i) * " - Single Network"
    labels[1,(iter+1)] = "t" * string(i) * " - 10 Networks"
    labels[1,(iter+2)] = "t" * string(i) * " - 25 Networks"
    iter+=3
end

ensCompDataSub = ensCompData[[1,10,20,30,40,50],:]
boxplotData = hcat(transpose(ensCompDataSub[:,1:89]), transpose(ensCompDataSub[:,90:178]), transpose(ensCompDataSub[:,179:267]))
boxplotData = boxplotData[:,[1,7,13,2,8,14,3,9,15,4,10,16,5,11,17,6,12,18]]

defaultCols = get_color_palette(:Dark2_5, 3)
pal = repeat([defaultCols[1], defaultCols[2], defaultCols[3]], outer = 6)
boxplot(labels, log1p.(boxplotData), legend = false, ylabel = "log MSE", palette = pal)
savefig("simDataEnsembleBoxplots.pdf")

#get an appropriate legend
plot(1:10, randn(10, 3), labels = ["Single Network" "10 Network Ensemble" "25 Network Ensemble"], palette = :Dark2_5)
savefig("legendEnsemble.pdf")


#getting statistical significance of loss level comparisons
testResults = Vector{Any}(undef, 12)
iter = 1
for i=1:2:12
    testResults[i] = UnequalVarianceTTest(boxplotData[:,iter], boxplotData[:,iter+1])
    testResults[i+1] = UnequalVarianceTTest(boxplotData[:,iter+1], boxplotData[:,iter+2])
    iter+=3
end


#look at just ensembles after 50 tps
unstableSims = append!(unstableSims, unstableSims25)
toUse = setdiff(1:117, unique(unstableSims))

ensCompData2 = hcat(ensemblePreds[:,toUse], ensemblePreds25[:, toUse])

#boxplot at 1,50,100,150,200
labels = Matrix{String}(undef, 1, 10)
iter = 1
for i in [1,50,100,150,200]
    labels[1,(iter)] = "t" * string(i) * " - 10 Networks"
    labels[1,(iter+1)] = "t" * string(i) * " - 25 Networks"
    iter+=2
end

ensCompDataSub2 = ensCompData2[[1,50,100,150,200],:]
boxplotData2 = hcat(transpose(ensCompDataSub2[:,1:89]), transpose(ensCompDataSub2[:,90:178]))
boxplotData2 = boxplotData2[:,[1,6,2,7,3,8,4,9,5,10]]

defaultCols = get_color_palette(:Dark2_5, 3)
pal = repeat([defaultCols[2], defaultCols[3]], outer = 5)
boxplot(labels, log1p.(boxplotData2), legend = false, ylabel = "log MSE", palette = pal)
savefig("simDataEnsembleBoxplots_200TPs.pdf")

#getting statistical significance of loss level comparisons
testResults = Vector{Any}(undef, 5)
iter = 1
for i=1:5
    testResults[i] = UnequalVarianceTTest(boxplotData2[:,iter], boxplotData2[:,iter+1])
    iter+=2
end
