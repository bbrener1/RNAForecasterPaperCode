using DelimitedFiles
ccPredsEnsMaxPred = readdlm("predExpTricycleScores_Ensemble_72hr.csv", ',')
ccPredsEnsMaxPred = float.(ccPredsEnsMaxPred[2:end, 2:end])

ccPredsEns = readdlm("predExpTricycleScores_Ensemble_NMP.csv", ',')
ccPredsEns = float.(ccPredsEns[2:end, 2:end])

ccPredsMaxPred = readdlm("predExpTricycleScores_MP_NE.csv", ',')
ccPredsMaxPred = float.(ccPredsMaxPred[2:end, 2:end])

ccPreds = readdlm("predExpTricycleScores_NMP_NE.csv", ',')
ccPreds = float.(ccPreds[2:end, 2:end])


#function to score how well each cell's predictions align witj the cell cycle
function scoreOrder(cellCycleScores)
    score = 0.0
    for i=2:length(cellCycleScores)
        #if prediction goes up, but not up too far
        if cellCycleScores[i] > cellCycleScores[i-1] && abs(cellCycleScores[i] - cellCycleScores[i-1]) < 0.75
            score +=1
        #if prediction goes from 2pi to 0 this just means the scores are looping around, so this should also get points
        elseif cellCycleScores[i] < 0.75 && cellCycleScores[i-1] > 5.5
            score+=1
        #if it goes down, but very little, this should get scored a bit higher than going down by a lot
        elseif cellCycleScores[i] > 6 && cellCycleScores[i-1] < 0.33
            score +=0.2
        elseif abs(cellCycleScores[i] - cellCycleScores[i-1]) < 0.25
            score +=0.2
        end
    end
    return score
end


scoresTotal = Vector{Float64}(undef, size(ccPredsEnsMaxPred)[1])
for i=1:size(ccPredsEnsMaxPred)[1]
    scoresTotal[i] = scoreOrder(ccPredsEnsMaxPred[i,:])
end

#scores for each day
scores24hr = Vector{Float64}(undef, size(ccPredsEnsMaxPred)[1])
for i=1:size(ccPredsEnsMaxPred)[1]
    scores24hr[i] = scoreOrder(ccPredsEnsMaxPred[i,1:24])
end

scores48hr = Vector{Float64}(undef, size(ccPredsEnsMaxPred)[1])
for i=1:size(ccPredsEnsMaxPred)[1]
    scores48hr[i] = scoreOrder(ccPredsEnsMaxPred[i,25:48])
end

scores72hr = Vector{Float64}(undef, size(ccPredsEnsMaxPred)[1])
for i=1:size(ccPredsEnsMaxPred)[1]
    scores72hr[i] = scoreOrder(ccPredsEnsMaxPred[i,49:72])
end

#scores of randomly generated sequences of numbers
using Random
Random.seed!(123);
randScores = Vector{Float64}(undef, size(ccPredsEnsMaxPred)[1])
for i=1:size(ccPredsEnsMaxPred)[1]
    randScores[i] = scoreOrder(2*pi*(1 .- rand(72)))
end

#scores w/o max preds
scoresEns = Vector{Float64}(undef, size(ccPredsEns)[1])
for i=1:size(ccPredsEns)[1]
    scoresEns[i] = scoreOrder(ccPredsEns[i,:])
end

#scores w/o ensembling
scoresMaxPred = Vector{Float64}(undef, size(ccPredsMaxPred)[1])
for i=1:size(ccPredsMaxPred)[1]
    scoresMaxPred[i] = scoreOrder(ccPredsMaxPred[i,:])
end

#scores w/o either
scoresRaw = Vector{Float64}(undef, size(ccPreds)[1])
for i=1:size(ccPreds)[1]
    scoresRaw[i] = scoreOrder(ccPreds[i,:])
end

using Statistics
Random.seed!(123);
for i=1:1e7
    if scoreOrder(2*pi*(1 .- rand(72))) >= median(scoresTotal)
        print(i)
        break
    end
end

using HypothesisTests
UnequalVarianceTTest(scoresTotal, randScores)
UnequalVarianceTTest(scoresTotal, scoresEns)
UnequalVarianceTTest(scoresTotal, scoresMaxPred)
UnequalVarianceTTest(scoresTotal, scoresRaw)

UnequalVarianceTTest(scores24hr, scores48hr)
UnequalVarianceTTest(scores24hr, scores72hr)
UnequalVarianceTTest(scores48hr, scores72hr)
UnequalVarianceTTest(scores72hr, randScores)


#boxplots of the scores
using Plots
using StatsPlots
bData = hcat(scoresTotal, scoresMaxPred, scoresEns, scoresRaw, randScores)
b1 = boxplot(["RNAForecaster" "Single Network" "No Max Prediction" "Plain Neural ODE" "Random Tricycle Scores"],
 bData, legend = false, ylabel = "Order Score", ylims = [0, 75])

savefig(b1, "scoring_ccPos.pdf")

#create random scores for 24 hr sets
Random.seed!(123);
randScores24 = Vector{Float64}(undef, size(ccPredsEnsMaxPred)[1])
for i=1:size(ccPredsEnsMaxPred)[1]
    randScores24[i] = scoreOrder(2*pi*(1 .- rand(24)))
end


defaultCols = get_color_palette(:auto, 5)
pal = [defaultCols[1], defaultCols[2], defaultCols[3], defaultCols[5]]

bData2 = hcat(scores24hr, scores48hr, scores72hr, randScores24)
b2 = boxplot(["1-24hrs" "25-48hrs" "49-72hrs" "Random Tricycle Scores"],
 bData2, legend = false, ylabel = "Order Score", ylims = [0, 25], palette = pal)

savefig(b2, "scoring_ccPos_byDay.pdf")

#plot the average log counts in each cell for each condition
using JLD2
predsEnsMP = load_object("rpe_72hour_Predictions_Ensemble.jld2")
predsNMP_NE = load_object("rpe_72hour_Predictions_noMaxPred_noEns.jld2")
predsMP_NE = load_object("rpe_72hour_Predictions_maxPred_noEns.jld2")
predsNMP_E = load_object("rpe_72hour_Predictions_Ensemble_noMaxPred.jld2")

function getMedianTotalCounts(mat::Matrix{Float32})
    sums = Vector{BigFloat}(undef, size(mat)[2])
    for i=1:size(mat)[2]
        sums[i] = sum(exp.(BigFloat.(mat[:,i])))
    end
    return median(sums)
end

ensMP_MTC = getMedianTotalCounts(predsEnsMP[:,:,72]) #28774.834
NE_NMP_MTC = getMedianTotalCounts(predsNMP_NE[:,:,72]) #>>>googol
NE_MP_MTC = getMedianTotalCounts(predsMP_NE[:,:,72]) #45460.85
ensNMP_MTC = getMedianTotalCounts(predsNMP_E[:,:,72]) #>>>googol

#compare to real data
initialCounts = load_object("totalCounts_RPE_1hrLab.jld2")
initialCount_MTC = getMedianTotalCounts(initialCounts) #11068

mtcs = [initialCount_MTC, ensMP_MTC, NE_MP_MTC, ensNMP_MTC, NE_NMP_MTC]
bar1 = bar(log.(log.(mtcs)), legend = false)

savefig(bar1, "totalCountsBarplot.pdf")
