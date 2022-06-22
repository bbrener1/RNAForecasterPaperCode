using Plots
using DelimitedFiles

##from tricycle
ccPreds = readdlm("predExpTricycleScores_Ensemble_72hr.csv", ',')
initialCellPos = readdlm("initialDataTricycleScores_Ensemble_72hr.csv", ',')

ccPreds = float.(ccPreds[2:end, 2:end])
initialCellPos = float.(initialCellPos[2:end, 2])

for i=1:size(ccPreds)[1]
    p = plot(1:24, ccPreds[i,1:24],
     title="Initial Tricycle Score: " * string(initialCellPos[i]),
     xlabel = "Time (hours)", ylabel = "Tricycle Score")
     savefig(p, "temp.pdf")
     append_pdf!("CCForecastsRPECells_TricycleScore_24hr_Ensemble.pdf",
      "temp.pdf", cleanup=true)
end

for i=1:size(ccPreds)[1]
    p = plot(1:72, ccPreds[i,1:72],
     title="Initial Tricycle Score: " * string(initialCellPos[i]),
     xlabel = "Time (hours)", ylabel = "Tricycle Score")
     savefig(p, "temp.pdf")
     append_pdf!("CCForecastsRPECells_TricycleScore_72hr_Ensemble.pdf",
      "temp.pdf", cleanup=true)
end
