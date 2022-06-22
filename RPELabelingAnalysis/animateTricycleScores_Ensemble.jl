using Plots

function circleShape(h, k, r)
    theta = LinRange(0, 2*pi, 500)
    h .+ r*sin.(theta), k .+ r*cos.(theta)
end

using DelimitedFiles
ccPreds = readdlm("predExpTricycleScores_Ensemble_72hr.csv", ',')
initialCellPos = readdlm("initialDataTricycleScores_Ensemble_72hr.csv", ',')

ccPreds = float.(ccPreds[2:end, 2:end])
initialCellPos = float.(initialCellPos[2:end, 2])

#make a function
function cellCyclePlot(cycleData, fps, fileName)
    anim = @animate for i in 1:size(cycleData)[2]
        plot(circleShape(0,0,1), legend = false)
        x = sin.(cycleData[:,i])
        y = cos.(cycleData[:,i])
        plot!(x,y, marker_z = 1:size(cycleData)[1], seriestype = :scatter)
    end

    gif(anim, fileName, fps = fps)
end

#plot all the cells
allData = hcat(initialCellPos, ccPreds)
cellCyclePlot(allData, 2, "RPE_tricycle_allData_Ensemble_72hr.gif")

#plot G2M cells
cellSub = findall(x->x > pi && x < 1.75*pi, initialCellPos)
#concat data
allDataG2M = hcat(initialCellPos[cellSub], ccPreds[cellSub,:])
cellCyclePlot(allDataG2M, 2, "RPE_tricycle_G2M_Ensemble_72hr.gif")

#plot G1 cells
cellSub = findall(x->x < 0.25*pi || x > 1.75*pi, initialCellPos)
#concat data
allDataG1 = hcat(initialCellPos[cellSub], ccPreds[cellSub,:])
cellCyclePlot(allDataG1, 2, "RPE_tricycle_G1_Ensemble_72hr.gif")

#plot S cells
cellSub = findall(x->x > 0.5*pi && x < pi, initialCellPos)
#concat data
allDataS = hcat(initialCellPos[cellSub], ccPreds[cellSub,:])
cellCyclePlot(allDataS, 2, "RPE_tricycle_S_Ensemble_72hr.gif")

##for first 24 hours
#plot all the cells
allData = hcat(initialCellPos, ccPreds[:,1:24])
cellCyclePlot(allData, 2, "RPE_tricycle_allData_Ensemble_24hr.gif")

#plot G2M cells
cellSub = findall(x->x > pi && x < 1.75*pi, initialCellPos)
#concat data
allDataG2M = hcat(initialCellPos[cellSub], ccPreds[cellSub,1:24])
cellCyclePlot(allDataG2M, 2, "RPE_tricycle_G2M_Ensemble_24hr.gif")

#plot G1 cells
cellSub = findall(x->x < 0.25*pi || x > 1.75*pi, initialCellPos)
#concat data
allDataG1 = hcat(initialCellPos[cellSub], ccPreds[cellSub,1:24])
cellCyclePlot(allDataG1, 2, "RPE_tricycle_G1_Ensemble_24hr.gif")

#plot S cells
cellSub = findall(x->x > 0.5*pi && x < pi, initialCellPos)
#concat data
allDataS = hcat(initialCellPos[cellSub], ccPreds[cellSub,1:24])
cellCyclePlot(allDataS, 2, "/RPE_tricycle_S_Ensemble_24hr.gif")


##plot without animation for static figure
using PDFmerger
function cellCyclePlotStatic(cycleData, fileName)
    for i in 1:size(cycleData)[2]
        p = plot(circleShape(0,0,1), legend = false)
        x = sin.(cycleData[:,i])
        y = cos.(cycleData[:,i])
        plot!(x,y, marker_z = 1:size(cycleData)[1], seriestype = :scatter)

        savefig("temp.pdf")
        append_pdf!(fileName, "temp.pdf", cleanup=true)
    end
end

allData = hcat(initialCellPos, ccPreds)
cellCyclePlotStatic(allData, "RPE_tricycle_allData_static_Ensemble.pdf")

#plot G2M cells
cellSub = findall(x->x > pi && x < 1.75*pi, initialCellPos)
#concat data
allDataG2M = hcat(initialCellPos[cellSub], ccPreds[cellSub,:])
cellCyclePlotStatic(allDataG2M, "RPE_tricycle_G2M_static_Ensemble.pdf")

#plot G1 cells
cellSub = findall(x->x < 0.25*pi || x > 1.75*pi, initialCellPos)
#concat data
allDataG1 = hcat(initialCellPos[cellSub], ccPreds[cellSub,:])
cellCyclePlotStatic(allDataG1, "RPE_tricycle_G1_static_Ensemble.pdf")

#plot S cells
cellSub = findall(x->x > 0.5*pi && x < pi, initialCellPos)
#concat data
allDataS = hcat(initialCellPos[cellSub], ccPreds[cellSub,:])
cellCyclePlotStatic(allDataS, "RPE_tricycle_S_static_Ensemble.pdf")
