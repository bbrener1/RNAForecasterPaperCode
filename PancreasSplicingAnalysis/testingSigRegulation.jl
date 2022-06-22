#drafting significantly regulated test function
using HypothesisTests
using DataFrames

OneSampleTTest(KOResult[3][2,1,:])

for i=1:100
    for j=1:100
        OneSampleTTest(KOResult[3][i,j,:])
    end
end

devGeneInds1 = findall(in(pancDevGenes), outDF[:,1])
devGeneInds2 = findall(in(pancDevGenes), outDF[:,2])
devGeneInds = intersect(devGeneInds1, devGeneInds2)
outDF[devGeneInds,:]

function findSigRegulation(perturbData, geneNames::Vector{String};
    pvalCut::Float64 = 0.05)

    pvalMat = Matrix{Float64}(undef, size(perturbData[3])[1], size(perturbData[3])[2])
    for i=1:size(perturbData[3])[1]
        for j=1:size(perturbData[3])[2]
            pvalMat[i,j] = pvalue(OneSampleTTest(perturbData[3][i,j,:]))
        end
    end

    sigRegs = findall(x->x < pvalCut, pvalMat)

    #remove self regulation
    toRemove = findall(x->x[1] == x[2], sigRegs)

    sigRegs = sigRegs[setdiff(1:end, toRemove)]

    #create output data frame
    outDF = DataFrame(Regulator = geneNames[getindex.(sigRegs,2)],
     Regulated = geneNames[getindex.(sigRegs,1)], PValue = pvalMat[sigRegs])

     #sort by pval
     sort!(outDF, [:PValue])

     return outDF
 end


pvalMatDev = pvalMat[findall(in(pancDevGenes), geneNames), findall(in(pancDevGenes), geneNames)]
using DelimitedFiles
writedlm("KOPancGenePvalMat.csv", pvalMatDev, ',')
