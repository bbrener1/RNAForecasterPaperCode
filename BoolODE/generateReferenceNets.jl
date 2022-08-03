using Random
using StatsBase
using DelimitedFiles

function randomRefNet(; minGenes::Int = 6, maxGenes::Int = 15, maxRegs::Int = 3,
     seed::Int = 123)
    Random.seed!(seed);
    #randomly select number of genes to use
    nGenes = rand(collect(minGenes:maxGenes), 1)[1]
    #create table to hold gene regulation rules
    rulesDF = Matrix{String}(undef, nGenes, 2)

    for i = 1:nGenes
     #name of gene
     rulesDF[i,1] = "G$i"
    end

    for i = 1:nGenes
        #determine whether the gene is positively regulated, neg, or both
        #1-pos, 2-neg, 3-both
        regDir = rand(collect(1:3), 1)[1]
        if regDir == 1
            nRegs = rand(collect(1:maxRegs), 1)[1]
            regulators = sample(collect(1:nGenes), nRegs, replace = false)
            regNames = rulesDF[regulators,1]
            rule = string("( ", join(regNames, " or "), " )")
            rulesDF[i,2] = rule
        elseif regDir == 2
            nRegs = rand(collect(1:maxRegs), 1)[1]
            regulators = sample(collect(1:nGenes), nRegs, replace = false)
            regNames = rulesDF[regulators,1]
            rule = string("not ", "( ", join(regNames, " or "), " )")
            rulesDF[i,2] = rule
        elseif regDir == 3
            nRegs = rand(collect(1:maxRegs), 1)[1]
            regulators = sample(collect(1:nGenes), nRegs, replace = false)
            regNames = rulesDF[regulators,1]
            rule1 = string("( ", join(regNames, " or "), " )")
            nRegs = rand(collect(1:maxRegs), 1)[1]
            regulators = sample(collect(1:nGenes), nRegs, replace = false)
            regNames = rulesDF[regulators,1]
            rule2 = string("not " , "( ", join(regNames, " or "), " )")
            rule = string("(", rule1, " and ", rule2, ")")
            rulesDF[i,2] = rule
        end
    end
    #add the column labels
    rulesDF = vcat(["Gene" "Rule"], rulesDF)
    return(rulesDF)
end

function generateRefNetSet(seedList::Vector{Int};
     minGenes::Int = 6, maxGenes::Int = 15, maxRegs::Int = 3)

     #create the initial conditions file
     #for now, it will just be gene 1 at value 1 for all
     ics = Matrix{String}(undef, 2,2)
     ics[1,:] = ["Genes", "Values"]
     ics[2,1] = "['G1']"
     ics[2,2] = "[1]"

     #create reference file and save to .txt
     for i=1:length(seedList)
         refNet = randomRefNet(minGenes = minGenes, maxGenes = maxGenes,
          maxRegs = maxRegs, seed = seedList[i])

        writedlm(string("randomRefNet", i,".txt"), refNet, "\t")
        writedlm(string("randomRefNet", i, "_ics",".txt"), ics, "\t")
    end
end

#seedList = collect(1:10000)
#generateRefNetSet(seedList)

#seedList = collect(1:2000)
#generateRefNetSet(seedList; minGenes=10, maxGenes=10)

#seedList = collect(1:2000)
#generateRefNetSet(seedList; minGenes=10, maxGenes=10, maxRegs = 5)


#seedList = collect(1:10)
#generateRefNetSet(seedList, minGenes=1000, maxGenes=1000, maxRegs = 10)
