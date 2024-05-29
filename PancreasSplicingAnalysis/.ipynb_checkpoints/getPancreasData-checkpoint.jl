using PyCall

py"""
import scvelo as scv
import numpy as np
import pandas as pd
adata = scv.datasets.pancreas()
spliced = adata.layers["spliced"].todense()
unspliced = adata.layers["unspliced"].todense()
hvg = adata.var['highly_variable_genes'].tolist()
cellMD = adata.obs

def getHVGenes():
    return hvg

def getSpliced():
    return spliced

def getUnspliced():
    return unspliced

def getNames():
    return adata.var.index.tolist()

def getCellMetaData():
    return cellMD['clusters'].tolist()

"""

splicedData = py"getSpliced"()
unsplicedData = py"getUnspliced"()
hvGenes = py"getHVGenes"()
geneNames = py"getNames"()
cellMetaData = py"getCellMetaData"()

splicedVarGenes = transpose(splicedData[:, findall(x->x == "True", hvGenes)])
unsplicedVarGenes = transpose(unsplicedData[:, findall(x->x == "True", hvGenes)])
varGeneNames = geneNames[findall(x->x == "True", hvGenes)]
splicedData = transpose(splicedData)
unsplicedData = transpose(unsplicedData)

#write
using DelimitedFiles
writedlm("pancSplicedCounts.csv", splicedData, ",")
writedlm("pancUnsplicedCounts.csv", splicedData, ",")
writedlm("pancGeneNames.csv", geneNames, ",")

writedlm("pancSplicedCountsHVGs.csv", splicedVarGenes, ",")
writedlm("pancUnsplicedCountsHVGs.csv", unsplicedVarGenes, ",")
writedlm("pancHVGNames.csv", varGeneNames, ",")
