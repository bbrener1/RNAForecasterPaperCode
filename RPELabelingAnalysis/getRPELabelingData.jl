using PyCall

py"""
from anndata import read_h5ad, read_loom
import numpy as np
import pandas as pd

adata = read_h5ad("./data/rpeLabeling.h5ad")
#subset to kinetics
adata = adata[adata.obs.exp_type=='Pulse', :]
#remove non-labeling cells
adata.obs['time'] = adata.obs['time'].astype(str)
adata.obs.loc[adata.obs['time'] == 'dmso', 'time'] = -1
adata.obs['time'] = adata.obs['time'].astype(float)
adata = adata[adata.obs.time != -1, :]
#collate labeled, unlabeled, and total transcripts
adata.layers['new'], adata.layers['old'] = adata.layers['ul'] + adata.layers['sl'], adata.layers['su'] + adata.layers['uu']
adata.layers['total'] = adata.layers['ul'] + adata.layers['sl'] + adata.layers['su'] + adata.layers['uu']

new = adata.layers["new"].todense()
old = adata.layers["old"].todense()
total = adata.layers["total"].todense()
geneNames = adata.var['Gene_Id'].tolist()

def getNewCounts():
    return new

def getOldCounts():
    return old

def getTotalCounts():
    return total

def getGeneNames():
    return geneNames

def getCellCyclePosition():
    return adata.obs['Cell_cycle_possition'].tolist()

#how long was the cell in labeling media
def getCellLabelingTime():
    return adata.obs['time'].tolist()
"""

newData = py"getNewCounts"()
oldData = py"getOldCounts"()
totalData = py"getTotalCounts"()
geneNames = py"getGeneNames"()
ccPos = py"getCellCyclePosition"()
time = py"getCellLabelingTime"()

using DelimitedFiles
writedlm("rpe_LabeledTranscripts.csv", transpose(newData), ',')
writedlm("rpe_UnlabeledTranscripts.csv", transpose(oldData), ',')
writedlm("rpe_TotalTranscripts.csv", transpose(totalData), ',')
writedlm("rpe_geneNames.csv", geneNames, ',')
writedlm("rpe_CellCyclePosition.csv", ccPos, ',')
writedlm("rpe_cellLabelingTime.csv", time, ',')
