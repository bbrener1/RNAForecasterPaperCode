from anndata import read_h5ad, read_loom
from urllib.request import urlretrieve
from pathlib import Path
import os
import ntpath
import pandas as pd

urlretrieve("https://www.dropbox.com/s/25enev458c8egn7/rpe1.h5ad?dl=1",
"./data/rpeLabeling.h5ad")


adata = read_h5ad("./data/rpeLabeling.h5ad")
adata.layers['new'], adata.layers['total'] = adata.layers['ul'] + adata.layers['sl'], adata.layers['su'] + adata.layers['sl'] + adata.layers['uu'] + adata.layers['ul']
