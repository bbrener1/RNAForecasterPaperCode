import tempfile
import sys
import os
import json
import subprocess as sp
import numpy as np


def default_prefix():
    return os.path.join(os.path.dirname(__file__),"tmp")

def julia_prefix():
    return os.path.dirname(__file__)

def write_parameters(param_dict,file_name,prefix=default_prefix()):
    p_file_path = os.path.join(prefix,file_name)
    with open(p_file_path, 'w') as file:
        json.dump(param_dict, file)    

def write_training_data(t1,t2,prefix=default_prefix()):
    t1_path = os.path.join(prefix,"t1.tsv")
    t2_path = os.path.join(prefix,"t2.tsv")
    np.savetxt(t1_path,t1)
    np.savetxt(t2_path,t2)

def write_prediction_data(pt1,prefix=default_prefix()):
    pt1_path = os.path.join(prefix,"pt1.tsv")
    np.savetxt(pt1_path,pt1)
    
def load_futures(n_futures=None,prefix=default_prefix()):

    if n_futures is None:
        future_glob = os.path.join(prefix,"ft*.tsv")
        n_futures = len(glob.glob(future_glob))
    
    futures = []
    # Julia is 1-indexed
    for i in range(1,n_futures+1):
        future_path = os.path.join(prefix,f"ft{i}.tsv")
        future = np.loadtxt(future_path)
        futures.append(future)
    
    return np.array(futures)

def train(t1,t2, prefix=default_prefix(), user_params=None):
    params = {
        "hiddenLayerNodes":1000,
        "seed":123,
        "learningRate": 5e-3,
        "nEpochs":10,
        "batchSize":100,
        "useGPU":False,
    }
    user_params = {} if user_params is None else user_params
    for key in user_params:
        params[str(key)]= user_params[key]

    write_training_data(t1,t2,prefix=prefix)
    write_parameters(params,"training_params.txt",prefix=prefix,)
    julia_path = os.path.join(julia_prefix(),"python_interface_train.jl")
    # tmp_dir = tempfile.mkdtemp(prefix=default_prefix())
    sp.run(["julia",julia_path,prefix])


def train_predict(t1,t2,prefix=default_prefix(), training_params=None, prediction_params=None):
    train(t1,t2,prefix=prefix,params=training_params)

def predict(t1,prefix=default_prefix(),user_params=None):
    params = {
        "tSteps":6,
        "useGPU":False,
        "batchSize":100,
        "damping":0.3,
    }
    user_params = {} if user_params is None else user_params
    for key in user_params:
        params[key]= user_params[key]    

    write_prediction_data(t1,prefix=prefix)
    write_parameters(params,"prediction_params.txt",prefix=prefix)

    julia_path = os.path.join(julia_prefix(),"python_interface_predict.jl")
    sp.run(["julia",julia_path,prefix])

    n_futures = int(params["tSteps"])
    futures = load_futures(n_futures = n_futures,prefix=prefix)
    return 






#     default_dict = {
#         trainingProp : 0.8,
# #        hiddenLayerNodes : 2*size(expressionDataT1)[1],  # TODO we're leaving this out for now, but need to compute it
#         shuffleData : true,
#         seed : 123,
#         learningRate : 0.005,
#         nEpochs : 10,
#         batchsize : 100,
#         checkStability : false,
#         iterToCheck : 50,
#         stabilityThreshold : 2*maximum(expressionDataT1),
#         stabilityChecksBeforeFail : 5,
#         useGPU : false
#     }
