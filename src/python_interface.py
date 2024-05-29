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

def write_parameters(param_dict,prefix=default_prefix()):
    
    p_file_path = os.path.join(prefix,"run_params.txt")

    with open(p_file_path, 'w') as file:
        json.dump(param_dict, file)    

def write_training_data(t1,t2,prefix=default_prefix()):
    t1_path = os.path.join(prefix,"t1.csv")
    t2_path = os.path.join(prefix,"t2.csv")
    np.savetxt(t1_path,t1,delimiter=",")
    np.savetxt(t2_path,t2,delimiter=",")

def write_prediction_data(pt1,prefix=default_prefix()):
    pt1_path = os.path.join(prefix,"pt1.csv")
    np.savetxt(pt1_path,pt1,delimiter=",")
    

def train(prefix=default_prefix()):
    julia_path = os.path.join(julia_prefix(),"python_interface_train.jl")
    sp.run(["julia",julia_path,prefix])

def predict(prefix=default_prefix()):
    julia_path = os.path.join(julia_prefix(),"python_interface_predict.jl")
    sp.run(["julia",julia_path,prefix])

def predict():
    pass

def exfiltrate():
    pass






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
