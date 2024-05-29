using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using JSON
using DelimitedFiles

# That can't be right =/
include(joinpath(@__DIR__,"trainRNAForecasterV2.jl"))
include(joinpath(@__DIR__,"python_interface_train.jl"))

prefix = joinpath(@__DIR__, "tmp")
if length(ARGS) > 0
    prefix = ARGS[1]
end

function load_model(inputNodes,hiddenLayerNodes)
    model_path = joinpath(prefix,"trained.jld2")
    model = defaultNetwork(inputNodes,hiddenLayerNodes)
    model = loadmodel!(model, load_object(fileName))
    return model
end

function load_data(prefix)
    path_pt1 = joinpath(prefix,"pt1.csv")
    pt1 = readdlm(path_pt1, ',')
    return pt1
end


parameters = load_params()
model = defaultNetwork(parameters...)