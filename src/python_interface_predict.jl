using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using JSON
using JLD2
using DelimitedFiles
using Flux: loadmodel!



# Figure out ways to do imports properly dogg
include(joinpath(@__DIR__,"trainRNAForecasterV2.jl"))
include(joinpath(@__DIR__,"recursivePredictionsV2.jl"))
# include(joinpath(@__DIR__,"python_interface_train.jl"))

prefix = joinpath(@__DIR__, "tmp")
if length(ARGS) > 0
    prefix = ARGS[1]
end

function load_model(inputNodes,hiddenLayerNodes,ensemblePrefix)
    model_path = joinpath(prefix,"trained_e$(ensemblePrefix).jld2")
    _,blank_model = defaultNetwork(inputNodes,hiddenLayerNodes)
    model = loadmodel!(blank_model, load_object(model_path))
    return model
end

function load_data(prefix)
    path_pt1 = joinpath(prefix,"pt1.tsv")
    pt1 = transpose(readdlm(path_pt1))
    pt1 = Float32.(pt1)
    return pt1
end

function load_training_params()
    file = open(joinpath(prefix,"training_params.txt"), "r")
    dict_obj = JSON.parse(read(file, String))
    close(file)
    translated = Dict(Symbol(k) => v for (k,v) in dict_obj)
    return translated
end

function load_prediction_params()
    params = Dict(
        :tSteps => 6,
        :useGPU => false,
        :damping => 0.7f0,
        :batchSize => 100
    )

    file = open(joinpath(prefix,"prediction_params.txt"), "r")
    dict_obj = JSON.parse(read(file, String))
    close(file)
    translated = Dict(Symbol(k) => v for (k,v) in dict_obj)
    for (key,value) in translated
        params[key] = value
    end
    params[:damping] = Float32.(params[:damping])
    return params
end

function write_futures(futures) 
    for i = 1:(size(futures)[3])
        println("Writing timepoint $i")
        writedlm(joinpath(prefix,"./ft$i.tsv"),transpose(futures[:,:,i]))
    end
end


println("Working at $prefix")
training_params = load_training_params()
prediction_params = load_prediction_params()
println("Read params $prediction_params, $training_params")

ensemble_prefix = pop!(prediction_params,:ensembleSuffix,"1")

t1 = load_data(prefix) 
model = load_model(size(t1)[1],training_params[:hiddenLayerNodes],ensemble_prefix)

futures = predictSimplified(model,t1;prediction_params...)
# futures = predictSimplified(model,t1,damping=Float32.(prediction_params[:damping]))
write_futures(futures)
