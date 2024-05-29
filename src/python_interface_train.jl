using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using JSON
using JLD2
using DelimitedFiles

# That can't be right =/
include(joinpath(@__DIR__,"trainRNAForecasterV2.jl"))

prefix = joinpath(@__DIR__,"tmp")
if length(ARGS) > 0
    prefix = ARGS[1]
end

function load_params()
    file = open(joinpath(prefix,"run_params.txt"), "r")
    dict_obj = JSON.parse(read(file, String))
    close(file)
    translated = Dict(Symbol(k) => v for (k,v) in dict_obj)
    return translated
end

function load_data(prefix)
    path_t1 = joinpath(prefix,"t1.csv")
    path_t2 = joinpath(prefix,"t2.csv")
    t1 = readdlm(path_t1, ',')
    t2 = readdlm(path_t2, ',')
    return (t1,t2)
end


println("Working at $prefix")
params = load_params()
println("Read params $params")

(t1,t2) = load_data(prefix) 
(t1,t2) = (Float32.(t1),Float32.(t2))
println("Read data $(summary(t1)),$(summary(t2))")

trained = trainRNAForecaster(t1,t2;params...)

outModel = cpu(trained[1])
model_path = joinpath(prefix,"trained.jld2")
save_object(model_path, Flux.state(outModel))
