##benchmark GPU vs CPU performance by data set size
include("trainRNAForecaster.jl");

#small data set
t1Small = Float32.(randn(10,1000))
t2Small = Float32.(randn(10,1000))

#medium data set
t1Medium = Float32.(randn(1000,1000))
t2Medium = Float32.(randn(1000,1000))

#large data set
t1Large = Float32.(randn(2000,20000))
t2Large = Float32.(randn(2000,20000))

##first time doesn't count because of compilation time
#get compilation out of the way
@time trainRNAForecaster(t1Small, t2Small, batchsize = 200,
            checkStability = false, useGPU = true)

@time trainRNAForecaster(t1Small, t2Small, batchsize = 200,
            checkStability = false, useGPU = false)
##

@time trainRNAForecaster(t1Small, t2Small, batchsize = 200,
            checkStability = false, useGPU = true)

@time trainRNAForecaster(t1Small, t2Small, batchsize = 200,
            checkStability = false, useGPU = false)

@time trainRNAForecaster(t1Medium, t2Medium, batchsize = 200,
            checkStability = false, useGPU = true)

@time trainRNAForecaster(t1Medium, t2Medium, batchsize = 200,
            checkStability = false, useGPU = false)

@time trainRNAForecaster(t1Large, t2Large, batchsize = 600,
            checkStability = false, useGPU = true)

#the below takes >1 week to run
@time trainRNAForecaster(t1Large, t2Large, batchsize::Int = 600, checkStability = false, useGPU = false)



##plots
using Plots

timeGPU = [4.46, 288.6, 94157]
timeCPU = [1.17, 12050, 691200]

memoryGPU = [0.28, 2.98, 563.3]
memoryCPU = [0.9, 4559, 4559]

dataSizes = ["10 x 1000", "1000  x 1000", "2000 x 20000"]

p1 = plot(dataSizes, log1p.(hcat(timeGPU, timeCPU)), xlabel = "Data Size genes x cells",
      ylabel = "Computational Time log(seconds)", labels = ["GPU" "CPU"], legend = :topleft)

savefig(p1, "GPUvCPUCompTime.pdf")

p2 = plot(dataSizes, log1p.(hcat(memoryGPU, memoryCPU)), xlabel = "Data Size genes x cells",
      ylabel = "Memory Allocations Required log(Gigabytes)", labels = ["GPU" "CPU"], legend = :topleft)

savefig(p2, "GPUvCPUMemory.pdf")
