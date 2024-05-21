using Flux, CUDA
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())
include("import.jl")

function train()

x_train, y_train, x_test, y_test = import_data("dataset")

model = Chain(
    Conv((5, 5), 3 => 16, relu),
    MaxPool((2, 2)),
    Dropout(0.1),
    Conv((3, 3), 16 => 32, relu),
    MaxPool((2, 2)),
    Conv((5, 5), 32 => 64, relu),
    MaxPool((2, 2)),
    Conv((3, 3), 64 => 128, relu),
    MaxPool((4, 4)),
    Flux.flatten,
    Dropout(0.01),
    Dense(512 => 128, relu),
    BatchNorm(128),
    Dense(128 => 64, relu),
    BatchNorm(64),
    Dense(64 => 32, relu),
    BatchNorm(32),
    Dense(32 => 2, relu),
    softmax
) |> gpu

# model = Chain(
#     Conv((3, 3), 1 => 16, relu),
#     Conv((3, 3), 16 => 32, relu),
#     MaxPool((2, 2)),
#     Flux.flatten,
#     Dropout(0.0001),
#     Dense(28800 => 14400, tanh),
#     BatchNorm(14400),
#     Dense(14400 => 7700, relu),
#     BatchNorm(7700),
#     Dense(7700 => 3600, relu),
#     BatchNorm(3600),
#     Dense(3600 => 1024, tanh),
#     BatchNorm(1024),
#     Dense(1024 => 128, tanh),
#     BatchNorm(128),
#     Dense(128 => 64, relu),
#     BatchNorm(64),
#     Dense(64 => 32, relu),
#     BatchNorm(32),
#     Dense(32 => 2, relu),
#     softmax
# ) |> gpu


# training
target = collect(Flux.onehotbatch(y_train, ["Daisy", "Dandelion"]))

loader = Flux.DataLoader((x_train, target), batchsize=64, shuffle=true) |> gpu

optim = Flux.setup(Flux.Adam(0.01), model)


loss3(m, x, y) = Flux.logitcrossentropy(m(x), y)
for epoch in 1:1000
    Flux.train!(loss3, model, loader, optim)
end
return model
end

using JLD2

model = train() |> cpu
state = Flux.state(model)
jldsave("recognizer.jld2"; state)
