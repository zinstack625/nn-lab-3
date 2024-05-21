module predictor

using Flux, JLD2

mutable struct State
    model
end

globalState = State(Nothing)

function restore(modelPath::String)
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
    )

    state = JLD2.load(modelPath, "state")
    Flux.loadmodel!(model, state)
    return model
end

Base.@ccallable function init(modelPath::Cstring)::Cvoid
    globalState.model = restore(modelPath)
    return
end

Base.@ccallable function predict(image::Ptr{Ptr{Ptr{Cfloat}}})::Ptr{Cfloat}
    image = unsafe_wrap(Array, image, (128, 128, 3))
    result = model(cat(image, dims=4))
    return result
end

function init(modelPath::String)
    globalState.model = restore(modelPath)
    globalState.model(cat(rand(Float32, 128, 128, 3), dims=4))
end

function predict(image::Array{Float32, 3})
    return globalState.model(cat(image, dims=4))[:,1]
end

end
