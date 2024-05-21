using Images, FileIO, Random

# imagePreprocess(x) = reshape(
#     Float32.(
#         channelview(
#             imresize(
#                 Gray.(x),
#                 ratio=1/8
#             )
#         )
#     ),
#     64, 64, 1)
imagePreprocess(x) = reshape(
    Float32.(
        channelview(
            imresize(
                x,
                ratio=1/4
            )
        )
    ),
    128, 128, 3)

function import_data(dataset_path)
    dataset = Dict("Daisy" => "daisy", "Dandelion" => "dandelion")
    train_paths = Dict()
    train_files = Dict()
    test_paths = Dict()
    test_files = Dict()
        
    for (k, v) in dataset
        train_paths[k] = joinpath(dataset_path, "train", v)
        train_files[k] = map(x -> joinpath(dataset_path, "train", v, x), readdir(train_paths[k]))
        test_paths[k] = joinpath(dataset_path, "test", v)
        test_files[k] = map(x -> joinpath(dataset_path, "test", v, x), readdir(test_paths[k]))
        test_paths[k] = joinpath(dataset_path, "valid", v)
        test_files[k] = append!(test_files[k], map(x -> joinpath(dataset_path, "valid", v, x), readdir(test_paths[k])))
    end
    dataset_files = []
    for (k, v) in dataset
        push!(dataset_files, zip(Iterators.cycle([k]), train_files[k])...)
    end
    # Random.shuffle!(Random.default_rng(), dataset_files)
    x_train = cat(map(x -> imagePreprocess(load(x[2])), dataset_files)..., dims=4)
    y_train = map(x -> x[1], dataset_files)
    
    dataset_files = []
    for (k, v) in dataset
        push!(dataset_files, zip(Iterators.cycle([k]), test_files[k])...)
    end
    # Random.shuffle!(Random.default_rng(), dataset_files)
    x_test = cat(map(x -> imagePreprocess(load(x[2])), dataset_files)..., dims=4)
    y_test = map(x -> x[1], dataset_files)
    return x_train, y_train, x_test, y_test
end
