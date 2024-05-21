# NN Lab 3
## Image classification

Image classifier using Julia and Flux. Included model parameters are to classify daisies and dandelions. Estimated accuracy is 79%.

Dataset is taken from [Kaggle](https://www.kaggle.com/datasets/alsaniipe/flowers-dataset), with all due regards to Al Sani.

Training script is at `src/train.jl`, it uses `src/import.jl` to load images. While `src/train.jl` should be dataset-agnostic to a large extent, `src/import.jl` is not, but can be relatively easily made to support others. Mentioned dataset is expected to reside in `./dataset`, easy to change in `src/train.jl`, didn't bother making it a parameter. Model parameters are saved to "recognizer.jld2" to then be loaded in the end deployment.

Telegram bot resides in `src/nnlab3.jl`, it uses `src/predict.jl` as a separate module to get access to the model. In a desparate maniacal delusion I tried making it into a C library but gave up midway. Model is executed on the CPU because I didn't bother. Making it run on the GPU is trivial, but unnecessary.

Packaged into [Docker](https://quay.zinstack.ru/repository/zinstack625/nn-lab3), has a primitive helm chart for kubernetes deployment. Image build can be performed with `docker build -t <whatever> . -f build/Dockerfile`, and later launched with `docker run -d -e BOT_TOKEN=<snip> <whatever>`. After a minute it'll be up and running. Optionally, model parameters can be overriden with a mount and an envvar `MODEL_PATH` (`/recognizer.jld2` by default)

