FROM julia:latest

RUN apt-get update && \
    apt-get install -y clang

COPY ./Manifest.toml ./Project.toml /src/
COPY build/build.jl /src/build/
COPY src /src/src/
RUN cd /src && julia -t auto -q --project /src/build/build.jl

FROM debian:latest

ENV MODEL_PATH=/recognizer.jld2
COPY --chmod=555 build/entrypoint.sh /
COPY src/recognizer.jld2 /
COPY --from=0 /src/artifact /usr/

ENTRYPOINT ["/entrypoint.sh"]
