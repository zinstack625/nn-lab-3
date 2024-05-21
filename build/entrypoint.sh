#!/usr/bin/env sh

die() {
    echo $1 && exit 1
}

shutdown() {
    kill -INT $1
}

run() {
    [ -z "$BOT_TOKEN" ] && die "BOT_TOKEN unset! Cannot continue"

    /usr/bin/nnlab3 &
    PID=$!

    trap 'shutdown $PID' TERM
    wait $PID
}

run


