#!/usr/bin/env sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

cd "$SCRIPT_DIR" || exit

docker build -f Dockerfile.RL -t kuavo_rl_img:0.1 .
