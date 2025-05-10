#!/bin/bash
docker run --rm --gpus all -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/gpt2_export:/models \
  nvcr.io/nvidia/tritonserver:24.03-py3 \
  tritonserver --model-repository=/models
