# Set sglang server for local LLM
# No need to run this file if using proprietary model

CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path $MODEL_NAME --disable-cuda-graph --attention-backend flashinfer
