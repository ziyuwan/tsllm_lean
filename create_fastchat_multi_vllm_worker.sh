CONTROLER_PORT=21801
echo PYTHON_EXECUTABLE=$(which python3)
PYTHON_EXECUTABLE=$(which python3)
WORKER_BASE_PORT=30010
MODEL_PATH="/data/workspace/muning/GloveInDark/MATH_PJ/ReProver/sft_leandojo_deepseek_math_7b/checkpoint-1463"

tmux start-server

tmux new-session -s FastChat -n controller -d
# tmux send-keys "unset http_proxy" Enter
tmux send-keys "$PYTHON_EXECUTABLE -m fastchat.serve.controller --port ${CONTROLER_PORT}" Enter

echo "Wait 10 seconds ..."
sleep 10

N_GPUS=1
START_GPU_ID=7
echo "Starting workers"
for i in $(seq 0 $(($N_GPUS-1)))
do
  WORKER_PORT=$((WORKER_BASE_PORT+i))
  # echo "CUDA_VISIBLE_DEVICES=$(($START_GPU_ID+$i)) $PYTHON_EXECUTABLE -m fastchat.serve.vllm_worker --model-path $MODEL_PATH --controller-address http://localhost:$CONTROLER_PORT --port $WORKER_PORT --worker-address http://localhost:$WORKER_PORT --swap-space 32" Enter
  tmux new-window -n worker_$i
#   tmux send-keys "unset http_proxy" Enter
  tmux send-keys "CUDA_VISIBLE_DEVICES=$(($START_GPU_ID+$i)) $PYTHON_EXECUTABLE -m fastchat.serve.vllm_worker --model-path $MODEL_PATH --controller-address http://localhost:$CONTROLER_PORT --port $WORKER_PORT --worker-address http://localhost:$WORKER_PORT --swap-space 32" Enter
done
