set -x

expoert VLLM_ATTENTION_BACKEND=XFORMERS

ray stop
sleep 5

# ray start --head --node-ip-address 0.0.0.0 --num-gpus 1 --ray-debugger-external --port 6380



MODEL_PATH=/data1/bowei/QUILL_ALL/model/Qwen2-0.5B-Instruct  # replace it with your local file path

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

CUDA_VISIBLE_DEVICES=6,7 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.n_gpus_per_node=2
