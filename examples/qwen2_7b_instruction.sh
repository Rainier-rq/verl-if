set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_PATH=/fs-computility/DeepSeek-R1-Distill-Qwen-7B
ray job submit --address="http://127.0.0.1:8265" \
    -- python3 -m verl.trainer.main \
        config=examples/config.yaml \
        worker.actor.model.model_path=${MODEL_PATH} \
        trainer.n_gpus_per_node=8 \


    
