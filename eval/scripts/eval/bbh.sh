# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# evaluating llama 7B model using chain-of-thought
python -m eval.bbh.run_eval \
    --data_dir data/eval/bbh \
    --save_dir results/bbh/llama-7B-cot/ \
    --model ../hf_llama_models/7B \
    --tokenizer ../hf_llama_models/7B \
    --max_num_examples_per_task 40 \
    --use_vllm
