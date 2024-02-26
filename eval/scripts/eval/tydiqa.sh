# Here we use 1 GPU for demonstration, but you can use multiple GPUs and larger eval_batch_size to speed up the evaluation.
export CUDA_VISIBLE_DEVICES=0


# Evaluating llama 7B model, with gold passage provided
# By default, we use 1-shot setting, and 100 examples per language
python -m eval.tydiqa.run_eval \
    --data_dir data/eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir results/tydiqa/llama-7B-goldp \
    --model ../hf_llama_model/7B \
    --tokenizer ../hf_llama_model/7B \
    --eval_batch_size 20 \
    --load_in_8bit

