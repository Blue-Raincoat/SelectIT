# Please make sure OPENAI_API_KEY is set in your environment variables

# Use V1 of alpaca farm evaluation.
export IS_ALPACA_EVAL_2=False

# use vllm for generation
python -m eval.alpaca_farm.run_eval \
    --model_name_or_path ../checkpoints \
    --save_dir results/alpaca_farm/checkpoints/ \
    --eval_batch_size 20 \
    --use_vllm \
    --use_chat_format \
    --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
