# Core test — OLS vs trained W_K across all 24 layers
python linear_attention_ols.py --model gpt2-medium

# With training comparison on small model
python linear_attention_ols.py --compare_training
