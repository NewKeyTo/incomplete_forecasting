model_name=Transformer

for method in ZERO LOCF SAITS BRITS Transformer
do
    python -u run.py \
    --task_name incomplete_long_term_forecast \
    --is_training 1 \
    --origin_root_path /data/pdz/incomplete/prepared_dataset/ETTm1/ \
    --imputed_root_path /data/pdz/incomplete/imputed_dataset/ETTm1/ \
    --impute_method ZERO \
    --model_id ETTm1_96_96 \
    --model Transformer \
    --data ETTm1_missing0.5 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --itr 1
done


python -u run.py \
  --task_name incomplete_long_term_forecast \
  --is_training 1 \
  --origin_root_path ./incomplete/prepared_dataset/ETTm1/ \
  --imputed_root_path ./incomplete/imputed_dataset/ETTm1/ \
  --impute_method ZERO \
  --model_id ETTm1_96_96 \
  --model Autoformer \
  --data ETTm1_missing0.5 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1