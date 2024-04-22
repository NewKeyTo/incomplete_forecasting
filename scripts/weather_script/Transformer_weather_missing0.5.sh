model_name=Transformer

for method in ZERO LOCF SAITS BRITS Transformer
do
    python -u run.py \
    --task_name incomplete_long_term_forecast \
    --is_training 1 \
    --origin_root_path ./incomplete/prepared_dataset/weather/ \
    --imputed_root_path ./incomplete/imputed_dataset/weather/ \
    --impute_method $method \
    --model_id weather_96_96 \
    --model $model_name \
    --data weather_missing0.5 \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 5
done