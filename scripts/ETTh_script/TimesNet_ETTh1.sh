export CUDA_VISIBLE_DEVICES=1
PUBLIC_DATA_PATH=/data/pdz/incomplete
model_name=TimesNet
data_name=ETTh1

impute_method=SAITS
# missing_rate=0.5

for missing_rate in 0.1 0.3 0.5 0.7 0.9
do
python -u run.py \
  --task_name incomplete_long_term_forecast \
  --is_training 1 \
  --origin_root_path $PUBLIC_DATA_PATH/prepared_dataset/$data_name/ \
  --imputed_root_path $PUBLIC_DATA_PATH/imputed_dataset/$data_name/ \
  --impute_method $impute_method \
  --model_id $data_name'_96_96' \
  --model $model_name \
  --data $data_name'_missing'$missing_rate \
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
  --d_model 16 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 3 \
  --top_k 5
done

# python -u run.py \
#   --task_name incomplete_long_term_forecast \
#   --is_training 1 \
#   --origin_root_path ./incomplete/prepared_dataset/weather/ \
#   --imputed_root_path ./incomplete/imputed_dataset/weather/ \
#   --model_id weather_96_96 \
#   --model TimesNet \
#   --impute_method LOCF \
#   --data weather_missing0.5 \
#   --features M \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --d_model 32 \
#   --d_ff 32 \
#   --top_k 5 \
#   --des 'Exp' \
#   --itr 1 