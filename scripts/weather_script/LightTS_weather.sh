model_name=LightTS
PUBLIC_DATA_PATH=/data/pdz/incomplete
data_name=weather

# impute_method=SAITS
# missing_rate=0.5
pred_len=48

for missing_rate in 0.5
do
for impute_method in SAITS
do
python -u run.py \
  --task_name incomplete_long_term_forecast \
  --origin_root_path $PUBLIC_DATA_PATH/prepared_dataset/$data_name/ \
  --imputed_root_path $PUBLIC_DATA_PATH/imputed_dataset/$data_name/ \
  --impute_method $impute_method \
  --model_id $data_name'_96_'$pred_len \
  --model $model_name \
  --data $data_name'_missing'$missing_rate \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --itr 3
done
done