model_name=DBT_DMAE
PUBLIC_DATA_PATH=/data/pdz/incomplete
data_name=weather

impute_method=ZERO
pred_len=48

for missing_rate in 0.5
do
python -u run.py \
  --task_name incomplete_long_term_forecast \
  --is_training 1 \
  --origin_root_path $PUBLIC_DATA_PATH/prepared_dataset/$data_name/ \
  --imputed_root_path $PUBLIC_DATA_PATH/imputed_dataset/$data_name/ \
  --impute_method $impute_method \
  --data $data_name'_missing'$missing_rate \
  --model_id $data_name'_96_'$pred_len \
  --model $model_name \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len $pred_len \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --pretrain \
  --itr 1
done