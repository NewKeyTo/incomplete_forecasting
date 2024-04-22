export CUDA_VISIBLE_DEVICES=0
model_name=PatchTST
PUBLIC_DATA_PATH=/data/pdz/incomplete
data_name=ETTm1

# missing_rate=0.5
impute_method=SAITS
pred_len=48

for missing_rate in 0.5
do
python -u run.py \
  --task_name incomplete_long_term_forecast \
  --is_training 1 \
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
  --e_layers 1 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 2 \
  --batch_size 32 \
  --itr 3
done
