export CUDA_VISIBLE_DEVICES=2
model_name=PatchTST
PUBLIC_DATA_PATH=/data/pdz/incomplete
data_name=illness

# missing_rate=0.5
# impute_method=SAITS
# pred_len=24

for missing_rate in 0.5
do
for impute_method in SAITS
do
for pred_len in 24
do
python -u run.py \
  --task_name incomplete_long_term_forecast \
  --is_training 1 \
  --origin_root_path $PUBLIC_DATA_PATH/prepared_dataset/$data_name/ \
  --imputed_root_path $PUBLIC_DATA_PATH/imputed_dataset/$data_name/ \
  --impute_method $impute_method \
  --model_id $data_name'_36_'$pred_len \
  --model $model_name \
  --data $data_name'_missing'$missing_rate \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len $pred_len \
  --e_layers 4 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_heads 4 \
  --d_model 1024\
  --itr 3
done
done
done
  # --train_epoch 100 \