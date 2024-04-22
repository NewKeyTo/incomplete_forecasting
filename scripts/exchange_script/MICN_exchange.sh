export CUDA_VISIBLE_DEVICES=0
PUBLIC_DATA_PATH=/data/pdz/incomplete
model_name=MICN
data_name=exchange

pred_len=48
# impute_method=SAITS
# missing_rate=0.5

for missing_rate in 0.5
do
for impute_method in VAE
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
    --label_len 96 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --dec_in 8 \
    --c_out 8 \
    --d_model 64 \
    --d_ff 64 \
    --top_k 5 \
    --des 'Exp' \
    --itr 1
done
done