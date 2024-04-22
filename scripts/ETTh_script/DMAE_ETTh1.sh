export CUDA_VISIBLE_DEVICES=0
model_name=DBT_DMAE
PUBLIC_DATA_PATH=/data/pdz/incomplete
data_name=ETTh1

# impute_method=ZERO
missing_rate=0.5

# impute_method=Transformer

for impute_method in ZERO
do
  python -u run.py \
    --task_name incomplete_long_term_forecast \
    --is_training 1 \
    --origin_root_path $PUBLIC_DATA_PATH/prepared_dataset/$data_name/ \
    --imputed_root_path $PUBLIC_DATA_PATH/imputed_dataset/$data_name/ \
    --impute_method $impute_method \
    --data $data_name'_missing'$missing_rate \
    --model_id $data_name'_96_96' \
    --model $model_name \
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
    --itr 3
done

# for impute_method in Transformer CSDI
# do
#   python -u run.py \
#     --task_name incomplete_long_term_forecast \
#     --is_training 1 \
#     --origin_root_path $PUBLIC_DATA_PATH/prepared_dataset/$data_name/ \
#     --imputed_root_path $PUBLIC_DATA_PATH/imputed_dataset/$data_name/ \
#     --impute_method $impute_method \
#     --data $data_name'_missing'$missing_rate \
#     --model_id $data_name'_96_96' \
#     --model $model_name \
#     --features M \
#     --seq_len 96 \
#     --label_len 48 \
#     --pred_len 96 \
#     --e_layers 2 \
#     --d_layers 1 \
#     --factor 3 \
#     --enc_in 7 \
#     --dec_in 7 \
#     --c_out 7 \
#     --des 'Exp' \
#     --itr 3
# done