import argparse
import os
import torch
import random
import numpy as np
import time
from utils.tools import setup_logger
from exp.exp_common_forecasting import Exp_Common_forecasting
from exp.exp_dmae_forecasting import Exp_Dmae_forecasting

def set_seed(fix_seed):
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed) 
    torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False 
    np.random.seed(fix_seed)

def set_gpu(args):
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        
def getlogger(args, flag='train'):
    log_path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if(flag == 'train'):
        mode = 'w'
    else:
        mode = 'a'
    logger = setup_logger(
        os.path.join(log_path, flag + '.log'),
        flag + " log",
        mode= mode,
    )
    return logger 

def get_setting(args, rand_seed):
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des, 
        args.impute_method,
        rand_seed
    )
    return setting

def get_parser():
    parser = argparse.ArgumentParser(description='TimesNet')
    parser.add_argument('--task_name', type=str, required=False, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--chunk_size', type=int, default=24, help='chunk_size of LightTS')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=20, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight_decay of optimizer')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    
    # incomplete data params
    parser.add_argument('--origin_root_path', type=str, default='/data/pdz/incomplete/prepared_dataset/ETTm1/', help='orgin data path')
    parser.add_argument('--imputed_root_path', type=str, default='/data/pdz/incomplete/imputed_dataset/ETTm1/', help='imputed data path')
    parser.add_argument('--impute_method', type=str, default='SAITS', help='imputed method')
    
    parser.add_argument('--pretrain', action='store_true', help='if pretrain', default=False)
    parser.add_argument('--pretrain_epoch', type=int, help='pretrain_epoch', default=20)
    parser.add_argument('--mask_ratio', type=float, help='mask_ratio', default=0.1)
    parser.add_argument('--warmup', type=int, help='warmup', default=20)

    return parser
    

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    set_gpu(args)
    if args.model == 'DBT_DMAE':
        Exp = Exp_Dmae_forecasting
    else:
        Exp = Exp_Common_forecasting
    
    for randseed in range(2021, args.itr + 2021):
        args.randseed = randseed
        set_seed(randseed)
        setting = get_setting(args, randseed)
        args.setting = setting 
        
        logger = getlogger(args)
        logger.info(args)
        
        exp = Exp(args, logger)
        
        start_time = time.time()
        logger.info('>>>>>>>Start Training At Time {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(start_time))
        exp.train(setting)
        logger.info('>>>>>>>End training at time {} elapsed time {}>>>>>>>>>>>>>>>>>'.format(time.time(), time.time()-start_time))
        
        logger.info('>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
        end_time = time.time()
        logger.info('>>>>>>>End At Time {} Elapsed Time {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(end_time, end_time - start_time))