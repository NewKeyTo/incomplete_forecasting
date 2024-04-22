from data_provider.incomplete_data_loader import Incomplete_Forecasting_Dataset
from torch.utils.data import DataLoader

# data_dict = {
#     'PhysioNet': PhysioNetloader,
# }

def data_provider(args, flag):
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = False
        batch_size = args.batch_size  # bsz for train and valid
    
    data_set = Incomplete_Forecasting_Dataset(
        data=args.data,
        origin_root_path=args.origin_root_path,
        imputed_root_path=args.imputed_root_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        impute_method=args.impute_method
    )
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    print('{}:{}'.format(flag, len(data_set)))
    return data_set, data_loader
    
