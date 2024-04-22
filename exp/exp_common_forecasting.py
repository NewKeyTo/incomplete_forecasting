from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import csv
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Common_forecasting(Exp_Basic):
    def __init__(self, args, logger):
        super(Exp_Common_forecasting, self).__init__(args, logger)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        mask_num = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mask_x, mask_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                mask_x = mask_x.float().to(self.device)
                mask_y = mask_y.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_x)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_x)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                mask = mask_y[:, -self.args.pred_len:, f_dim:].detach().cpu()

                loss = criterion(pred[mask == 1], true[mask == 1]) * mask.sum()
                mask_num = mask_num + mask.sum()
                total_loss.append(loss)
        total_loss = np.sum(total_loss) / mask_num
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, logger=self.logger)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mask_x, mask_y) in enumerate(train_loader):
                # print(iter_count)
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                mask_x = mask_x.float().to(self.device)
                mask_y = mask_y.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_x)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_x)[0]
                    else:
                        # s_time = time.time()
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs[mask_y[:, -self.args.pred_len:, f_dim:] == 1], \
                    batch_y[mask_y[:, -self.args.pred_len:, f_dim:] == 1])
                train_loss.append(loss.item())
                

                if (i + 1) % 100 == 0:
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    self.logger.info(
                        f'iters: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}'
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    self.logger.info(
                        f'speed: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.4f}'
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            self.logger.info(f'Epoch: {epoch + 1} cost time: {time.time() - epoch_time}')
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            self.logger.info(
                f'Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}'
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.logger.info(f'Early stopping')
                break
        
            adjust_learning_rate(model_optim, epoch + 1, self.args, self.logger)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.logger.info(f'loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        a_trues = []
        masks = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mask_x, mask_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                mask_x = mask_x.float().to(self.device)
                mask_y = mask_y.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_x)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_x)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_x)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                mask_y = mask_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                mask_y = mask_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y
                
                preds.append(pred)
                trues.append(true)
                a_trues.append(batch_x[:,-self.args.pred_len:, f_dim:].detach().cpu().numpy())
                masks.append(mask_y)

                # inverse_pred = test_data.inverse(outputs[0])
                # inverse_true = test_data.inverse(batch_y[0])
                # inverse_preds.append(inverse_pred)
                # inverse_trues.append(inverse_true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        atrues = np.concatenate(a_trues, axis=0)
        masks = np.concatenate(masks, axis=0)
       
        # inverse_preds = np.array(inverse_preds).reshape(-1, inverse_preds.shape[-2], inverse_preds.shape[-1])
        # inverse_trues = np.array(inverse_trues).reshape(-1, inverse_trues.shape[-2], inverse_trues.shape[-1])
        
        self.logger.info(f'Test Shape: {preds.shape}')

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds[masks == 1], trues[masks == 1])
        amae, amse, armse, amape, amspe = metric(preds[masks == 1], atrues[masks == 1])
        print('aaaaa:', amae, amse)
        self.logger.info(
            f'--TEST METRIC INFO--\n\
            MSE:{mse}, MAE:{mae} \n\
            rmse:{rmse}, mape:{mape}, mspe:{mspe}'
        )
        
        save_dict = {
            'metric':{'mae':mae, 'mse':mse, 'rmase':rmse, 'mape':mape, 'mspe':mspe},
            'pred': preds,
            'true': trues,
            # 'inverse_pred': inverse_preds,
            # 'inverse_true': inverse_trues,
            'scaler': test_data.scaler
        }
        folder_path = './results/res/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        import pickle
        with open(os.path.join(folder_path, "result.pkl"), "wb") as f:
            pickle.dump(save_dict, f)
        
        data_head = ['model', 'data_set', 'impute_method', 'randseed', 
                    'mae', 'mse', 'rmse', 'mape', 'mspe',
                    'seq_len', 'pred_len','setting']
        data_result = [[self.args.model, self.args.data, self.args.impute_method, self.args.randseed,
                    mae, mse, rmse, mape, mspe,
                    self.args.seq_len, self.args.pred_len,self.args.setting]]
        filename = "./results/incomplete_forecasting_metrics.csv"
        
        with open(filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not os.path.isfile(filename):  # 如果文件不存在，则写入标题行
                writer.writerow(data_head)
            writer.writerows(data_result)
            print('Csv Result saved to {}'.format(filename))
        return