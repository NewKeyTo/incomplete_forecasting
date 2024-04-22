from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.dmae_utils.transform import *
import torch
import torch.nn as nn
from torch import optim
import os
import csv
import time
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

warnings.filterwarnings('ignore')

class Exp_Dmae_forecasting(Exp_Basic):
    def __init__(self, args, logger):
        super(Exp_Dmae_forecasting, self).__init__(args, logger)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        self.saving_path = os.path.join(self.args.checkpoints, self.args.setting)
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)
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
    
    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        mask_num = 0
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mask_x, mask_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                mask_x = mask_x.to(self.device)
                mask_y = mask_y.to(self.device)

                outputs, _ = self.model(batch_x, mask_x, dynamic_uniform=False)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                mask = mask_y[:, -self.args.pred_len:, f_dim:].detach().cpu()

                loss = criterion(pred[mask == 1], \
                    true[mask == 1]) * mask.sum()
                mask_num = mask_num + mask.sum()
                total_loss.append(loss)
        total_loss = np.sum(total_loss) / mask_num
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        if(self.args.pretrain):
            self.pretrain(train_loader, vali_loader)
            
            self.logger.info(f'loading pretrain model')
            pretrain_state = torch.load(os.path.join(self.saving_path + '/' + 'checkpoint_pretrain.pth'))
            pretrained_para = {k: v for k, v in pretrain_state.items() if not k.startswith('decoder')}
            self.model.state_dict().update(pretrained_para)

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
                
                mask_x = mask_x.to(self.device)
                mask_y = mask_y.to(self.device)
                
                outputs, _ = self.model(batch_x, mask_x, dynamic_uniform=False)
                
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                loss = criterion(outputs[mask_y[:, -self.args.pred_len:, f_dim:] == 1], \
                    batch_y[mask_y[:, -self.args.pred_len:, f_dim:] == 1])
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    self.logger.info(
                        f'iters: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}'
                    )
                    speed = (time.time() - time_now) / iter_count
                    self.logger.info(
                        f'speed: {speed}, epoch: {epoch + 1} | loss: {loss.item():.4f}'
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

            self.logger.info(
                f'Epoch: {epoch + 1} cost time: {time.time() - epoch_time}'
            )
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            self.logger.info(
                f'Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}'
            )
            early_stopping(vali_loss, self.model, self.saving_path)
            if early_stopping.early_stop:
                self.logger.info(f'Early stopping')
                break
            
            # adjust_learning_rate(model_optim, epoch + 1, self.args, self.logger)
            # self.logger.info('')

        best_model_path = self.saving_path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            self.logger.info(f'loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.saving_path, 'checkpoint.pth')))

        preds = []
        trues = []
        masks = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, mask_x, mask_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                mask_x = mask_x.float().to(self.device)
                mask_y = mask_y.to(self.device)

                outputs, _ = self.model(batch_x, mask_x, dynamic_uniform=False)

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
                masks.append(mask_y)

                # inverse_pred = test_data.inverse(outputs[0])
                # inverse_true = test_data.inverse(batch_y[0])
                # inverse_preds.append(inverse_pred)
                # inverse_trues.append(inverse_true)

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        masks = np.concatenate(masks, axis=0)
       
        # inverse_preds = np.array(inverse_preds).reshape(-1, inverse_preds.shape[-2], inverse_preds.shape[-1])
        # inverse_trues = np.array(inverse_trues).reshape(-1, inverse_trues.shape[-2], inverse_trues.shape[-1])
        
        self.logger.info(f'Test Shape: {preds.shape}')

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds[masks == 1], trues[masks == 1])
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
  
    def pretrain(self, dl_pretrain_train, dl_pretrain_val):
        self.logger.info(f'Pretrain Start')
        pretrain_model = self.model_dict[self.args.model].Model(self.args, mode='pretrain').float().to(self.device)
        best_v_loss = np.inf
        optimizer = optim.Adam(pretrain_model.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=5, verbose=True, threshold=1e-2)
        for e in range(self.args.pretrain_epoch):
            pretrain_model.train()
            t_s = t_a = 0
            for step, (batch_x, batch_y, batch_x_mark, batch_y_mark, mask_x, mask_y) in enumerate(dl_pretrain_train):
                batch, seq_len, in_feature = batch_x.shape
                gd = batch_x
                mask = mask_x
                
                new_mask = torch.rand(*mask.shape)
                new_mask = new_mask * mask
                new_mask = 1 * (new_mask >= self.args.mask_ratio)
                x = batch_x.clone().detach() * new_mask
                
                x = x.to(self.device).float()
                mask = mask.to(self.device).float()
                mask_ = new_mask.to(self.device).float()
                gd = gd.to(self.device).float()

                if e < self.args.warmup:
                    predict, embedding_res = pretrain_model(x, mask_, dynamic_uniform=True)
                    predict = predict.view(batch, seq_len, in_feature)
                    predict_full, embedding_res_full = pretrain_model(gd * mask, mask, dynamic_uniform=True)
                    predict_full = predict_full.view(batch, seq_len, in_feature)
                else:
                    predict, embedding_res = pretrain_model(x, mask_, dynamic_uniform=False)
                    predict = predict.view(batch, seq_len, in_feature)
                    predict_full, embedding_res_full = pretrain_model(gd * mask, mask, dynamic_uniform=False)
                    predict_full = predict_full.view(batch, seq_len, in_feature)

                s_f_v = fullinput_Valid_MSELoss(predict_full, gd, mask)
                a_f_v = fullinput_Valid_AbsLoss(predict_full, gd, mask)
                s_m_am = maskinput_AM_MSELoss(predict, gd, mask, mask_)
                a_m_am = maskinput_AM_AbsLoss(predict, gd, mask, mask_)
                
                a = ( a_m_am + self.args.mask_ratio * a_f_v)/(1 + self.args.mask_ratio)
                s = ( s_m_am + self.args.mask_ratio * s_f_v)/(1 + self.args.mask_ratio)  

                t_a += a.item()
                t_s += s.item()
                
                optimizer.zero_grad()
                s.backward()
                optimizer.step()
                
                if (step + 1) % 100 == 0:
                    self.logger.info(f'Pretrain iters: {step + 1}, epoch: {e + 1} | loss: {s.item():.7f}')
            
            t_s /= (step + 1)
            t_a /= (step + 1)
            scheduler.step(t_s)
            
            pretrain_val_loss = self.pretrain_vali(pretrain_model, dl_pretrain_val, e)
            if(pretrain_val_loss < best_v_loss):
                best_v_loss = pretrain_val_loss
                torch.save(pretrain_model.state_dict(), self.saving_path + '/' + 'checkpoint_pretrain.pth')
            self.logger.info(
                f'Pretrain Epoch: {e + 1}, Steps: {step} | Train Loss: {t_s:.7f} Vali Loss: {pretrain_val_loss:.7f}'
            )

    def pretrain_vali(self, pretrain_model, vali_loader, e):
        total_loss = []
        pretrain_model.eval()
        with torch.no_grad():
            pretrain_model.eval()
            torch.cuda.empty_cache()
            for step, (batch_x, batch_y, batch_x_mark, batch_y_mark, mask_x, mask_y) in enumerate(vali_loader):
                batch, seq_len, in_feature = batch_x.shape
                gd = batch_x
                mask = mask_x
                
                new_mask = torch.rand(*mask.shape)
                new_mask = new_mask * mask
                new_mask = 1 * (new_mask >= self.args.mask_ratio)
                x = batch_x.clone().detach() * new_mask
                
                x = x.to(self.device).float()
                mask = mask.to(self.device).float()
                mask_ = new_mask.to(self.device).float()
                gd = gd.to(self.device).float()
                if e < self.args.warmup:
                    predict, embedding_res = pretrain_model(x, mask_, dynamic_uniform=True)
                    predict = predict.view(batch, seq_len, in_feature)
                    predict_full, embedding_res_full = pretrain_model(gd * mask, mask, dynamic_uniform=True)
                    predict_full = predict_full.view(batch, seq_len, in_feature)
                else:
                    predict, embedding_res = pretrain_model(x, mask_, dynamic_uniform=False)
                    predict = predict.view(batch, seq_len, in_feature)
                    predict_full, embedding_res_full = pretrain_model(gd * mask, mask, dynamic_uniform=False)
                    predict_full = predict_full.view(batch, seq_len, in_feature)

                s_m_am = maskinput_AM_MSELoss(predict, gd, mask, mask_)
                a_m_am = maskinput_AM_AbsLoss(predict, gd, mask, mask_)

                s_f_v = fullinput_Valid_MSELoss(predict_full, gd, mask)
                a_f_v = fullinput_Valid_AbsLoss(predict_full, gd, mask)

                # a = (0.8 * a_m_am + 0.2 * a_f_v + 0.1 * a_m_d)
                a = ( a_m_am + self.args.mask_ratio * a_f_v)/(1 + self.args.mask_ratio)
                s = ( s_m_am + self.args.mask_ratio * s_f_v)/(1 + self.args.mask_ratio)

                
                total_loss.append(s.item())
            total_loss = np.average(total_loss)
            pretrain_model.train()
            return total_loss