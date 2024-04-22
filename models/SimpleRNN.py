import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.hidden_size = configs.d_model
        self.rnn = torch.nn.LSTM(
            configs.enc_in, hidden_size=configs.d_model, num_layers = configs.e_layers, batch_first=True
        )
        if self.task_name == 'classification':
            self.fcn = torch.nn.Linear(configs.d_model, configs.class_num)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.fcn = torch.nn.Linear(configs.d_model, configs.c_out)
        if self.task_name == 'incomplete_long_term_forecast':
            self.fcn = torch.nn.Linear(configs.d_model, configs.c_out)
    
    def forecast(self, x):
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        predictions = x
        for step in range(self.pred_len):
            out, (h0, c0) = self.lstm(predictions[:, step:, :], (h0, c0))
            pred_out = self.fcn(out[:, -1:, :])
            predictions = predictions.cat(pred_out, dim=1)

        return predictions[:, -self.pred_len:, :]
    
    def incomplete_forecast(self, x):
        # init state: [layers, batch_size, hidden_size]
        h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)

        # 前向传播过程
        out, _ = self.lstm(x, (h0, c0))

        # 使用最后一个时间步的隐藏状态进行预测
        predictions = []
        for _ in range(self.pred_len):
            out, (h0, c0) = self.lstm(out[:, -1:, :], (h0, c0))
            prediction = self.fc(out[:, -1:, :])
            predictions.append(prediction)

        return torch.cat(predictions, dim=1)

    def classification(self, data):
        hidden_states, _ = self.rnn(data)
        logits = self.fcn(hidden_states[:, -1, :])
        return logits
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'classification' or self.task_name == 'incomplete_classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        if self.task_name == 'incomplete_long_term_forecast':
            dec_out = self.incomplete_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]s
        return None
    