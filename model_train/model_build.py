import torch
import torch.nn as nn

def build_model_from_yaml(yaml_config, seq_len):
    class DynamicModel(nn.Module):
        def __init__(self, yaml_config, seq_len):
            super().__init__()
            self.input_channels = yaml_config['model']['input_channels']
            layers = []
            self.lstm_index = None
            self.cnn_output_size = None

            # 處理 backbone
            for layer_config in yaml_config['model']['backbone']:
                from_idx, number, module, args = layer_config
                for _ in range(number):
                    if module == 'Conv1d':
                        layers.append(nn.Conv1d(
                            in_channels=self.input_channels,
                            out_channels=args[0],
                            kernel_size=args[1],
                            padding=args[2]
                        ))
                        self.input_channels = args[0]  # 更新通道數
                        layers.append(nn.ReLU())  # Conv1d 後添加 ReLU
                    elif module == 'MaxPool1d':
                        layers.append(nn.MaxPool1d(kernel_size=args[0]))
                    elif module == 'AdaptiveAvgPool1d':
                        layers.append(nn.AdaptiveAvgPool1d(output_size=args[0]))
                        self.cnn_output_size = args[0]

            self.lstm_index = len(layers)  # 記錄 LSTM 開始位置
            self.cnn_layers = nn.Sequential(*layers) if layers else None

            # 處理 head
            layers = []
            lstm_output_size = None
            for layer_config in yaml_config['model']['head']:
                from_idx, number, module, args = layer_config
                for _ in range(number):
                    if module == 'LSTM':
                        layers.append(nn.LSTM(
                            input_size=args[0],
                            hidden_size=args[1],
                            num_layers=args[2],
                            batch_first=args[3],
                            bidirectional=args[4]
                        ))
                        lstm_output_size = args[1] * (2 if args[4] else 1)
                    elif module == 'Dropout':
                        layers.append(nn.Dropout(p=args[0]))
                    elif module == 'Linear':
                        layers.append(nn.Linear(in_features=args[0], out_features=args[1]))
                    elif module == 'ReLU':
                        layers.append(nn.ReLU())

            self.head_layers = nn.Sequential(*layers) if layers and lstm_output_size is None else None
            self.lstm = layers[0] if lstm_output_size is not None else None
            self.fc_layers = nn.Sequential(*layers[1:]) if lstm_output_size is not None else None
            self.lstm_output_size = lstm_output_size

        def forward(self, x):
            x = x.unsqueeze(1)  # (B, 1, seq_len)
            if self.cnn_layers:
                x = self.cnn_layers(x)  # (B, channels, seq_len)
                x = x.transpose(1, 2)  # (B, seq_len, channels)
            if self.lstm:
                x, (hn, cn) = self.lstm(x)  # (B, seq_len, hidden_size)
                x = x[:, -1, :]  # Take last time step
            if self.fc_layers:
                x = self.fc_layers(x)
            elif self.head_layers:
                x = self.head_layers(x)
            return x.squeeze()

    return DynamicModel(yaml_config, seq_len)