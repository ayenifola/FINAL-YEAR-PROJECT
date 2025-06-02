import torch
import torch.nn as nn
from typing import Literal

__all__ = ["model_factory", "LSTMModel", "GRUModel", "CNN1DModel"]


def model_factory(
    model_type: Literal["lstm", "cnn1d", "gru"],
    input_dim: int,
    sequence_length: int,
) -> nn.Module:
    if model_type == "lstm":
        return LSTMModel(input_dim=input_dim)
    if model_type == "cnn1d":
        return CNN1DModel(input_dim=input_dim, seq_length=sequence_length)
    if model_type == "gru":
        return GRUModel(input_dim=input_dim)
    raise ValueError(f"Unknown model type: {model_type}")


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_lstm_layers: int = 2,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.LayerNorm(hidden_dim) if input_dim > 1 else nn.Identity(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, input_):
        lstm_out, (hidden, cell) = self.lstm(input_)
        x = lstm_out[:, -1, :]  # Take output from last time step: (batch_size, hidden_size)
        output = self.fc(x)
        return output.squeeze(-1)


class CNN1DModel(nn.Module):
    def __init__(self, input_dim: int, seq_length: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (seq_length // 2), 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, num_features, seq_length)
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_size, num_layers=2, batch_first=True
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x: (batch_size, seq_length, num_features)
        gru_out, h_n = self.gru(x)  # gru_out: (batch_size, seq_length, hidden_size)
        x = gru_out[:, -1, :]  # Take last time step
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(-1)  # (batch_size,)
