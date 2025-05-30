from collections import OrderedDict
from typing import Literal

import torch
import torch.nn as nn

__all__ = ["model_factory", "SimpleNN", "ModernMLP", "AdvancedResMLP"]


def model_factory(
    model_type: Literal["mlp", "resnet-mlp", "simple-nn", "lstm"], input_dim: int
) -> nn.Module:
    if model_type == "mlp":
        return ModernMLP(input_dim=input_dim, hidden_dims=[128, 64], dropout_p=0.2)
    if model_type == "resnet-mlp":
        return AdvancedResMLP(input_features=input_dim, num_resnet_blocks=3, resnet_block_dim=128)
    if model_type == "simple-nn":
        return SimpleNN(input_features=input_dim, hidden_units1=128, hidden_units2=64)
    if model_type == "lstm":
        return LSTM(
            embedding_dim=32,
            lstm_hidden_dim=64,
            output_dim=1,
            lstm_layers=1,
            dropout_rate=0.3,
        )

    raise ValueError(f"Unknown model type: {model_type}")


class SimpleNN(nn.Module):
    def __init__(
        self, input_features: int, hidden_units1: int, hidden_units2: int, dropout_p: float = 0.2
    ):
        super().__init__()
        self.layer1 = nn.Linear(input_features, hidden_units1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_p)
        self.layer2 = nn.Linear(hidden_units1, hidden_units2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_p)
        self.output_layer = nn.Linear(hidden_units2, 1)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.dropout2(self.relu2(self.layer2(x)))
        x = self.output_layer(x)
        return x


class ModernMLP(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: list[int], output_dim: int = 1, dropout_p: float = 0.3
    ):
        super().__init__()
        layers = OrderedDict()
        current_dim = input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers[f"fc{i + 1}"] = nn.Linear(current_dim, h_dim)
            layers[f"bn{i + 1}"] = nn.BatchNorm1d(h_dim)
            layers[f"relu{i + 1}"] = nn.ReLU()
            layers[f"dropout{i + 1}"] = nn.Dropout(p=dropout_p)
            current_dim = h_dim
        layers["output_fc"] = nn.Linear(current_dim, output_dim)
        self.network = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout_rate, use_batchnorm, activation_fn):
        super().__init__()
        layers = [nn.Linear(dim, dim)]
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        layers.append(activation_fn)
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dim, dim))  # Second linear layer
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(dim))
        # No final activation before adding skip connection, often done after sum
        self.block = nn.Sequential(*layers)
        self.activation_after_skip = activation_fn  # Store for use after skip
        self.dropout_after_skip = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Skip connection
        out = self.activation_after_skip(out)  # Activation after skip
        out = self.dropout_after_skip(out)
        return out


class AdvancedResMLP(nn.Module):
    def __init__(
        self,
        input_features: int,
        num_resnet_blocks: int,
        resnet_block_dim: int,  # Dimension within ResNet blocks
        output_features: int = 1,
        dropout_rate: float = 0.3,
        use_batchnorm: bool = True,
        activation_fn_str: str = "gelu",
    ):
        super().__init__()

        if activation_fn_str.lower() == "gelu":
            activation = nn.GELU()
        elif activation_fn_str.lower() == "relu":
            activation = nn.ReLU()
        else:
            raise ValueError("Unsupported activation_fn. Choose 'relu' or 'gelu'.")

        self.initial_layer = nn.Linear(input_features, resnet_block_dim)
        self.initial_bn = nn.BatchNorm1d(resnet_block_dim) if use_batchnorm else nn.Identity()
        self.initial_activation = activation
        self.initial_dropout = nn.Dropout(dropout_rate)

        self.resnet_blocks = nn.Sequential(
            *[
                ResNetBlock(resnet_block_dim, dropout_rate, use_batchnorm, activation)
                for _ in range(num_resnet_blocks)
            ]
        )
        self.final_layer = nn.Linear(resnet_block_dim, output_features)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.initial_bn(x)
        x = self.initial_activation(x)
        x = self.initial_dropout(x)
        x = self.resnet_blocks(x)
        x = self.final_layer(x)
        return x


class LSTM(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        lstm_hidden_dim: int,
        output_dim: int = 1,
        lstm_layers: int = 1,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = nn.LSTM(
            1,
            lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
        )

        self.combined_feature_size = lstm_hidden_dim

        self.fc1 = nn.Linear(self.combined_feature_size, self.combined_feature_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(self.combined_feature_size // 2, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_):
        lstm_out, (hidden, cell) = self.lstm(input_[..., None])

        x = self.fc1(hidden[-1])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        output = self.sigmoid(x)

        return output


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
