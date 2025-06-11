import torch
import torch.nn as nn

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.data_handling.scalers import CustomScalingMultivariate
from src.data_handling.time_features import compute_batch_time_features
from src.models.blocks import (
    ConcatLayer,
    DilatedConv1dBlock,
    EncoderFactory,
    SinPositionalEncoding,
)
from src.utils.utils import device


class BaseModel(nn.Module):
    """
    Base model for multivariate time series forecasting with GluonTS time features.
    """

    def __init__(
        self,
        epsilon=1e-3,
        scaler="custom_robust",
        sin_pos_enc=False,
        sin_pos_const=10000.0,
        encoding_dropout=0.0,
        handle_constants_model=False,
        embed_size=128,  # Default embed_size, adjust as needed
        K_max=6,
        **kwargs,
    ):
        super().__init__()
        assert encoding_dropout >= 0.0 and encoding_dropout <= 1.0
        self.epsilon = epsilon
        self.encoding_dropout = encoding_dropout
        self.handle_constants_model = handle_constants_model
        self.embed_size = embed_size

        # Create multivariate scaler
        self.scaler = CustomScalingMultivariate(scaler)

        # Initial embedding layers for time series values
        self.expand_values = nn.Linear(1, embed_size, bias=True)

        # Projection layer for GluonTS time features
        self.time_feature_projection = nn.Linear(K_max, embed_size)

        # Sinusoidal positional encoding (optional)
        self.sin_pos_encoder = None
        self.sin_pos_flag = sin_pos_enc
        if sin_pos_enc:
            self.sin_pos_encoder = SinPositionalEncoding(
                d_model=embed_size, max_len=5000, sin_pos_const=sin_pos_const
            )

        # Concatenation layers
        self.concat_pos = ConcatLayer(dim=-1, name="ConcatPos")
        self.concat_embed = ConcatLayer(dim=-1, name="ConcatEmbed")
        self.concat_targets = ConcatLayer(dim=1, name="ConcatTargets")

    def get_positional_embeddings(
        self, time_features, num_channels, drop_enc_allow=False
    ):
        """
        Generate positional embeddings from GluonTS time features.

        Args:
            time_features: Tensor [batch_size, seq_len, K_max]
            num_channels: Number of channels
            drop_enc_allow: Whether to allow dropout of encodings

        Returns:
            Tensor [batch_size, seq_len, num_channels, embed_size]
        """
        batch_size, seq_len, _ = time_features.shape
        if (torch.rand(1).item() < self.encoding_dropout) and drop_enc_allow:
            return torch.zeros(
                batch_size, seq_len, num_channels, self.embed_size, device=device
            ).to(torch.float32)

        if self.sin_pos_flag and self.sin_pos_encoder is not None:
            return self.sin_pos_encoder(time_features, num_channels).to(torch.float32)

        # Project GluonTS time features to embed_size
        pos_embed = self.time_feature_projection(
            time_features
        )  # [batch_size, seq_len, embed_size]
        return pos_embed.unsqueeze(2).expand(-1, -1, num_channels, -1)

    def forward(
        self,
        data_container: BatchTimeSeriesContainer,
        training=False,
        drop_enc_allow=False,
    ):
        """
        Forward pass through the model.

        Args:
            data_container: BatchTimeSeriesContainer with history and target data
            training: Whether in training mode
            drop_enc_allow: Whether to allow dropping encodings

        Returns:
            Dictionary with model outputs and scaling information
        """
        history_values = data_container.history_values
        target_values = data_container.target_values
        target_index = data_container.target_index
        history_time_features, target_time_features = compute_batch_time_features(
            data_container.start,
            data_container.history_length,
            data_container.target_length,
            data_container.batch_size,
            data_container.frequency,
        )

        batch_size, seq_len, num_channels = history_values.shape
        pred_len = target_values.shape[1] if target_values is not None else 0

        # Handle constant channels if needed
        if self.handle_constants_model:
            is_constant = torch.all(history_values == history_values[:, 0:1, :], dim=1)
            noise = torch.randn_like(history_values) * 0.1 * is_constant.unsqueeze(1)
            history_values = history_values + noise

        # Scale history values
        history_scale_params, history_scaled = self.scaler(history_values, self.epsilon)

        # Scale target values
        target_scaled = None
        if target_values is not None:
            assert target_values.dim() == 2, (
                "target_values should be [batch_size, pred_len]"
            )
            target_scaled = torch.zeros_like(target_values)
            for b in range(batch_size):
                channel_idx = target_index[b].long()
                if self.scaler.name == "custom_robust":
                    median = history_scale_params[0][b, 0, channel_idx]
                    iqr = history_scale_params[1][b, 0, channel_idx]
                    target_scaled[b, :] = (target_values[b, :] - median) / iqr
                else:  # min_max scaler
                    max_val = history_scale_params[0][b, 0, channel_idx]
                    min_val = history_scale_params[1][b, 0, channel_idx]
                    target_scaled[b, :] = (target_values[b, :] - min_val) / (
                        max_val - min_val
                    )

        # Get positional embeddings
        history_pos_embed = self.get_positional_embeddings(
            history_time_features, num_channels, drop_enc_allow
        )
        target_pos_embed = self.get_positional_embeddings(
            target_time_features, num_channels, drop_enc_allow
        )

        # Process embeddings
        history_scaled = history_scaled.unsqueeze(-1)
        channel_embeddings = self.expand_values(history_scaled)
        channel_embeddings = channel_embeddings + history_pos_embed
        all_channels_embedded = channel_embeddings.view(batch_size, seq_len, -1)

        # Generate predictions
        predictions = self.forecast(
            all_channels_embedded, target_pos_embed, target_index, pred_len
        )

        return {
            "result": predictions,
            "scale_params": history_scale_params,
            "target_index": target_index,
            "target_scaled": target_scaled,
            "target_values": target_values,
        }

    def compute_loss(self, y_true, y_pred):
        """
        Compute loss between predictions and scaled ground truth.

        Args:
            y_true: Ground truth target values [batch_size, pred_len]
            y_pred: Dictionary containing predictions and scaling info

        Returns:
            Loss value
        """
        predictions = y_pred["result"]
        target_scaled = y_pred["target_scaled"]

        if target_scaled is None:
            return torch.tensor(0.0, device=predictions.device)

        if predictions.shape != target_scaled.shape:
            predictions = predictions.view(target_scaled.shape)

        loss = nn.functional.mse_loss(predictions, target_scaled)
        return loss

    def forecast(self, embedded, target_pos_embed, target_index, prediction_length):
        raise NotImplementedError("Subclasses must implement forecast method")


class MultiStepModel(BaseModel):
    """
    Multivariate time series forecasting model with encoder layers.
    """

    def __init__(
        self,
        base_model_config,
        encoder_config,
        scaler="custom_robust",
        num_encoder_layers=2,
        hidden_dim=128,
        token_embed_dim=1024,
        use_gelu=True,
        use_input_projection_norm=False,
        use_global_residual=False,
        linear_sequence_length=2,
        use_dilated_conv=True,
        dilated_conv_kernel_size=3,
        dilated_conv_max_dilation=3,
        **kwargs,
    ):
        super().__init__(scaler=scaler, **base_model_config)
        self.num_encoder_layers = num_encoder_layers
        self.use_gelu = use_gelu
        self.hidden_dim = hidden_dim
        self.channel_embed_dim = self.embed_size

        if use_dilated_conv:
            self.input_projection_layer = DilatedConv1dBlock(
                self.channel_embed_dim,
                token_embed_dim,
                dilated_conv_kernel_size,
                dilated_conv_max_dilation,
                single_conv=False,
            )
        else:
            self.input_projection_layer = nn.Linear(
                self.channel_embed_dim, token_embed_dim
            )

        self.global_residual = None
        if use_global_residual:
            self.global_residual = nn.Linear(
                self.channel_embed_dim * (linear_sequence_length + 1), token_embed_dim
            )

        self.linear_sequence_length = linear_sequence_length
        self.initial_gelu = nn.GELU() if use_gelu else None
        self.input_projection_norm = (
            nn.LayerNorm(token_embed_dim)
            if use_input_projection_norm
            else nn.Identity()
        )

        self.encoder_layers = nn.ModuleList(
            [
                EncoderFactory.create_encoder(**encoder_config)
                for _ in range(num_encoder_layers)
            ]
        )

        self.target_projection = nn.Linear(self.embed_size, token_embed_dim)
        self.final_output_layer = nn.Linear(token_embed_dim, 1)
        self.final_activation = nn.Identity()

    def forecast(self, embedded, target_pos_embed, target_index, prediction_length):
        """
        Generate forecasts for target channels specified by target_index.

        Args:
            embedded: Embedded history values [batch_size, seq_len, num_channels * embed_size]
            target_pos_embed: Target positional embeddings [batch_size, pred_len, num_channels, embed_size]
            target_index: Indices of target channels [batch_size, 1]
            prediction_length: Length of prediction horizon

        Returns:
            Predictions tensor [batch_size, pred_len]
        """
        batch_size, seq_len, _ = embedded.shape
        embedded = embedded.view(batch_size, seq_len, -1, self.embed_size)
        batch_indices = torch.arange(batch_size, device=device)
        target_channel_indices = target_index.view(-1)
        target_embedded = embedded[batch_indices, :, target_channel_indices].unsqueeze(
            2
        )

        if self.global_residual:
            glob_res = target_embedded[
                :, -(self.linear_sequence_length + 1) :, :, :
            ].reshape(batch_size, -1)
            glob_res = (
                self.global_residual(glob_res)
                .unsqueeze(1)
                .repeat(1, prediction_length, 1)
            )

        target_embedded = target_embedded.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.embed_size
        )
        x = self.input_projection_layer(target_embedded)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        if self.use_gelu and self.initial_gelu is not None:
            x = self.input_projection_norm(x)
            x = self.initial_gelu(x)

        x = x.view(batch_size, 1, seq_len, -1)
        target_pos_embed_selected = (
            target_pos_embed[batch_indices, :, target_channel_indices]
            .unsqueeze(2)
            .permute(0, 2, 1, 3)
        )
        target_repr = self.target_projection(target_pos_embed_selected)
        x_sliced = x[:, :, -prediction_length:, :]

        if x_sliced.shape != target_repr.shape:
            raise ValueError(
                f"Shape mismatch: x_sliced {x_sliced.shape}, target_repr {target_repr.shape}"
            )

        predictions = self.final_output_layer(x_sliced + target_repr)
        return predictions.squeeze(-1).squeeze(1)
