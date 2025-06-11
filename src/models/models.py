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
        sub_day=False,
        encoding_dropout=0.0,
        handle_constants_model=False,
        time_feature_dim=64,  # Projection dimension for time features
        **kwargs,
    ):
        super().__init__()
        assert encoding_dropout >= 0.0 and encoding_dropout <= 1.0
        self.epsilon = epsilon
        self.sub_day = sub_day
        self.encoding_dropout = encoding_dropout
        self.handle_constants_model = handle_constants_model
        self.time_feature_dim = time_feature_dim

        # Create multivariate scaler
        self.scaler = CustomScalingMultivariate(scaler)

        # Embedding layers
        self.value_embedding = nn.Linear(1, time_feature_dim, bias=True)
        self.time_feature_proj = nn.Linear(1, time_feature_dim, bias=False)

        # Sinusoidal positional encoding
        self.sin_pos_encoder = None
        self.sin_pos_flag = sin_pos_enc
        if sin_pos_enc:
            self.sin_pos_encoder = SinPositionalEncoding(
                d_model=time_feature_dim, max_len=5000, sin_pos_const=sin_pos_const
            )

        # Concatenation layers
        self.concat_embed = ConcatLayer(dim=-1, name="ConcatEmbed")
        self.concat_targets = ConcatLayer(dim=1, name="ConcatTargets")

    def get_positional_embeddings(
        self,
        time_features: torch.Tensor,
        reference_year=None,
        num_channels=1,
        drop_enc_allow=False,
    ) -> torch.Tensor:
        """
        Generate positional embeddings for time features.

        Args:
            time_features: Time features tensor [batch_size, seq_len, num_time_features]
            reference_year: Not used (kept for compatibility)
            num_channels: Number of channels for channel-specific encodings
            drop_enc_allow: Whether to allow dropout of encodings

        Returns:
            Positional embeddings tensor [batch_size, seq_len, num_channels, embed_dim]
        """
        if time_features is None:
            return torch.zeros(
                time_features.size(0),
                time_features.size(1),
                num_channels,
                self.time_feature_dim,
                device=device,
            ).float()

        batch_size, seq_len, num_features = time_features.shape

        if drop_enc_allow and (torch.rand(1).item() < self.encoding_dropout):
            return torch.zeros(
                batch_size, seq_len, num_channels, self.time_feature_dim, device=device
            ).float()

        # Process time features [batch, seq, features] -> [batch*seq*features, 1]
        flat_features = time_features.reshape(-1, 1)

        # Project each feature dimension separately
        projected = self.time_feature_proj(
            flat_features
        )  # [batch*seq*features, embed_dim]

        # Reshape back and sum across feature dimension
        projected = projected.view(
            batch_size, seq_len, num_features, self.time_feature_dim
        )
        time_embed = torch.sum(projected, dim=2)  # [batch, seq, embed_dim]

        # Expand for channels and add sinusoidal encoding if enabled
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, num_channels, -1)

        if self.sin_pos_flag and self.sin_pos_encoder is not None:
            time_embed = self.sin_pos_encoder(time_embed, num_channels)

        return time_embed

    def forward(
        self,
        data_container: BatchTimeSeriesContainer,
        training=False,
        drop_enc_allow=False,
    ):
        """
        Forward pass through the model.
        """
        history_values = data_container.history_values
        target_values = data_container.target_values
        target_index = data_container.target_index

        # Compute time features using GluonTS
        history_time_features, target_time_features = compute_batch_time_features(
            data_container.start,
            data_container.history_length,
            data_container.target_length,
            data_container.batch_size,
            data_container.frequency,
            include_subday=self.sub_day,
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
            history_time_features,
            None,  # reference_year not used
            num_channels,
            drop_enc_allow,
        )

        target_pos_embed = (
            self.get_positional_embeddings(
                target_time_features,
                None,  # reference_year not used
                num_channels,
                drop_enc_allow,
            )
            if target_time_features is not None
            else None
        )

        # Process value embeddings
        history_scaled = history_scaled.unsqueeze(-1)  # [batch, seq, channels, 1]
        channel_embeddings = self.value_embedding(
            history_scaled
        )  # [batch, seq, channels, embed_dim]
        channel_embeddings = channel_embeddings + history_pos_embed

        # Combine channel and time embeddings
        all_channels_embedded = channel_embeddings.view(
            batch_size, seq_len, -1
        )  # [batch, seq, channels*embed_dim]

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
        predictions = y_pred["result"]
        target_scaled = y_pred["target_scaled"]

        if target_scaled is None:
            return torch.tensor(0.0, device=predictions.device)

        if predictions.shape != target_scaled.shape:
            predictions = predictions.view(target_scaled.shape)

        return nn.functional.mse_loss(predictions, target_scaled)

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

        # Input projection layer
        if use_dilated_conv:
            self.input_projection_layer = DilatedConv1dBlock(
                self.time_feature_dim,
                token_embed_dim,
                dilated_conv_kernel_size,
                dilated_conv_max_dilation,
                single_conv=False,
            )
        else:
            self.input_projection_layer = nn.Linear(
                self.time_feature_dim, token_embed_dim
            )

        # Global residual connection
        self.global_residual = None
        if use_global_residual:
            self.global_residual = nn.Linear(
                self.time_feature_dim * (linear_sequence_length + 1), token_embed_dim
            )

        self.linear_sequence_length = linear_sequence_length
        self.initial_gelu = nn.GELU() if use_gelu else None
        self.input_projection_norm = (
            nn.LayerNorm(token_embed_dim)
            if use_input_projection_norm
            else nn.Identity()
        )

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                EncoderFactory.create_encoder(**encoder_config)
                for _ in range(self.num_encoder_layers)
            ]
        )

        # Target-specific projection
        self.target_projection = nn.Linear(self.time_feature_dim, token_embed_dim)

        # Output layers
        self.final_output_layer = nn.Linear(token_embed_dim, 1)
        self.final_activation = nn.Identity()

    def forecast(self, embedded, target_pos_embed, target_index, prediction_length):
        batch_size, seq_len, _ = embedded.shape
        num_channels = embedded.shape[-1] // self.time_feature_dim

        # Reshape to separate channels
        embedded = embedded.view(
            batch_size, seq_len, num_channels, self.time_feature_dim
        )

        # Select target channel embeddings
        batch_indices = torch.arange(batch_size, device=device)
        target_channel_indices = target_index.view(-1)
        target_embedded = embedded[batch_indices, :, target_channel_indices]

        # Add channel dimension back
        target_embedded = target_embedded.unsqueeze(2)  # [batch, seq, 1, embed_dim]

        # Global residual connection if used
        if self.global_residual:
            glob_res = target_embedded[
                :, -(self.linear_sequence_length + 1) :, :, :
            ].reshape(batch_size, -1)
            glob_res = (
                self.global_residual(glob_res)
                .unsqueeze(1)
                .repeat(1, prediction_length, 1)
            )

        # Process through projection and encoder
        target_embedded = target_embedded.reshape(
            batch_size, seq_len, self.time_feature_dim
        )
        x = self.input_projection_layer(target_embedded)

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        if self.use_gelu and self.initial_gelu is not None:
            x = self.input_projection_norm(x)
            x = self.initial_gelu(x)

        # Prepare target features
        target_pos_embed_selected = target_pos_embed[
            batch_indices, :, target_channel_indices
        ].unsqueeze(2)  # [batch, pred_len, 1, embed_dim]

        target_repr = self.target_projection(
            target_pos_embed_selected
        )  # [batch, pred_len, 1, token_dim]

        # Slice and combine
        x_sliced = x[:, -prediction_length:, :].unsqueeze(
            2
        )  # [batch, pred_len, 1, token_dim]
        predictions = self.final_output_layer(x_sliced + target_repr)

        return predictions.squeeze(-1).squeeze(1)  # [batch, pred_len]
