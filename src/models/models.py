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
        time_feature_config=None,
        **kwargs,
    ):
        super().__init__()
        assert encoding_dropout >= 0.0 and encoding_dropout <= 1.0
        self.epsilon = epsilon
        self.encoding_dropout = encoding_dropout
        self.handle_constants_model = handle_constants_model
        self.embed_size = embed_size
        self.time_feature_config = time_feature_config or {}

        # Create multivariate scaler
        self.scaler = CustomScalingMultivariate(scaler)

        # Initial embedding layers for time series values
        self.expand_values = nn.Linear(1, embed_size, bias=True)

        # Projection layer for GluonTS time features
        # K_max might be auto-adjusted, so we'll set this after we know the actual value
        self.K_max = K_max
        self.time_feature_projection = nn.Linear(K_max, embed_size)
        self._k_max_initialized = False

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

    def _update_k_max_if_needed(self, actual_k_max: int):
        """Update K_max and time feature projection layer if auto-adjustment changed it."""
        if not self._k_max_initialized or actual_k_max != self.K_max:
            self.K_max = actual_k_max
            self.time_feature_projection = nn.Linear(actual_k_max, self.embed_size).to(
                device
            )
            self._k_max_initialized = True

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
        drop_enc_allow=False,
    ):
        """
        Forward pass through the model.

        Args:
            data_container: BatchTimeSeriesContainer with history and future data
            drop_enc_allow: Whether to allow dropping encodings

        Returns:
            Dictionary with model outputs and scaling information
        """
        history_values = data_container.history_values
        future_values = data_container.future_values
        history_time_features, target_time_features = compute_batch_time_features(
            data_container.start,
            data_container.history_length,
            data_container.future_length,
            data_container.batch_size,
            data_container.frequency,
            K_max=self.K_max,
            time_feature_config=self.time_feature_config,
        )

        # Update K_max if auto-adjustment changed it
        actual_k_max = history_time_features.shape[-1]
        self._update_k_max_if_needed(actual_k_max)

        batch_size, seq_len, num_channels = history_values.shape
        pred_len = future_values.shape[1] if future_values is not None else 0

        # Handle constant channels if needed
        if self.handle_constants_model:
            is_constant = torch.all(history_values == history_values[:, 0:1, :], dim=1)
            noise = torch.randn_like(history_values) * 0.1 * is_constant.unsqueeze(1)
            history_values = history_values + noise

        # Scale history values
        history_scale_params, history_scaled = self.scaler(history_values, self.epsilon)

        # Scale future values (multivariate)
        future_scaled = None
        if future_values is not None:
            assert future_values.dim() == 3, (
                f"future_values should be [batch_size, pred_len, num_channels], got shape {future_values.shape}"
            )
            assert future_values.shape[2] == num_channels, (
                f"Channel size mismatch: future_values has {future_values.shape[2]} channels, "
                f"but history_values has {num_channels} channels"
            )

            # Apply scaling using the same parameters as history
            if self.scaler.name == "custom_robust":
                medians, iqrs = history_scale_params
                future_scaled = (future_values - medians) / iqrs
            else:  # min_max scaler
                max_vals, min_vals = history_scale_params
                future_scaled = (future_values - min_vals) / (max_vals - min_vals)

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

        # Generate predictions for all channels
        predictions = self.forecast(
            all_channels_embedded, target_pos_embed, pred_len, num_channels
        )

        return {
            "result": predictions,
            "scale_params": history_scale_params,
            "future_scaled": future_scaled,
            "future_values": future_values,
        }

    def compute_loss(self, y_true, y_pred):
        """
        Compute loss between predictions and scaled ground truth.

        Args:
            y_true: Ground truth future values [batch_size, pred_len, num_channels]
            y_pred: Dictionary containing predictions and scaling info

        Returns:
            Loss value
        """
        predictions = y_pred["result"]
        scale_params = y_pred["scale_params"]
        future_values = y_true

        if future_values is None:
            return torch.tensor(0.0, device=predictions.device)

        # Scale the ground truth future values using the same parameters as history
        if self.scaler.name == "custom_robust":
            medians, iqrs = scale_params
            future_scaled = (future_values - medians) / iqrs
        else:  # min_max scaler
            max_vals, min_vals = scale_params
            future_scaled = (future_values - min_vals) / (max_vals - min_vals)

        if predictions.shape != future_scaled.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs future_scaled {future_scaled.shape}"
            )

        loss = nn.functional.mse_loss(predictions, future_scaled)
        return loss

    def forecast(self, embedded, target_pos_embed, prediction_length, num_channels):
        """
        Generate forecasts for all channels.

        Args:
            embedded: Embedded history values [batch_size, seq_len, num_channels * embed_size]
            target_pos_embed: Target positional embeddings [batch_size, pred_len, num_channels, embed_size]
            prediction_length: Length of prediction horizon
            num_channels: Number of channels

        Returns:
            Predictions tensor [batch_size, pred_len, num_channels]
        """
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
                self.channel_embed_dim * self.num_channels
                if hasattr(self, "num_channels")
                else self.channel_embed_dim,
                token_embed_dim,
                dilated_conv_kernel_size,
                dilated_conv_max_dilation,
                single_conv=False,
            )
        else:
            # For multivariate processing, we need to handle all channels
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
        # Output layer now predicts all channels
        self.final_output_layer = nn.Linear(token_embed_dim, 1)
        self.final_activation = nn.Identity()

    def forecast(self, embedded, target_pos_embed, prediction_length, num_channels):
        """
        Generate forecasts for all channels simultaneously.

        Args:
            embedded: Embedded history values [batch_size, seq_len, num_channels * embed_size]
            target_pos_embed: Target positional embeddings [batch_size, pred_len, num_channels, embed_size]
            prediction_length: Length of prediction horizon
            num_channels: Number of channels

        Returns:
            Predictions tensor [batch_size, pred_len, num_channels]
        """
        batch_size, seq_len, _ = embedded.shape

        # Reshape embedded to separate channels
        embedded = embedded.view(batch_size, seq_len, num_channels, self.embed_size)

        # Process each channel through the encoder
        all_channel_predictions = []

        for c in range(num_channels):
            # Get embeddings for current channel
            channel_embedded = embedded[:, :, c, :]  # [batch_size, seq_len, embed_size]

            # Apply input projection
            x = self.input_projection_layer(channel_embedded)

            # Apply global residual if enabled
            if self.global_residual:
                glob_res = channel_embedded[
                    :, -(self.linear_sequence_length + 1) :, :
                ].reshape(batch_size, -1)
                glob_res = (
                    self.global_residual(glob_res)
                    .unsqueeze(1)
                    .repeat(1, prediction_length, 1)
                )

            # Pass through encoder layers
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)

            if self.use_gelu and self.initial_gelu is not None:
                x = self.input_projection_norm(x)
                x = self.initial_gelu(x)

            # Get target positional embeddings for current channel
            target_pos_embed_channel = target_pos_embed[
                :, :, c, :
            ]  # [batch_size, pred_len, embed_size]
            target_repr = self.target_projection(target_pos_embed_channel)

            # Use the last part of the sequence for prediction
            x_sliced = x[
                :, -prediction_length:, :
            ]  # [batch_size, pred_len, token_embed_dim]

            if x_sliced.shape != target_repr.shape:
                # If sequence is shorter than prediction length, repeat the last timestep
                if x.shape[1] < prediction_length:
                    x_last = x[:, -1:, :].repeat(1, prediction_length, 1)
                    x_sliced = x_last
                else:
                    raise ValueError(
                        f"Shape mismatch: x_sliced {x_sliced.shape}, target_repr {target_repr.shape}"
                    )

            # Generate predictions for this channel
            channel_pred = self.final_output_layer(
                x_sliced + target_repr
            )  # [batch_size, pred_len, 1]
            all_channel_predictions.append(
                channel_pred.squeeze(-1)
            )  # [batch_size, pred_len]

        # Stack all channel predictions
        predictions = torch.stack(
            all_channel_predictions, dim=-1
        )  # [batch_size, pred_len, num_channels]

        return predictions
