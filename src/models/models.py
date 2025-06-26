import torch
import torch.nn as nn

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.data_handling.scalers import MinMaxScaler, RobustScaler
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
        scaler_clamp_value=None,
        sin_pos_enc=False,
        sin_pos_const=10000.0,
        encoding_dropout=0.0,
        handle_constants_model=False,
        embed_size=128,
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

        # Create the appropriate scaler based on config
        if scaler == "custom_robust":
            self.scaler = RobustScaler(epsilon=epsilon)
        elif scaler == "min_max":
            self.scaler = MinMaxScaler(epsilon=epsilon)
        else:
            raise ValueError(f"Unknown scaler: {scaler}")

        self.scaler_name = scaler
        self.scaler_clamp_value = scaler_clamp_value

        # Initial embedding layers for time series values
        self.expand_values = nn.Linear(1, embed_size, bias=True)

        # Projection layer for GluonTS time features
        self.K_max = K_max
        # Single projection layer - time_features.py handles k_max determination internally
        self.time_feature_projection = nn.Linear(self.K_max, self.embed_size)

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
        pos_embed = self.time_feature_projection(time_features)

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
        history_mask = data_container.history_mask
        history_time_features, target_time_features = compute_batch_time_features(
            data_container.start,
            data_container.history_length,
            data_container.future_length,
            data_container.batch_size,
            data_container.frequency,
            K_max=self.K_max,
            time_feature_config=self.time_feature_config,
        )

        batch_size, seq_len, num_channels = history_values.shape
        pred_len = future_values.shape[1] if future_values is not None else 0

        # Pass num_channels to MultiStepModel on first forward pass
        if hasattr(self, "set_num_channels") and not hasattr(self, "num_channels"):
            self.set_num_channels(num_channels)

        # Handle constant channels if needed
        if self.handle_constants_model:
            is_constant = torch.all(history_values == history_values[:, 0:1, :], dim=1)
            noise = torch.randn_like(history_values) * 0.1 * is_constant.unsqueeze(1)
            history_values = history_values + noise

        # Compute scaling statistics from history values
        scale_statistics = self.scaler.compute_statistics(history_values, history_mask)

        # Scale history values
        history_scaled = self.scaler.scale(history_values, scale_statistics)

        # Apply optional clamping if configured
        if self.scaler_clamp_value is not None:
            history_scaled = torch.clamp(
                history_scaled, -self.scaler_clamp_value, self.scaler_clamp_value
            )

        # Scale future values if present (for training)
        future_scaled = None
        if future_values is not None:
            assert future_values.dim() == 3, (
                f"future_values should be [batch_size, pred_len, num_channels], got shape {future_values.shape}"
            )
            assert future_values.shape[2] == num_channels, (
                f"Channel size mismatch: future_values has {future_values.shape[2]} channels, "
                f"but history_values has {num_channels} channels"
            )

            # Scale future values using the same statistics from history
            future_scaled = self.scaler.scale(future_values, scale_statistics)

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
            "scale_statistics": scale_statistics,
            "future_scaled": future_scaled,
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
        scale_statistics = y_pred["scale_statistics"]
        future_values = y_true

        if future_values is None:
            return torch.tensor(0.0, device=predictions.device)

        # Scale the ground truth future values using history scale_statistics
        future_scaled = self.scaler.scale(future_values, scale_statistics)

        if predictions.shape != future_scaled.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs future_scaled {future_scaled.shape}"
            )

        loss = nn.functional.mse_loss(predictions, future_scaled)
        return loss

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

        # Reshape embedded to separate channels: [B, S, N, E]
        embedded = embedded.view(batch_size, seq_len, num_channels, self.embed_size)

        # Permute and reshape for joint channel processing: [B*N, S, E]
        channel_embedded_all = (
            embedded.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * num_channels, seq_len, self.embed_size)
        )

        # Apply input projection
        x = self.input_projection_layer(channel_embedded_all)

        # Apply global residual if enabled
        glob_res = 0.0
        if self.global_residual:
            glob_res_input = channel_embedded_all[
                :, -(self.linear_sequence_length + 1) :, :
            ].reshape(batch_size * num_channels, -1)
            glob_res = (
                self.global_residual(glob_res_input)
                .unsqueeze(1)
                .repeat(1, prediction_length, 1)
            )

        # Pass through encoder layers (temporal processing)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # Reshape back to [B, N, S, token_embed_dim] for cross-channel attention
        x = x.view(batch_size, num_channels, seq_len, -1)

        # Apply cross-channel attention layers (after temporal processing)
        for i in range(self.cross_channel_attention_layers):
            # Reshape for attention: [B, S, N, token_embed_dim] -> [B*S, N, token_embed_dim]
            attention_input = (
                x.permute(0, 2, 1, 3)
                .contiguous()
                .view(batch_size * seq_len, num_channels, -1)
            )

            # Apply cross-channel attention
            attended_channels, _ = self.cross_channel_attentions[i](
                query=attention_input, key=attention_input, value=attention_input
            )

            # Add residual connection and layer norm
            attended_channels = self.channel_attention_norms[i](
                attention_input + attended_channels
            )

            # Reshape back to [B, N, S, token_embed_dim]
            x = attended_channels.view(batch_size, seq_len, num_channels, -1).permute(
                0, 2, 1, 3
            )

        # Reshape back to [B*N, S, token_embed_dim] for final processing
        x = (
            x.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * num_channels, seq_len, -1)
        )

        if self.use_gelu and self.initial_gelu is not None:
            x = self.input_projection_norm(x)
            x = self.initial_gelu(x)

        # Reshape target positional embeddings for joint processing: [B*N, P, E]
        target_pos_embed_all = (
            target_pos_embed.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * num_channels, prediction_length, self.embed_size)
        )
        target_repr = self.target_projection(target_pos_embed_all)

        # Use the last part of the sequence for prediction
        x_sliced = x[
            :, -prediction_length:, :
        ]  # [batch_size * num_channels, pred_len, token_embed_dim]

        if x_sliced.shape != target_repr.shape:
            # If sequence is shorter than prediction length, repeat the last timestep
            if x.shape[1] < prediction_length:
                x_last = x[:, -1:, :].repeat(1, prediction_length, 1)
                x_sliced = x_last
            else:
                raise ValueError(
                    f"Shape mismatch: x_sliced {x_sliced.shape}, target_repr {target_repr.shape}"
                )

        # Generate predictions for all channels
        final_representation = x_sliced + target_repr
        if self.global_residual:
            final_representation += glob_res

        predictions_flat = self.final_output_layer(final_representation).squeeze(-1)

        # Reshape back to [B, P, N] before returning
        predictions = (
            predictions_flat.view(batch_size, num_channels, prediction_length)
            .permute(0, 2, 1)
            .contiguous()
        )

        return predictions


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
        cross_channel_attention_heads: int = 4,
        cross_channel_attention_layers: int = 2,
        **kwargs,
    ):
        super().__init__(scaler=scaler, **base_model_config)
        self.num_encoder_layers = num_encoder_layers
        self.use_gelu = use_gelu
        self.hidden_dim = hidden_dim
        self.channel_embed_dim = self.embed_size
        self.use_dilated_conv = use_dilated_conv
        self.cross_channel_attention_layers = cross_channel_attention_layers

        # Create multiple cross-channel attention layers with layer norms
        self.cross_channel_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=token_embed_dim,  # Use token_embed_dim instead of channel_embed_dim
                    num_heads=cross_channel_attention_heads,
                    batch_first=True,
                )
                for _ in range(cross_channel_attention_layers)
            ]
        )

        self.channel_attention_norms = nn.ModuleList(
            [
                nn.LayerNorm(token_embed_dim)
                for _ in range(cross_channel_attention_layers)
            ]
        )

        self.input_projection_layer = None
        if use_dilated_conv:
            self.dilated_conv_config = {
                "token_embed_dim": token_embed_dim,
                "dilated_conv_kernel_size": dilated_conv_kernel_size,
                "dilated_conv_max_dilation": dilated_conv_max_dilation,
            }
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
        # Output layer now predicts all channels jointly
        self.final_output_layer = nn.Linear(token_embed_dim, 1)
        self.final_activation = nn.Identity()

    def set_num_channels(self, num_channels: int):
        """Set num_channels and initialize layers that depend on it."""
        self.num_channels = num_channels
        if self.use_dilated_conv and self.input_projection_layer is None:
            self.input_projection_layer = DilatedConv1dBlock(
                in_channels=self.channel_embed_dim,
                out_channels=self.dilated_conv_config["token_embed_dim"],
                kernel_size=self.dilated_conv_config["dilated_conv_kernel_size"],
                max_dilation=self.dilated_conv_config["dilated_conv_max_dilation"],
                single_conv=False,
            )
            # Move the newly created layer to the same device as the rest of the model
            device = self.expand_values.weight.device
            self.input_projection_layer.to(device)

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

        # Reshape embedded to separate channels: [B, S, N, E]
        embedded = embedded.view(batch_size, seq_len, num_channels, self.embed_size)

        # Permute and reshape for joint channel processing: [B*N, S, E]
        channel_embedded_all = (
            embedded.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * num_channels, seq_len, self.embed_size)
        )

        # Apply input projection
        x = self.input_projection_layer(channel_embedded_all)

        # Apply global residual if enabled
        glob_res = 0.0
        if self.global_residual:
            glob_res_input = channel_embedded_all[
                :, -(self.linear_sequence_length + 1) :, :
            ].reshape(batch_size * num_channels, -1)
            glob_res = (
                self.global_residual(glob_res_input)
                .unsqueeze(1)
                .repeat(1, prediction_length, 1)
            )

        # Pass through encoder layers (temporal processing)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # Reshape back to [B, N, S, token_embed_dim] for cross-channel attention
        x = x.view(batch_size, num_channels, seq_len, -1)

        # Apply cross-channel attention layers (after temporal processing)
        for i in range(self.cross_channel_attention_layers):
            # Reshape for attention: [B, S, N, token_embed_dim] -> [B*S, N, token_embed_dim]
            attention_input = (
                x.permute(0, 2, 1, 3)
                .contiguous()
                .view(batch_size * seq_len, num_channels, -1)
            )

            # Apply cross-channel attention
            attended_channels, _ = self.cross_channel_attentions[i](
                query=attention_input, key=attention_input, value=attention_input
            )

            # Add residual connection and layer norm
            attended_channels = self.channel_attention_norms[i](
                attention_input + attended_channels
            )

            # Reshape back to [B, N, S, token_embed_dim]
            x = attended_channels.view(batch_size, seq_len, num_channels, -1).permute(
                0, 2, 1, 3
            )

        # Reshape back to [B*N, S, token_embed_dim] for final processing
        x = (
            x.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * num_channels, seq_len, -1)
        )

        if self.use_gelu and self.initial_gelu is not None:
            x = self.input_projection_norm(x)
            x = self.initial_gelu(x)

        # Reshape target positional embeddings for joint processing: [B*N, P, E]
        target_pos_embed_all = (
            target_pos_embed.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * num_channels, prediction_length, self.embed_size)
        )
        target_repr = self.target_projection(target_pos_embed_all)

        # Use the last part of the sequence for prediction
        x_sliced = x[
            :, -prediction_length:, :
        ]  # [batch_size * num_channels, pred_len, token_embed_dim]

        if x_sliced.shape != target_repr.shape:
            # If sequence is shorter than prediction length, repeat the last timestep
            if x.shape[1] < prediction_length:
                x_last = x[:, -1:, :].repeat(1, prediction_length, 1)
                x_sliced = x_last
            else:
                raise ValueError(
                    f"Shape mismatch: x_sliced {x_sliced.shape}, target_repr {target_repr.shape}"
                )

        # Generate predictions for all channels
        final_representation = x_sliced + target_repr
        if self.global_residual:
            final_representation += glob_res

        predictions_flat = self.final_output_layer(final_representation).squeeze(-1)

        # Reshape back to [B, P, N] before returning
        predictions = (
            predictions_flat.view(batch_size, num_channels, prediction_length)
            .permute(0, 2, 1)
            .contiguous()
        )

        return predictions
