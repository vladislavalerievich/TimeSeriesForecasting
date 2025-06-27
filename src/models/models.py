import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.data_handling.scalers import MinMaxScaler, RobustScaler
from src.data_handling.time_features import compute_batch_time_features
from src.models.blocks import (
    ConcatLayer,
    DilatedConv1dBlock,
    GatedDeltaNetEncoder,
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
        max_history_length=1024,
        max_prediction_length=900,
        **kwargs,
    ):
        super().__init__()
        assert encoding_dropout >= 0.0 and encoding_dropout <= 1.0
        self.epsilon = epsilon
        self.encoding_dropout = encoding_dropout
        self.handle_constants_model = handle_constants_model
        self.embed_size = embed_size
        self.time_feature_config = time_feature_config or {}
        self.max_history_length = max_history_length
        self.max_prediction_length = max_prediction_length

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

    def pad_sequence(self, sequence, target_length, pad_value=0.0):
        """
        Pad sequence to target length on the right side.

        Args:
            sequence: Tensor of shape [batch_size, seq_len, ...]
            target_length: Target sequence length
            pad_value: Value to use for padding

        Returns:
            Padded sequence and padding mask
        """
        batch_size, seq_len = sequence.shape[:2]

        if seq_len >= target_length:
            # Truncate if longer than target
            return sequence[:, :target_length], torch.ones(
                batch_size, target_length, device=sequence.device, dtype=torch.bool
            )

        # Calculate padding needed
        pad_length = target_length - seq_len

        # Create padding mask (True for valid positions, False for padded)
        padding_mask = torch.cat(
            [
                torch.ones(
                    batch_size, seq_len, device=sequence.device, dtype=torch.bool
                ),
                torch.zeros(
                    batch_size, pad_length, device=sequence.device, dtype=torch.bool
                ),
            ],
            dim=1,
        )

        # Pad the sequence
        pad_shape = list(sequence.shape)
        pad_shape[1] = pad_length
        padding = torch.full(
            pad_shape, pad_value, device=sequence.device, dtype=sequence.dtype
        )

        padded_sequence = torch.cat([sequence, padding], dim=1)

        return padded_sequence, padding_mask

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

        batch_size, original_seq_len, num_channels = history_values.shape
        original_pred_len = future_values.shape[1] if future_values is not None else 0

        # Pad history values to max_history_length
        padded_history_values, history_padding_mask = self.pad_sequence(
            history_values, self.max_history_length, pad_value=0.0
        )

        # Pad history mask if provided
        if history_mask is not None:
            padded_history_mask, _ = self.pad_sequence(
                history_mask.unsqueeze(-1), self.max_history_length, pad_value=0.0
            )
            padded_history_mask = padded_history_mask.squeeze(-1)
            # Combine original mask with padding mask
            combined_history_mask = padded_history_mask * history_padding_mask.float()
        else:
            combined_history_mask = history_padding_mask.float()

        # Pad future values if present (for training)
        if future_values is not None:
            padded_future_values, future_padding_mask = self.pad_sequence(
                future_values, self.max_prediction_length, pad_value=0.0
            )
        else:
            padded_future_values = None
            future_padding_mask = None

        # Compute time features with padded lengths
        history_time_features, target_time_features = compute_batch_time_features(
            data_container.start,
            self.max_history_length,  # Use max length
            self.max_prediction_length,  # Use max length
            data_container.batch_size,
            data_container.frequency,
            K_max=self.K_max,
            time_feature_config=self.time_feature_config,
        )

        # Handle constant channels if needed
        if self.handle_constants_model:
            is_constant = torch.all(
                padded_history_values == padded_history_values[:, 0:1, :], dim=1
            )
            noise = (
                torch.randn_like(padded_history_values) * 0.1 * is_constant.unsqueeze(1)
            )
            padded_history_values = padded_history_values + noise

        # Compute scaling statistics from original (non-padded) history values
        scale_statistics = self.scaler.compute_statistics(history_values, history_mask)

        # Scale padded history values
        history_scaled = self.scaler.scale(padded_history_values, scale_statistics)

        # Apply padding mask to scaled values (set padded positions to 0)
        history_scaled = history_scaled * history_padding_mask.unsqueeze(-1).float()

        # Apply optional clamping if configured
        if self.scaler_clamp_value is not None:
            history_scaled = torch.clamp(
                history_scaled, -self.scaler_clamp_value, self.scaler_clamp_value
            )

        # Scale padded future values if present (for training)
        future_scaled = None
        if padded_future_values is not None:
            assert padded_future_values.dim() == 3, (
                f"future_values should be [batch_size, pred_len, num_channels], got shape {padded_future_values.shape}"
            )
            assert padded_future_values.shape[2] == num_channels, (
                f"Channel size mismatch: future_values has {padded_future_values.shape[2]} channels, "
                f"but history_values has {num_channels} channels"
            )

            # Scale future values using the same statistics from history
            future_scaled = self.scaler.scale(padded_future_values, scale_statistics)
            # Apply padding mask
            future_scaled = future_scaled * future_padding_mask.unsqueeze(-1).float()

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
        all_channels_embedded = channel_embeddings.view(
            batch_size, self.max_history_length, -1
        )

        # Generate predictions for all channels
        predictions = self.forecast(
            all_channels_embedded,
            target_pos_embed,
            self.max_prediction_length,
            num_channels,
            history_padding_mask,
            original_pred_len,
        )

        return {
            "result": predictions[
                :, :original_pred_len
            ],  # Truncate predictions to original length
            "scale_statistics": scale_statistics,
            "future_scaled": future_scaled[:, :original_pred_len]
            if future_scaled is not None
            else None,
            "original_lengths": {
                "history": original_seq_len,
                "future": original_pred_len,
            },
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

        # Get original lengths
        original_pred_len = y_pred["original_lengths"]["future"]

        # Truncate predictions to original length
        predictions = predictions[:, :original_pred_len]

        # Scale the ground truth future values using history scale_statistics
        future_scaled = self.scaler.scale(future_values, scale_statistics)

        if predictions.shape != future_scaled.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs future_scaled {future_scaled.shape}"
            )
        loss = nn.functional.huber_loss(predictions, future_scaled)
        return loss

    def forecast(
        self,
        embedded,
        target_pos_embed,
        prediction_length,
        num_channels,
        history_padding_mask=None,
        original_pred_len=None,
    ):
        """
        Generate forecasts for all channels simultaneously.
        This method should be implemented by subclasses.

        Args:
            embedded: Embedded history values [batch_size, seq_len, num_channels * embed_size]
            target_pos_embed: Target positional embeddings [batch_size, pred_len, num_channels, embed_size]
            prediction_length: Length of prediction horizon
            num_channels: Number of channels
            history_padding_mask: Optional mask for padded positions in history
            original_pred_len: Original prediction length before padding

        Returns:
            Predictions tensor [batch_size, pred_len, num_channels]
        """
        raise NotImplementedError("Subclasses must implement the forecast method")


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
        max_history_length=1024,
        max_prediction_length=900,
        **kwargs,
    ):
        # Add max lengths to base_model_config
        base_model_config = base_model_config.copy()
        base_model_config["max_history_length"] = max_history_length
        base_model_config["max_prediction_length"] = max_prediction_length

        super().__init__(scaler=scaler, **base_model_config)
        self.num_encoder_layers = num_encoder_layers
        self.use_gelu = use_gelu
        self.hidden_dim = hidden_dim
        self.channel_embed_dim = self.embed_size
        self.use_dilated_conv = use_dilated_conv

        # Always initialize input projection layer
        if use_dilated_conv:
            self.input_projection_layer = DilatedConv1dBlock(
                in_channels=self.channel_embed_dim,
                out_channels=token_embed_dim,
                kernel_size=dilated_conv_kernel_size,
                max_dilation=dilated_conv_max_dilation,
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
                GatedDeltaNetEncoder(layer_idx=layer_idx, **encoder_config)
                for layer_idx in range(num_encoder_layers)
            ]
        )

        self.target_projection = nn.Linear(self.embed_size, token_embed_dim)
        # Output layer now predicts all channels jointly
        self.final_output_layer = nn.Linear(token_embed_dim, 1)
        self.final_activation = nn.Identity()

    def forecast(
        self,
        embedded,
        target_pos_embed,
        prediction_length,
        num_channels,
        history_padding_mask=None,
        original_pred_len=None,
    ):
        """
        Generate forecasts for all channels simultaneously.

        Args:
            embedded: Embedded history values [batch_size, seq_len, num_channels * embed_size]
            target_pos_embed: Target positional embeddings [batch_size, pred_len, num_channels, embed_size]
            prediction_length: Length of prediction horizon
            num_channels: Number of channels
            history_padding_mask: Optional mask for padded positions in history
            original_pred_len: Original prediction length before padding

        Returns:
            Predictions tensor [batch_size, pred_len, num_channels]
        """
        batch_size, seq_len, _ = embedded.shape

        # Reshape embedded to separate channels: [B, S, N, E]
        embedded = embedded.view(batch_size, seq_len, num_channels, self.embed_size)

        # Process each channel independently
        channel_outputs = []

        for channel_idx in range(num_channels):
            # Extract single channel: [B, S, E]
            channel_embedded = embedded[:, :, channel_idx, :]

            # Apply input projection
            x = self.input_projection_layer(channel_embedded)

            # Apply padding mask if provided to zero out padded positions
            if history_padding_mask is not None:
                x = x * history_padding_mask.unsqueeze(-1).float()

            # Apply global residual if enabled
            if self.global_residual is not None:
                # Get valid positions for global residual computation
                if history_padding_mask is not None:
                    # Find last valid position for each batch
                    valid_lengths = history_padding_mask.sum(dim=1).long()
                    glob_res_list = []
                    for b in range(batch_size):
                        valid_len = valid_lengths[b].item()
                        start_idx = max(0, valid_len - self.linear_sequence_length - 1)
                        end_idx = valid_len
                        if end_idx > start_idx:
                            glob_res_input = channel_embedded[
                                b, start_idx:end_idx, :
                            ].reshape(-1)
                            # Pad if needed
                            if glob_res_input.shape[0] < self.channel_embed_dim * (
                                self.linear_sequence_length + 1
                            ):
                                pad_size = (
                                    self.channel_embed_dim
                                    * (self.linear_sequence_length + 1)
                                    - glob_res_input.shape[0]
                                )
                                glob_res_input = F.pad(glob_res_input, (0, pad_size))
                        else:
                            glob_res_input = torch.zeros(
                                self.channel_embed_dim
                                * (self.linear_sequence_length + 1),
                                device=x.device,
                            )
                        glob_res_list.append(glob_res_input)

                    glob_res_input = torch.stack(glob_res_list, dim=0)
                else:
                    glob_res_input = channel_embedded[
                        :, -(self.linear_sequence_length + 1) :, :
                    ].reshape(batch_size, -1)

                glob_res = (
                    self.global_residual(glob_res_input)
                    .unsqueeze(1)
                    .repeat(1, prediction_length, 1)
                )
            else:
                glob_res = 0.0

            # Pass through encoder layers (temporal processing)
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)
                # Re-apply padding mask after each encoder layer
                if history_padding_mask is not None:
                    x = x * history_padding_mask.unsqueeze(-1).float()

            if self.use_gelu and self.initial_gelu is not None:
                x = self.input_projection_norm(x)
                x = self.initial_gelu(x)

            # Get target representation for this channel
            target_repr = self.target_projection(target_pos_embed[:, :, channel_idx, :])

            # Use the last part of the sequence for prediction
            if x.shape[1] >= prediction_length:
                x_sliced = x[:, -prediction_length:, :]
            else:
                # If sequence is shorter than prediction length, repeat the last timestep
                x_last = x[:, -1:, :].repeat(1, prediction_length, 1)
                x_sliced = x_last

            # Generate predictions for this channel
            final_representation = x_sliced + target_repr
            if self.global_residual is not None:
                final_representation += glob_res

            channel_predictions = self.final_output_layer(final_representation).squeeze(
                -1
            )
            channel_outputs.append(channel_predictions)

        # Stack channel predictions: [B, P, N]
        predictions = torch.stack(channel_outputs, dim=-1)

        return predictions
