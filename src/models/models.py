import torch
import torch.nn as nn

from src.data_handling.data_containers import TimeSeriesDataContainer
from src.data_handling.scalers import CustomScalingMultivariate
from src.models.blocks import (
    ConcatLayer,
    DilatedConv1dBlock,
    EncoderFactory,
    PositionExpansion,
    SinPositionalEncoding,
)
from src.models.constants import DAY, DOW, HOUR, MINUTE, MONTH, SECOND, YEAR
from src.utils.utils import device


class BaseModel(nn.Module):
    """
    Base model for multivariate time series forecasting with channel-specific positional encodings.
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
        **kwargs,
    ):
        super().__init__()
        assert encoding_dropout >= 0.0 and encoding_dropout <= 1.0
        self.epsilon = epsilon
        self.sub_day = sub_day
        self.encoding_dropout = encoding_dropout
        self.handle_constants_model = handle_constants_model

        # Create position embeddings for time features
        if sub_day:
            self.pos_minute = PositionExpansion(60, 4)
            self.pos_hour = PositionExpansion(24, 6)
        self.pos_year = PositionExpansion(10, 4)
        self.pos_month = PositionExpansion(12, 4)
        self.pos_day = PositionExpansion(31, 6)
        self.pos_dow = PositionExpansion(7, 4)

        # Calculate embedding size
        if sub_day:
            self.embed_size = sum(
                emb.channels
                for emb in (
                    self.pos_minute,
                    self.pos_hour,
                    self.pos_year,
                    self.pos_month,
                    self.pos_day,
                    self.pos_dow,
                )
            )
        else:
            self.embed_size = sum(
                emb.channels
                for emb in (self.pos_year, self.pos_month, self.pos_day, self.pos_dow)
            )

        # Create multivariate scaler
        self.scaler = CustomScalingMultivariate(scaler)

        # Initial embedding layers for time series values
        self.expand_values = nn.Linear(1, self.embed_size, bias=True)

        # Sinusoidal positional encoding
        self.sin_pos_encoder = None
        self.sin_pos_flag = sin_pos_enc
        if sin_pos_enc:
            self.sin_pos_encoder = SinPositionalEncoding(
                d_model=self.embed_size, max_len=5000, sin_pos_const=sin_pos_const
            )

        # Concatenation layers
        self.concat_pos = ConcatLayer(dim=-1, name="ConcatPos")
        self.concat_embed = ConcatLayer(dim=-1, name="ConcatEmbed")
        self.concat_targets = ConcatLayer(dim=1, name="ConcatTargets")

    @staticmethod
    def tc(time_features: torch.Tensor, time_index: int):
        """Extract a specific time feature from the time feature tensor."""
        return time_features[:, :, time_index]

    def get_positional_embeddings(
        self, time_features, reference_year=None, num_channels=1, drop_enc_allow=False
    ):
        """
        Generate positional embeddings for time features, channel-specific if sin_pos_enc=True.

        Args:
            time_features: Time features tensor [batch_size, seq_len, num_time_features]
            reference_year: Optional reference year for computing year delta
            num_channels: Number of channels for channel-specific encodings
            drop_enc_allow: Whether to allow dropout of encodings

        Returns:
            Positional embeddings tensor [batch_size, seq_len, num_channels, embed_size]
        """
        batch_size, seq_len, _ = time_features.shape
        if (torch.rand(1).item() < self.encoding_dropout) and drop_enc_allow:
            return torch.zeros(
                batch_size,
                seq_len,
                num_channels,
                self.embed_size,
                device=device,
            ).to(torch.float32)

        if self.sin_pos_flag and self.sin_pos_encoder is not None:
            # Generate channel-specific sinusoidal encodings
            return self.sin_pos_encoder(time_features, num_channels).to(torch.float32)

        # Using custom time feature embeddings
        year = self.tc(time_features, YEAR)
        if reference_year is not None:
            delta_year = torch.clamp(
                reference_year - year, min=0, max=self.pos_year.periods
            )
        else:
            delta_year = torch.zeros_like(year)

        if self.sub_day:
            pos_embedding = self.concat_pos(
                [
                    self.pos_minute(self.tc(time_features, MINUTE)),
                    self.pos_hour(self.tc(time_features, HOUR)),
                    self.pos_year(delta_year),
                    self.pos_month(self.tc(time_features, MONTH)),
                    self.pos_day(self.tc(time_features, DAY)),
                    self.pos_dow(self.tc(time_features, DOW)),
                ]
            ).to(torch.float32)
        else:
            pos_embedding = self.concat_pos(
                [
                    self.pos_year(delta_year),
                    self.pos_month(self.tc(time_features, MONTH)),
                    self.pos_day(self.tc(time_features, DAY)),
                    self.pos_dow(self.tc(time_features, DOW)),
                ]
            ).to(torch.float32)

        # Expand to channel dimension
        return pos_embedding.unsqueeze(2).expand(-1, -1, num_channels, -1)

    def forward(
        self,
        data_container: TimeSeriesDataContainer,
        training=False,
        drop_enc_allow=False,
    ):
        """
        Forward pass through the model.

        Args:
            data_container: TimeSeriesDataContainer with history and target data
            training: Whether in training mode
            drop_enc_allow: Whether to allow dropping encodings

        Returns:
            Dictionary with model outputs
        """
        print("Forward pass through BaseModel")
        print(
            "Data container history_values shape:", data_container.history_values.shape
        )
        print(
            "Data container target_values shape:",
            data_container.target_values.shape,
        )
        print(
            "Data container target_channels_indices shape:",
            data_container.target_channels_indices.shape,
        )
        print(
            "Data container target_channels_indices shape:",
            data_container.target_channels_indices.shape,
        )
        print(
            "Data container target_time_features shape:",
            data_container.target_time_features.shape,
        )
        history_values = data_container.history_values
        target_values = data_container.target_values
        target_channels_indices = data_container.target_channels_indices
        history_time_features = data_container.history_time_features
        target_time_features = data_container.target_time_features

        batch_size, seq_len, num_channels = history_values.shape
        pred_len = target_values.shape[1] if target_values is not None else 0

        # Handle constant channels if needed
        if self.handle_constants_model:
            is_constant = torch.all(history_values == history_values[:, 0:1, :], dim=1)
            noise = torch.randn_like(history_values) * 0.1 * is_constant.unsqueeze(1)
            history_values = history_values + noise

        # Get year reference
        if history_time_features is not None:
            reference_year = self.tc(history_time_features, YEAR)[:, -1:]
        else:
            reference_year = None

        # Scale history values
        history_scale_params, history_scaled = self.scaler(history_values, self.epsilon)

        # Scale target values during training
        target_scaled = None
        if training and target_values is not None:
            _, num_targets = target_channels_indices.shape
            target_scaled = torch.zeros_like(target_values)
            for b in range(batch_size):
                for t in range(num_targets):
                    channel_idx = target_channels_indices[b, t].long()
                    median = history_scale_params[0][b, 0, channel_idx]
                    iqr = history_scale_params[1][b, 0, channel_idx]
                    target_scaled[b, :, t] = (target_values[b, :, t] - median) / iqr

        # Get positional embeddings for history
        if history_time_features is not None:
            history_pos_embed = self.get_positional_embeddings(
                history_time_features, reference_year, num_channels, drop_enc_allow
            )
        else:
            history_pos_embed = torch.zeros(
                batch_size,
                seq_len,
                num_channels,
                self.embed_size,
                device=device,
            ).to(torch.float32)

        # Get positional embeddings for targets
        if target_time_features is not None:
            target_pos_embed = self.get_positional_embeddings(
                target_time_features, reference_year, num_channels, drop_enc_allow
            )
        else:
            target_pos_embed = torch.zeros(
                batch_size,
                pred_len,
                num_channels,
                self.embed_size,
                device=device,
            ).to(torch.float32)

        # Vectorized channel embeddings
        history_scaled = history_scaled.unsqueeze(
            -1
        )  # [batch_size, seq_len, num_channels, 1]
        channel_embeddings = self.expand_values(
            history_scaled
        )  # [batch_size, seq_len, num_channels, embed_size]
        channel_embeddings = channel_embeddings + history_pos_embed
        all_channels_embedded = channel_embeddings.view(batch_size, seq_len, -1)

        # Forecast using the embeddings
        predictions = self.forecast(
            all_channels_embedded, target_pos_embed, target_channels_indices, pred_len
        )

        return {
            "result": predictions,
            "scale_params": history_scale_params,
            "target_indices": target_channels_indices,
            "target_scaled": target_scaled,
        }

    def compute_loss(self, y_true, y_pred):
        """
        Compute loss between predictions and pre-scaled ground truth.

        Args:
            y_true: Ground truth target values [batch_size, pred_len, num_targets]
            y_pred: Dictionary containing predictions and scaling info

        Returns:
            Loss value
        """
        predictions = y_pred["result"]
        target_scaled = y_pred["target_scaled"]
        batch_size, pred_len, num_targets = predictions.shape

        # Use pre-scaled target values
        loss = nn.functional.mse_loss(predictions, target_scaled)
        return loss / num_targets

    def forecast(
        self, embedded, target_pos_embed, target_channels_indices, prediction_length
    ):
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

        # Calculate input dimension
        self.channel_embed_dim = self.embed_size

        # Create projection layer
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

        # Create global residual connection
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

        # Create encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                EncoderFactory.create_encoder(**encoder_config)
                for _ in range(self.num_encoder_layers)
            ]
        )

        # Target-specific projection layers
        self.target_projection = nn.Linear(self.embed_size, token_embed_dim)

        # Final output layer
        self.final_output_layer = nn.Linear(token_embed_dim, 1)
        self.final_activation = nn.Identity()

    def forecast(
        self, embedded, target_pos_embed, target_channels_indices, prediction_length
    ):
        """
        Generate forecasts for target channels in parallel.

        Args:
            embedded: Embedded history values [batch_size, seq_len, num_channels * embed_size]
            target_pos_embed: Target positional embeddings [batch_size, pred_len, num_channels, embed_size]
            target_channels_indices: Indices of target channels [batch_size, num_targets]
            prediction_length: Length of prediction horizon

        Returns:
            Predictions tensor [batch_size, pred_len, num_targets]
        """
        batch_size, seq_len, _ = embedded.shape
        num_targets = target_channels_indices.shape[1]

        # Reshape embedded to separate channels
        embedded = embedded.view(batch_size, seq_len, -1, self.embed_size)

        # Select target channel embeddings
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1)
        target_embedded = embedded[
            batch_indices, :, target_channels_indices
        ]  # [batch_size, seq_len, num_targets, embed_size]

        # Global residual connection
        if self.global_residual:
            glob_res = target_embedded[
                :, -(self.linear_sequence_length + 1) :, :, :
            ].reshape(batch_size, -1)
            glob_res = (
                self.global_residual(glob_res)
                .unsqueeze(1)
                .repeat(1, prediction_length, 1)
            )

        # Process target embeddings
        target_embedded = target_embedded.permute(0, 2, 1, 3)
        target_embedded = target_embedded.reshape(
            batch_size * num_targets, seq_len, self.embed_size
        )

        x = self.input_projection_layer(target_embedded)

        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # Reshape back to separate batch and targets
        x = x.reshape(batch_size, num_targets, seq_len, -1)

        # Apply normalization and activation if needed
        if self.use_gelu and self.initial_gelu is not None:
            x = self.input_projection_norm(x)
            x = self.initial_gelu(x)

        # Extract the target-specific representations for only the target channels
        target_pos_embed_for_targets = torch.zeros(
            batch_size,
            prediction_length,
            num_targets,
            self.embed_size,
            device=target_pos_embed.device,
        )

        # For each batch and target, select the correct channel's positional embedding
        for b in range(batch_size):
            for t in range(num_targets):
                channel_idx = target_channels_indices[b, t].long()
                target_pos_embed_for_targets[b, :, t, :] = target_pos_embed[
                    b, :, channel_idx, :
                ]

        # Apply target projection to the correct positional embeddings
        target_repr = self.target_projection(target_pos_embed_for_targets)

        # Extract the predictions portion and add target representation
        predictions = self.final_output_layer(
            x[:, :, -prediction_length:, :] + target_repr
        )  # [batch_size, num_targets, pred_len, 1]

        # Reshape to expected output format [batch_size, pred_len, num_targets]
        predictions = predictions.squeeze(-1).permute(0, 2, 1)

        return predictions
