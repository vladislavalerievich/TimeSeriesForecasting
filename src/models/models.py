import torch
import torch.nn as nn

from src.data_handling.data_containers import BatchTimeSeriesContainer
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
            self.pos_second = PositionExpansion(60, 2)
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
                    self.pos_second,
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
                    self.pos_second(self.tc(time_features, SECOND)),
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
        data_container: BatchTimeSeriesContainer,
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
        history_values = data_container.history_values
        target_values = data_container.target_values
        target_index = data_container.target_index
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
            assert target_values.dim() == 2, (
                "target_values should be [batch_size, pred_len]"
            )
            target_scaled = torch.zeros_like(target_values)
            # Get scaling parameters for each target channel
            for b in range(batch_size):
                channel_idx = target_index[b].long()
                median = history_scale_params[0][b, 0, channel_idx]
                iqr = history_scale_params[1][b, 0, channel_idx]
                target_scaled[b, :] = (target_values[b, :] - median) / iqr

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
            all_channels_embedded, target_pos_embed, target_index, pred_len
        )

        return {
            "result": predictions,
            "scale_params": history_scale_params,
            "target_index": target_index,
            "target_scaled": target_scaled,
        }

    def compute_loss(self, y_true, y_pred):
        """
        Compute loss between predictions and pre-scaled ground truth.

        Args:
            y_true: Ground truth target values [batch_size, pred_len, 1]
            y_pred: Dictionary containing predictions and scaling info

        Returns:
            Loss value
        """
        predictions = y_pred["result"]
        target_scaled = y_pred["target_scaled"]

        # Use pre-scaled target values - no need to normalize by num_targets since it's always 1
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

    def forecast(self, embedded, target_pos_embed, target_index, prediction_length):
        """
        Generate forecasts for target channels specified by target_index.

        Args:
            embedded: Embedded history values [batch_size, seq_len, num_channels * embed_size]
            target_pos_embed: Target positional embeddings [batch_size, pred_len, num_channels, embed_size]
            target_index: Indices of target channels [batch_size, 1]
            prediction_length: Length of prediction horizon

        Returns:
            Predictions tensor [batch_size, pred_len, 1]
        """
        batch_size, seq_len, _ = embedded.shape

        # Reshape embedded to separate channels
        embedded = embedded.view(batch_size, seq_len, -1, self.embed_size)

        # Get batch indices for selection
        batch_indices = torch.arange(batch_size, device=device)

        # Select target channel embeddings for each item in batch
        # target_index shape is [batch_size, 1], so we flatten it to [batch_size]
        target_channel_indices = target_index.view(-1)

        # Select the target channel for each item in the batch
        # This gives us [batch_size, seq_len, embed_size]
        target_embedded = embedded[batch_indices, :, target_channel_indices]

        # Add channel dimension back: [batch_size, seq_len, 1, embed_size]
        target_embedded = target_embedded.unsqueeze(2)

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

        # Permute to [batch_size, 1, seq_len, embed_size]
        target_embedded = target_embedded.permute(0, 2, 1, 3)

        # Reshape for processing through layers - we can treat this as a batch with size (batch_size)
        # since we only have 1 target channel per batch item
        target_embedded = target_embedded.reshape(batch_size, seq_len, self.embed_size)

        # Project through input projection layer
        x = self.input_projection_layer(
            target_embedded
        )  # [batch_size, seq_len, token_embed_dim]

        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)  # [batch_size, seq_len, token_embed_dim]

        # Apply normalization and activation
        if self.use_gelu and self.initial_gelu is not None:
            x = self.input_projection_norm(x)
            x = self.initial_gelu(x)

        # Reshape to restore batch and channel dimensions [batch_size, 1, seq_len, token_embed_dim]
        x = x.view(batch_size, 1, seq_len, -1)

        # Get target channel-specific position embeddings
        # Select the target channel positional encoding for each item in the batch
        target_pos_embed_selected = target_pos_embed[
            batch_indices, :, target_channel_indices
        ]

        # Add channel dimension [batch_size, pred_len, 1, embed_size]
        target_pos_embed_selected = target_pos_embed_selected.unsqueeze(2)

        # Permute to [batch_size, 1, pred_len, embed_size]
        target_pos_embed_selected = target_pos_embed_selected.permute(0, 2, 1, 3)

        # Project to token embedding dimension
        target_repr = self.target_projection(
            target_pos_embed_selected
        )  # [batch_size, 1, pred_len, token_embed_dim]

        # Get the last prediction_length time steps from x
        x_sliced = x[
            :, :, -prediction_length:, :
        ]  # [batch_size, 1, pred_len, token_embed_dim]

        # Make sure shapes are compatible before adding
        if x_sliced.shape != target_repr.shape:
            raise ValueError(
                f"Shape mismatch: x_sliced {x_sliced.shape}, target_repr {target_repr.shape}"
            )

        # Final prediction
        predictions = self.final_output_layer(
            x_sliced + target_repr
        )  # [batch_size, 1, pred_len, 1]

        # Reshape to [batch_size, pred_len, 1]
        predictions = predictions.squeeze(-1).permute(0, 2, 1)

        return predictions
