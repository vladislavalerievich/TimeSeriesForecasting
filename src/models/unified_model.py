import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_handling.data_containers import BatchTimeSeriesContainer
from src.data_handling.scalers import MinMaxScaler, RobustScaler
from src.data_handling.time_features import compute_batch_time_features
from src.models.blocks import (
    DilatedConv1dBlock,
    GatedDeltaNetEncoder,
    SinPositionalEncoding,
)
from src.utils.utils import device


def create_scaler(scaler_type: str, epsilon: float = 1e-3):
    """Create scaler instance based on type."""
    if scaler_type == "custom_robust":
        return RobustScaler(epsilon=epsilon)
    elif scaler_type == "min_max":
        return MinMaxScaler(epsilon=epsilon)
    else:
        raise ValueError(f"Unknown scaler: {scaler_type}")


def pad_sequence(sequence: torch.Tensor, target_length: int, pad_value: float = 0.0):
    """Pad sequence to target length on the right side."""
    batch_size, seq_len = sequence.shape[:2]

    if seq_len >= target_length:
        return sequence[:, :target_length], torch.ones(
            batch_size, target_length, device=sequence.device, dtype=torch.bool
        )

    pad_length = target_length - seq_len
    padding_mask = torch.cat(
        [
            torch.ones(batch_size, seq_len, device=sequence.device, dtype=torch.bool),
            torch.zeros(
                batch_size, pad_length, device=sequence.device, dtype=torch.bool
            ),
        ],
        dim=1,
    )

    pad_shape = list(sequence.shape)
    pad_shape[1] = pad_length
    padding = torch.full(
        pad_shape, pad_value, device=sequence.device, dtype=sequence.dtype
    )
    padded_sequence = torch.cat([sequence, padding], dim=1)

    return padded_sequence, padding_mask


def apply_channel_noise(values: torch.Tensor, noise_scale: float = 0.1):
    """Add noise to constant channels to prevent model instability."""
    is_constant = torch.all(values == values[:, 0:1, :], dim=1)
    noise = torch.randn_like(values) * noise_scale * is_constant.unsqueeze(1)
    return values + noise


class TimeSeriesModel(nn.Module):
    """Time series forecasting model combining embedding, encoding, and prediction."""

    def __init__(
        self,
        # Core architecture
        embed_size: int = 128,
        token_embed_dim: int = 1024,
        num_encoder_layers: int = 2,
        # Sequence lengths
        max_history_length: int = 1024,
        max_prediction_length: int = 900,
        # Scaling and preprocessing
        scaler: str = "custom_robust",
        epsilon: float = 1e-3,
        scaler_clamp_value: float = None,
        handle_constants: bool = False,
        # Time features
        K_max: int = 6,
        time_feature_config: dict = None,
        sin_pos_enc: bool = False,
        sin_pos_const: float = 10000.0,
        encoding_dropout: float = 0.0,
        # Model architecture
        use_gelu: bool = True,
        use_input_projection_norm: bool = False,
        use_global_residual: bool = False,
        linear_sequence_length: int = 2,
        use_dilated_conv: bool = True,
        dilated_conv_kernel_size: int = 3,
        dilated_conv_max_dilation: int = 3,
        # Encoder configuration
        encoder_config: dict = None,
        **kwargs,
    ):
        super().__init__()

        # Core parameters
        self.embed_size = embed_size
        self.token_embed_dim = token_embed_dim
        self.max_history_length = max_history_length
        self.max_prediction_length = max_prediction_length
        self.epsilon = epsilon
        self.scaler_clamp_value = scaler_clamp_value
        self.handle_constants = handle_constants
        self.encoding_dropout = encoding_dropout
        self.K_max = K_max
        self.time_feature_config = time_feature_config or {}

        # Architecture flags
        self.use_gelu = use_gelu
        self.use_global_residual = use_global_residual
        self.linear_sequence_length = linear_sequence_length
        self.sin_pos_flag = sin_pos_enc

        # Initialize components
        self.scaler = create_scaler(scaler, epsilon)
        self._init_embedding_layers()
        self._init_encoder_layers(encoder_config or {}, num_encoder_layers)
        self._init_projection_layers(
            use_dilated_conv, dilated_conv_kernel_size, dilated_conv_max_dilation
        )
        self._init_positional_encoding(sin_pos_enc, sin_pos_const)
        self._init_auxiliary_layers(use_input_projection_norm, use_global_residual)

    def _init_embedding_layers(self):
        """Initialize value and time feature embedding layers."""
        self.expand_values = nn.Linear(1, self.embed_size, bias=True)
        self.time_feature_projection = nn.Linear(self.K_max, self.embed_size)

    def _init_encoder_layers(self, encoder_config: dict, num_encoder_layers: int):
        """Initialize encoder layers."""
        self.num_encoder_layers = num_encoder_layers
        self.encoder_layers = nn.ModuleList(
            [
                GatedDeltaNetEncoder(layer_idx=layer_idx, **encoder_config)
                for layer_idx in range(self.num_encoder_layers)
            ]
        )

    def _init_projection_layers(
        self, use_dilated_conv: bool, kernel_size: int, max_dilation: int
    ):
        """Initialize input and output projection layers."""
        if use_dilated_conv:
            self.input_projection_layer = DilatedConv1dBlock(
                in_channels=self.embed_size,
                out_channels=self.token_embed_dim,
                kernel_size=kernel_size,
                max_dilation=max_dilation,
                single_conv=False,
            )
        else:
            self.input_projection_layer = nn.Linear(
                self.embed_size, self.token_embed_dim
            )

        self.target_projection = nn.Linear(self.embed_size, self.token_embed_dim)
        self.final_output_layer = nn.Linear(self.token_embed_dim, 1)

    def _init_positional_encoding(self, sin_pos_enc: bool, sin_pos_const: float):
        """Initialize positional encoding components."""
        self.sin_pos_encoder = None
        if sin_pos_enc:
            self.sin_pos_encoder = SinPositionalEncoding(
                d_model=self.embed_size, max_len=5000, sin_pos_const=sin_pos_const
            )

    def _init_auxiliary_layers(
        self, use_input_projection_norm: bool, use_global_residual: bool
    ):
        """Initialize auxiliary layers and components."""
        self.initial_gelu = nn.GELU() if self.use_gelu else None
        self.input_projection_norm = (
            nn.LayerNorm(self.token_embed_dim)
            if use_input_projection_norm
            else nn.Identity()
        )

        self.global_residual = None
        if use_global_residual:
            self.global_residual = nn.Linear(
                self.embed_size * (self.linear_sequence_length + 1),
                self.token_embed_dim,
            )

    def _preprocess_data(self, data_container: BatchTimeSeriesContainer):
        """Handle data preprocessing including padding, scaling, and noise injection."""
        history_values = data_container.history_values
        future_values = data_container.future_values
        history_mask = data_container.history_mask

        batch_size, original_seq_len, num_channels = history_values.shape
        original_pred_len = future_values.shape[1] if future_values is not None else 0

        # Pad sequences
        padded_history_values, history_padding_mask = pad_sequence(
            history_values, self.max_history_length, pad_value=0.0
        )

        # Handle history mask
        combined_history_mask = history_padding_mask.float()
        if history_mask is not None:
            padded_history_mask, _ = pad_sequence(
                history_mask.unsqueeze(-1), self.max_history_length, pad_value=0.0
            )
            combined_history_mask = (
                padded_history_mask.squeeze(-1) * history_padding_mask.float()
            )

        # Pad future values if present
        padded_future_values = None
        if future_values is not None:
            padded_future_values, _ = pad_sequence(
                future_values, self.max_prediction_length, pad_value=0.0
            )
        # Handle constants
        if self.handle_constants:
            padded_history_values = apply_channel_noise(padded_history_values)

        return {
            "padded_history_values": padded_history_values,
            "padded_future_values": padded_future_values,
            "combined_history_mask": combined_history_mask,
            "num_channels": num_channels,
            "original_lengths": {
                "history": original_seq_len,
                "future": original_pred_len,
            },
        }

    def _compute_scaling(
        self, history_values: torch.Tensor, history_mask: torch.Tensor = None
    ):
        """Compute scaling statistics and apply scaling."""
        scale_statistics = self.scaler.compute_statistics(history_values, history_mask)
        return scale_statistics

    def _apply_scaling_and_masking(
        self, values: torch.Tensor, scale_statistics: dict, mask: torch.Tensor = None
    ):
        """Apply scaling and optional masking to values."""
        scaled_values = self.scaler.scale(values, scale_statistics)

        if mask is not None:
            scaled_values = scaled_values * mask.unsqueeze(-1).float()

        if self.scaler_clamp_value is not None:
            scaled_values = torch.clamp(
                scaled_values, -self.scaler_clamp_value, self.scaler_clamp_value
            )

        return scaled_values

    def _get_positional_embeddings(
        self,
        time_features: torch.Tensor,
        num_channels: int,
        batch_size: int,
        drop_enc_allow: bool = False,
    ):
        """Generate positional embeddings from time features."""
        seq_len, _ = time_features.shape

        # Expand time features to batch size: [seq_len, K_max] -> [batch_size, seq_len, K_max]
        expanded_time_features = time_features.unsqueeze(0).expand(batch_size, -1, -1)

        if (torch.rand(1).item() < self.encoding_dropout) and drop_enc_allow:
            return torch.zeros(
                batch_size, seq_len, num_channels, self.embed_size, device=device
            ).to(torch.float32)

        if self.sin_pos_flag and self.sin_pos_encoder is not None:
            return self.sin_pos_encoder(expanded_time_features, num_channels).to(
                torch.float32
            )

        pos_embed = self.time_feature_projection(expanded_time_features)
        return pos_embed.unsqueeze(2).expand(-1, -1, num_channels, -1)

    def _compute_embeddings(
        self, scaled_history: torch.Tensor, history_pos_embed: torch.Tensor
    ):
        """Compute value embeddings and combine with positional embeddings."""
        history_scaled = scaled_history.unsqueeze(-1)
        channel_embeddings = self.expand_values(history_scaled)
        channel_embeddings = channel_embeddings + history_pos_embed

        batch_size = scaled_history.shape[0]
        all_channels_embedded = channel_embeddings.view(
            batch_size, self.max_history_length, -1
        )

        return all_channels_embedded

    def _compute_global_residual(
        self,
        channel_embedded: torch.Tensor,
        history_padding_mask: torch.Tensor,
        prediction_length: int,
    ):
        """Compute global residual connection."""
        batch_size = channel_embedded.shape[0]

        if history_padding_mask is not None:
            valid_lengths = history_padding_mask.sum(dim=1).long()
            glob_res_list = []

            for b in range(batch_size):
                valid_len = valid_lengths[b].item()
                start_idx = max(0, valid_len - self.linear_sequence_length - 1)
                end_idx = valid_len

                if end_idx > start_idx:
                    glob_res_input = channel_embedded[b, start_idx:end_idx, :].reshape(
                        -1
                    )
                    if glob_res_input.shape[0] < self.embed_size * (
                        self.linear_sequence_length + 1
                    ):
                        pad_size = (
                            self.embed_size * (self.linear_sequence_length + 1)
                            - glob_res_input.shape[0]
                        )
                        glob_res_input = F.pad(glob_res_input, (0, pad_size))
                else:
                    glob_res_input = torch.zeros(
                        self.embed_size * (self.linear_sequence_length + 1),
                        device=channel_embedded.device,
                    )
                glob_res_list.append(glob_res_input)

            glob_res_input = torch.stack(glob_res_list, dim=0)
        else:
            glob_res_input = channel_embedded[
                :, -(self.linear_sequence_length + 1) :, :
            ].reshape(batch_size, -1)

        return (
            self.global_residual(glob_res_input)
            .unsqueeze(1)
            .repeat(1, prediction_length, 1)
        )

    def _generate_predictions(
            self,
            embedded: torch.Tensor,
            target_pos_embed: torch.Tensor,
            prediction_length: int,
            num_channels: int,
            history_padding_mask: torch.Tensor = None,
    ):
        """
        Generate predictions for all channels using vectorized operations.
        """
        batch_size, seq_len, _ = embedded.shape
        # embedded shape: [B, S, N*E] -> Reshape to [B, S, N, E]
        embedded = embedded.view(batch_size, seq_len, num_channels, self.embed_size)

        # Vectorize across channels by merging the batch and channel dimensions.
        # [B, S, N, E] -> [B*N, S, E]
        channel_embedded = (
            embedded.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * num_channels, seq_len, self.embed_size)
        )

        # Reshape target positional embeddings similarly: [B, P, N, E] -> [B*N, P, E]
        target_pos_embed = (
            target_pos_embed.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size * num_channels, prediction_length, self.embed_size)
        )
        target_repr = self.target_projection(target_pos_embed)

        # Reshape padding mask to match the vectorized input: [B, S] -> [B*N, S]
        if history_padding_mask is not None:
            history_padding_mask = (
                history_padding_mask.unsqueeze(1)
                .expand(-1, num_channels, -1)
                .reshape(batch_size * num_channels, seq_len)
            )

        # --- Process all channels in a single pass ---
        x = self.input_projection_layer(channel_embedded)

        glob_res = 0.0
        if self.global_residual is not None:
            glob_res = self._compute_global_residual(
                channel_embedded, history_padding_mask, prediction_length
            )

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        if self.use_gelu and self.initial_gelu is not None:
            x = self.initial_gelu(x)

        x_sliced = x[:, :174, :].mean(dim=1, keepdim=True)
        target_repr = self.input_projection_norm(target_repr)
        final_representation = x_sliced + target_repr
        if self.global_residual is not None:
            final_representation += glob_res

        predictions = self.final_output_layer(final_representation)

        # Reshape the output back to [B, P, N]
        predictions = predictions.view(batch_size, num_channels, prediction_length)
        predictions = predictions.permute(0, 2, 1)

        return predictions

    def forward(
        self, data_container: BatchTimeSeriesContainer, drop_enc_allow: bool = False
    ):
        """Main forward pass."""
        # Preprocess data
        preprocessed = self._preprocess_data(data_container)

        # Compute time features
        history_time_features, target_time_features = compute_batch_time_features(
            start=data_container.start,
            history_length=self.max_history_length,
            future_length=self.max_prediction_length,
            frequency=data_container.frequency,
            K_max=self.K_max,
            time_feature_config=self.time_feature_config,
        )

        # Compute scaling
        scale_statistics = self._compute_scaling(
            data_container.history_values, data_container.history_mask
        )

        # Apply scaling
        history_scaled = self._apply_scaling_and_masking(
            preprocessed["padded_history_values"],
            scale_statistics,
            preprocessed["combined_history_mask"],
        )

        # Scale future values if present
        future_scaled = None
        if preprocessed["padded_future_values"] is not None:
            future_scaled = self.scaler.scale(
                preprocessed["padded_future_values"], scale_statistics
            )

        # Get positional embeddings
        batch_size = preprocessed["padded_history_values"].shape[0]
        history_pos_embed = self._get_positional_embeddings(
            history_time_features,
            preprocessed["num_channels"],
            batch_size,
            drop_enc_allow,
        )
        target_pos_embed = self._get_positional_embeddings(
            target_time_features,
            preprocessed["num_channels"],
            batch_size,
            drop_enc_allow,
        )

        # Compute embeddings
        all_channels_embedded = self._compute_embeddings(
            history_scaled, history_pos_embed
        )

        # Generate predictions
        predictions = self._generate_predictions(
            all_channels_embedded,
            target_pos_embed,
            self.max_prediction_length,
            preprocessed["num_channels"],
            preprocessed["combined_history_mask"],
        )

        # Truncate to original lengths
        original_pred_len = preprocessed["original_lengths"]["future"]
        if original_pred_len > 0:
            predictions = predictions[:, :original_pred_len]
            if future_scaled is not None:
                future_scaled = future_scaled[:, :original_pred_len]

        return {
            "result": predictions,
            "scale_statistics": scale_statistics,
            "future_scaled": future_scaled,
            "original_lengths": preprocessed["original_lengths"],
        }

    def compute_loss(self, y_true: torch.Tensor, y_pred: dict):
        """Compute loss between predictions and scaled ground truth."""
        predictions = y_pred["result"]
        scale_statistics = y_pred["scale_statistics"]

        if y_true is None:
            return torch.tensor(0.0, device=predictions.device)

        original_pred_len = y_pred["original_lengths"]["future"]
        predictions = predictions[:, :original_pred_len]
        future_scaled = self.scaler.scale(y_true, scale_statistics)

        if predictions.shape != future_scaled.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs future_scaled {future_scaled.shape}"
            )
        return nn.functional.huber_loss(predictions, future_scaled)
