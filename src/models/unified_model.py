import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules import GatedMLP

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
        self.epsilon = epsilon
        self.scaler_clamp_value = scaler_clamp_value
        self.handle_constants = handle_constants
        self.encoding_dropout = encoding_dropout
        self.K_max = K_max
        self.time_feature_config = time_feature_config or {}
        self.encoder_config = encoder_config or {}

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
        self.mlp = GatedMLP(
            hidden_size=self.token_embed_dim,
            hidden_ratio=4,
            hidden_act="swish",
            fuse_swiglu=True,
        )

        # Initialize learnable initial hidden state for the first encoder layer
        # This will be expanded to match batch size during forward pass
        head_dim = self.token_embed_dim // self.encoder_config["num_heads"]
        self.initial_hidden_state = nn.Parameter(
            torch.randn(1, self.encoder_config["num_heads"], head_dim, head_dim)
            / head_dim,
            requires_grad=True,
        )

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
        """Extract data shapes and handle constants without padding."""
        history_values = data_container.history_values
        future_values = data_container.future_values
        history_mask = data_container.history_mask

        batch_size, history_length, num_channels = history_values.shape
        future_length = future_values.shape[1] if future_values is not None else 0

        # Handle constants
        if self.handle_constants:
            history_values = apply_channel_noise(history_values)

        return {
            "history_values": history_values,
            "future_values": future_values,
            "history_mask": history_mask,
            "num_channels": num_channels,
            "history_length": history_length,
            "future_length": future_length,
            "batch_size": batch_size,
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

        batch_size, seq_len = scaled_history.shape[:2]
        all_channels_embedded = channel_embeddings.view(batch_size, seq_len, -1)

        return all_channels_embedded

    def _compute_global_residual(
        self,
        channel_embedded: torch.Tensor,
        history_mask: torch.Tensor,
        prediction_length: int,
    ):
        """Compute global residual connection."""
        batch_size, seq_len = channel_embedded.shape[:2]

        if history_mask is not None:
            # Use mask to find valid sequence lengths
            valid_lengths = history_mask.sum(dim=1).long()
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
            # Use the last linear_sequence_length + 1 positions
            start_idx = max(0, seq_len - self.linear_sequence_length - 1)
            glob_res_input = channel_embedded[:, start_idx:, :].reshape(batch_size, -1)

            # Pad if necessary
            expected_size = self.embed_size * (self.linear_sequence_length + 1)
            if glob_res_input.shape[1] < expected_size:
                pad_size = expected_size - glob_res_input.shape[1]
                glob_res_input = F.pad(glob_res_input, (0, pad_size))

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
        history_mask: torch.Tensor = None,
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

        # Reshape mask to match the vectorized input: [B, S] -> [B*N, S]
        if history_mask is not None:
            history_mask = (
                history_mask.unsqueeze(1)
                .expand(-1, num_channels, -1)
                .reshape(batch_size * num_channels, seq_len)
            )

        # --- Process all channels in a single pass ---
        x = self.input_projection_layer(channel_embedded)

        if history_mask is not None:
            x = x * history_mask.unsqueeze(-1).float()

        glob_res = 0.0
        if self.global_residual is not None:
            glob_res = self._compute_global_residual(
                channel_embedded, history_mask, prediction_length
            )

        # Initialize with learnable initial state for the first layer
        hidden_state = self.initial_hidden_state.repeat(
            batch_size * num_channels, 1, 1, 1
        )

        for layer_idx, encoder_layer in enumerate(self.encoder_layers):
            x, hidden_state = encoder_layer(x, hidden_state)

            if history_mask is not None:
                x = x * history_mask.unsqueeze(-1).float()

        if self.use_gelu and self.initial_gelu is not None:
            x = self.input_projection_norm(x)
            x = self.initial_gelu(x)

        # Use the last prediction_length positions
        x_sliced = x[:, -prediction_length:, :]

        final_representation = x_sliced + target_repr
        if self.global_residual is not None:
            final_representation += glob_res

        predictions = self.final_output_layer(self.mlp(final_representation))

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

        # Compute time features dynamically based on actual lengths
        history_time_features, target_time_features = compute_batch_time_features(
            start=data_container.start,
            history_length=preprocessed["history_length"],
            future_length=preprocessed["future_length"],
            frequency=data_container.frequency,
            K_max=self.K_max,
            time_feature_config=self.time_feature_config,
        )

        # Compute scaling
        scale_statistics = self._compute_scaling(
            preprocessed["history_values"], preprocessed["history_mask"]
        )

        # Apply scaling
        history_scaled = self._apply_scaling_and_masking(
            preprocessed["history_values"],
            scale_statistics,
            preprocessed["history_mask"],
        )

        # Scale future values if present
        future_scaled = None
        if preprocessed["future_values"] is not None:
            future_scaled = self.scaler.scale(
                preprocessed["future_values"], scale_statistics
            )

        # Get positional embeddings
        history_pos_embed = self._get_positional_embeddings(
            history_time_features,
            preprocessed["num_channels"],
            preprocessed["batch_size"],
            drop_enc_allow,
        )
        target_pos_embed = self._get_positional_embeddings(
            target_time_features,
            preprocessed["num_channels"],
            preprocessed["batch_size"],
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
            preprocessed["future_length"],
            preprocessed["num_channels"],
            preprocessed["history_mask"],
        )

        return {
            "result": predictions,
            "scale_statistics": scale_statistics,
            "future_scaled": future_scaled,
            "history_length": preprocessed["history_length"],
            "future_length": preprocessed["future_length"],
        }

    def compute_loss(self, y_true: torch.Tensor, y_pred: dict):
        """Compute loss between predictions and scaled ground truth."""
        predictions = y_pred["result"]
        scale_statistics = y_pred["scale_statistics"]

        if y_true is None:
            return torch.tensor(0.0, device=predictions.device)

        future_scaled = self.scaler.scale(y_true, scale_statistics)

        if predictions.shape != future_scaled.shape:
            raise ValueError(
                f"Shape mismatch: predictions {predictions.shape} vs future_scaled {future_scaled.shape}"
            )

        return nn.functional.huber_loss(predictions, future_scaled)
