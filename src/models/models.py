import numpy as np
import torch
import torch.nn as nn

from src.models.blocks import (
    ConcatLayer,
    DilatedConv1dBlock,
    EncoderFactory,
    SinPositionalEncoding,
)
from src.models.constants import DAY, DOW, HOUR, MINUTE, MONTH, NUM_TASKS, YEAR
from src.utils.utils import CustomScaling, PositionExpansion, device


class BaseModel(nn.Module):
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
        super().__init__(**kwargs)
        assert encoding_dropout >= 0.0 and encoding_dropout <= 1.0
        self.epsilon = epsilon
        # pos embeddings
        self.sub_day = sub_day
        self.encoding_dropout = encoding_dropout
        self.handle_constants_model = handle_constants_model
        if sub_day:
            self.pos_minute = PositionExpansion(60, 4)
            self.pos_hour = PositionExpansion(24, 6)
        self.pos_year = PositionExpansion(10, 4)
        self.pos_month = PositionExpansion(12, 4)
        self.pos_day = PositionExpansion(31, 6)
        self.pos_dow = PositionExpansion(7, 4)
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
        # scaler
        self.scaler = CustomScaling(scaler)
        # time series initial embedding
        self.expand_target_nopos = nn.Linear(1, self.embed_size, bias=True)
        self.expand_target_nopos.name = "NoPosEnc"
        self.expand_target_forpos = nn.Linear(1, self.embed_size, bias=True)
        self.expand_target_forpos.name = "ForPosEnc"
        # sinuosoidal positional encoding
        self.sin_pos_encoder = None
        self.sin_pos_flag = sin_pos_enc
        if sin_pos_enc:
            self.sin_pos_encoder = SinPositionalEncoding(
                d_model=self.embed_size, max_len=5000, sin_pos_const=sin_pos_const
            )
        self.concat_pos = ConcatLayer(dim=-1, name="ConcatPos")
        self.concat_embed = ConcatLayer(dim=-1, name="ConcatEmbed")
        self.concat_target = ConcatLayer(dim=1, name="AppendTarget")
        self.target_marker = nn.Embedding(NUM_TASKS, self.embed_size)

    @staticmethod
    def tc(ts: torch.Tensor, time_index: int):
        return ts[:, :, time_index]

    def forward(self, x, prediction_length=None, training=False, drop_enc_allow=False):
        # drop enc allow is a flag to allow dropping of positional encodings as we use it in case of patching the preds so we can't drop the encodings
        ts, history, target_dates, task = (
            x["ts"],
            x["history"],
            x["target_dates"],
            x["task"],
        )
        # Build position encodings
        if self.handle_constants_model:
            idx_const_in = torch.nonzero(
                torch.all(history == history[:, 0].unsqueeze(1), dim=1)
            ).squeeze(1)
            if idx_const_in.size(0) > 0:
                # not sure yet, to be tested
                history[idx_const_in, 0] += np.random.uniform(0.1, 1)

        if (np.random.rand() < self.encoding_dropout) and training and drop_enc_allow:
            pos_embedding = torch.zeros(
                ts.shape[0], ts.shape[1], self.embed_size, device=device
            ).to(torch.float32)
            target_pos_embed = torch.zeros(
                target_dates.shape[0],
                target_dates.shape[1],
                self.embed_size,
                device=device,
            ).to(torch.float32)
        else:
            if self.sin_pos_flag:
                input_total = torch.concatenate([history, task], dim=1)
                total_pos_embedding = self.sin_pos_encoder(input_total).to(
                    torch.float32
                )
                pos_embedding = total_pos_embedding[:, : history.shape[1]]
                target_pos_embed = total_pos_embedding[:, history.shape[1] :]
            else:
                year = self.tc(ts, YEAR)
                delta_year = torch.clamp(
                    year[:, -1:] - year, min=0, max=self.pos_year.periods
                )
                target_year = torch.clamp(
                    year[:, -1:] - self.tc(target_dates, YEAR),
                    min=0,
                    max=self.pos_year.periods,
                )

                if self.sub_day:
                    pos_embedding = self.concat_pos(
                        [
                            self.pos_minute(self.tc(ts, MINUTE)),
                            self.pos_hour(self.tc(ts, HOUR)),
                            self.pos_year(delta_year),
                            self.pos_month(self.tc(ts, MONTH)),
                            self.pos_day(self.tc(ts, DAY)),
                            self.pos_dow(self.tc(ts, DOW)),
                        ]
                    ).to(torch.float32)

                    target_pos_embed = self.concat_pos(
                        [
                            self.pos_minute(self.tc(target_dates, MINUTE)),
                            self.pos_hour(self.tc(target_dates, HOUR)),
                            self.pos_year(target_year),
                            self.pos_month(self.tc(target_dates, MONTH)),
                            self.pos_day(self.tc(target_dates, DAY)),
                            self.pos_dow(self.tc(target_dates, DOW)),
                        ]
                    ).to(torch.float32)

                else:
                    pos_embedding = self.concat_pos(
                        [
                            self.pos_year(delta_year),
                            self.pos_month(self.tc(ts, MONTH)),
                            self.pos_day(self.tc(ts, DAY)),
                            self.pos_dow(self.tc(ts, DOW)),
                        ]
                    ).to(torch.float32)

                    target_pos_embed = self.concat_pos(
                        [
                            self.pos_year(target_year),
                            self.pos_month(self.tc(target_dates, MONTH)),
                            self.pos_day(self.tc(target_dates, DAY)),
                            self.pos_dow(self.tc(target_dates, DOW)),
                        ]
                    ).to(torch.float32)

        # Embed history
        history_channels = history.unsqueeze(-1)
        med_scale, scaled = self.scaler(history_channels, self.epsilon)

        embed_nopos = self.expand_target_nopos(scaled)
        embed_pos = (self.expand_target_forpos(scaled) + pos_embedding).to(
            torch.float32
        )
        embedded = self.concat_embed([embed_nopos, embed_pos])  # 72  dil cov ->  4*72

        task_embed = self.target_marker(task)
        target = self.concat_embed([task_embed, task_embed + target_pos_embed])
        full_embedded = self.concat_target([embedded, target])

        result = self.forecast(full_embedded, prediction_length)
        # TODO apply rescaling to the resul if needed
        return {"result": result, "scale": med_scale}

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        scale = y_pred["scale"]
        return nn.functional.mse_loss(
            y_pred["result"] / scale, y / scale, reduction="mean"
        )

    def forecast(
        self,
        ts: torch.Tensor,
        scale: torch.Tensor,
        embedded: torch.Tensor,
        target: torch.Tensor,
    ):
        raise NotImplementedError


class MultiStepModel(BaseModel):
    def __init__(
        self,
        base_model_config,
        encoder_config,
        scaler="custom_robust",
        num_encoder_layers=2,
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
        token_input_dim = self.embed_size * 2

        if use_dilated_conv:
            self.input_projection_layer = DilatedConv1dBlock(
                token_input_dim,
                token_embed_dim,
                dilated_conv_kernel_size,
                dilated_conv_max_dilation,
                single_conv=False,
            )
        else:
            self.input_projection_layer = nn.Linear(token_input_dim, token_embed_dim)

        self.global_residual = (
            nn.Linear(token_input_dim * (linear_sequence_length + 1), token_embed_dim)
            if use_global_residual
            else None
        )

        self.linear_sequence_length = linear_sequence_length
        self.initial_gelu = nn.GELU() if use_gelu else None
        self.input_projection_norm = (
            nn.LayerNorm(token_embed_dim)
            if use_input_projection_norm
            else nn.Identity()
        )

        # Create encoder layers using the factory
        self.encoder_layers = nn.ModuleList(
            [
                EncoderFactory.create_encoder(
                    token_embed_dim=token_embed_dim, **encoder_config
                )
                for _ in range(self.num_encoder_layers)
            ]
        )

        self.final_output_layer = (
            nn.Linear(token_embed_dim * 2, 1)
            if use_global_residual
            else nn.Linear(token_embed_dim, 1)
        )
        self.final_output_layer.name = "FinalOutput"
        self.final_activation = nn.Identity()

    def forecast(self, embedded: torch.Tensor, prediction_length: int):
        if self.global_residual:
            glob_res = embedded[:, -(self.linear_sequence_length + 1) :, :].reshape(
                embedded.shape[0], -1
            )
            glob_res = (
                self.global_residual(glob_res)
                .unsqueeze(1)
                .repeat(1, prediction_length, 1)
            )

        x = self.input_projection_layer(embedded)

        if self.use_gelu and self.initial_gelu is not None:
            x = self.input_projection_norm(x)
            x = self.initial_gelu(x)

        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        if self.global_residual:
            x = self.final_output_layer(
                torch.concat([x[:, -prediction_length:, :], glob_res], dim=-1)
            )
        else:
            x = self.final_output_layer(x[:, -prediction_length:, :])

        x = self.final_activation(x)

        # Keep the feature dimension to match target shape [batch_size, seq_len, 1]
        # Instead of squeezing the last dimension
        return x
