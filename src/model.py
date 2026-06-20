

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Channel-group indices  (SCENE_VARIABLES order, 26 channels total)
#
#   0–2   SAR      : nersc_sar_primary, nersc_sar_secondary, sar_incidenceangle
#   3–7  AMRS     : distance_map + AMSR2 bands (5 channels)
#   8–12 ENV      : u10m_rotated, v10m_rotated, t2m, tcwv, tclw
#   13–15 AUX      : aux_time, aux_lat, aux_long
# =============================================================================

SAR_IDX  = [0, 1, 2]
AMRS_IDX = [3, 4, 5, 6, 7]
ENV_IDX  = [8, 9, 10, 11, 12]
AUX_IDX  = [13, 14, 15]

N_SAR   = len(SAR_IDX)    # 3
N_AMRS  = len(AMRS_IDX)   # 5
N_ENV   = len(ENV_IDX)    # 5
N_AUX   = len(AUX_IDX)    # 3
N_GROUPS = 4

TASKS = ['SIC', 'SOD', 'FLOE']
N_TASKS = len(TASKS)


# =============================================================================
# Basic blocks borrowed from : github.com/echonax07/MMSeaIce
# =============================================================================

class DoubleConv(nn.Module):
    def __init__(self, input_n, output_n):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_n, output_n, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_n),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_n, output_n, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_n),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class ResDoubleConv(nn.Module):
    """DoubleConv with residual projection — used as expert body and FLOE decoder."""
    def __init__(self, input_n, output_n):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_n, output_n, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_n),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_n, output_n, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_n),
        )
        self.proj = (nn.Conv2d(input_n, output_n, 1, bias=False)
                     if input_n != output_n else nn.Identity())
        self.bn   = nn.BatchNorm2d(output_n)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x) + self.proj(x)))


class ContractingBlock(nn.Module):
    def __init__(self, input_n, output_n):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(input_n, output_n)

    def forward(self, x):
        return self.conv(self.pool(x))


class ExpandingBlock(nn.Module):
    def __init__(self, input_n, output_n):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(input_n + output_n, output_n)

    def forward(self, x, skip):
        x = self.up(x)
        x = _pad_to(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class ExpandingBlockRes(nn.Module):
    def __init__(self, input_n, output_n):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ResDoubleConv(input_n + output_n, output_n)

    def forward(self, x, skip):
        x = self.up(x)
        x = _pad_to(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


def _pad_to(x, ref):
    """Pad x so its spatial dims match ref."""
    dY = ref.size(2) - x.size(2)
    dX = ref.size(3) - x.size(3)
    return F.pad(x, [dX // 2, dX - dX // 2, dY // 2, dY - dY // 2])


class FeatureMap(nn.Module):
    def __init__(self, input_n, output_n):
        super().__init__()
        self.conv = nn.Conv2d(input_n, output_n, 1)

    def forward(self, x):
        return self.conv(x)


# =============================================================================
# ASPP  (Atrous Spatial Pyramid Pooling (Chen et al. 2017).
# =============================================================================
class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1   = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.conv6   = nn.Conv2d(in_ch, out_ch, 3, padding=6,  dilation=6,  bias=False)
        self.conv12  = nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False)
        self.conv18  = nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False)
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.pool_conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.out  = nn.Conv2d(out_ch * 5, out_ch, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.shape[-2:]
        f5 = F.interpolate(self.pool_conv(self.pool(x)),
                           size=size, mode='bilinear', align_corners=True)
        return self.relu(self.bn(self.out(torch.cat([
            self.conv1(x), self.conv6(x), self.conv12(x), self.conv18(x), f5
        ], dim=1))))


# =============================================================================
# CBAM  (Convolutional Block Attention Module (Woo et al. 2018).
# Applied sequentially: channel first, then spatial (order based on paper).
# =============================================================================

class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16, spatial_kernel=7):
        super().__init__()
        reduced = max(channels // reduction_ratio, 1)

        # --- Channel attention ---
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
        )
        self.channel_sigmoid = nn.Sigmoid()

        # --- Spatial attention ---
        assert spatial_kernel % 2 == 1, "spatial_kernel must be odd"
        padding = spatial_kernel // 2
        self.spatial_conv = nn.Conv2d(2, 1, spatial_kernel,
                                      padding=padding, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.channel_mlp(self.avg_pool(x))
        max_out = self.channel_mlp(self.max_pool(x))
        x = x * self.channel_sigmoid(avg_out + max_out)

        # Spatial attention
        avg_s = x.mean(dim=1, keepdim=True)
        max_s = x.max(dim=1, keepdim=True).values
        x = x * self.spatial_sigmoid(
            self.spatial_conv(torch.cat([avg_s, max_s], dim=1))
        )
        return x


class SkipAttentionGate(nn.Module):
    def __init__(self, F_skip, F_gate):
        super().__init__()
        F_int = max(F_skip // 2, 1)
        self.W_g  = nn.Conv2d(F_gate, F_int, 1, bias=False)
        self.W_x  = nn.Conv2d(F_skip, F_int, 1, bias=False)
        self.psi  = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.bn   = nn.BatchNorm2d(F_int)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_skip, g):
        if g.shape[-2:] != x_skip.shape[-2:]:
            g = F.interpolate(g, size=x_skip.shape[-2:],
                              mode='bilinear', align_corners=True)
        att = self.psi(self.relu(self.bn(self.W_g(g) + self.W_x(x_skip))))
        return x_skip * att



# One way(detached tensors)
class CrossDecoderGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, recipient, donor):
        if donor.shape[-2:] != recipient.shape[-2:]:
            donor = F.interpolate(donor, size=recipient.shape[-2:],
                                  mode='bilinear', align_corners=True)
        w = self.gate(torch.cat([recipient, donor], dim=1))
        return recipient + w * self.proj(donor)



class DualCrossDecoderFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        for name in ('sod', 'sic'):
            setattr(self, f'gate_{name}', nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.Sigmoid(),
            ))
            setattr(self, f'proj_{name}',
                    nn.Conv2d(channels, channels, 1, bias=False))

    def _align(self, t, ref):
        if t.shape[-2:] != ref.shape[-2:]:
            t = F.interpolate(t, size=ref.shape[-2:],
                              mode='bilinear', align_corners=True)
        return t

    def forward(self, recipient, sod, sic):
        sod = self._align(sod, recipient)
        sic = self._align(sic, recipient)
        w_sod = self.gate_sod(torch.cat([recipient, sod], dim=1))
        w_sic = self.gate_sic(torch.cat([recipient, sic], dim=1))
        return (recipient
                + w_sod * self.proj_sod(sod)
                + w_sic * self.proj_sic(sic))



# Make decoder for FLOE a bit deeper to see if that helps
class FLOERefinement(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1),  nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, padding=4, dilation=4), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)



class GroupEncoder(nn.Module):

    def __init__(self, in_ch, filters):
        super().__init__()
        self.input_block     = DoubleConv(in_ch, filters[0])
        self.contract_blocks = nn.ModuleList(
            ContractingBlock(filters[i - 1], filters[i])
            for i in range(1, len(filters))
        )

    def forward(self, x):
        skips = [self.input_block(x)]
        for blk in self.contract_blocks:
            skips.append(blk(skips[-1]))
        return skips          # list[tensor], len = n_skip


class MMoESkipLayer(nn.Module):

    def __init__(self, in_ch, out_ch, n_experts=4):
        """
        in_ch    : channels of each group's encoder skip at this scale
        out_ch   : output channels  (= filters[i], same for all tasks)
        n_experts: number of experts per group
        """
        super().__init__()
        self.n_experts = n_experts
        self.n_groups  = N_GROUPS

        # Expert pools: N_GROUPS × n_experts ResDoubleConv blocks
        self.experts = nn.ModuleList([
            nn.ModuleList([ResDoubleConv(in_ch, out_ch) for _ in range(n_experts)])
            for _ in range(N_GROUPS)
        ])

        # Per-group gating MLP: gap → [B, n_experts] softmax weights
        # Input is the raw group skip (in_ch channels)
        self.group_gates = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),           # [B, in_ch, 1, 1]
                nn.Flatten(),                       # [B, in_ch]
                nn.Linear(in_ch, n_experts),
                nn.Softmax(dim=-1),                 # [B, n_experts]
            )
            for _ in range(N_GROUPS)
        ])

        # Per-task gating MLP: gap on concatenated group features → [B, N_GROUPS] softmax
        # Input is all group fused features cat'd: out_ch * N_GROUPS channels
        self.task_gates = nn.ModuleDict({
            task: nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_ch * N_GROUPS, N_GROUPS),
                nn.Softmax(dim=-1),                 # [B, N_GROUPS]
            )
            for task in TASKS
        })

    def forward(self, group_skips):

        expert_outs = [
            [self.experts[g][e](group_skips[g]) for e in range(self.n_experts)]
            for g in range(self.n_groups)
        ]


        group_fused = []
        for g in range(self.n_groups):
            weights = self.group_gates[g](group_skips[g])  # [B, n_experts]
            # Stack experts: [B, n_experts, out_ch, H, W]
            stacked = torch.stack(expert_outs[g], dim=1)
            # Weighted sum: [B, out_ch, H, W]
            fused = (stacked * weights[:, :, None, None, None]).sum(dim=1)
            group_fused.append(fused)

        # Concatenate all group features for gate input: [B, out_ch*N_GROUPS, H, W]
        all_groups_cat = torch.cat(group_fused, dim=1)

        task_skips = {}
        for task in TASKS:
            weights = self.task_gates[task](all_groups_cat)  # [B, N_GROUPS]
            # Stack groups: [B, N_GROUPS, out_ch, H, W]
            stacked = torch.stack(group_fused, dim=1)
            task_skips[task] = (stacked * weights[:, :, None, None, None]).sum(dim=1)

        return task_skips   # dict: task → [B, out_ch, H, W]


# =============================================================================
# MAIN MODEL
# =============================================================================

class UNet_MTL_MMoE(nn.Module):


    def __init__(self, options):
        super().__init__()

        filters  = options['unet_conv_filters']   
        n_experts = options.get('n_experts', 4)
        C        = filters[-1]                    # bottleneck channels
        n_skip   = len(filters)

        # ----------------------------------------------------------------
        # Four parallel group encoders
        # ----------------------------------------------------------------
        self.enc_sar  = GroupEncoder(N_SAR,  filters)
        self.enc_amrs = GroupEncoder(N_AMRS, filters)
        self.enc_env  = GroupEncoder(N_ENV,  filters)
        self.enc_aux  = GroupEncoder(N_AUX,  filters)

        # ----------------------------------------------------------------
        # Per-scale MMoE layers
        #   in_ch  = filters[i]  (each group encoder outputs this at level i)
        #   out_ch = filters[i]  (restore standard channel count)
        # ----------------------------------------------------------------
        self.mmoe_layers = nn.ModuleList([
            MMoESkipLayer(filters[i], filters[i], n_experts=n_experts)
            for i in range(n_skip)
        ])


        self.bridge = ContractingBlock(filters[-1], filters[-1])
        self.aspp   = ASPP(C, C)


        self.attn_sic  = CBAM(C)
        self.attn_sod  = CBAM(C)
        self.attn_floe = CBAM(C)

        self.floe_refine = FLOERefinement(C)

        # ----------------------------------------------------------------
        # Decoders
        # ----------------------------------------------------------------
        self.expand_sic  = self._make_decoder(filters, residual=False)
        self.expand_sod  = self._make_decoder(filters, residual=False)
        self.expand_floe = self._make_decoder(filters, residual=True)

        # ----------------------------------------------------------------
        # Per-task skip attention gates
        #
        # At decode step k (0 = closest to bottleneck) the gating signal x
        # has channels:
        #   k = 0   → filters[-1]   (bottleneck output, first upsample)
        #   k = 1   → filters[-1]   (output of first expanding block)
        #   k >= 2  → filters[-(k)] (output of block k-1)
        #
        # Gate index j maps to skip level:  j = n_skip - 1 - k
        # F_gate[j] = filters[ min(j+1, n_skip-1) ]
        # ----------------------------------------------------------------
        def _gate_ch(j):
            return filters[min(j + 1, n_skip - 1)]

        self.gates_sic  = nn.ModuleList([
            SkipAttentionGate(filters[i], _gate_ch(i)) for i in range(n_skip)
        ])
        self.gates_sod  = nn.ModuleList([
            SkipAttentionGate(filters[i], _gate_ch(i)) for i in range(n_skip)
        ])
        self.gates_floe = nn.ModuleList([
            SkipAttentionGate(filters[i], _gate_ch(i)) for i in range(n_skip)
        ])

        # ----------------------------------------------------------------
        # Cross-decoder fusion
        #   SIC → SOD  post-decode  [DETACHED]
        #   SOD + SIC → FLOE at mid (level 1) and late (level 2) [DETACHED]
        # ----------------------------------------------------------------
        assert n_skip >= 3, "filters must have at least 3 levels for cross-decoder fusion"
        self.sic_to_sod_gate  = CrossDecoderGate(filters[0])
        self.fusion_mid        = DualCrossDecoderFusion(filters[-2])
        self.fusion_late       = DualCrossDecoderFusion(filters[-3])


        self.regression_layer = nn.Linear(filters[0], 1)   # SIC [B,H,W,1]
        self.sod_out           = FeatureMap(filters[0], options['n_classes']['SOD'])
        self.floe_out          = FeatureMap(filters[0], options['n_classes']['FLOE'])


    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _make_decoder(self, filters, residual=False):
        Block  = ExpandingBlockRes if residual else ExpandingBlock
        blocks = nn.ModuleList()
        blocks.append(Block(filters[-1], filters[-1]))
        for i in range(len(filters), 1, -1):
            blocks.append(Block(filters[i - 1], filters[i - 2]))
        return blocks

    def _decode_capture(self, x, task_skips, decoder, gates):
        """
        Gated decode; returns (final_feat, intermediates).
        task_skips : list of per-task skip tensors [skip_level_0 .. skip_level_n-1]
                     ordered shallowest → deepest  (index 0 = filters[0] resolution)
        intermediates[k] is captured after expanding block k (0 = closest to bottleneck).
        """
        intermediates = []
        n = len(task_skips)
        for k, (block, gate) in enumerate(zip(decoder, reversed(gates))):
            skip = task_skips[n - 1 - k]
            gated_skip = gate(skip, x)
            x = block(x, gated_skip)
            intermediates.append(x)
        return x, intermediates

    def _decode_floe(self, x, task_skips, sod_ints, sic_ints):
        """
        FLOE decode with gated skips + dual cross-decoder fusion at
        levels 1 (mid) and 2 (late).  Donor intermediates must already
        be detached by the caller.
        """
        n = len(task_skips)
        for k, (block, gate) in enumerate(
                zip(self.expand_floe, reversed(self.gates_floe))):
            skip = task_skips[n - 1 - k]
            gated_skip = gate(skip, x)
            x = block(x, gated_skip)
            if k == 1 and len(sod_ints) > 1 and len(sic_ints) > 1:
                x = self.fusion_mid(x, sod_ints[1], sic_ints[1])
            elif k == 2 and len(sod_ints) > 2 and len(sic_ints) > 2:
                x = self.fusion_late(x, sod_ints[2], sic_ints[2])
        return x

    # ----------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------

    def forward(self, x):
        # ---- Split input into channel groups -------------------------
        x_sar  = x[:, SAR_IDX,  :, :]    # [B,  3, H, W]
        x_amrs = x[:, AMRS_IDX, :, :]    # [B, 5, H, W]
        x_env  = x[:, ENV_IDX,  :, :]    # [B,  5, H, W]
        x_aux  = x[:, AUX_IDX,  :, :]    # [B,  3, H, W]

        # ---- Four parallel encoders ----------------------------------
        # Each returns a list of n_skip tensors (shallowest → deepest)
        skips_sar  = self.enc_sar(x_sar)
        skips_amrs = self.enc_amrs(x_amrs)
        skips_env  = self.enc_env(x_env)
        skips_aux  = self.enc_aux(x_aux)

        # print("Encoder output shapes (shallowest → deepest):")
        # for i in range(len(skips_sar)):
        #     print(f"Level {i}: SAR {skips_sar[i].shape}, "
        #           f"AMRS {skips_amrs[i].shape}, "
        #           f"ENV {skips_env[i].shape}, "
        #           f"AUX {skips_aux[i].shape}")
        # ---- MMoE per scale ------------------------------------------
        # mmoe_skips[i] = dict  task → [B, filters[i], H/2^i, W/2^i]
        n_skip = len(skips_sar)
        mmoe_skips = [
            self.mmoe_layers[i]([
                skips_sar[i], skips_amrs[i], skips_env[i], skips_aux[i]
            ])
            for i in range(n_skip)
        ]
        # for skip in mmoe_skips:
        #     print(skip['SIC'].shape, skip['SOD'].shape, skip['FLOE'].shape)

        task_skips = {
            task: [mmoe_skips[i][task] for i in range(n_skip)]
            for task in TASKS
        }

        deepest = torch.stack(
            [task_skips[t][-1] for t in TASKS], dim=0
        ).mean(dim=0)
        bottleneck = self.aspp(self.bridge(deepest))

        # ---- Task-specific bottleneck attention (CBAM) ---------------
        x_sic  = self.attn_sic(bottleneck)
        x_sod  = self.attn_sod(bottleneck)
        x_floe = self.attn_floe(bottleneck)
        x_floe = self.floe_refine(x_floe)

        # ---- Decode SIC (capture intermediates) ----------------------
        x_sic, sic_ints = self._decode_capture(
            x_sic, task_skips['SIC'], self.expand_sic, self.gates_sic
        )

        # ---- Decode SOD (capture intermediates) ----------------------
        x_sod, sod_ints = self._decode_capture(
            x_sod, task_skips['SOD'], self.expand_sod, self.gates_sod
        )

        # SIC → SOD post-decode refinement (DETACHED: SOD loss ≠ SIC grad)
        x_sod = self.sic_to_sod_gate(x_sod, x_sic.detach())

        # ---- Decode FLOE with dual cross-talk (DETACHED) -------------
        sic_ints_d = [f.detach() for f in sic_ints]
        sod_ints_d = [f.detach() for f in sod_ints]
        x_floe = self._decode_floe(
            x_floe, task_skips['FLOE'], sod_ints_d, sic_ints_d
        )

        # ---- Output heads --------------------------------------------
        return {
            'SIC':  self.regression_layer(x_sic.permute(0, 2, 3, 1)),  # [B,H,W,1]
            'SOD':  self.sod_out(x_sod),                                # [B,C,H,W]
            'FLOE': self.floe_out(x_floe),                              # [B,C,H,W]
        }