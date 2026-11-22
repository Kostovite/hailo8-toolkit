## Shape sweep: chip vs SDK_BIT_EXACT (frac mismatching / max LSB); only cases with any mismatch

C128_H32_W32: 57 chip-run, 2 with mismatch; stages: 1 "stage": "compile", 60 "stage": "ok", 24 "stage": "parse", 5 "stage": "quantize"
C24_H30_W50: 55 chip-run, 4 with mismatch; stages: 1 "stage": "compile", 58 "stage": "ok", 24 "stage": "parse", 4 "stage": "quantize"
C256_H12_W12: 56 chip-run, 4 with mismatch; stages: 1 "stage": "compile", 59 "stage": "ok", 24 "stage": "parse", 5 "stage": "quantize"
C32_H40_W40: 58 chip-run, 3 with mismatch; stages: 1 "stage": "compile", 61 "stage": "ok", 24 "stage": "parse", 4 "stage": "quantize"
C64_H16_W16: 58 chip-run, 4 with mismatch; stages: 1 "stage": "compile", 61 "stage": "ok", 24 "stage": "parse", 4 "stage": "quantize"
C8_H8_W8: 58 chip-run, 2 with mismatch; stages: 1 "stage": "compile", 61 "stage": "ok", 24 "stage": "parse", 4 "stage": "quantize"

| case | C128_H32_W32 | C24_H30_W50 | C256_H12_W12 | C32_H40_W40 | C64_H16_W16 | C8_H8_W8 |
|---|---|---|---|---|---|---|
| T_Conv_dil3 | 0 | 0.197/148 | 0 | 0 | 0 | 0 |
| T_GroupNorm | 0 | 0 | 0.000/1 | 0 | 0 | 0 |
| T_GroupNorm1 | 0.194/3 | 0.496/123 | 0.493/66 | 0.039/1 | 0.494/122 | 0.546/121 |
| T_InstanceNorm2d | 0 | 0.000/1 | 0.000/1 | 0.000/1 | 0.000/1 | 0.000/1 |
| T_LayerNorm_C | 0 | 0 | 0 | 0 | 0.000/1 | 0 |
| T_RMSNorm_C | 0 | 0.000/1 | 0 | 0 | 0 | 0 |
| T_ViT_block | 0.020/3 | 0 | 0.018/3 | 0.003/2 | 0.018/2 | 0 |

## GroupNorm(1) grid: frac mismatching (max LSB), rows C, cols H=W

| C \ H | 8 | 12 | 16 | 20 | 24 | 32 | 40 |
|---|---|---|---|---|---|---|---|
| 8 | 0.5161(77) | 0.5113(100) | 0.4965(68) | 0.5021(128) | 0.4944(129) | 0.4941(129) | 0.4931(67) |
| 16 | 0.5116(120) | 0.4965(121) | 0.4946(104) | 0.4964(119) | 0.4938(131) | 0.4947(128) | 0.4938(121) |
| 32 | 0.4981(69) | 0.4951(116) | 0.4936(68) | 0.4937(131) | 0.4938(129) | 0.4936(129) | 0.4229(65) |
| 64 | 0.4942(68) | 0.4953(130) | 0.4941(66) | 0.3734(125) | 0.4934(119) | 0.4924(66) | 0.4924(65) |
| 128 | 0.4937(68) | 0.4936(129) | 0.4933(66) | 0.4925(101) | 0.4924(111) | 0.4923(128) | 0.4923(121) |
