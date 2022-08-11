#!/bin/bash

python test.py \
    ./config/ddpm_cosine_hybird_timestep-4k_drop0.3_customdata_64x64_b8x8_50k.py \
    ./work_dirs/experiments/ddpm_test_50k/ckpt/ddpm_test_50k/latest.pth \
    --save-path ./work_dirs/experiments/ddpm_test_50k/ddpm_samples.gif # --sample-model [orig | ema | ema/orig] --same-noise

python test.py \
    ./config/ddpm_cosine_hybird_timestep-4k_drop0.3_customdata_64x64_b8x4_100k.py \
    ./work_dirs/experiments/ddpm_test_100k/ckpt/ddpm_test_100k/latest.pth \
    --save-path ./work_dirs/experiments/ddpm_test_100k/ddpm_ema-original_same-noise_samples.gif \
    --sample-model ema/orig # --same-noise

python test.py \
    ./config/ddpm_cosine_hybird_timestep-4k_drop0.3_customdata_128x128_b8x4_100k.py \
    ./work_dirs/experiments/ddpm_128_test_100k/ckpt/ddpm_128_test_100k/latest.pth \
    --save-path ./work_dirs/experiments/ddpm_128_test_100k/ddpm_ema-original_samples.gif \
    --sample-model ema/orig