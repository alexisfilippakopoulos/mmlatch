#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmlatch

python run_original_mosei.py --config configs/noisy_test/base_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/l1_01_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/l1_001_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/l1_0001_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/l1_00001_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/l1_000001_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/l1_0000001_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/l2_01_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/l2_001_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/l2_0001_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/l2_00001_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/l2_000001_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/l2_0000001_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/base_noise_0_5.yaml
python run_original_mosei.py --config configs/noisy_test/l1_01_noise_0_5.yaml
python run_original_mosei.py --config configs/noisy_test/l1_001_noise_0_5.yaml
python run_original_mosei.py --config configs/noisy_test/l1_0001_noise_0_5.yaml
python run_original_mosei.py --config configs/noisy_test/l1_00001_noise_0_5.yaml
python run_original_mosei.py --config configs/noisy_test/l1_000001_noise_0_5.yaml
python run_original_mosei.py --config configs/noisy_test/l1_0000001_noise_0_5.yaml
python run_original_mosei.py --config configs/noisy_test/l2_01_noise_0_5.yaml
python run_original_mosei.py --config configs/noisy_test/l2_001_noise_0_5.yaml
python run_original_mosei.py --config configs/noisy_test/l2_0001_noise_0_5.yaml
python run_original_mosei.py --config configs/noisy_test/l2_00001_noise_0_5.yaml
python run_original_mosei.py --config configs/noisy_test/l2_000001_noise_0_5.yaml
python run_original_mosei.py --config configs/noisy_test/l2_0000001_noise_0_5.yaml