#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmlatch

python run_original_mosei.py --config configs/noisy_test/l1_3_16e-5_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/l1_5_62e-5_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/l1_1_78e-4_noise_0_3.yaml
python run_original_mosei.py --config configs/noisy_test/l1_3_16e-4_noise_0_3.yaml


python run_original_mosei.py --config configs/noisy_test/l1_3_16e-6_noise_0_5.yaml
python run_original_mosei.py --config configs/noisy_test/l1_5_62e-6_noise_0_5.yaml
python run_original_mosei.py --config configs/noisy_test/l1_1_78e-5_noise_0_5.yaml
python run_original_mosei.py --config configs/noisy_test/l1_3_16e-5_noise_0_5.yaml