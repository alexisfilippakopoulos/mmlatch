#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mmlatch

python run_original_mosei.py --config configs/missing_modal_test/base_0_5.yaml
python run_original_mosei.py --config configs/missing_modal_test/l1_01_0_5.yaml
python run_original_mosei.py --config configs/missing_modal_test/l1_001_0_5.yaml
python run_original_mosei.py --config configs/missing_modal_test/l1_0001_0_5.yaml
python run_original_mosei.py --config configs/missing_modal_test/l1_00001_0_5.yaml
python run_original_mosei.py --config configs/missing_modal_test/l1_000001_0_5.yaml
python run_original_mosei.py --config configs/missing_modal_test/l1_0000001_0_5.yaml
python run_original_mosei.py --config configs/missing_modal_test/l2_01_0_5.yaml
python run_original_mosei.py --config configs/missing_modal_test/l2_001_0_5.yaml
python run_original_mosei.py --config configs/missing_modal_test/l2_0001_0_5.yaml
python run_original_mosei.py --config configs/missing_modal_test/l2_00001_0_5.yaml
python run_original_mosei.py --config configs/missing_modal_test/l2_000001_0_5.yaml
python run_original_mosei.py --config configs/missing_modal_test/l2_0000001_0_5.yaml