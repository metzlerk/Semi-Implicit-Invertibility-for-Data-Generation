#!/bin/bash
set -euo pipefail

# Prepare encoder_split5 data folder: copy split 5 to train/val/test
mkdir -p /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/encoder_split5/
cp -n /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/Data/clean_split_5_of_5.feather /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/encoder_split5/train_data.feather
cp -n /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/Data/clean_test_data.feather /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/encoder_split5/val_data.feather
cp -n /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/Data/clean_test_data.feather /home/kjmetzler/Semi-Implicit-Invertibility-for-Data-Generation/encoder_split5/test_data.feather

echo "Prepared encoder_split5 data folder"
