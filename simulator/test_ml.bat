@echo off
REM script for running python script multiple times with arguments:
python main.py 0.32 0 1 --ml_test_APS --folder nu_standard -sf --paths 180000
python main.py 0.32 0 1 --ml_test_APS --folder nu_standard -sf -diff --paths 180000
python main.py 0.32 0 1 --ml_test_KD --folder nu_standard -sf --paths 180000
python main.py 0.1 0 1 --ml_test_APS --folder nu_standard -sf --paths 180000
python main.py 0.1 0 1 --ml_test_APS --folder nu_standard -sf -diff --paths 180000
python main.py 0.1 0 1 --ml_test_KD --folder nu_standard -sf --paths 180000
python main.py 0.032 0 1 --ml_test_APS --folder nu_standard -sf --paths 180000
python main.py 0.032 0 1 --ml_test_APS --folder nu_standard -sf -diff --paths 180000
python main.py 0.032 0 1 --ml_test_KD --folder nu_standard -sf --paths 180000
python main.py 0.32 10 0.1 --ml_test_APS --folder nu_standard -sf --paths 180000
python main.py 0.32 10 0.1 --ml_test_APS --folder nu_standard -sf -diff --paths 180000
python main.py 0.32 10 0.1 --ml_test_KD --folder nu_standard -sf --paths 180000
python main.py 0.1 10 0.1 --ml_test_APS --folder nu_standard -sf --paths 180000
python main.py 0.1 10 0.1 --ml_test_APS --folder nu_standard -sf -diff --paths 180000
python main.py 0.1 10 0.1 --ml_test_KD --folder nu_standard -sf --paths 180000
python main.py 0.032 10 0.1 --ml_test_APS --folder nu_standard -sf --paths 180000
python main.py 0.032 10 0.1 --ml_test_APS --folder nu_standard -sf -diff --paths 180000
python main.py 0.032 10 0.1 --ml_test_KD --folder nu_standard -sf --paths 180000
shutdown /s
