@echo off
REM script for running python script multiple times with arguments:
REM python main.py 0.5 0 10 --ml_test_APS --folder Logfiles_new -sf
REM python main.py 0.5 0 10 --ml_test_APS --folder Logfiles_new -sf -rev
REM python main.py 0.5 0 10 --ml_test_APS --folder Logfiles_new -sf -diff
REM python main.py 0.5 0 10 --ml_test_APS --folder Logfiles_new -sf -rev -diff
REM python main.py 0.03 0 10 --ml_test_APS --folder Logfiles_new -sf
REM python main.py 0.03 0 10 --ml_test_APS --folder Logfiles_new -sf -rev
REM python main.py 0.03 0 10 --ml_test_APS --folder Logfiles_new -sf -diff
REM python main.py 0.03 0 10 --ml_test_APS --folder Logfiles_new -sf -rev -diff
REM python main.py 0.3 1000 10 --ml_test_APS --folder Logfiles_new -sf
REM python main.py 0.3 1000 10 --ml_test_APS --folder Logfiles_new -sf -rev
python main.py 0.3 1000 10 --ml_test_APS --folder Logfiles_new --paths 180000 -sf -diff
python main.py 0.3 1000 10 --ml_test_APS --folder Logfiles_new --paths 180000 -sf -rev -diff
python main.py 0.5 0 10 --ml_test_KD --folder Logfiles_new -sf
python main.py 0.03 0 10 --ml_test_KD --folder Logfiles_new -sf
python main.py 0.3 1000 10 --ml_test_KD --folder Logfiles_new --paths 180000 -sf
python main.py 0.5 0 10 -de --paths 600000 --folder Logfiles_new -sf
python main.py 0.03 0 10 -de --paths 600000 --folder Logfiles_new -sf
python main.py 0.3 1000 10 -de --paths 600000 --folder Logfiles_new -sf
