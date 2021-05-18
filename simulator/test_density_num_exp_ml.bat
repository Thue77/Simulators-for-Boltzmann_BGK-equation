@echo off
REM script for running python script multiple times with arguments:
python main.py 0.5 0 10 -de --paths 600000 --folder Logfiles_new
python main.py 0.03 0 10 -de --paths 600000 --folder Logfiles_new
python main.py 0.3 1000 10 -de --paths 600000 --folder Logfiles_new
