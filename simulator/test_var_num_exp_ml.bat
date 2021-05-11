@echo off
REM script for running python script multiple times with arguments:
set epsilon=10,1,0.1,0.05
(for %%e in (%epsilon%) do (
  python example_script.py %%e A 0 10 "num_exp_ml" 1
))
