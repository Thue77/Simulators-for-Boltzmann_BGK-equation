@echo off
REM script for running python script multiple times with arguments:
set A=0 0.1 0.2 0.5 1 2 5 10
set B=1
(for %%b in (%B%) do (
  (for %%a in (%A%) do (
  python example_script.py 1 B1 %%a %%b "figure 5" 1000000
  )
)))
