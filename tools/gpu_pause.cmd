@echo off
rem Frees the GPU: running wind solves park their state in RAM within about 30 s (pipeline/wind_lbm.py pause_file).
type nul > "%~dp0..\PAUSE_GPU"
echo Wind runs pausing; the GPU is free in about 30 seconds. Run gpu_resume.cmd when done.
