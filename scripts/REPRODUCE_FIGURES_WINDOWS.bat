@echo off
cd /d "%~dp0.."
python source\generate_corrected_main_figures.py --pred data\MAIN_4MODEL_PREDICTIONS.npz --out reproduced_figures
if errorlevel 1 goto :err
echo.
echo Figures written to reproduced_figures\
pause
exit /b 0
:err
echo Figure reproduction failed. Install requirements first: python -m pip install -r requirements.txt
pause
exit /b 1
