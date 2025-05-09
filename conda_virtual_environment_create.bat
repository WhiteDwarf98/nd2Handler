@echo off

:: ===============================
:: Generalized Conda Environment Setup Script
:: ===============================

:: Set environment name here
set ENV_NAME=nd2venv

echo ===============================
echo %ENV_NAME% environment installation
echo ===============================

:: Check if the environment exists
conda env list | findstr "%ENV_NAME%" >nul
if %ERRORLEVEL% EQU 0 (
    echo The environment "%ENV_NAME%" already exists.
    echo The following will reinstall the environment. If you would like to abort the installation, please close the window.
    pause
    echo =======================
    echo Removing the existing environment...
    call conda env remove -n %ENV_NAME% -y
)

:: Clean the cache and recreate the environment
echo =======================
echo Cleaning cache...
call conda clean --all -y
echo =======================
echo Installing environment "%ENV_NAME%" from environment.yml...
call conda env create -n %ENV_NAME% -f environment.yml
pause
