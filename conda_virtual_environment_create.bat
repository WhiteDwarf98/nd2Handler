@echo off
REM Initialize Conda
CALL "%ProgramData%\Anaconda3\Scripts\activate.bat"

REM Import the environment
echo Importing environment from requirements.yml ...
conda env create --file requirements.yml

REM Inform the user
echo The environment has been successfully imported.
pause
