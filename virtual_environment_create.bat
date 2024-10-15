@echo off
REM Create a virtual environment
python -m venv venv

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Install the required packages
pip install -r requirements.txt

REM Optional: Inform the user that the process is complete
echo Virtual environment setup complete and packages installed.
pause