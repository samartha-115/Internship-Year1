@echo off
title 3D Intelligent Radar-Target Scheduling Simulation [Point 2 & 3]
echo ===============================================================================
echo     3D INTELLIGENT RADAR-TARGET SCHEDULING SIMULATION
echo     Point 2: 3D Airspace, Radar Detection Depth & Geometric Redundancy
echo     Point 3: Carrier Aircraft Maneuvers & Enemy Cluster Target Swarms
echo ===============================================================================
echo Launching simulation...
python main.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Simulation exited with code %ERRORLEVEL%.
    pause
)
