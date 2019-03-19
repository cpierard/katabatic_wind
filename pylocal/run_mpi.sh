#!/bin/sh
echo "**STARTING**"
mpiexec -n 2 python katabatic_wind.py
mpiexec -n 2 python merge.py ../katabatic_wind_2/ --cleanup
echo "**DONE**"
