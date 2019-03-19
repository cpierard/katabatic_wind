#!/bin/sh
echo "**STARTING**"
mpiexec -n 2 python katabatic_wind.py
mpiexec -n 2 python merge.py ../katabatic_rand_ic/ --cleanup
echo "**DONE**"
