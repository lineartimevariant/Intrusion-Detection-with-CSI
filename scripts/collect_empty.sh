#!/bin/bash

echo "Recording empty room CSI for 4.5 hours..."

stty -F /dev/ttyUSB0 115200 raw -echo

timeout 16200 bash -c 'cat /dev/ttyUSB0' | tee ../data/raw/empty_room.csv

echo "Empty room collection finished."
