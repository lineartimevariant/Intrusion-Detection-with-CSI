#!/bin/bash

echo "Recording intrusion CSI..."

stty -F /dev/ttyUSB0 115200 raw -echo

cat /dev/ttyUSB0 | tee ../data/raw/intrusion.csv


