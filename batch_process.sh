#!/bin/bash


cd ~/PycharmProjects/EMS-Pred-ICDE-21



for t in 0 1 2 3
do
python Main.py -device=cuda:3 -model=STIAM_Net -date 0101 0630 0701 0731 -sdw 3 $t 1
python Main.py -device=cuda:3 -model=STIAM_Net -date 0201 0731 0801 0831 -sdw 3 $t 1
python Main.py -device=cuda:3 -model=STIAM_Net -date 0301 0831 0901 0930 -sdw 3 $t 1
python Main.py -device=cuda:3 -model=STIAM_Net -date 0401 0930 1001 1031 -sdw 3 $t 1
python Main.py -device=cuda:3 -model=STIAM_Net -date 0501 1031 1101 1130 -sdw 3 $t 1
python Main.py -device=cuda:3 -model=STIAM_Net -date 0601 1130 1201 1231 -sdw 3 $t 1
done


