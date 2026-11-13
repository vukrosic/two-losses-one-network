#!/bin/bash
# Multi-token prediction runs (finding 23 in LAB-NOTES.md). Not run yet.
#   bash mtp_runs.sh 0   time two runs (smallest own, biggest shared); if either takes over ~30 s, lower STEPS or the widths
#   bash mtp_runs.sh 1   core: own model per offset + shared K=4 at four head weights, widths 32/64/128   (24 runs)
#   bash mtp_runs.sh 2   cheap prediction: stage 1 at 1/8 of the steps                                   (24 short runs)
#   bash mtp_runs.sh 3   K=2, and Gloeckle-style heads with their own transformer block                   (18 runs)
#   bash mtp_runs.sh 4   easier (ALPHA=0.1) and harder (ALPHA=2) chains at width 64                        (14 runs)
# Then: python3 mtpfit.py
cd "$(dirname "$0")"
R="nice -n 20 python3 mtp.py"
S=${STEPS:-1500}; Q=$((S / 8))
case "$1" in
0) time $R own 1 32 $S; time W=1 $R shared 4 128 $S ;;
1|2) [ "$1" = 2 ] && S=$Q
   for d in 32 64 128; do
     for k in 1 2 3 4; do $R own $k $d $S; done
     for w in 0.1 0.3 1 3; do W=$w $R shared 4 $d $S; done
   done ;;
3) for d in 32 64 128; do
     W=1 $R shared 2 $d $S
     for k in 1 2 3 4; do HEADBLOCK=1 $R own $k $d $S; done
     HEADBLOCK=1 W=1 $R shared 4 $d $S
   done ;;
4) for a in 0.1 2; do
     for k in 1 2 3 4; do ALPHA=$a $R own $k 64 $S; done
     for w in 0.3 1 3; do ALPHA=$a W=$w $R shared 4 64 $S; done
   done ;;
*) echo "usage: bash mtp_runs.sh 0|1|2|3|4" ;;
esac
