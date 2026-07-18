#!/bin/bash
LOG=/home/ma/ma_ma/ma_mpandya/Thesis/logs/rclone_sync_$(date +%Y%m%d_%H%M%S).log
rclone sync /pfs/data6/home/ma/ma_ma/ma_mpandya/Thesis gdrive:Thesis-Backup \
  --exclude "thesis/**" \
  --exclude "thesis_broken_*/**" \
  --exclude ".git/**" \
  --exclude "__pycache__/**" \
  --exclude "*.pyc" \
  --transfers=8 --checkers=8 --log-file="$LOG"
