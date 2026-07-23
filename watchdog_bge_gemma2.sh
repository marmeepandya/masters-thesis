#!/bin/bash
# Resubmits run8_short.sh if BGE-Gemma2 isn't running/queued and isn't finished yet.
# Needed because a hard SLURM TIMEOUT kill (e.g. model loading alone takes >30min on
# a slow read) terminates the job before its own "resubmit if unfinished" line at the
# end of run8_short.sh can run -- that self-chaining only works on a graceful exit.
cd /pfs/data6/home/ma/ma_ma/ma_mpandya/Thesis

if [ -f "result/15_baseline_bge_gemma2/company_embeddings.npy" ]; then
    exit 0
fi

RUNNING=$(squeue -u ma_mpandya -n 15_baseline_bge_gemma2_short -h | wc -l)
if [ "$RUNNING" -eq 0 ]; then
    echo "$(date): no BGE-Gemma2 job running/queued and not finished -- resubmitting" >> logs/watchdog_bge_gemma2.log
    sbatch run8_short.sh >> logs/watchdog_bge_gemma2.log 2>&1
fi
