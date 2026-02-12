#!/bin/bash
set -e

# Per-node IJCAR experiment runner for 2-node cluster with shared storage.
# Node 1 collects traces, then both nodes run their pipeline configs.
#
# Usage:
#   scripts/cluster_node.sh 1               # Node 1: collect traces + 8 configs
#   scripts/cluster_node.sh 2               # Node 2: wait for traces + 7 configs
#   scripts/cluster_node.sh --status [N]    # Check status (optionally for node N)
#   scripts/cluster_node.sh --kill [N]      # Stop jobs (optionally for node N)
#   scripts/cluster_node.sh --foreground 1  # Run in foreground (no daemon)

WORKERS=60
GPU_WORKERS=4
BATCH_SIZE=16M

# Node-specific files to avoid conflicts on shared storage
node_log() { echo ".data/cluster_node$1.log"; }
node_pid() { echo ".data/cluster_node$1.pid"; }
node_prefix() { echo "node$1"; }

case "$1" in
    1) NODE=1; CONFIGS="gcn_mlp gcn_attention gcn_transformer gcn_struct_mlp gcn_struct_attention gcn_struct_transformer gcn_symbol_mlp gcn_symbol_attention" ;;
    2) NODE=2; CONFIGS="gcn_symbol_transformer features_mlp features_attention features_transformer sentence_mlp sentence_attention sentence_transformer" ;;
    --status)
        NODES="${2:-1 2}"
        for N in $NODES; do
            PIDFILE=$(node_pid $N)
            LOG=$(node_log $N)
            echo "=== Node $N ==="
            if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
                echo "Trace collection running (PID: $(cat "$PIDFILE"))"
                echo "  Log: tail -f $LOG"
                tail -3 "$LOG" 2>/dev/null | sed 's/^/  /'
            else
                rm -f "$PIDFILE"
                proofatlas-pipeline --status --job-prefix "$(node_prefix $N)" 2>/dev/null || echo "No active jobs"
            fi
            echo
        done
        exit 0
        ;;
    --kill)
        NODES="${2:-1 2}"
        for N in $NODES; do
            PIDFILE=$(node_pid $N)
            echo "=== Node $N ==="
            if [ -f "$PIDFILE" ]; then
                kill "$(cat "$PIDFILE")" 2>/dev/null && echo "Killed trace collection"
                rm -f "$PIDFILE"
            fi
            proofatlas-pipeline --kill --job-prefix "$(node_prefix $N)" 2>/dev/null || true
        done
        exit 0
        ;;
    --foreground)
        shift
        case "$1" in
            1) NODE=1; CONFIGS="gcn_mlp gcn_attention gcn_transformer gcn_struct_mlp gcn_struct_attention gcn_struct_transformer gcn_symbol_mlp gcn_symbol_attention" ;;
            2) NODE=2; CONFIGS="gcn_symbol_transformer features_mlp features_attention features_transformer sentence_mlp sentence_attention sentence_transformer" ;;
            *) echo "Usage: $0 --foreground <1|2>"; exit 1 ;;
        esac
        if [ "$NODE" = "1" ]; then
            echo "=== Phase 1: Traces ==="
            proofatlas-bench --config age_weight --trace --foreground --cpu-workers "$WORKERS"
        else
            echo "=== Phase 1: Waiting for traces ==="
            while [ ! -d .data/traces ] || [ -z "$(ls -A .data/traces 2>/dev/null)" ]; do
                echo "Waiting for node 1 to collect traces..."
                sleep 30
            done
            echo "Traces found, proceeding"
        fi
        echo "=== Phase 2: Pipeline ==="
        proofatlas-pipeline --configs $CONFIGS --job-prefix "$(node_prefix $NODE)" \
            --use-cuda --gpu-workers "$GPU_WORKERS" --cpu-workers "$WORKERS" --batch-size "$BATCH_SIZE"
        exit 0
        ;;
    *)
        echo "Usage: $0 <1|2|--status|--kill|--foreground N>"
        exit 1
        ;;
esac

LOG=$(node_log $NODE)
PIDFILE=$(node_pid $NODE)
PREFIX=$(node_prefix $NODE)

# Daemonize: re-exec in background with nohup
mkdir -p .data
nohup bash -c "
    set -e
    if [ '$NODE' = '1' ]; then
        echo \"[\$(date +%H:%M:%S)] Phase 1: Collecting traces (workers=$WORKERS)\"
        proofatlas-bench --config age_weight --trace --foreground --cpu-workers $WORKERS
    else
        echo \"[\$(date +%H:%M:%S)] Phase 1: Waiting for traces from node 1\"
        while [ ! -d .data/traces ] || [ -z \"\$(ls -A .data/traces 2>/dev/null)\" ]; do
            echo \"[\$(date +%H:%M:%S)] Waiting...\"
            sleep 30
        done
        echo \"[\$(date +%H:%M:%S)] Traces found\"
    fi

    echo \"[\$(date +%H:%M:%S)] Phase 2: Launching pipeline ($CONFIGS)\"
    proofatlas-pipeline --configs $CONFIGS --job-prefix $PREFIX \
        --use-cuda --gpu-workers $GPU_WORKERS --cpu-workers $WORKERS --batch-size $BATCH_SIZE

    echo \"[\$(date +%H:%M:%S)] Done. Pipeline daemonized.\"
    rm -f $PIDFILE
" > "$LOG" 2>&1 &

echo "$!" > "$PIDFILE"
echo "Started node $NODE (PID: $!), configs: $CONFIGS"
echo "  Log:    tail -f $LOG"
echo "  Status: $0 --status $NODE"
echo "  Kill:   $0 --kill $NODE"
