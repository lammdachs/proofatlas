#!/bin/bash
set -e

# Per-node IJCAR experiment runner.
# Collects traces, then launches the train+evaluate pipeline.
#
# Usage:
#   scripts/cluster_node.sh 1               # Node 1 configs
#   scripts/cluster_node.sh 2               # Node 2 configs
#   scripts/cluster_node.sh 3               # Node 3 configs
#   scripts/cluster_node.sh --status        # Check status
#   scripts/cluster_node.sh --kill          # Stop everything
#   scripts/cluster_node.sh --foreground 1  # Run in foreground (no daemon)

WORKERS=60
GPU_WORKERS=4
BATCH_SIZE=16M
LOG=.data/cluster_node.log
PIDFILE=.data/cluster_node.pid

case "$1" in
    1) CONFIGS="gcn_mlp gcn_attention gcn_transformer gcn_struct_mlp gcn_struct_attention" ;;
    2) CONFIGS="gcn_struct_transformer gcn_symbol_mlp gcn_symbol_attention gcn_symbol_transformer features_mlp" ;;
    3) CONFIGS="features_attention features_transformer sentence_mlp sentence_attention sentence_transformer" ;;
    --status)
        if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "Trace collection running (PID: $(cat "$PIDFILE"))"
            echo "  Log: tail -f $LOG"
            tail -5 "$LOG" 2>/dev/null | sed 's/^/  /'
        else
            rm -f "$PIDFILE"
            proofatlas-pipeline --status
        fi
        exit 0
        ;;
    --kill)
        if [ -f "$PIDFILE" ]; then
            kill "$(cat "$PIDFILE")" 2>/dev/null && echo "Killed trace collection"
            rm -f "$PIDFILE"
        fi
        proofatlas-pipeline --kill
        exit 0
        ;;
    --foreground)
        shift
        case "$1" in
            1) CONFIGS="gcn_mlp gcn_attention gcn_transformer gcn_struct_mlp gcn_struct_attention" ;;
            2) CONFIGS="gcn_struct_transformer gcn_symbol_mlp gcn_symbol_attention gcn_symbol_transformer features_mlp" ;;
            3) CONFIGS="features_attention features_transformer sentence_mlp sentence_attention sentence_transformer" ;;
            *) echo "Usage: $0 --foreground <1|2|3>"; exit 1 ;;
        esac
        echo "=== Phase 1: Traces ==="
        proofatlas-bench --config age_weight --trace --foreground --cpu-workers "$WORKERS"
        echo "=== Phase 2: Pipeline ==="
        proofatlas-pipeline --configs $CONFIGS \
            --use-cuda --gpu-workers "$GPU_WORKERS" --cpu-workers "$WORKERS" --batch-size "$BATCH_SIZE"
        exit 0
        ;;
    *)
        echo "Usage: $0 <1|2|3|--status|--kill|--foreground N>"
        exit 1
        ;;
esac

# Daemonize: re-exec in background with nohup
mkdir -p .data
nohup bash -c "
    set -e
    echo \"[\$(date +%H:%M:%S)] Phase 1: Collecting traces (workers=$WORKERS)\"
    proofatlas-bench --config age_weight --trace --foreground --cpu-workers $WORKERS

    echo \"[\$(date +%H:%M:%S)] Phase 2: Launching pipeline ($CONFIGS)\"
    proofatlas-pipeline --configs $CONFIGS \
        --use-cuda --gpu-workers $GPU_WORKERS --cpu-workers $WORKERS --batch-size $BATCH_SIZE

    echo \"[\$(date +%H:%M:%S)] Done. Pipeline daemonized.\"
    rm -f $PIDFILE
" > "$LOG" 2>&1 &

echo "$!" > "$PIDFILE"
echo "Started (PID: $!), configs: $CONFIGS"
echo "  Log:    tail -f $LOG"
echo "  Status: $0 --status"
echo "  Kill:   $0 --kill"
