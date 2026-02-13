#!/bin/bash
set -e

# Per-node IJCAR experiment runner for 2-node cluster with shared storage.
# Traces must already exist (collected via proofatlas-bench --config age_weight --trace).
#
# Usage:
#   scripts/cluster_node.sh 1               # Node 1: 9 configs (GPU)
#   scripts/cluster_node.sh 2               # Node 2: 6 configs (CPU)
#   scripts/cluster_node.sh --status [N]    # Check status (optionally for node N)
#   scripts/cluster_node.sh --kill [N]      # Stop jobs (optionally for node N)
#   scripts/cluster_node.sh --foreground 1  # Run in foreground (no daemon)

ulimit -n 65536

WORKERS=60
GPU_WORKERS=4
BATCH_SIZE=64K

# Node-specific files to avoid conflicts on shared storage
node_log() { echo ".data/cluster_node$1.log"; }
node_pid() { echo ".data/cluster_node$1.pid"; }
node_prefix() { echo "node$1"; }

case "$1" in
    1) NODE=1; CONFIGS="gcn_transformer gcn_struct_transformer gcn_symbol_transformer features_transformer sentence_mlp sentence_attention sentence_transformer"; GPU_ARGS="--use-cuda --gpu-workers $GPU_WORKERS" ;;
    2) NODE=2; CONFIGS="gcn_mlp gcn_attention gcn_struct_mlp gcn_struct_attention gcn_symbol_mlp gcn_symbol_attention features_mlp features_attention"; GPU_ARGS="" ;;
    --status)
        NODES="${2:-1 2}"
        for N in $NODES; do
            PIDFILE=$(node_pid $N)
            LOG=$(node_log $N)
            echo "=== Node $N ==="
            if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
                echo "Running (PID: $(cat "$PIDFILE"))"
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
                PID=$(cat "$PIDFILE")
                # Kill entire process group (negative PID)
                kill -TERM -"$PID" 2>/dev/null && echo "Killed process group $PID"
                # Also kill by PID in case process group kill failed
                kill -TERM "$PID" 2>/dev/null || true
                rm -f "$PIDFILE"
            fi
            proofatlas-pipeline --kill --job-prefix "$(node_prefix $N)" 2>/dev/null || true
        done
        exit 0
        ;;
    --foreground)
        shift
        case "$1" in
            1) NODE=1; CONFIGS="gcn_transformer gcn_struct_transformer gcn_symbol_transformer features_transformer sentence_mlp sentence_attention sentence_transformer"; GPU_ARGS="--use-cuda --gpu-workers $GPU_WORKERS" ;;
            2) NODE=2; CONFIGS="gcn_mlp gcn_attention gcn_struct_mlp gcn_struct_attention gcn_symbol_mlp gcn_symbol_attention features_mlp features_attention"; GPU_ARGS="" ;;
            *) echo "Usage: $0 --foreground <1|2>"; exit 1 ;;
        esac
        proofatlas-pipeline --configs $CONFIGS --job-prefix "$(node_prefix $NODE)" \
            $GPU_ARGS --cpu-workers "$WORKERS" --batch-size "$BATCH_SIZE"
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

# Daemonize: run in new process group for clean kill
mkdir -p .data
setsid bash -c "
    set -e
    # Trap signals to clean up child processes
    cleanup() {
        echo \"[\$(date +%H:%M:%S)] Received signal, cleaning up...\"
        # Kill all children in our process group
        kill -TERM 0 2>/dev/null || true
        rm -f $PIDFILE
        exit 0
    }
    trap cleanup TERM INT QUIT

    echo \"[\$(date +%H:%M:%S)] Launching pipeline ($CONFIGS)\"
    proofatlas-pipeline --configs $CONFIGS --job-prefix $PREFIX \
        $GPU_ARGS --cpu-workers $WORKERS --batch-size $BATCH_SIZE

    echo \"[\$(date +%H:%M:%S)] Done. Pipeline daemonized.\"
    rm -f $PIDFILE
" > "$LOG" 2>&1 &

echo "$!" > "$PIDFILE"
echo "Started node $NODE (PID: $!), configs: $CONFIGS"
echo "  Log:    tail -f $LOG"
echo "  Status: $0 --status $NODE"
echo "  Kill:   $0 --kill $NODE"
