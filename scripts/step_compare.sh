#!/bin/bash
# For each ProofAtlas-failing problem, record steps done by each prover in 10s.
# Output: problem pa_iters vampire_activations spass_givenclauses
ROOT=/home/apluska/proofatlas/.tptp/TPTP-v9.0.0
VAMP=/home/apluska/proofatlas/.vampire/vampire
SPASS=/home/apluska/proofatlas/.spass/SPASS

run_one() {
  local name="$1" pa="$2"
  local P
  P=$(ls "$ROOT"/Problems/*/"$name".p 2>/dev/null | head -1)
  [ -z "$P" ] && { echo "$name $pa NA NA"; return; }
  local va
  va=$(timeout 14 "$VAMP" --include "$ROOT" --statistics full -t 10s "$P" 2>/dev/null \
        | grep -m1 -i "Activations started:" | grep -oE "[0-9]+$")
  [ -z "$va" ] && va=NA
  local sp
  sp=$(TPTP="$ROOT" timeout 14 "$SPASS" -TPTP -TimeLimit=10 "$P" 2>/dev/null \
        | grep -c "Given clause:")
  echo "$name $pa $va $sp"
}
export -f run_one
export ROOT VAMP SPASS

awk '{print $1, $2}' /tmp/pa_fail.txt | \
  xargs -P 12 -n 2 bash -c 'run_one "$0" "$1"'
