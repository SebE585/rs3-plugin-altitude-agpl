#!/usr/bin/env bash

set -euo pipefail

# Force numeric locale to POSIX to avoid comma decimal separators with fr_FR locales
export LC_ALL=C
export LANG=C
export LC_NUMERIC=C

# Defaults
URL="http://localhost:5004/sample"
LAT="49.45"
LON="1.1"
PROVIDER="rge1"
N=200            # total requests (hors warmup)
C=8              # concurrency (xargs -P)
WARMUP=10        # warmup requests (non comptées)
TIMEOUT=10       # curl timeout seconds

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -u URL          Base URL (default: $URL)
  -a LAT          Latitude (default: $LAT)
  -o LON          Longitude (default: $LON)
  -p PROVIDER     Provider (rge1|rge5|srtm30|srtm90) (default: $PROVIDER)
  -n N            Number of measured requests (default: $N)
  -c C            Concurrency (default: $C)
  -w WARMUP       Warmup requests (default: $WARMUP)
  -t TIMEOUT      Per request timeout seconds (default: $TIMEOUT)
  -h              Help

Examples:
  $(basename "$0") -p rge1 -n 500 -c 16
  $(basename "$0") -u http://localhost:5004/sample -a 49.4431 -o 1.0993 -p rge5
EOF
}

while getopts ":u:a:o:p:n:c:w:t:h" opt; do
  case $opt in
    u) URL="$OPTARG" ;;
    a) LAT="$OPTARG" ;;
    o) LON="$OPTARG" ;;
    p) PROVIDER="$OPTARG" ;;
    n) N="$OPTARG" ;;
    c) C="$OPTARG" ;;
    w) WARMUP="$OPTARG" ;;
    t) TIMEOUT="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 1 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 1 ;;
  esac
done

QUERY="$URL?lat=${LAT}&lon=${LON}&provider=${PROVIDER}"

echo "== bench_curl =="
echo "URL        : $QUERY"
echo "Requests   : $N  (warmup: $WARMUP)"
echo "Concurrency: $C"
echo

# Warmup (non mesuré)
if [ "$WARMUP" -gt 0 ]; then
  echo "[warmup] $WARMUP requests..."
  seq "$WARMUP" | xargs -I{} -n1 -P"$C" \
    sh -c "curl -m $TIMEOUT -s -o /dev/null '$QUERY'" >/dev/null
fi

# Mesure
# -w %{time_total} renvoie un nombre en secondes (float) avec un point décimal.
# On force LC_ALL=C pour que awk/sort lisent les décimales correctement.
TMP_TIMES="$(mktemp)"
TMP_CODES="$(mktemp)"
cleanup() { rm -f "$TMP_TIMES" "$TMP_CODES"; }
trap cleanup EXIT

echo "[run] $N requests..."
LC_ALL=C seq "$N" | xargs -I{} -n1 -P"$C" \
  sh -c "curl -m $TIMEOUT -s -w '%{http_code} %{time_total}\n' -o /dev/null '$QUERY'" \
| tee >(awk '{print $2}' > "$TMP_TIMES") \
      >(awk '{print $1}' > "$TMP_CODES") >/dev/null

# Stats codes HTTP
TOTAL=$N
SUCCESS=$(awk 'BEGIN{ok=0} {if ($1 ~ /^2[0-9][0-9]$/) ok++} END{print ok}' "$TMP_CODES")
FAIL=$((TOTAL - SUCCESS))
SUCCESS_RATE=$(awk -v s="$SUCCESS" -v t="$TOTAL" 'BEGIN{printf("%.2f", (t>0)?(100*s/t):0)}')

# Stats latence
read -r MIN MAX AVG <<<"$(LC_ALL=C awk '
  NR==1 {min=$1; max=$1; sum=0}
  { if($1<min) min=$1; if($1>max) max=$1; sum+=$1 }
  END { printf("%.6f %.6f %.6f", min, max, sum/NR) }
' "$TMP_TIMES")"

# Percentiles
percentile() {
  local p="$1"
  LC_ALL=C sort -n "$TMP_TIMES" | awk -v p="$p" '
    BEGIN { i=0 }
    { a[++i]=$1 }
    END {
      if (i==0) { print "nan"; exit }
      idx = int((p/100.0)*i); if (idx<1) idx=1; if (idx>i) idx=i;
      printf("%.6f", a[idx]);
    }'
}
P50=$(percentile 50)
P90=$(percentile 90)
P95=$(percentile 95)
P99=$(percentile 99)

# R/s approx = succès / temps total mur (approx via moyenne et concurrence)
# Mieux: mesurer le temps mur réel, mais on donne une estimation simple:
RPS_EST=$(awk -v ok="$SUCCESS" -v avg="$AVG" -v c="$C" \
  'BEGIN{ if (avg>0) printf("%.1f", (ok)/( (ok/c)*avg )); else print "inf"}')

echo
echo "== Results =="
printf "HTTP 2xx           : %d/%d (%.2f%%)\n" "$SUCCESS" "$TOTAL" "$SUCCESS_RATE"
printf "Latency (s)        : min=%.4f  p50=%.4f  p90=%.4f  p95=%.4f  p99=%.4f  max=%.4f  avg=%.4f\n" \
  "$MIN" "$P50" "$P90" "$P95" "$P99" "$MAX" "$AVG"
printf "Throughput (est.)  : %s req/s\n" "$RPS_EST"
echo
echo "Done."