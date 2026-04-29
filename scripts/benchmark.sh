#!/usr/bin/env bash

usage() {
  printf "Usage: scripts/benchmark.sh <output>.json [LANG1] [LANG2] ...\n"
  printf "Languages: cirq cudaq guppy pennylane pyquil qsharp qiskit qualtran qrisp silq\n"
  exit 1
}

if [ "$#" -lt 2 ]; then
  usage
fi

RUNS=1
JSON="${1?}"
shift
LANGS=("$@")
CASES=( tfim_trotter tfim_lcu heis_trotter heis_lcu shors21_2 )
NCASES=$((${#LANGS[@]} * ${#CASES[@]}))

echo "Executing $RUNS runs for $NCASES programs."

times_sum=()
# Initialize sums to all zeros
for ((i=0; i<NCASES; i++)); do
  times_sum[i]=0
done

for i in $(seq $RUNS); do
  mkdir -p benchmarks
  benchmark="benchmarks/_benchmark${i}.json"
  rm -f "$benchmark"

  for lang in "${LANGS[@]}"; do
    for case in "${CASES[@]}"; do
      echo "Run #${i}"
      python harness/run_tests.py --languages="$lang" --cases="$case" --json "$benchmark"
    done
  done

  readarray -t times_arr < <(jq '.[] | .time_mean' "$benchmark")

  times_sum=($(paste <(printf "%s\n" "${times_sum[@]}") <(printf "%s\n" "${times_arr[@]}") | awk '{print $1 + $2}'))
  rm -f "$benchmark"
done

times_mean=($(printf "%s\n" "${times_sum[@]}" | awk -v n="$RUNS" '{print $1 / n}'))


### Output ###

printf >"$JSON" "{"
i=0
for lang in "${LANGS[@]}"; do
  for case in "${CASES[@]}"; do
    printf >>"$JSON" "\"%s/%s\":{" "$lang" "$case"
    printf >>"$JSON" "\"language\":\"%s\"," "$lang"
    printf >>"$JSON" "\"case\":\"%s\"," "$case"
    printf >>"$JSON" "\"time_mean\":%s," "${times_mean[i]}"
    printf >>"$JSON" "\"success\":true}"
    if [ $((i+1)) -ne "$NCASES" ]; then
      printf >>"$JSON" ","
    fi

    i=$((i+1))
  done
done
printf >>"$JSON" "}"
