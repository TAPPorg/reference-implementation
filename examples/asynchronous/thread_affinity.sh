#!/usr/bin/env bash
#
# Shows which CPUs the threads of a running process may run on (their affinity), which CPU each one ran on last and
# how busy it was since the previous sample. Waits for the process to start, then samples it until it exits.
#
# usage: thread_affinity.sh [-i seconds] [program name or pid]
#
# The default program is tapp-reference-async: start this script, then the example in another terminal.

set -euo pipefail

usage() { echo "usage: $0 [-i seconds] [program name or pid]" >&2; exit 2; }

interval=1
while getopts "i:h" opt; do
    case $opt in
        i) interval=$OPTARG ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))
(( $# <= 1 )) || usage
target=${1:-tapp-reference-async}

if [[ $target =~ ^[0-9]+$ ]]; then
    pid=$target
else
    # Matched on the command line: the process name the kernel keeps is cut to 15 characters
    echo "Waiting for $target to start..." >&2
    until pid=$(pgrep -n -f "^([^ ]*/)?${target}( |$)"); do sleep 0.1; done
fi

hz=$(getconf CLK_TCK)
declare -A prev_ticks
prev_time=$EPOCHREALTIME

while [[ -d /proc/$pid ]]; do
    now=$EPOCHREALTIME
    elapsed=$(awk -v a="$now" -v b="$prev_time" 'BEGIN { print a - b }')
    rows=""

    for task in /proc/"$pid"/task/*; do
        tid=${task##*/}
        # A thread can exit while it is read: skip it
        stat=$(cat "$task/stat" 2>/dev/null) || continue
        name=$(cat "$task/comm" 2>/dev/null) || continue
        allowed=$(awk '/^Cpus_allowed_list/ { print $2 }' "$task/status" 2>/dev/null) || continue

        # The fields after the name, which is in parentheses and may contain spaces. The first one is field 3 of stat.
        read -ra field <<< "${stat##*) }"
        ticks=$(( field[11] + field[12] ))   # utime + stime
        cpu=${field[36]}                     # processor: the CPU the thread ran on last

        if [[ -n ${prev_ticks[$tid]:-} ]]; then
            busy=$(awk -v d=$(( ticks - prev_ticks[$tid] )) -v hz="$hz" -v t="$elapsed" \
                   'BEGIN { printf "%.0f", 100 * d / hz / t }')
        else
            busy="-"
        fi
        prev_ticks[$tid]=$ticks

        rows+=$(printf '%-8s %-16s %4s %6s  %s' "$tid" "$name" "$cpu" "$busy" "$allowed")$'\n'
    done
    prev_time=$now

    printf '\n%s  pid %s\n' "$(date +%T)" "$pid"
    printf '%-8s %-16s %4s %6s  %s\n' TID NAME CPU %CPU ALLOWED
    # Grouped by the allowed CPUs, so the threads of each executor end up together
    printf '%s' "$rows" | sort -k5,5 -k1,1n

    sleep "$interval"
done

echo "Process $pid has exited" >&2
