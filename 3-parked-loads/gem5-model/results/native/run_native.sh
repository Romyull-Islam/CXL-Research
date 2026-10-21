#!/bin/bash
# Native calibration of the software protocols: hosts on cores A and B, shared memory on node M.
A=${A:-216}; B=${B:-220}; M=${M:-1}; REPS=${REPS:-5}
run2() {  # role1-args role2-args
    rm -f /dev/shm/sbn
    SB_SHM=/sbn SB_N=2 timeout 60 numactl --membind=$M taskset -c $A ./sync_bench_native $1 &
    SB_SHM=/sbn SB_N=2 timeout 60 numactl --membind=$M taskset -c $B ./sync_bench_native $2
    wait; rm -f /dev/shm/sbn
}
echo "# cores $A $B memnode $M $(date +%F_%T)"
for r in $(seq $REPS); do
    rm -f /dev/shm/sbn; SB_SHM=/sbn SB_N=1 timeout 60 numactl --membind=$M taskset -c $A ./sync_bench_native rtt - - 20000; rm -f /dev/shm/sbn
    for p in flush lines; do
        for s in 64 256 1024 4096; do run2 "oneway consumer $p 2000 $s" "oneway producer $p 2000 $s"; done
        for sv in 0 2000 10000; do run2 "rpc client $p 2000 $sv" "rpc server $p 2000 $sv"; done
    done
done
echo "# done $(date +%T)"
