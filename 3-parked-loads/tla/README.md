# Model checking the five rules

`ParkedLoads.tla` models a CXL device that parks reads of an empty synchronization line, consumer
hosts that wait on it, and a producer host that first reads an ordinary line and then writes the
sync line; the producer may think for an unbounded time and may fail. Request credits, the
separate RwD write channel, a bounded parked table, the device timeout and the Root Port
completion timeout are explicit. Time advances only when nothing urgent can happen; a retry after
a "not yet" answer costs at least one tick; credit acquisition is strongly fair (round-robin link
arbitration).

The rules are switches (R1 a parked read frees its credit; R2 writes need no request credit;
R3 every parked read is answered before the Root Port timeout; R4 that answer says "not yet";
R5 a read for ownership is never parked). WBWrite makes the producer write through a write-back
mapping, so its store first reads the line for ownership, as on real hardware (found in gem5).

Properties: NoRootPortTimeout, NoStaleRead, ProducerReadNotBlocked, ProducerWriteNotBlocked
(invariants) and EventuallyDone (liveness: if the producer does not fail, every consumer gets the
data).

Run (needs Java 11+; TLC 1.8.0 is downloaded to tools/, not in git):

    curl -L -o tools/tla2tools.jar https://github.com/tlaplus/tlaplus/releases/download/v1.8.0/tla2tools.jar
    python3 run_tlc.py        # writes results.csv

Result (results.csv; 2 and 3 consumers, 2 credits, 2-entry table, Timeout 2, Root Port timeout 4
ticks; 72 checks hold, 12 fail as expected): with all rules every property holds, including
liveness, with a write-through or a write-back writer. Removing a rule breaks one guarantee:
without R1 the producer's ordinary read waits for the timeout; without R1 and R2 its releasing
write does too; without R3 a read outlives the Root Port timeout; without R4 a consumer accepts
stale data; without R5 a write-back writer's release waits for the timeout. R2 alone is redundant
once R1 holds, because a parked read then holds no credit; CXL provides it anyway (with BI, RwD must pass Req).
