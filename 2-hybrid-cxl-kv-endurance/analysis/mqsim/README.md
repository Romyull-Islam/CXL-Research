# MQSim cross-check of analysis/ftl.py

MQSim (CMU-SAFARI, FAST'18) replays the KV write streams exported by analysis/export_streams.py.

    python analysis/export_streams.py                      # data/streams/*.npy and index.csv
    mkdir -p data/mqsim && cd data/mqsim
    git clone --depth 1 https://github.com/CMU-SAFARI/MQSim.git
    cd MQSim && git apply ../../../analysis/mqsim/greedy_fix.patch && make -j8 && cd ../../..
    python analysis/mqsim/driver.py setup && python analysis/mqsim/driver.py run && python analysis/mqsim/driver.py report

Configuration (ssdconfig.tmpl): 1 channel, chip, die and plane; 4 KB pages, 64 pages per block (one
KV block per page); greedy GC; over-provisioning set so the logical space equals the stream's
units; ideal mapping table, no data cache, no preconditioning. Steady-state WAF is the difference
between a warm-up run (2 x physical capacity) and the full run.

greedy_fix.patch: stock MQSim can pick a GC victim whose relocation programs are still in flight
(crash in Set_barrier_for_accessing_physical_block on the LRU oracle stream at 7% OP); the patch
counts GC programs in Ongoing_user_program_count. Runs that finish on the stock binary are
bit-identical with the patch.

Results: results/mqsim_crosscheck.csv. MQSim keeps about 10 free blocks per plane
(max_ongoing_gc_reqs_per_plane), so analysis/test_ftl.py compares with ftl.py at reserve=10.
