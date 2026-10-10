"""Several hosts sharing one CXL memory device that can park reads (gem5 SE mode).

Each host is its own gem5 System (own cores, caches, local DRAM, coherence domain). Hosts reach
the shared device through a link (a Bridge with the one-way flight delay and a crossbar with the
per-host link bandwidth) and a shared crossbar in front of the device's port, so hosts contend
there. The device (ParkedMemory) is not coherent with any host: software uses clflushopt for
write-back data and uncached loads/stores for the sync windows, which the config maps into every
process at fixed virtual addresses (see bench/sync_bench.c).

Usage:
  gem5.opt configs/multihost.py --hosts "oneway consumer parked 40 64" "oneway producer parked 40 64"
Each --hosts entry is one host running one process with those arguments; prefix the first word
with N* to give a host N cores each running the same arguments with the core index added to the
second argument (for waiters and posters), and with T: to use timing (in-order, one access at a
time) cores instead of out-of-order cores on that host.
"""
import argparse
import os

import m5
from m5.objects import *

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.environ.get("BENCH", os.path.join(HERE, "..", "bench", "sync_bench"))

DEV_BASE = 0x100000000
DEV_SIZE = 1 << 30
DATA_SIZE = 160 << 20   # payloads 0-32 MiB, tenant buffers 64-128 MiB, flush flags from 128 MiB
SYNC_OFF = 512 << 20
SYNC_SIZE = 32 << 20     # one sync line per 4 KiB page
VA_DATA, VA_SYNC0, VA_SYNC1 = 0x100000000000, 0x110000000000, 0x120000000000
VA_SYNC0_WB = 0x130000000000   # window 0 again, write-back cacheable (protocol wbpark)

ap = argparse.ArgumentParser()
ap.add_argument("--hosts", nargs="+", required=True)
ap.add_argument("--cpu", default="o3", choices=["o3", "timing"])
ap.add_argument("--park", type=int, default=1)
ap.add_argument("--table", type=int, default=4096)
ap.add_argument("--timeout", default="25us")
ap.add_argument("--link-delay", default="200ns", help="one-way host-device flight time")
ap.add_argument("--dev-latency", default="100ns")
ap.add_argument("--link-bw-bytes", type=int, default=32, help="bytes per ns per host link")
ap.add_argument("--dev-bw", default="64GiB/s")
ap.add_argument("--max-ticks", type=int, default=0)
ap.add_argument("--pf-degree", type=int, default=16, help="L2 stride prefetcher degree")
args = ap.parse_args()


def cpu_class(kind):
    return X86O3CPU if kind == "o3" else X86TimingSimpleCPU


class L1(Cache):
    size = "32KiB"
    assoc = 8
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    mshrs = 16
    tgts_per_mshr = 16


class L2(Cache):
    size = "1MiB"
    assoc = 16
    tag_latency = 12
    data_latency = 12
    response_latency = 12
    mshrs = 128
    tgts_per_mshr = 16


root = Root(full_system=False)
root.shared = NoncoherentXBar(
    width=64, clk_domain=SrcClockDomain(clock="1GHz", voltage_domain=VoltageDomain()), frontend_latency=1, forward_latency=1,
    response_latency=1, header_latency=1,
)
root.dev = ParkedMemory(
    clk_domain=SrcClockDomain(clock="1GHz", voltage_domain=VoltageDomain()),
    range=AddrRange(DEV_BASE, size=DEV_SIZE),
    latency=args.dev_latency,
    bandwidth=args.dev_bw,
    sync_base=DEV_BASE + SYNC_OFF,
    sync_size=f"{SYNC_SIZE}B",
    park_enable=bool(args.park),
    park_timeout=args.timeout,
    table_entries=args.table,
    collect_stats=False,
)
root.dev.port = root.shared.mem_side_ports

procs = []
hosts = []
pid = 100
for h, spec in enumerate(args.hosts):
    kind = args.cpu
    if spec.startswith("T:"):
        kind, spec = "timing", spec[2:]
    ncores, cmd = 1, spec
    if "*" in spec.split()[0]:
        n, first = spec.split()[0].split("*")
        ncores, cmd = int(n), " ".join([first] + spec.split()[1:])
    s = System()
    s.clk_domain = SrcClockDomain(clock="3GHz", voltage_domain=VoltageDomain())
    s.mem_mode = "timing"
    s.mem_ranges = [AddrRange("128MiB")]
    s.membus = SystemXBar()
    s.l2bus = L2XBar()
    s.l2 = L2()
    s.l2.prefetcher = StridePrefetcher(degree=args.pf_degree)
    s.l2.cpu_side = s.l2bus.mem_side_ports
    s.l2.mem_side = s.membus.cpu_side_ports
    s.cpu = [cpu_class(kind)(cpu_id=j) for j in range(ncores)]
    s.workload = SEWorkload.init_compatible(BENCH)
    for j, c in enumerate(s.cpu):
        c.icache = L1()
        c.dcache = L1()
        c.icache_port = c.icache.cpu_side
        c.dcache_port = c.dcache.cpu_side
        c.icache.mem_side = s.l2bus.cpu_side_ports
        c.dcache.mem_side = s.l2bus.cpu_side_ports
        c.createInterruptController()
        c.interrupts[0].pio = s.membus.mem_side_ports
        c.interrupts[0].int_requestor = s.membus.cpu_side_ports
        c.interrupts[0].int_responder = s.membus.mem_side_ports
        words = cmd.split()
        if ncores > 1:
            words = [words[0], str(int(words[1]) + j)] + words[2:]
        p = Process(pid=pid, cmd=[BENCH] + words)
        pid += 1
        c.workload = p
        c.createThreads()
        procs.append(p)
    s.system_port = s.membus.cpu_side_ports
    s.local = SimpleMemory(range=s.mem_ranges[0], latency="80ns", bandwidth="64GiB/s")
    s.local.port = s.membus.mem_side_ports
    s.link = Bridge(delay=args.link_delay, ranges=[AddrRange(DEV_BASE, size=DEV_SIZE)],
                    req_size=512, resp_size=512)
    s.link.cpu_side_port = s.membus.mem_side_ports
    s.linkbus = NoncoherentXBar(
        width=args.link_bw_bytes, clk_domain=SrcClockDomain(clock="1GHz", voltage_domain=VoltageDomain()), frontend_latency=1,
        forward_latency=0, response_latency=1, header_latency=1,
    )
    s.link.mem_side_port = s.linkbus.cpu_side_ports
    s.linkbus.mem_side_ports = root.shared.cpu_side_ports
    # the shared device needs one owning System for its backing store and stat registration;
    # per-requestor stats are off because requestor ids are only unique within a System
    s.memories = [s.local, root.dev] if h == 0 else [s.local]
    setattr(root, f"host{h}", s)
    hosts.append(s)

m5.instantiate()
for p in procs:
    p.map(VA_DATA, DEV_BASE, DATA_SIZE, True)
    p.map(VA_SYNC0, DEV_BASE + SYNC_OFF, SYNC_SIZE, False)
    p.map(VA_SYNC1, DEV_BASE + SYNC_OFF + SYNC_SIZE, SYNC_SIZE, False)
    p.map(VA_SYNC0_WB, DEV_BASE + SYNC_OFF, SYNC_SIZE, True)

remaining = len(hosts)
while remaining:
    ev = m5.simulate(args.max_ticks) if args.max_ticks else m5.simulate()
    cause = ev.getCause()
    if "exiting with last active thread context" in cause:
        remaining -= 1
        continue
    if "simulate() limit reached" in cause and not args.max_ticks:
        break                       # every process has exited
    print(f"EXIT {cause} at {m5.curTick()}")
    break
print(f"SIM_END tick={m5.curTick()}")
