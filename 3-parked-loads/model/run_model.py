"""Print the go/no-go tables for parked loads. Usage: python3 run_model.py"""

from handoff_model import PROTOCOLS, Link, parked, poll_traffic

# Idle load latency classes, from published device measurements (see README.md).
DEVICES = {"ASIC expander 250 ns": 250.0, "FPGA expander 400 ns": 400.0,
           "switched pool 600 ns": 600.0}
SIZES = [64, 1024, 4096, 16384, 65536]
DEVICE_BW = 60.0   # Samsung CMM-B, GB/s


def link(rtt):
    return Link(load_rtt=rtt, poll_overhead=50.0, host_bw=32.0, max_outstanding=48)


def best_software(n, k):
    name = min(("poll_flag", "poll_lines", "coherent_spin"), key=lambda p: PROTOCOLS[p](n, k))
    return name, PROTOCOLS[name](n, k)


def latency_table():
    print("One-way handoff latency (ns), and parked vs the best software protocol")
    print(f"{'device':22} {'bytes':>6} {'poll_flag':>10} {'poll_lines':>11} {'coh_spin':>9} "
          f"{'parked':>7}  best software      speedup")
    for dev, rtt in DEVICES.items():
        k = link(rtt)
        for size in SIZES:
            n = size // 64
            t = {p: f(n, k) for p, f in PROTOCOLS.items()}
            name, best = best_software(n, k)
            print(f"{dev:22} {size:>6} {t['poll_flag']:>10.0f} {t['poll_lines']:>11.0f} "
                  f"{t['coherent_spin']:>9.0f} {t['parked']:>7.0f}  {name:17} {best / t['parked']:.2f}x")
    print()


def traffic_table():
    print(f"Device bandwidth burned while waiting (device = {DEVICE_BW:.0f} GB/s)")
    print(f"{'device':22} {'flag poll GB/s':>15} {'waiters for 25%':>16} {'waiters for 50%':>16} "
          f"{'4 KB line-poll GB/s':>20}")
    for dev, rtt in DEVICES.items():
        flag, lines = poll_traffic(4096 // 64, link(rtt))
        print(f"{dev:22} {flag:>15.3f} {0.25 * DEVICE_BW / flag:>16.0f} {0.5 * DEVICE_BW / flag:>16.0f} "
              f"{lines:>20.2f}")
    print()


def end_to_end():
    print("End-to-end upper bound: step = compute + handoffs x one-way latency (switched pool, 600 ns)")
    print("Compute times are assumptions spanning plausible ranges, not measurements.")
    k = link(600.0)
    cases = [
        ("cross-host RPC, 256 B, service 0.5 us", 256, 2, 500.0),
        ("cross-host RPC, 256 B, service 2 us", 256, 2, 2_000.0),
        ("cross-host RPC, 256 B, service 10 us", 256, 2, 10_000.0),
        ("MoE dispatch+combine, 64 KB, layer 100 us", 65536, 2, 100_000.0),
        ("MoE dispatch+combine, 64 KB, layer 300 us", 65536, 2, 300_000.0),
        ("pipeline stage handoff, 16 KB, stage 200 us", 16384, 1, 200_000.0),
        ("pipeline stage handoff, 16 KB, stage 1 ms", 16384, 1, 1_000_000.0),
    ]
    for label, size, handoffs, compute in cases:
        n = size // 64
        _, best = best_software(n, k)
        base = compute + handoffs * best
        new = compute + handoffs * parked(n, k)
        print(f"  {label:46} handoff share {handoffs * best / base:6.1%}   speedup {base / new:.3f}x")
    print()


if __name__ == "__main__":
    latency_table()
    traffic_table()
    end_to_end()
