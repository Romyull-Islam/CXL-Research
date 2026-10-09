"""First-order model of one cross-host handoff over CXL shared memory.

A producer on one host writes an S-byte message into shared CXL memory; a consumer on another
host must observe all of it. Four protocols are compared:

  poll_flag      payload, fence, flag write; consumer flushes and re-reads the flag, then reads
                 the payload (TraCT, cMPI style)
  poll_lines     every 64 B line carries its own valid/sequence field; consumer re-reads all lines
                 until they are all valid (the fastest software scheme, at a bandwidth cost)
  coherent_spin  flag in a hardware-coherent (HDM-DB) region; consumer spins on its cached copy
                 and reloads after the back-invalidation
  parked         full/empty lines; consumer has parked reads outstanding on the message lines,
                 and the device answers them when the writes land

Times are in ns, sizes in bytes, bandwidths in GB/s (= bytes/ns). Latency inputs come from
published measurements; see README.md. This is a model for a go/no-go decision, not a simulator.
"""

from dataclasses import dataclass
from math import ceil

LINE = 64


@dataclass
class Link:
    load_rtt: float          # idle load round trip from a host core to the device and back
    poll_overhead: float     # extra per poll iteration: clflushopt + fence
    host_bw: float           # per-host link bandwidth, GB/s
    max_outstanding: int     # line requests one consumer can keep in flight


def fetch(n, k):
    """Time to read n lines with at most k.max_outstanding in flight."""
    return ceil(n / k.max_outstanding) * k.load_rtt + n * LINE / k.host_bw


def write_and_fence(n, k):
    return n * LINE / k.host_bw + k.load_rtt


def land(n, k):
    """Posted writes: time until the last of n lines reaches the device."""
    return k.load_rtt / 2 + n * LINE / k.host_bw


def poll_flag(n, k):
    flag_poll = k.load_rtt + k.poll_overhead
    detect = flag_poll / 2 + k.load_rtt / 2
    return write_and_fence(n, k) + k.load_rtt / 2 + detect + fetch(n, k)


def poll_lines(n, k):
    iteration = fetch(n, k) + k.poll_overhead
    return land(n, k) + iteration / 2 + fetch(n, k) - k.load_rtt / 2


def coherent_spin(n, k):
    invalidation_reaches_consumer = k.load_rtt
    reload_flag = k.load_rtt
    return write_and_fence(n, k) + invalidation_reaches_consumer + reload_flag + fetch(n, k)


def parked(n, k):
    first_wave = min(n, k.max_outstanding)
    rest = n - first_wave
    answered = land(n, k) + k.load_rtt / 2
    return answered + (fetch(rest, k) if rest else 0.0)


PROTOCOLS = {"poll_flag": poll_flag, "poll_lines": poll_lines,
             "coherent_spin": coherent_spin, "parked": parked}


def poll_traffic(n, k):
    """GB/s of device bandwidth one waiting consumer burns, for the two polling protocols."""
    flag = LINE / (k.load_rtt + k.poll_overhead)
    lines = n * LINE / (fetch(n, k) + k.poll_overhead)
    return flag, lines
