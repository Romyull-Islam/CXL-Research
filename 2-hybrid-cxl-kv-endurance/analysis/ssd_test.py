"""Real-SSD test of how KV dies (standard library only; the run phase needs root).

  python3 analysis/ssd_test.py check            find the test drive, print it, write nothing
  sudo python3 analysis/ssd_test.py run sdX     run every phase on /dev/sdX (resumes after a stop)

Only a drive that matches every check is used: USB, vendor 0781 product 55ae (SanDisk Extreme),
exactly 1,000,171,323,904 bytes over USB, nothing mounted. Phases, each on a freshly trimmed and filled drive:
  discard   time BLKDISCARD of freshly written 1 MiB to 4 GiB ranges, idle and under writes
  seq       sequential 8 MiB overwrites (write amplification about 1), one drive capacity
  rand      uniform random 8 MiB overwrites, two capacities
  lru_chat, scoped_trim, scoped_notrim   the KV streams of ssd_streams.py, three capacities each
Every 10 s the host write rate is logged; at garbage-collection-bound steady state the bandwidth
falls roughly as 1 / write amplification, so ratios to `seq` estimate it, and `rand` checks that
estimate against greedy garbage collection under uniform writes.
"""
import fcntl
import json
import mmap
import os
import random
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OPS = ROOT / "data" / "ssd"
OUT = ROOT / "results" / "ssd"
SIZE = 1_000_171_323_904          # as the USB bridge reports it (the NVMe drive inside: 1,000,204,886,016)
VENDOR, PRODUCT = "0781", "55ae"
ALIGN = 256 * 1024
BLKDISCARD = 0x1277
PASSES = {"seq": 1, "rand": 2, "lru_chat": 1.5, "scoped_trim": 1.5, "scoped_notrim": 1.5}
# rand was stopped at 1.03 TB on 2026-10-13, in steady state (median 44 MB/s over its last third)


def props(dev):
    out = subprocess.run(["udevadm", "info", "--query=property", f"--name=/dev/{dev}"],
                         capture_output=True, text=True).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


def mounted(dev):
    return [l.split()[1] for l in open("/proc/mounts") if l.split()[0].startswith(f"/dev/{dev}")]


def find():
    hits = []
    for d in sorted(os.listdir("/sys/block")):
        if not d.startswith("sd"):
            continue
        p = props(d)
        size = int(open(f"/sys/block/{d}/size").read()) * 512
        ok = (p.get("ID_BUS") == "usb" and p.get("ID_VENDOR_ID") == VENDOR and p.get("ID_MODEL_ID") == PRODUCT
              and size == SIZE)
        print(f"/dev/{d}: bus={p.get('ID_BUS')} id={p.get('ID_VENDOR_ID')}:{p.get('ID_MODEL_ID')} "
              f"model={p.get('ID_MODEL')} serial={p.get('ID_SERIAL_SHORT')} size={size} "
              f"mounted={mounted(d)} -> {'TEST DRIVE' if ok else 'not used'}")
        if ok:
            hits.append(d)
    return hits


class Log:
    def __init__(self, name, dev):
        self.f = open(OUT / f"{name}.csv", "a")
        self.dev, self.t0, self.last, self.bytes, self.lastb = dev, time.time(), time.time(), 0, 0
        self.f.write("t_s,host_bytes,interval_MBps,dev_sectors_written\n")

    def add(self, n):
        self.bytes += n
        now = time.time()
        if now - self.last >= 10:
            sect = int(open(f"/sys/block/{self.dev}/stat").read().split()[6])
            self.f.write(f"{now - self.t0:.1f},{self.bytes},{(self.bytes - self.lastb) / (now - self.last) / 1e6:.1f},"
                         f"{sect}\n")
            self.f.flush()
            self.last, self.lastb = now, self.bytes


def discard(fd, off, n):
    fcntl.ioctl(fd, BLKDISCARD, struct.pack("QQ", off, n))


def wipe_and_fill(fd, dev, buf, name):
    """Trim the whole drive, then write it once sequentially (the steady starting state)."""
    t = time.time()
    discard(fd, 0, SIZE)
    log = Log(f"{name}_fill", dev)
    chunk = len(buf)
    for off in range(0, SIZE - SIZE % chunk, chunk):
        os.pwrite(fd, buf, off)
        log.add(chunk)
    print(f"{name}: trimmed and filled in {time.time() - t:.0f} s", flush=True)


def phase_discard(fd, dev, buf, loads=("idle", "writing"), out="discard.csv"):
    rows = []
    discard(fd, 0, SIZE)
    sizes = [1 << 20, 4 << 20, 16 << 20, 64 << 20, 256 << 20, 1 << 30, 4 << 30]
    busy = threading.Event()

    base = SIZE // 2 // ALIGN * ALIGN           # aligned: SIZE is not a multiple of 4 KiB
    span = SIZE // 4 // len(buf) * len(buf)

    def writer():
        k = 0
        while busy.is_set():
            os.pwrite(fd, buf, base + k)
            k = (k + len(buf)) % span
    for load in loads:
        th = None
        if load == "writing":
            busy.set()
            th = threading.Thread(target=writer)
            th.start()
        rng = random.Random(1)
        for n in sizes:
            for rep in range(5):
                off = rng.randrange(0, SIZE // 4 - n) // ALIGN * ALIGN
                for o in range(off, off + n, min(n, len(buf))):
                    os.pwrite(fd, buf[:min(n, len(buf))], o)
                os.fsync(fd)
                t = time.perf_counter()
                discard(fd, off, n)
                rows.append(dict(load=load, bytes=n, rep=rep, seconds=time.perf_counter() - t))
            print(f"discard {load} {n >> 20} MiB: median "
                  f"{sorted(r['seconds'] for r in rows if r['bytes'] == n and r['load'] == load)[2] * 1e3:.1f} ms",
                  flush=True)
        if th:
            busy.clear()
            th.join()
    with open(OUT / out, "w") as f:
        f.write("load,bytes,rep,seconds\n")
        for r in rows:
            f.write(f"{r['load']},{r['bytes']},{r['rep']},{r['seconds']:.6f}\n")


def phase_stream(fd, dev, buf, name):
    wipe_and_fill(fd, dev, buf, name)
    target = PASSES[name] * SIZE
    log = Log(name, dev)
    if name in ("seq", "rand"):
        unit = 8 << 20
        units = SIZE // unit
        rng = random.Random(2)
        i = 0
        while log.bytes < target:
            u = i % units if name == "seq" else rng.randrange(units)
            os.pwrite(fd, buf[:unit], u * unit)
            log.add(unit)
            i += 1
    else:
        meta = json.loads((OPS / "streams.json").read_text())[name]
        unit = SIZE // meta["units"] // ALIGN * ALIGN
        ops = open(OPS / f"{name}.ops", "rb").read()
        n = len(ops) // 4
        i = 0
        while log.bytes < target:
            u = struct.unpack_from("<i", ops, (i % n) * 4)[0]
            if u >= 0:
                os.pwrite(fd, buf[:unit], u * unit)
                log.add(unit)
            else:
                discard(fd, (-u - 1) * unit, unit)
            i += 1
    os.fsync(fd)
    print(f"{name}: {log.bytes / 1e12:.2f} TB in {time.time() - log.t0:.0f} s", flush=True)


def run(dev):
    hits = find()
    if hits != [dev]:
        sys.exit(f"refusing: the only matching drive must be {dev}, found {hits}")
    if mounted(dev):
        sys.exit(f"refusing: /dev/{dev} has mounted partitions {mounted(dev)}; unmount them first")
    if os.geteuid() != 0:
        sys.exit("run needs root")
    OUT.mkdir(parents=True, exist_ok=True)
    for p in Path(f"/sys/block/{dev}/device/scsi_disk").glob("*/provisioning_mode"):
        p.write_text("unmap")
    if int(open(f"/sys/block/{dev}/queue/discard_max_bytes").read()) == 0:
        sys.exit("refusing: TRIM (UNMAP) could not be enabled on this drive")
    raw = mmap.mmap(-1, 24 << 20)
    raw.write(os.urandom(24 << 20))
    buf = memoryview(raw)
    try:                                   # O_EXCL on a block device: one test at a time
        fd = os.open(f"/dev/{dev}", os.O_RDWR | os.O_DIRECT | os.O_EXCL)
    except OSError as e:
        sys.exit(f"refusing: /dev/{dev} is already open exclusively, another test is running ({e})")
    done = OUT / "done.json"
    state = json.loads(done.read_text()) if done.exists() else []
    for name in ["discard"] + list(PASSES) + ["discard_load"]:
        if name in state:
            continue
        print(f"== {name} {time.strftime('%H:%M:%S')}", flush=True)
        if name == "discard":
            phase_discard(fd, dev, buf)
        elif name == "discard_load":            # the first runs' writer thread failed on alignment
            phase_discard(fd, dev, buf, loads=("writing",), out="discard_load.csv")
        else:
            phase_stream(fd, dev, buf, name)
        state.append(name)
        done.write_text(json.dumps(state))
    os.close(fd)
    print("all phases done", flush=True)


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "check":
        hits = find()
        print("test drive:", hits if hits else "none found")
    elif len(sys.argv) == 3 and sys.argv[1] == "run":
        try:
            run(sys.argv[2])
        finally:
            user = os.environ.get("SUDO_USER")
            if user and OUT.exists():
                subprocess.run(["chown", "-R", f"{user}:{user}", str(OUT)])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
