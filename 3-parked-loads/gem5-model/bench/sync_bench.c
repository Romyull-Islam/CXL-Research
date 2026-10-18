/*
 * Cross-host synchronization benchmark for gem5 SE mode (one process per simulated core).
 *
 * The shared CXL device is mapped by the gem5 config at fixed virtual addresses:
 *   DATA  write-back cacheable, for payloads (software keeps it coherent with clflushopt)
 *   SYNC0 uncacheable, sync window 0 (a read waits while the line is empty if parking is on;
 *         a write sets the line full)
 *   SYNC1 uncacheable, sync window 1 (consuming read; a write sets the line empty)
 *
 * Wait protocols for a flag:
 *   flush   flag in DATA: clflushopt + mfence + load, repeated (today's software practice)
 *   uc      flag in SYNC0 with parking off: uncached load, repeated
 *   parked  flag in SYNC0/SYNC1 with parking on: one load that returns when the flag is written
 *           (or after the device timeout, then re-issued)
 *   wbpark  parking on ordinary cache-line fills: the doorbell line lives in SYNC0 mapped
 *           write-back and carries the sequence number, send time and up to 48 B of payload
 *           (longer payloads go to write-back memory first). The consumer flushes its copy and
 *           loads the line; the device parks the fill until the line is written and the wake-up
 *           returns the whole line. Reads are non-consuming (a hardware prefetch must not consume
 *           a message); the consumer empties the line afterwards with an uncached write to SYNC1,
 *           off the critical path. Every sync line sits on its own 4 KiB page, which hardware
 *           prefetchers do not cross.
 *   lines   no flag: every payload line carries the sequence number and the consumer re-reads
 *           (clflushopt + mfence + loads) until all lines are new; the fastest software scheme
 *           on x86 (oneway and rpc only)
 *
 * Usage: sync_bench TEST ROLE PROTO ITERS [ARG]
 *   oneway  producer|consumer  PROTO ITERS SIZE   one-way handoff of SIZE bytes, consumer waiting
 *   rpc     client|server      PROTO ITERS SERVICE_NS
 *   barrier ID NHOSTS PROTO ITERS                  central barrier, host 0 is the master
 *   rtt     -  -  ITERS                              idle round trip of a poll
 *   tenant  ID -  US MB                             streaming reads of DATA for US microseconds
 *   waiter  ID PROTO ITERS                          waits for ITERS messages on a private flag
 *   poster  ID PROTO ITERS MEAN_NS N                posts to waiters ID..ID+N-1, each message
 *                                                    after an exponential delay
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>

#define DATA ((volatile uint8_t *)0x100000000000ULL)
#define SYNC0 ((volatile uint8_t *)0x110000000000ULL)
#define SYNC1 ((volatile uint8_t *)0x120000000000ULL)
#define SYNC0WB ((volatile uint8_t *)0x130000000000ULL)
#define LINE 64
#define PAGE 4096   /* every sync line sits on its own page, out of reach of prefetchers */

static inline uint64_t now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + t.tv_nsec;
}

static uint64_t jseed = 88172645463325252ULL;
static inline uint64_t jitter(uint64_t max)
{
    jseed ^= jseed << 13;
    jseed ^= jseed >> 7;
    jseed ^= jseed << 17;
    return jseed % (max + 1);
}

static inline void spin_until(uint64_t t)
{
    while (now_ns() < t)
        ;
}

static inline void flush_range(volatile uint8_t *p, size_t n)
{
    for (size_t i = 0; i < n; i += LINE)
        _mm_clflushopt((void *)(p + i));
    _mm_mfence();
}

/* flags: one 64 B line each; flag f of a protocol lives at DATA_FLAGS or SYNC0/SYNC1 offsets */
#define DATA_FLAGS (DATA + (128u << 20))   /* flush-protocol flags, one per page */

static inline uint64_t read_flag(const char *proto, unsigned f, int consume)
{
    if (proto[0] == 'w') {
        volatile uint64_t *p = (volatile uint64_t *)(SYNC0WB + (size_t)f * PAGE);
        _mm_clflushopt((void *)p);
        _mm_mfence();
        return *p;
    }
    if (proto[0] == 'f') {
        volatile uint64_t *p = (volatile uint64_t *)(DATA_FLAGS + (size_t)f * PAGE);
        _mm_clflushopt((void *)p);
        _mm_mfence();
        return *p;
    }
    volatile uint8_t *base = (consume && proto[0] == 'p') ? SYNC1 : SYNC0;
    return *(volatile uint64_t *)(base + (size_t)f * PAGE);
}

static inline void write_flag(const char *proto, unsigned f, uint64_t v)
{
    if (proto[0] == 'f' || proto[0] == 'w') {
        volatile uint64_t *p = (volatile uint64_t *)((proto[0] == 'w' ? SYNC0WB : DATA_FLAGS)
                                                     + (size_t)f * PAGE);
        p[1] = now_ns();
        *p = v;
        _mm_clflushopt((void *)p);
        _mm_mfence();
        return;
    }
    *(volatile uint64_t *)(SYNC0 + (size_t)f * PAGE) = v;
    _mm_mfence();
}

static inline void reset_flag(const char *proto, unsigned f)
{
    if (proto[0] == 'p' || proto[0] == 'w')
        *(volatile uint64_t *)(SYNC1 + (size_t)f * PAGE) = 0;
}

/* wait until flag f equals v; parked loads consume the flag when `consume` */
static inline uint64_t wait_flag(const char *proto, unsigned f, uint64_t v, int consume)
{
    uint64_t polls = 0;
    while (read_flag(proto, f, consume) != v)
        polls++;
    if (consume && proto[0] == 'w')
        reset_flag(proto, f);
    return polls;
}

static void send_payload(volatile uint8_t *buf, size_t size, uint64_t seq, uint64_t t)
{
    for (size_t i = 0; i < size; i += LINE) {
        ((volatile uint64_t *)(buf + i))[0] = seq;
        ((volatile uint64_t *)(buf + i))[1] = t;
    }
    flush_range(buf, size);
}

static uint64_t recv_payload(volatile uint8_t *buf, size_t size, uint64_t seq)
{
    flush_range(buf, size);
    uint64_t t = 0;
    for (size_t i = 0; i < size; i += LINE) {
        if (((volatile uint64_t *)(buf + i))[0] != seq) {
            fprintf(stderr, "payload mismatch at %zu: %lu vs %lu\n", i,
                    ((volatile uint64_t *)(buf + i))[0], seq);
            exit(2);
        }
        t = ((volatile uint64_t *)(buf + i))[1];
    }
    return t;
}

/* wbpark: the doorbell line (sync line f) carries the sequence number, the send time and up to
   48 B of payload; a longer payload goes to write-back memory first and is read after the wake */
static void send_wb(unsigned f, volatile uint8_t *payload, size_t size, uint64_t seq, uint64_t t)
{
    if (size > LINE)
        send_payload(payload, size, seq, t);
    volatile uint64_t *d = (volatile uint64_t *)(SYNC0WB + (size_t)f * PAGE);
    d[1] = t;
    for (int k = 2; k < 8; k++)
        d[k] = seq;
    d[0] = seq;
    _mm_clflushopt((void *)d);
    _mm_mfence();
}

/* returns the sender's timestamp; *t_done is taken when the data has arrived, before the doorbell
   is emptied */
static uint64_t recv_wb(unsigned f, volatile uint8_t *payload, size_t size, uint64_t seq,
                        uint64_t *reads, uint64_t *t_done)
{
    while (read_flag("w", f, 0) != seq)
        (*reads)++;
    (*reads)++;
    uint64_t t = ((volatile uint64_t *)(SYNC0WB + (size_t)f * PAGE))[1];
    if (size > LINE)
        recv_payload(payload, size, seq);
    *t_done = now_ns();
    reset_flag("w", f);
    return t;
}

/* "lines" protocol: poll the payload itself until every line carries seq; returns polls */
static uint64_t poll_lines(volatile uint8_t *buf, size_t size, uint64_t seq)
{
    for (uint64_t polls = 0;; polls++) {
        flush_range(buf, size);
        int ok = 1;
        for (size_t i = 0; i < size; i += LINE)
            ok &= ((volatile uint64_t *)(buf + i))[0] == seq;
        if (ok)
            return polls;
    }
}

static double rng_exp(uint64_t *s, double mean)
{
    *s = *s * 6364136223846793005ULL + 1442695040888963407ULL;
    double u = ((*s >> 11) + 1) * (1.0 / 9007199254740993.0);
    return -mean * __builtin_log(u);
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr, "usage: see source\n");
        return 1;
    }
    const char *test = argv[1], *role = argv[2], *proto = argv[3];
    long iters = atol(argv[4]);
    long arg = argc > 5 ? atol(argv[5]) : 0;
    for (const char *c = argv[2]; *c; c++)
        jseed = jseed * 31 + (uint64_t)*c + 7;
    jseed ^= (uint64_t)strlen(argv[1]) << 40;
    volatile uint8_t *msg = DATA;               /* request / message payload */
    volatile uint8_t *rep = DATA + (16u << 20); /* reply payload */

    if (!strcmp(test, "oneway")) {
        size_t size = arg ? arg : 64;
        if (!strcmp(role, "producer")) {
            for (long i = 1; i <= iters; i++) {
                spin_until(now_ns() + 20000 + jitter(2000));   /* consumer is already waiting */
                if (proto[0] == 'w') {
                    send_wb(0, msg, size, i, now_ns());
                    continue;
                }
                send_payload(msg, size, i, now_ns());
                if (proto[0] != 'l')
                    write_flag(proto, 0, i);
            }
        } else {
            double sum = 0, polls = 0;
            for (long i = 1; i <= iters; i++) {
                uint64_t t0, t1 = 0;
                if (proto[0] == 'w') {
                    uint64_t r = 0;
                    t0 = recv_wb(0, msg, size, i, &r, &t1);
                    polls += r - 1;
                } else if (proto[0] == 'l') {
                    polls += poll_lines(msg, size, i);
                    t0 = ((volatile uint64_t *)msg)[1];
                } else {
                    polls += wait_flag(proto, 0, i, 1);
                    t0 = recv_payload(msg, size, i);
                }
                if (!t1)
                    t1 = now_ns();
                if (i > 2)
                    sum += t1 - t0;
            }
            printf("RESULT test=oneway proto=%s size=%zu iters=%ld lat_ns=%.1f polls=%.1f\n",
                   proto, size, iters, sum / (iters - 2), polls / iters);
        }
    } else if (!strcmp(test, "rpc")) {
        if (!strcmp(role, "server")) {
            for (long i = 1; i <= iters; i++) {
                uint64_t r = 0, td;
                if (proto[0] == 'w') {
                    recv_wb(0, msg, 256, i, &r, &td);
                } else if (proto[0] == 'l') {
                    poll_lines(msg, 256, i);
                } else {
                    wait_flag(proto, 0, i, 1);
                    recv_payload(msg, 256, i);
                }
                spin_until(now_ns() + arg);
                if (proto[0] == 'w') {
                    send_wb(1, rep, 256, i, 0);
                    continue;
                }
                send_payload(rep, 256, i, 0);
                if (proto[0] != 'l')
                    write_flag(proto, 1, i);
            }
        } else {
            double sum = 0;
            for (long i = 1; i <= iters; i++) {
                spin_until(now_ns() + 5000 + jitter(2000));
                uint64_t t0 = now_ns();
                if (proto[0] == 'w') {
                    uint64_t r = 0, td;
                    send_wb(0, msg, 256, i, t0);
                    recv_wb(1, rep, 256, i, &r, &td);
                    if (i > 2)
                        sum += td - t0;
                    continue;
                }
                send_payload(msg, 256, i, t0);
                if (proto[0] == 'l') {
                    poll_lines(rep, 256, i);
                } else {
                    write_flag(proto, 0, i);
                    wait_flag(proto, 1, i, 1);
                    recv_payload(rep, 256, i);
                }
                if (i > 2)
                    sum += now_ns() - t0;
            }
            printf("RESULT test=rpc proto=%s service_ns=%ld iters=%ld rtt_ns=%.1f\n",
                   proto, arg, iters, sum / (iters - 2));
        }
    } else if (!strcmp(test, "barrier")) {
        int id = atoi(role), n = atoi(proto);
        proto = argv[4];
        iters = atol(argv[5]);
        /* flags 3..n+2: arrivals; flags 0/1/2: release lines rotating by round. The master
           empties the line for round r+2 after releasing round r, off the critical path; every
           host has passed that line (round r-1) because all of them arrived at round r. */
        uint64_t t_start = 0;
        for (long r = 1; r <= iters; r++) {
            if (r == 3)
                t_start = now_ns();
            spin_until(now_ns() + jitter(1000));   /* arrival skew */
            if (id == 0) {
                for (int w = 1; w < n; w++)
                    wait_flag(proto, 3 + w, r, 1);
                write_flag(proto, r % 3, r);
                reset_flag(proto, (r + 2) % 3);
            } else {
                write_flag(proto, 3 + id, r);
                wait_flag(proto, r % 3, r, 0);
            }
        }
        if (id == 0)
            printf("RESULT test=barrier proto=%s hosts=%d iters=%ld round_ns=%.1f\n", proto,
                   n, iters, (double)(now_ns() - t_start) / (iters - 2));
    } else if (!strcmp(test, "rtt")) {
        /* idle round trip: uncached loads of a flag that nobody writes (parking off), and
           clflushopt + mfence + load of a write-back line */
        uint64_t sink = 0, t0 = now_ns();
        for (long i = 0; i < iters; i++)
            sink += *(volatile uint64_t *)(SYNC0 + 4 * LINE);
        double uc = (double)(now_ns() - t0) / iters;
        t0 = now_ns();
        for (long i = 0; i < iters; i++)
            sink += read_flag("flush", 4, 0);
        double fl = (double)(now_ns() - t0) / iters;
        printf("RESULT test=rtt uc_ns=%.1f flush_ns=%.1f sink=%lu\n", uc, fl, sink);
    } else if (!strcmp(test, "tenant")) {
        /* stream reads of a private buffer for ITERS microseconds of simulated time */
        size_t bytes = (size_t)arg << 20;
        volatile uint8_t *buf = DATA + (64u << 20) + (size_t)atoi(role) * bytes;
        uint64_t sink = 0, t0 = now_ns(), done = 0, t_end = t0 + (uint64_t)iters * 1000;
        while (now_ns() < t_end) {
            for (size_t i = 0; i < bytes && (i & 0xffff || now_ns() < t_end); i += LINE) {
                sink += *(volatile uint64_t *)(buf + i);
                done += LINE;
            }
        }
        double ns = now_ns() - t0;
        printf("RESULT test=tenant id=%s GBs=%.3f sink=%lu\n", role, done / ns, sink);
    } else if (!strcmp(test, "waiter")) {
        /* wait for messages on a private flag until ITERS have arrived; the poster may run ahead,
           so take the newest message */
        int id = atoi(role);
        unsigned f = 4096 + id;
        volatile uint8_t *m = msg + (size_t)id * LINE * 2;
        double sum = 0;
        long got = 0;
        uint64_t last = 0, reads = 0;
        while (last < (uint64_t)iters) {
            uint64_t v = read_flag(proto, f, 1);
            reads++;
            if (v <= last)
                continue;
            last = v;
            uint64_t t0, t1;
            if (proto[0] == 'w') {          /* the line itself carries the message */
                t0 = ((volatile uint64_t *)(SYNC0WB + (size_t)f * PAGE))[1];
                t1 = now_ns();
                reset_flag(proto, f);
            } else {
                flush_range(m, 64);
                t0 = ((volatile uint64_t *)m)[1];
                t1 = now_ns();
            }
            sum += t1 - t0;
            got++;
        }
        printf("RESULT test=waiter id=%d proto=%s msgs=%ld lat_ns=%.1f reads_per_msg=%.1f\n", id,
               proto, got, sum / got, (double)reads / got);
    } else if (!strcmp(test, "poster")) {
        /* post ITERS messages to each of N waiters (ids ID..ID+N-1), each after an exponential
           delay with mean ARG ns; usage: poster ID PROTO ITERS MEAN_NS N */
        int id = atoi(role), n = argc > 6 ? atoi(argv[6]) : 1;
        uint64_t seed = 12345 + id, seq[1024] = {0}, next[1024];
        uint64_t t = now_ns();
        for (int j = 0; j < n; j++)
            next[j] = t + (uint64_t)rng_exp(&seed, arg);
        for (long left = (long)n * iters; left > 0;) {
            uint64_t now = now_ns();
            for (int j = 0; j < n; j++) {
                if (seq[j] < (uint64_t)iters && now >= next[j]) {
                    volatile uint8_t *m = msg + (size_t)(id + j) * LINE * 2;
                    ++seq[j];
                    if (proto[0] != 'w')
                        send_payload(m, 64, seq[j], now_ns());
                    write_flag(proto, 4096 + id + j, seq[j]);
                    next[j] = now_ns() + (uint64_t)rng_exp(&seed, arg);
                    left--;
                }
            }
        }
    }
    return 0;
}
