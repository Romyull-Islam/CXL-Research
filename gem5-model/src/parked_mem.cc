#include "parked_mem.hh"

#include <algorithm>

#include "base/trace.hh"
#include "debug/Drain.hh"
#include "debug/ParkedMem.hh"

namespace gem5
{

namespace memory
{

ParkedMemory::ParkedMemory(const ParkedMemoryParams &p) :
    AbstractMemory(p),
    port(name() + ".port", *this), latency(p.latency),
    bandwidth(p.bandwidth), syncBase(p.sync_base), syncSize(p.sync_size),
    parkEnable(p.park_enable), parkTimeout(p.park_timeout),
    tableEntries(p.table_entries),
    isBusy(false), retryReq(false), retryResp(false),
    full(p.sync_size / 64, 0), parkedNow(0), nextId(0),
    releaseEvent([this]{ release(); }, name()),
    dequeueEvent([this]{ dequeue(); }, name()),
    timeoutEvent([this]{ timeout(); }, name() + ".timeout"),
    pstats(*this)
{
    fatal_if(syncSize % 64, "sync_size must be a multiple of 64");
    fatal_if(syncSize && (syncBase < range.start() ||
             syncBase + 2 * syncSize > range.end()),
             "both sync windows must lie inside the device range");
}

ParkedMemory::ParkedStats::ParkedStats(ParkedMemory &mem) :
    statistics::Group(&mem, "parked"),
    ADD_STAT(parks, statistics::units::Count::get(), "reads parked"),
    ADD_STAT(wakes, statistics::units::Count::get(),
             "parked reads answered by a write"),
    ADD_STAT(timeouts, statistics::units::Count::get(),
             "parked reads answered empty at the timeout"),
    ADD_STAT(tableFull, statistics::units::Count::get(),
             "reads of empty lines answered at once because the table was full"),
    ADD_STAT(emptyReads, statistics::units::Count::get(),
             "reads of empty lines answered at once (parking off)"),
    ADD_STAT(maxParked, statistics::units::Count::get(),
             "most reads parked at once"),
    ADD_STAT(syncReads, statistics::units::Count::get(),
             "reads to the sync region"),
    ADD_STAT(syncWrites, statistics::units::Count::get(),
             "writes to the sync region")
{
}

void
ParkedMemory::init()
{
    AbstractMemory::init();
    if (port.isConnected())
        port.sendRangeChange();
}

int
ParkedMemory::window(Addr a) const
{
    if (!syncSize || a < syncBase || a >= syncBase + 2 * syncSize)
        return -1;
    return a < syncBase + syncSize ? 0 : 1;
}

Addr
ParkedMemory::lineOf(Addr a) const
{
    return ((a - syncBase) % syncSize) / 64;
}

void
ParkedMemory::serve(PacketPtr pkt, bool consume)
{
    const int w = window(pkt->getAddr());
    if (w == 1) {
        const Addr a0 = pkt->getAddr() - syncSize;
        if (pkt->isRead()) {
            pkt->setData(toHostAddr(a0));
            if (consume)
                full[lineOf(a0)] = 0;
        } else if (pkt->isWrite()) {
            full[lineOf(a0)] = 0;
        }
        if (pkt->needsResponse())
            pkt->makeResponse();
        return;
    }
    access(pkt);
    if (w == 0 && pkt->isWrite())
        full[lineOf(pkt->getAddr())] = 1;
}

void
ParkedMemory::queueResponse(PacketPtr pkt, Tick when)
{
    auto i = packetQueue.end();
    while (i != packetQueue.begin()) {
        auto prev = std::prev(i);
        if (when >= prev->tick || prev->pkt->matchAddr(pkt))
            break;
        i = prev;
    }
    packetQueue.emplace(i, pkt, when);
    if (!retryResp && !dequeueEvent.scheduled())
        schedule(dequeueEvent, packetQueue.front().tick);
    else if (!retryResp && packetQueue.front().tick < dequeueEvent.when())
        reschedule(dequeueEvent, packetQueue.front().tick);
}

void
ParkedMemory::wake(Addr line)
{
    auto it = parked.find(line);
    if (it == parked.end())
        return;
    auto &q = it->second;
    while (!q.empty() && full[line]) {
        Parked e = q.front();
        q.pop_front();
        parkedNow--;
        pstats.wakes++;
        serve(e.pkt, e.consume);
        queueResponse(e.pkt, curTick() + latency);
        DPRINTF(ParkedMem, "wake line %#x consume %d\n", line, e.consume);
    }
    if (q.empty())
        parked.erase(it);
}

void
ParkedMemory::armTimer()
{
    if (deadlines.empty()) {
        if (timeoutEvent.scheduled())
            deschedule(timeoutEvent);
        return;
    }
    Tick t = deadlines.begin()->first;
    if (!timeoutEvent.scheduled())
        schedule(timeoutEvent, t);
    else if (timeoutEvent.when() != t)
        reschedule(timeoutEvent, t);
}

void
ParkedMemory::timeout()
{
    while (!deadlines.empty() && deadlines.begin()->first <= curTick()) {
        auto [line, id] = deadlines.begin()->second;
        deadlines.erase(deadlines.begin());
        auto it = parked.find(line);
        if (it == parked.end())
            continue;
        auto &q = it->second;
        auto e = std::find_if(q.begin(), q.end(),
                              [id](const Parked &x) { return x.id == id; });
        if (e == q.end())
            continue;
        PacketPtr pkt = e->pkt;
        q.erase(e);
        if (q.empty())
            parked.erase(it);
        parkedNow--;
        pstats.timeouts++;
        serve(pkt, false);
        queueResponse(pkt, curTick() + latency);
    }
    armTimer();
}

Tick
ParkedMemory::recvAtomic(PacketPtr pkt)
{
    panic_if(pkt->cacheResponding(), "Should not see packets where cache "
             "is responding");
    serve(pkt, window(pkt->getAddr()) == 1);
    return latency;
}

void
ParkedMemory::recvFunctional(PacketPtr pkt)
{
    pkt->pushLabel(name());
    functionalAccess(pkt);
    bool done = false;
    for (auto p = packetQueue.begin(); !done && p != packetQueue.end(); ++p)
        done = pkt->trySatisfyFunctional(p->pkt);
    pkt->popLabel();
}

bool
ParkedMemory::recvTimingReq(PacketPtr pkt)
{
    panic_if(pkt->cacheResponding(), "Should not see packets where cache "
             "is responding");
    panic_if(!(pkt->isRead() || pkt->isWrite()),
             "Should only see read and writes at memory controller, "
             "saw %s to %#llx\n", pkt->cmdString(), pkt->getAddr());

    if (retryReq)
        return false;
    if (isBusy) {
        retryReq = true;
        return false;
    }

    Tick receive_delay = pkt->headerDelay + pkt->payloadDelay;
    pkt->headerDelay = pkt->payloadDelay = 0;

    // CXL.mem moves whole 64 B lines, so every access costs at least a line
    Tick duration = std::max<unsigned>(pkt->getSize(), 64) * bandwidth;
    if (duration != 0) {
        schedule(releaseEvent, curTick() + duration);
        isBusy = true;
    }

    const int w = window(pkt->getAddr());
    if (w >= 0) {
        const Addr line = lineOf(pkt->getAddr());
        if (pkt->isRead()) {
            pstats.syncReads++;
            // a read for ownership (a store missing in a write-back mapping) is never
            // parked: only the writer issues it, and parking it would block the release
            if (!full[line] && !pkt->needsWritable()) {
                if (parkEnable && parkedNow < tableEntries) {
                    uint64_t id = nextId++;
                    parked[line].push_back({pkt, w == 1, id});
                    parkedNow++;
                    pstats.parks++;
                    if (parkedNow > pstats.maxParked.value())
                        pstats.maxParked = parkedNow;
                    deadlines.emplace(curTick() + receive_delay + parkTimeout,
                                      std::make_pair(line, id));
                    armTimer();
                    DPRINTF(ParkedMem, "park line %#x consume %d\n", line,
                            w == 1);
                    return true;
                }
                if (parkEnable)
                    pstats.tableFull++;
                else
                    pstats.emptyReads++;
                // answered with the current contents; an empty line stays
                // empty even for a consuming read
                serve(pkt, false);
                queueResponse(pkt, curTick() + receive_delay + latency);
                return true;
            }
            serve(pkt, w == 1);
            queueResponse(pkt, curTick() + receive_delay + latency);
            return true;
        }
        pstats.syncWrites++;
        bool needs_response = pkt->needsResponse();
        serve(pkt, false);
        if (w == 0)
            wake(line);
        if (needs_response)
            queueResponse(pkt, curTick() + receive_delay + latency);
        else
            pendingDelete.reset(pkt);
        return true;
    }

    bool needs_response = pkt->needsResponse();
    access(pkt);
    if (needs_response)
        queueResponse(pkt, curTick() + receive_delay + latency);
    else
        pendingDelete.reset(pkt);
    return true;
}

void
ParkedMemory::release()
{
    assert(isBusy);
    isBusy = false;
    if (retryReq) {
        retryReq = false;
        port.sendRetryReq();
    }
}

void
ParkedMemory::dequeue()
{
    assert(!packetQueue.empty());
    DeferredPacket d = packetQueue.front();
    retryResp = !port.sendTimingResp(d.pkt);
    if (!retryResp) {
        packetQueue.pop_front();
        if (!packetQueue.empty()) {
            reschedule(dequeueEvent,
                       std::max(packetQueue.front().tick, curTick()), true);
        } else if (drainState() == DrainState::Draining && parkedNow == 0) {
            signalDrainDone();
        }
    }
}

void
ParkedMemory::recvRespRetry()
{
    assert(retryResp);
    dequeue();
}

Port &
ParkedMemory::getPort(const std::string &if_name, PortID idx)
{
    if (if_name != "port")
        return AbstractMemory::getPort(if_name, idx);
    return port;
}

DrainState
ParkedMemory::drain()
{
    if (!packetQueue.empty() || parkedNow) {
        DPRINTF(Drain, "ParkedMemory has requests, waiting to drain\n");
        return DrainState::Draining;
    }
    return DrainState::Drained;
}

ParkedMemory::MemoryPort::MemoryPort(const std::string &_name,
                                     ParkedMemory &_memory)
    : ResponsePort(_name), mem(_memory)
{ }

AddrRangeList
ParkedMemory::MemoryPort::getAddrRanges() const
{
    AddrRangeList ranges;
    ranges.push_back(mem.getAddrRange());
    return ranges;
}

Tick
ParkedMemory::MemoryPort::recvAtomic(PacketPtr pkt)
{
    return mem.recvAtomic(pkt);
}

void
ParkedMemory::MemoryPort::recvFunctional(PacketPtr pkt)
{
    mem.recvFunctional(pkt);
}

bool
ParkedMemory::MemoryPort::recvTimingReq(PacketPtr pkt)
{
    return mem.recvTimingReq(pkt);
}

void
ParkedMemory::MemoryPort::recvRespRetry()
{
    mem.recvRespRetry();
}

} // namespace memory
} // namespace gem5
