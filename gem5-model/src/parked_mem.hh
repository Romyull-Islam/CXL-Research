/*
 * ParkedMemory: a CXL Type-3 memory device that can park reads of empty lines.
 *
 * Timing follows gem5's SimpleMemory (fixed access latency, bandwidth limit on accepted
 * requests, in-order response queue), except that every access costs at least 64 B of bandwidth
 * because CXL.mem transfers whole lines. A synchronization region of sync_size bytes at sync_base
 * keeps one full/empty bit per 64 B line and is reachable through two address windows:
 *   window 0 [sync_base, sync_base + sync_size): a read waits while the line is empty; a write
 *            stores the data and sets the line full, which answers the reads parked on it;
 *   window 1 [sync_base + sync_size, sync_base + 2 sync_size): a consuming read returns the
 *            window-0 data and sets the line empty; a write sets the line empty.
 * A read for ownership (a store that misses in a write-back mapping) is never parked. A shared
 * read of an empty line is parked if park_enable is set and the table has room; otherwise it is
 * answered at once with the line's current contents. A parked read is answered with the current
 * contents after park_timeout ("empty" reply: software sees an unchanged sequence number and
 * re-issues). Wake-up is FIFO per line, and a consuming read stops the wake-up of later ones.
 */

#ifndef __PARKED_MEM_HH__
#define __PARKED_MEM_HH__

#include <deque>
#include <list>
#include <map>
#include <unordered_map>
#include <vector>

#include "base/statistics.hh"
#include "mem/abstract_mem.hh"
#include "mem/port.hh"
#include "params/ParkedMemory.hh"

namespace gem5
{

namespace memory
{

class ParkedMemory : public AbstractMemory
{
  private:
    struct DeferredPacket
    {
        Tick tick;
        PacketPtr pkt;
        DeferredPacket(PacketPtr _pkt, Tick _tick) : tick(_tick), pkt(_pkt) {}
    };

    class MemoryPort : public ResponsePort
    {
      private:
        ParkedMemory &mem;

      public:
        MemoryPort(const std::string &_name, ParkedMemory &_memory);

      protected:
        Tick recvAtomic(PacketPtr pkt) override;
        void recvFunctional(PacketPtr pkt) override;
        bool recvTimingReq(PacketPtr pkt) override;
        void recvRespRetry() override;
        AddrRangeList getAddrRanges() const override;
    };

    struct Parked
    {
        PacketPtr pkt;
        bool consume;
        uint64_t id;
    };

    MemoryPort port;
    const Tick latency;
    const double bandwidth;
    const Addr syncBase;
    const Addr syncSize;
    const bool parkEnable;
    const Tick parkTimeout;
    const unsigned tableEntries;

    std::list<DeferredPacket> packetQueue;
    bool isBusy;
    bool retryReq;
    bool retryResp;

    std::vector<uint8_t> full;
    std::unordered_map<Addr, std::deque<Parked>> parked;
    std::multimap<Tick, std::pair<Addr, uint64_t>> deadlines;
    unsigned parkedNow;
    uint64_t nextId;

    void release();
    EventFunctionWrapper releaseEvent;
    void dequeue();
    EventFunctionWrapper dequeueEvent;
    void timeout();
    EventFunctionWrapper timeoutEvent;

    std::unique_ptr<Packet> pendingDelete;

    // window of an address: -1 outside the sync region, else 0 or 1
    int window(Addr a) const;
    Addr lineOf(Addr a) const;
    // perform the access now and turn pkt into a response (no timing)
    void serve(PacketPtr pkt, bool consume);
    void queueResponse(PacketPtr pkt, Tick when);
    void wake(Addr line);
    void armTimer();

    struct ParkedStats : public statistics::Group
    {
        ParkedStats(ParkedMemory &mem);
        statistics::Scalar parks;
        statistics::Scalar wakes;
        statistics::Scalar timeouts;
        statistics::Scalar tableFull;
        statistics::Scalar emptyReads;
        statistics::Scalar maxParked;
        statistics::Scalar syncReads;
        statistics::Scalar syncWrites;
    } pstats;

  public:
    ParkedMemory(const ParkedMemoryParams &p);

    DrainState drain() override;
    Port &getPort(const std::string &if_name,
                  PortID idx = InvalidPortID) override;
    void init() override;

  protected:
    Tick recvAtomic(PacketPtr pkt);
    void recvFunctional(PacketPtr pkt);
    bool recvTimingReq(PacketPtr pkt);
    void recvRespRetry();
};

} // namespace memory
} // namespace gem5

#endif // __PARKED_MEM_HH__
