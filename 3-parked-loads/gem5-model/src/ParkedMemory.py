from m5.objects.AbstractMemory import *
from m5.params import *


class ParkedMemory(AbstractMemory):
    type = "ParkedMemory"
    cxx_header = "parked_mem.hh"
    cxx_class = "gem5::memory::ParkedMemory"

    port = ResponsePort("This port sends responses and receives requests")
    latency = Param.Latency("100ns", "Device access latency")
    bandwidth = Param.MemoryBandwidth(
        "64GiB/s", "Combined read and write bandwidth of the device port"
    )
    sync_base = Param.Addr(0, "Start of the synchronization region (window 0)")
    sync_size = Param.MemorySize("0B", "Size of each sync window")
    park_enable = Param.Bool(True, "Park reads of empty lines")
    park_timeout = Param.Latency("25us", "Bound on how long a read stays parked")
    table_entries = Param.Unsigned(4096, "Parked-table entries")
