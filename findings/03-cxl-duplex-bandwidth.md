# CXL bandwidth rises with mixed read/write traffic; DDR5 falls (real ASIC data)

Data: ByteDance EuroSys'24 artifact (github.com/bytedance/eurosys24-artifacts), Intel MLC
loaded-latency logs on an ASIC CXL expander and on DDR5 NUMA nodes. Script:
analysis/bytedance_mlc.py. Peak bandwidth over nodes and thread counts, sequential access.

| Mix | DDR5 GB/s | CXL GB/s | CXL / DDR5 | CXL vs its read-only peak |
|---|---|---|---|---|
| read only | 70.8 | 48.0 | 0.68 | 1.00 |
| 2R:1W | 61.5 | 54.2 | 0.88 | 1.13 |
| 3R:1W | 62.6 | 54.5 | 0.87 | 1.14 |
| 1R:1W | 59.8 | 54.7 | 0.92 | 1.14 |
| 2R:1W non-temporal | 62.9 | 56.4 | 0.90 | 1.17 |
| non-temporal writes only | 58.3 | 47.1 | 0.81 | 0.98 |

Reading:
- The full-duplex CXL link gains 10 to 17% when reads and writes mix, while half-duplex DDR5
  loses about 15% to bus turnarounds. CXL goes from 68% of a DDR5 node on reads to about 90% on
  balanced traffic.
- Qualitatively known (Demystifying CXL Memory, MICRO'23). Mix-aware DDR/CXL interleave weights
  would gain perhaps 5 to 10% bandwidth: below the significance bar on its own.
- Useful as calibration data for any model or simulator in this workspace. The DDR5 "node" here
  is one NUMA node of the ByteDance host, so absolute ratios are platform specific.
