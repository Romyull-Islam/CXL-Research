# CXL research projects

One folder per project, each self-contained, in one repository; each folder keeps its own history.
Every idea here went through an adversarial novelty search (papers, arXiv, patents) and a
first-order significance check. None is promised to be accepted anywhere; each PROJECT.md states
which venue tier it can realistically reach and what work remains.

| Folder | Project | Novelty | Realistic target | State (2026-10-18) |
|---|---|---|---|---|
| 1-tts-cxl-tiering | Keeping preempted KV of reasoning workloads in a CXL tier | mechanism already shipped | IEEE CAL letter | gate 2 failed for preempted KV (2 to 10% gain); 4-page letter, model tested on real vLLM with seven models of four architectures on two GPUs (RTX 4500 Ada, A100) |
| 2-hybrid-cxl-kv-endurance | Endurance and provisioning of hybrid DRAM-over-NAND CXL KV caches | narrow but present | ACM TOS | paper draft done; key effects checked on a real NVMe drive |
| 3-parked-loads | Full/empty parked loads for multi-host CXL | novel in CXL, small latency gain, large bandwidth gain | IEEE CAL letter | model-checked (TLA+) and gem5-evaluated; 4-page letter done |

Each project's paper/ folder holds its draft; CLAUDE.md and PROJECT.md list what the author still
has to do (department in the author line, final bibliography check, a fresh novelty search before submission).

notes/ holds material shared by all projects: the public data inventory and the record of the
ideas checked and rejected on 2026-10-08 and 2026-10-09.

Downloaded data sits in each project's data/ folder (not in git; fetch_data.sh re-creates it).
To run the analyses, create a virtualenv in a project and pip install -r requirements.txt.
