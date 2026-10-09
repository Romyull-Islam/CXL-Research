# ParkedLoads

Full/empty synchronization for multi-host CXL memory.

Hosts in a CXL pod share memory, but hardware coherence across hosts is absent or limited to a
small region. Today a host that waits for data written by another host polls: it flushes its
cached copy and re-reads the line over the link until the value changes. This project studies a
different contract. The CXL memory device keeps a full/empty bit for each line of a
synchronization region, and a read of an empty line is parked inside the device until a
producer's write fills it. The idea comes from the HEP (1978) and Tera MTA machines and from
I-structures; CXL brings it back because the memory controller now lives in the device, where
every host's requests meet.

Status: idea checked and parked. Novel in CXL, but the significance model (model/) shows too little
gain for a top venue; see the status note at the top of IDEA.md.

## Layout

    GOAL.md                          current goal and target venue (IEEE CAL letter)
    IDEA.md                          problem and gap, goal, motivation, related work
    PROPOSAL.md                      research questions, workloads, baselines, evaluation plan, expected objections
    model/                           first-order significance model and its result
    notes/idea-search-2026-10.md     the 16 ideas checked on 2026-10-08 and why this one was chosen
    chat/                            readable transcript of the conversation that produced the idea (not tracked by git)
