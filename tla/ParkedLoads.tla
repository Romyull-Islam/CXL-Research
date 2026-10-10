---------------------------- MODULE ParkedLoads ----------------------------
(* A CXL device that parks reads of an empty synchronization line, consumer hosts that wait on
   the line, and a producer host that first reads an ordinary line (for example a lock or a
   payload pointer) and then writes the sync line. The producer may think for an unbounded time
   before acting, and may fail.

   The four rules of the letter are switches:
     R1  a parked read leaves the request queue, so it frees its request credit;
     R2  writes travel on the RwD channel and need no request credit;
     R3  the device answers every parked read within Timeout ticks, below the Root Port timeout;
     R4  that answer says "not yet" (explicit empty, or an unchanged sequence number), so the
         consumer re-issues instead of taking it as the written data;
     R5  a read for ownership is never parked. With WBWrite the producer writes the sync line
         through a write-back mapping, so its store first reads the line for ownership.
   A read that finds the parked table full (TableSize entries) is answered "not yet" at once.

   Time: Tick advances only when no device or host action that cannot wait is enabled, so ages
   grow only while something is genuinely blocked or the producer is thinking. A consumer that
   gets a "not yet" answer re-issues after the next tick (a retry costs at least a round trip).
   Ages saturate at RPTimeout + 1 to keep the state space finite.

   Fairness: every action is weakly fair, and taking a request credit is strongly fair, as the
   round-robin arbitration of a CXL link provides. *)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Cons, R1, R2, R3, R4, R5, WBWrite, Credits, TableSize, Timeout, RPTimeout,
          ProducerMayFail

VARIABLES full, val, reqQ, rwd, table, resp, cpc, ppc, cAge, pAge, seen, failed

vars == <<full, val, reqQ, rwd, table, resp, cpc, ppc, cAge, pAge, seen, failed>>

Cap == RPTimeout + 1
Inc(a) == IF a < Cap THEN a + 1 ELSE a

Init ==
    /\ full = FALSE /\ val = 0
    /\ reqQ = <<>> /\ rwd = <<>> /\ table = {} /\ resp = {}
    /\ cpc = [c \in Cons |-> "issue"] /\ ppc = "think"
    /\ cAge = [c \in Cons |-> 0] /\ pAge = 0 /\ seen = [c \in Cons |-> 0] /\ failed = FALSE

Remove(s, i) == [j \in 1..(Len(s) - 1) |-> IF j < i THEN s[j] ELSE s[j + 1]]

(* ---- hosts ---- *)
ConsIssue(c) ==
    /\ cpc[c] = "issue" /\ Len(reqQ) < Credits
    /\ reqQ' = Append(reqQ, [op |-> "rd", src |-> c, line |-> "L", age |-> 0, parked |-> FALSE])
    /\ cpc' = [cpc EXCEPT ![c] = "wait"] /\ cAge' = [cAge EXCEPT ![c] = 0]
    /\ UNCHANGED <<full, val, rwd, table, resp, ppc, pAge, seen, failed>>

ConsRecv(c) ==
    /\ cpc[c] = "wait"
    /\ \E r \in resp :
        /\ r.dst = c
        /\ resp' = resp \ {r}
        /\ IF r.empty \/ (R4 /\ r.data # 1)
             THEN cpc' = [cpc EXCEPT ![c] = "backoff"] /\ UNCHANGED seen
             ELSE cpc' = [cpc EXCEPT ![c] = "done"] /\ seen' = [seen EXCEPT ![c] = r.data]
    /\ UNCHANGED <<full, val, reqQ, rwd, table, ppc, cAge, pAge, failed>>

ProdStart ==            \* leaving the think state is the producer's choice, not urgent
    /\ ppc = "think" /\ ppc' = "read" /\ pAge' = 0
    /\ UNCHANGED <<full, val, reqQ, rwd, table, resp, cpc, cAge, seen, failed>>

ProdRead ==
    /\ ppc = "read" /\ Len(reqQ) < Credits
    /\ reqQ' = Append(reqQ, [op |-> "rd", src |-> "p", line |-> "N", age |-> 0, parked |-> FALSE])
    /\ ppc' = "rwait"
    /\ UNCHANGED <<full, val, rwd, table, resp, cpc, cAge, pAge, seen, failed>>

ProdRfo ==
    /\ ppc = "rfo" /\ Len(reqQ) < Credits
    /\ reqQ' = Append(reqQ, [op |-> "own", src |-> "p", line |-> "L", age |-> 0, parked |-> FALSE])
    /\ ppc' = "ofwait"
    /\ UNCHANGED <<full, val, rwd, table, resp, cpc, cAge, pAge, seen, failed>>

ProdWrite ==
    /\ ppc = "write"
    /\ IF R2
         THEN /\ rwd = <<>>
              /\ rwd' = <<[op |-> "wr", src |-> "p", line |-> "L"]>>
              /\ UNCHANGED reqQ
         ELSE /\ Len(reqQ) < Credits
              /\ reqQ' = Append(reqQ, [op |-> "wr", src |-> "p", line |-> "L", age |-> 0,
                                       parked |-> FALSE])
              /\ UNCHANGED rwd
    /\ ppc' = "wwait"
    /\ UNCHANGED <<full, val, table, resp, cpc, cAge, pAge, seen, failed>>

ProdRecv ==
    /\ ppc \in {"rwait", "ofwait", "wwait"}
    /\ \E r \in resp :
        /\ r.dst = "p"
        /\ resp' = resp \ {r}
        /\ CASE ppc = "rwait" -> ppc' = (IF WBWrite THEN "rfo" ELSE "write") /\ pAge' = 0
              [] ppc = "ofwait" -> ppc' = "write" /\ UNCHANGED pAge
              [] OTHER -> ppc' = "done" /\ UNCHANGED pAge
    /\ UNCHANGED <<full, val, reqQ, rwd, table, cpc, cAge, seen, failed>>

ProdFail ==
    /\ ProducerMayFail /\ ~failed /\ ppc \in {"think", "read", "rfo", "write"}
    /\ failed' = TRUE /\ ppc' = "dead"
    /\ UNCHANGED <<full, val, reqQ, rwd, table, resp, cpc, cAge, pAge, seen>>

(* ---- device ---- *)
ParkedInQ == {i \in 1..Len(reqQ) : reqQ[i].parked}
NParked == Cardinality(table) + Cardinality(ParkedInQ)

(* write the line, answer every parked read (in the table or parked in place), ack the writer *)
DoWrite(newQ) ==
    /\ full' = TRUE /\ val' = 1
    /\ resp' = resp \cup {[dst |-> "p", data |-> 0, empty |-> FALSE]}
                    \cup {[dst |-> e.src, data |-> 1, empty |-> FALSE] : e \in table}
                    \cup {[dst |-> newQ[i].src, data |-> 1, empty |-> FALSE] :
                              i \in {j \in 1..Len(newQ) : newQ[j].parked}}
    /\ table' = {}
    /\ reqQ' = SelectSeq(newQ, LAMBDA e : ~e.parked)

ServeReq ==
    \E i \in 1..Len(reqQ) :
        LET e == reqQ[i] IN
        /\ ~e.parked
        /\ \/ /\ \/ e.op = "rd" /\ (e.line = "N" \/ full)
                 \/ e.op = "own" /\ (full \/ R5)
              /\ reqQ' = Remove(reqQ, i)
              /\ resp' = resp \cup {[dst |-> e.src, data |-> IF e.line = "N" THEN 0 ELSE val,
                                     empty |-> FALSE]}
              /\ UNCHANGED <<full, val, rwd, table>>
           \/ /\ e.op \in {"rd", "own"} /\ e.line = "L" /\ ~full /\ (e.op = "rd" \/ ~R5)
              /\ NParked < TableSize
              /\ IF R1
                   THEN /\ table' = table \cup {[src |-> e.src, age |-> 0]}
                        /\ reqQ' = Remove(reqQ, i)
                   ELSE /\ reqQ' = [reqQ EXCEPT ![i].parked = TRUE]
                        /\ UNCHANGED table
              /\ UNCHANGED <<full, val, rwd, resp>>
           \/ /\ e.op \in {"rd", "own"} /\ e.line = "L" /\ ~full /\ (e.op = "rd" \/ ~R5)
              /\ NParked >= TableSize
              /\ reqQ' = Remove(reqQ, i)
              /\ resp' = resp \cup {[dst |-> e.src, data |-> val, empty |-> R4]}
              /\ UNCHANGED <<full, val, rwd, table>>
           \/ /\ e.op = "wr"
              /\ DoWrite(Remove(reqQ, i))
              /\ UNCHANGED rwd
    /\ UNCHANGED <<cpc, ppc, cAge, pAge, seen, failed>>

ServeRwd ==
    /\ rwd # <<>>
    /\ DoWrite(reqQ)
    /\ rwd' = <<>>
    /\ UNCHANGED <<cpc, ppc, cAge, pAge, seen, failed>>

TimeoutReply(src) == [dst |-> src, data |-> val, empty |-> R4]

DevTimeout ==
    /\ R3
    /\ \/ \E e \in table :
            /\ e.age >= Timeout
            /\ table' = table \ {e}
            /\ resp' = resp \cup {TimeoutReply(e.src)}
            /\ UNCHANGED reqQ
       \/ \E i \in ParkedInQ :
            /\ reqQ[i].age >= Timeout
            /\ reqQ' = Remove(reqQ, i)
            /\ resp' = resp \cup {TimeoutReply(reqQ[i].src)}
            /\ UNCHANGED table
    /\ UNCHANGED <<full, val, rwd, cpc, ppc, cAge, pAge, seen, failed>>

Urgent == (\E c \in Cons : ConsIssue(c) \/ ConsRecv(c)) \/ ProdRead \/ ProdRfo \/ ProdWrite
          \/ ProdRecv
          \/ ServeReq \/ ServeRwd \/ DevTimeout

Tick ==
    /\ ~ENABLED Urgent
    /\ table' = {[src |-> e.src, age |-> Inc(e.age)] : e \in table}
    /\ reqQ' = [i \in 1..Len(reqQ) |-> [reqQ[i] EXCEPT !.age = IF reqQ[i].parked
                                                              THEN Inc(reqQ[i].age)
                                                              ELSE reqQ[i].age]]
    /\ cAge' = [c \in Cons |-> IF cpc[c] = "wait" THEN Inc(cAge[c]) ELSE cAge[c]]
    /\ pAge' = IF ppc \in {"read", "rwait", "rfo", "ofwait", "write", "wwait"} THEN Inc(pAge)
               ELSE pAge
    /\ cpc' = [c \in Cons |-> IF cpc[c] = "backoff" THEN "issue" ELSE cpc[c]]
    /\ UNCHANGED <<full, val, rwd, resp, ppc, seen, failed>>

Next == Urgent \/ ProdStart \/ ProdFail \/ Tick

Spec == Init /\ [][Next]_vars
        /\ WF_vars(ServeReq) /\ WF_vars(ServeRwd) /\ WF_vars(DevTimeout) /\ WF_vars(ProdRecv)
        /\ WF_vars(ProdStart) /\ WF_vars(Tick)
        /\ \A c \in Cons : WF_vars(ConsRecv(c)) /\ SF_vars(ConsIssue(c))
        /\ SF_vars(ProdRead) /\ SF_vars(ProdRfo) /\ SF_vars(ProdWrite)

(* ---- properties ---- *)
TypeOK == Len(reqQ) <= Credits /\ NParked <= TableSize /\ pAge <= Cap

(* no read is still outstanding when the Root Port completion timeout expires *)
NoRootPortTimeout == \A c \in Cons : cpc[c] = "wait" => cAge[c] < RPTimeout

(* a consumer accepts only data the producer wrote *)
NoStaleRead == \A c \in Cons : cpc[c] = "done" => seen[c] = 1

(* the producer's ordinary read and its releasing write are never held up by parked reads for
   as long as the parking bound *)
ProducerReadNotBlocked == ppc \in {"read", "rwait"} => pAge < Timeout
ProducerWriteNotBlocked == ppc \in {"rfo", "ofwait", "write", "wwait"} => pAge < Timeout

(* if the producer does not fail, every consumer eventually gets the data *)
EventuallyDone == <>((\A c \in Cons : cpc[c] = "done") \/ failed)
=============================================================================
