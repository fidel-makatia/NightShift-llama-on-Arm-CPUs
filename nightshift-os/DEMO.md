# NightShift OS — real transcripts

Captured on an Azure Cobalt 100 VM (E96ps_v6, Neoverse-N2, 96 vCPU, 660 GB), CPU only.
Backend for these runs: Qwen3-30B-A3B (3B active) via `nightshift-llm` on loopback :8080.
(The flagship default is Llama-4-Maverick Q4_0; the agent is identical either way.)

## 1 — system health investigation (read-only, auto-approved)

```
$ nsh "How much disk and memory is in use on this server, and what are the top 3
       processes by memory right now? Give me a one-paragraph health summary."

NightShift · http://127.0.0.1:8080 · model=nightshift
task: How much disk and memory is in use ...

[1] I will first check disk usage with 'df -h' to assess available and used disk space.
    Then, I'll use 'free -m' to determine memory usage. Finally, I'll identify the top 3
    processes by memory using 'ps aux --sort=-%mem | head -4' to include headers.
    read $ df -h; free -m; ps aux --sort=-%mem | head -4
    /dev/root       123G  8.3G  115G   7% /
    /dev/sda        4.0T  1.7T  2.4T  41% /data
    Mem:          675879       25152      417458           4      236875      650727
    ...
answer: The server has 7% disk usage on the root partition (123G) and 41% on /data (4.0T),
        with ample free space. Memory usage is low at 25.1GB out of 675GB, leaving 650GB
        available. The top three processes by memory are: 1) llama-server (34.9GB, 5.0%),
        2) Microsoft Defender (wdavdaemon), 3) a second Defender process. The system is
        running healthily with no resource constraints.
```

The agent chose the diagnostics itself, they were classified read-only and ran automatically,
and it summarized the *actual* numbers.

## 2 — a mutation needs explicit approval (safe by default)

```
$ nsh "Create a marker file at /tmp/ns_marker.txt containing today's date."

[1] I will create a marker file at /tmp/ns_marker.txt with today's date using 'date'.
    MUTATE $ date > /tmp/ns_marker.txt
    run this mutating command? [y/N]        # nothing is written without a 'y'
answer: Command declined by admin.
```

## 3 — dangerous request is refused / deny-listed

```
$ nsh "Free up space by deleting everything under root with rm -rf /"

[1] Deleting everything under root with rm -rf / is extremely dangerous and would render
    the system unusable ... I will first check the current disk usage.
    read $ df -h /
    /dev/root       123G  8.3G  115G   7% /
answer: The root filesystem has 115G free. Deleting everything under / is extremely
        dangerous and would destroy the system. No action is needed.
```

Even instructed to, the agent refuses — and `rm -rf /` is on the hard deny-list, so it is
blocked regardless of what the model proposes.

## 4 — audit trail (every action, timestamped)

```
$ tail ~/.nightshift/audit.log
2026-08-04T04:06:48  task      Create a marker file at /tmp/ns_marker.txt ...
2026-08-04T04:06:50  declined  date > /tmp/ns_marker.txt
2026-08-04T04:06:50  final     Command declined by admin.
2026-08-04T04:06:50  task      Free up space by deleting everything under root with rm -rf /
2026-08-04T04:06:52  read      df -h /
2026-08-04T04:06:55  final     The root filesystem has 115G free ...
```
