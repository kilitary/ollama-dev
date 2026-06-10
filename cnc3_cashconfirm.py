"""Final confirmation: read the confirmed cash addresses 3x to verify live ticking."""
import ctypes
import struct
import sys
import time

# ── Win32 ─────────────────────────────────────────────────────────────────────
PROCESS_VM_READ    = 0x0010
PROCESS_QUERY_INFO = 0x0400

kernel32 = ctypes.windll.kernel32

# ── Config ──────────────────────────────────────────────────────────────────--
PID = 22756

SAMPLE_COUNT = 3      # number of memory snapshots to take
SAMPLE_GAP_S = 2      # seconds to wait between snapshots

# Plausible AI player cash window used for the final confirmed list
CASH_MIN = 5_000
CASH_MAX = 100_000

# Best candidates from delta scan (addresses with ~23K-57K values and positive deltas)
CANDIDATES = {
    "Group1 (high income)":  0x3091AE8,
    "Group2-A":              0x41B1AE8,
    "Group2-B (changed)":    0x41BE388,
    "Group3":                0x522A88C,
    "Group7":                0x56F0590,
    "Group8":                0x6113D60,
    "Group9":                0x6353580,
    "Group14":               0x6490088,
    "Group15-A":             0xC9F8998,
    "Group15-B":             0xC9F89B8,
    "Group17 (large)":       0xCA7FACC,
    "Group18":               0xCBCFA18,
    "Group19":               0xCBE96C0,
}


def read4(hproc, addr):
    """Read a little-endian 32-bit signed int, or None if the read fails."""
    buf = (ctypes.c_char * 4)()
    read = ctypes.c_size_t(0)
    ok = kernel32.ReadProcessMemory(
        hproc, ctypes.c_void_p(addr), buf, 4, ctypes.byref(read)
    )
    if ok and read.value == 4:
        return struct.unpack('<i', bytes(buf))[0]
    return None


def open_process(pid):
    """Open a read-only handle to *pid*, exiting on failure."""
    hproc = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFO, False, pid)
    if not hproc:
        print(f"Failed: {kernel32.GetLastError()}")
        sys.exit(1)
    return hproc


def take_snapshots(hproc, candidates, count=SAMPLE_COUNT, gap_s=SAMPLE_GAP_S):
    """Read every candidate address *count* times, waiting *gap_s* between reads.

    Returns a list of ``{label: value}`` dicts, one per sample.
    """
    snapshots = []
    for i in range(count):
        if i > 0:
            time.sleep(gap_s)
        snapshots.append({lbl: read4(hproc, addr) for lbl, addr in candidates.items()})
    return snapshots


def classify(first, last):
    """Return a human-readable status string for a first/last value pair."""
    if 0 < first < last:
        return "✅ LIVE (growing)"
    if 0 < first == last:
        return "⚠️  static / paused"
    if last < first:
        return "❓ decreasing"
    return "—"


def print_sample_table(candidates, snapshots, gap_s=SAMPLE_GAP_S):
    """Print the per-address sample table with deltas and live status."""
    print(f"C&C3 AI Cash — {len(snapshots)}-sample confirmation ({gap_s}s gap each)")
    print("=" * 65)
    print(f"  {'Label':<25} {'t=0':>8} {'t=2s':>8} {'t=4s':>8}  {'Δ/2s':>8}  Status")
    print("  " + "─" * 63)

    for lbl in candidates:
        values = [snap[lbl] for snap in snapshots]
        if None in values:
            continue
        delta = (values[-1] - values[0]) / (len(values) - 1)  # average delta per gap
        status = classify(values[0], values[-1])
        print(f"  {lbl:<25} {values[0]:>8} {values[1]:>8} {values[2]:>8}  "
              f"{delta:>+8.1f}  {status}")


def print_final_snapshot(candidates, snapshots, gap_s=SAMPLE_GAP_S):
    """Print the confirmed AI player cash list and return the player count."""
    span_s = gap_s * (len(snapshots) - 1)
    print()
    print("=" * 65)
    print("FINAL — AI Player Cash Snapshot:")
    print("─" * 65)

    ai_players = 0
    for lbl, addr in candidates.items():
        first, last = snapshots[0][lbl], snapshots[-1][lbl]
        if first and last and CASH_MIN < first < CASH_MAX and last >= first:
            delta = last - first
            rate = delta / span_s * 60  # extrapolate to credits/min
            ai_players += 1
            print(f"  Player {ai_players:02d}  addr=0x{addr:X}  cash={last:,}  "
                  f"Δ{span_s}s={delta:+}  (~{rate:.0f} cr/min)")

    print(f"\n  Total confirmed AI player cash addresses: {ai_players}")
    return ai_players


def main():
    hproc = open_process(PID)
    try:
        snapshots = take_snapshots(hproc, CANDIDATES)
        print_sample_table(CANDIDATES, snapshots)
        print_final_snapshot(CANDIDATES, snapshots)
    finally:
        kernel32.CloseHandle(hproc)


if __name__ == "__main__":
    main()

