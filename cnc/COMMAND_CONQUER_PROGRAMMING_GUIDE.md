# Command and Conquer 3 Programming Guide

This guide documents a practical workflow for CNC3 research in this repository, from process validation to symbol analysis and cash-address triage.

## 1) Goals and Safety Model

- Objective: inspect runtime behavior and identify candidate game-state fields for analysis.
- Preferred operating mode: read-only process access using `OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFO, ...)`.
- Baseline validation script: `cnc3_memtest.py`.
- Safety boundary: run experiments in single-player contexts where possible.

### Why this matters

The current toolchain is designed around low-risk visibility first:

1. Verify process access and memory map stability.
2. Scan and narrow value candidates.
3. Confirm liveness via repeated snapshots.
4. Correlate with engine symbols and reports.

## 2) Repository Map for CNC3 Work

### Core analysis scripts

- `cnc3_memtest.py`: safe stage test for process open + `ReadProcessMemory`.
- `cnc3_cashscan.py`: broad region scan of aligned float candidates in cash ranges.
- `cnc3_cashdelta.py`: snapshot-delta scan to detect changing int32 values.
- `cnc3_human_cash.py`: filters for low-occurrence values to isolate human-like addresses.
- `cnc3_allplayers.py`: compares known addresses over three samples (2-second spacing).
- `cnc3_analyze.py`: reads PE exports/imports from target binaries and saves `cnc3_analysis.json`.
- `cnc3_report.py`: keyword-filtered summary report from `cnc3_analysis.json`.
- `cnc3_entrypoints.py`: categorized export/import review with optional demangling.

### Primary report artifacts

- `cnc3_analysis.json`: raw import/export data.
- `cnc3_report.txt`: real-time oriented keyword report.
- `cnc3_entrypoints_out.txt`: categorized entrypoint report output.
- `cnc3_human_result.txt`: sample output from human cash candidate search.

### Strategic references

- `CNC3_Process_Thread_Analysis.md`
- `CNC3_Network_Anticheat_Analysis.md`
- `CNC3_GameAPI_SinglePlayer.md`

## 3) Typical End-to-End Workflow

### Step A: Validate process access

Run `cnc3_memtest.py` first. It checks:

- process handle open (read-only)
- readable committed region enumeration
- smoke-test read from a candidate base address

Expected result shape:

- handle opened successfully
- non-zero readable region count
- successful 64-byte test read

### Step B: Discover candidate cash values

Use `cnc3_cashscan.py` for broad discovery:

- scans committed RW-like regions
- reads in chunks
- evaluates aligned 4-byte values as float
- buckets values to identify repetition patterns

Use this stage to get rough neighborhoods, not final addresses.

### Step C: Confirm liveness with deltas

Use `cnc3_cashdelta.py`:

- capture snapshot 1
- wait (income tick window)
- capture snapshot 2
- keep addresses that changed

If values do not change:

- verify game is not paused
- extend observation window
- repeat with active spending/harvesting events

### Step D: Separate likely human values

Use `cnc3_human_cash.py`:

- focuses on unique or near-unique value occurrence
- excludes known AI addresses (`KNOWN_AI`)
- uses three snapshots (`t=0s`, `t=2s`, `t=4s`)
- ranks by spend/income behavior patterns

Interpretation tip:

- negative swings often indicate purchases
- stable monotonic increases can indicate periodic income

### Step E: Cross-check known player addresses

Use `cnc3_allplayers.py`:

- quick tri-sample report over curated target addresses
- useful to compare "outlier" versus AI-like update rates

## 4) Binary and Symbol Analysis Workflow

### Generate raw symbol data

`cnc3_analyze.py` parses target binaries and writes `cnc3_analysis.json`.

Key target set in script includes:

- `CNC3.exe`
- `VistaShellSupport.dll`
- `patchw32.dll`
- `RetailExe\1.9\cnc3game.dat`
- `RetailExe\1.9\dbghelp.dll`

### Build focused reports

- `cnc3_report.py` filters names via `RT_KEYWORDS` and prints real-time relevant imports/exports.
- `cnc3_entrypoints.py` adds category grouping and optional `undname` demangling.

### Understand `cnc3_entrypoints.py`

Important elements:

- `CATEGORIES`: semantic bucket mapping for export names.
- `IMPORTANT_IAT`: selected DLL import families to inspect first.
- `demangle(sym)`: attempts `undname` and falls back to regex decode.
- `STRATEGIES`: conceptual summary for investigation paths.

Output destination:

- `cnc3_entrypoints.txt`

## 5) Reading and Interpreting Existing Outputs

### `cnc3_report.txt`

Look for:

- high-count import families in `CNC3.exe`
- patch-related exports in `patchw32.dll`
- clusters of function names aligned with your objective (timing, UI, networking)

### `cnc3_entrypoints_out.txt`

Use category headings to quickly navigate:

- game logic
- economy/resource
- UI/APT
- network/sync

This is useful for shortlisting symbols before deeper static analysis.

### `cnc3_human_result.txt`

Treat "best guess" as provisional. Validate with additional cycles:

- repeat during active gameplay events
- compare against known addresses
- watch for implausible outliers (very large or negative transitions)

## 6) Practical Patterns Used in Scripts

- Chunked `ReadProcessMemory` scans to keep memory reads manageable.
- 4-byte alignment when interpreting int32/float fields.
- Region filtering with `VirtualQueryEx` (`MEM_COMMIT`, no guard/no access pages).
- Multi-snapshot comparisons to distinguish static tables from live state.
- Address clustering by proximity to infer struct/array locality.

## 7) Known Limitations

- PID values are hardcoded in multiple scripts and require local adjustment.
- Candidate detection is heuristic and can produce false positives.
- Memory layout can vary by build/runtime session.
- Some scripts include conceptual sections not intended as direct operational playbooks.

## 8) Suggested Improvements

1. Centralize shared constants (`PID`, protection masks, chunk size) in one module.
2. Add CLI arguments for PID, ranges, and snapshot timings.
3. Persist structured scan results (JSON/CSV) for diffable runs.
4. Add a small runner script to orchestrate staged execution.
5. Add confidence scoring that combines delta stability and neighborhood features.

## 9) Minimal Command Set

Use from repository root:

```powershell
python cnc3_memtest.py
python cnc3_cashscan.py
python cnc3_cashdelta.py
python cnc3_human_cash.py
python cnc3_allplayers.py
python cnc3_analyze.py
python cnc3_report.py > cnc3_report.txt
python cnc3_entrypoints.py > cnc3_entrypoints_out.txt
```

## 10) Fast Navigation

- Index: `cnc/README.md`
- Main guide: `cnc/COMMAND_CONQUER_PROGRAMMING_GUIDE.md`
- Symbol categorization implementation: `cnc3_entrypoints.py`
- Process architecture reference: `CNC3_Process_Thread_Analysis.md`
- Network/anticheat reference: `CNC3_Network_Anticheat_Analysis.md`

