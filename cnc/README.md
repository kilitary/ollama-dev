# Command and Conquer Research Docs

This directory centralizes the Command and Conquer 3 (CNC3) reverse-engineering and tooling guides used in this repository.

## Start Here

- [Command and Conquer Programming Guide](COMMAND_CONQUER_PROGRAMMING_GUIDE.md) - End-to-end workflow for binary analysis, memory scanning, and result interpretation.

## Related Scripts

- `cnc3_memtest.py` - Read-only memory access smoke test.
- `cnc3_cashscan.py` - Broad float scan for cash-like values.
- `cnc3_cashdelta.py` - Double-snapshot delta method to find live values.
- `cnc3_human_cash.py` - Human cash candidate isolation from unique values.
- `cnc3_allplayers.py` - Multi-sample comparison across known player addresses.
- `cnc3_analyze.py` - PE export/import extraction into `cnc3_analysis.json`.
- `cnc3_report.py` - Real-time keyword report from analysis JSON.
- `cnc3_entrypoints.py` - Categorized symbol and IAT hook candidate report.

## Related Reports

- `cnc3_report.txt`
- `cnc3_entrypoints_out.txt`
- `cnc3_human_result.txt`
- `CNC3_Process_Thread_Analysis.md`
- `CNC3_Network_Anticheat_Analysis.md`
- `CNC3_GameAPI_SinglePlayer.md`

## Scope

- Primary focus: single-player research workflows.
- Primary safety posture: read-only memory inspection scripts.
- Multiplayer context and desync/anticheat behavior are documented in the linked network analysis report.

