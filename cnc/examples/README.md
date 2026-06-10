# Custom Unit + AI Example (CNC3)

This example demonstrates a legal Mod SDK style workflow for a custom unit with custom AI behavior in single-player/skirmish.

## What is included

- `shadow_striker/unit_spec.json` - Source-of-truth unit stats.
- `shadow_striker/ai_profile.json` - Rule-based AI behavior profile.
- `shadow_striker/generated/ShadowStriker.ini` - Example unit config output.
- `shadow_striker/generated/shadow_striker_ai.scr` - Example script hooks (`OnAIThink`, `OnEnemySpotted`).
- `shadow_striker/test_state.json` - Local simulation fixture.
- `validate_custom_unit.py` - Local validator/simulator (no SDK required).
- `scr_interface.py` - Python interface for parsing/generating SCR scripts.
- `test_scr_interface.py` - Test suite for SCR interface functionality.

## Behavior summary

The `ShadowStriker` AI uses three rules:

1. `RETREAT` when health < 35%.
2. `RUSH` when our combat power >= 120% of enemy power.
3. `HARASS` by default (fallback).

## Quick run

```powershell
python P:\ollama-dev\cnc\examples\validate_custom_unit.py
```

Expected result:

- Validation passes.
- `test_state.json` decisions include `RUSH`, `RETREAT`, and `HARASS` in their respective scenarios.

## Notes

- This is for offline/single-player modding patterns only.
- The generated files are examples of SDK-style assets; you can adapt naming/layout to your Mod SDK project structure.

## SCR Interface Usage

The `scr_interface.py` module provides Python bindings for CNC3 script files:

```python
from scr_interface import SCRInterface, convert_ai_profile_to_scr

# Load existing SCR script
script = SCRInterface.load_scr("path/to/script.scr")

# Generate SCR from AI profile JSON
convert_ai_profile_to_scr("ai_profile.json", "output.scr")

# Parse SCR text
scr_text = "function MyFunc(unit): IssueCommand(unit, \"Move\", null, {100, 200}) end"
script = SCRInterface.parse_scr(scr_text)
```

## Testing the SCR Interface

Run the test suite to verify SCR interface functionality:

```powershell
python P:\ollama-dev\cnc\examples\test_scr_interface.py
```

Expected result: All tests pass, demonstrating parsing, generation, and conversion capabilities.
