# Excel Symbol Scroll Effect

This project creates a visual effect in Excel with scrolling backtick symbols that are gradually replaced with comments from a CSV file.

## Features

1. **Symbol Scrolling (0-3 seconds)**: Creates randomly scrolling backtick symbols (`) moving from bottom to top
2. **Asynchronous Movement**: Each column moves at a different speed for a more dynamic effect
3. **Random Symbol Count**: Each cell contains between 0-300 backtick symbols
4. **Comment Replacement (10-20 seconds)**: After 10 seconds, random symbols are replaced with comments from comments.csv
5. **Top-to-Bottom Replacement**: Comments appear to move from top to bottom as they replace symbols

## Installation

1. Open Excel
2. Press `Alt + F11` to open the VBA editor
3. Insert a new module (`Insert > Module`)
4. Copy and paste the code from `EnhancedSymbolEffect.bas` into the module
5. Save the workbook as a macro-enabled workbook (`.xlsm`)

## Usage

1. Ensure the `comments.csv` file is located at `p:\ollama-dev\comments.csv`
2. In Excel, press `Alt + F8` to open the macro dialog
3. Select `StartEnhancedSymbolEffect` and click "Run"
4. To stop the effect early, run the `StopEnhancedEffect` macro

## How It Works

### Phase 1: Symbol Scrolling (0-3 seconds)
- Each column scrolls independently at a random speed
- New rows of backtick symbols are generated at the bottom
- Each cell contains a random number of backticks (0-300)
- Symbols move upward as new ones are added at the bottom

### Phase 2: Comment Replacement (10-20 seconds)
- After 10 seconds, the replacement effect begins
- Comments from the CSV file gradually replace the symbols
- Replacement starts from the top and moves downward
- Comments have a higher chance of appearing at the top of the sheet

## Customization

You can modify the following parameters in the code:

- **Number of columns**: Change the `For i = 1 To 10` loop in `StartEnhancedSymbolEffect`
- **Scrolling duration**: Modify the `If elapsedSeconds < 3` condition in `UpdateEnhancedEffect`
- **Replacement start time**: Change the `elapsedSeconds >= 10` condition
- **Symbol count range**: Adjust the `count = Int((300 * Rnd))` line in `ScrollSymbolsAsync`
- **Comment source**: Change the file path in `LoadCommentsFromCSV`

## Requirements

- Microsoft Excel with VBA support
- comments.csv file in the specified location

## Troubleshooting

If the effect doesn't start:
1. Ensure macros are enabled in Excel
2. Check that the comments.csv file exists at the specified path
3. Verify that the VBA code was copied correctly

If Excel becomes unresponsive:
1. Press `Ctrl + Break` to interrupt the macro
2. Run the `StopEnhancedEffect` macro to clean up timers