# Update NirSoft Script Improvements

> **Latest Update (2026-02-13)**: Added comprehensive operation tracking, detailed statistics, and real-time operation visibility

## Changes Made

### 1. Timestamps with Seconds
- Added `get_timestamp()` function that returns current time in `HH:MM:SS` format
- All log messages now include timestamps at the beginning

### 2. Icons & Systemized Messages
All messages now use categorized icons and color-coded formatting:

- 🌐 **NETWORK** (green) - Network operations (downloads, fetches)
- ⬇️ **DOWNLOAD** (green) - File downloads
- ⬆️ **UPLOAD** (green) - Upload operations
- 💾 **WRITE** (magenta) - File write operations
- 📂 **UNPACK** (cyan) - Extraction operations
- 📦 **PACK** (cyan) - Packing operations
- 🔍 **SEARCH** (blue) - File/executable searches
- 🔧 **ANALYZE** (blue) - PE file analysis
- ℹ️ **INFO** (blue) - General information
- 📈 **PROGRESS** (blue) - Progress updates
- 🗑️ **CLEANUP/DELETE** (yellow) - Cleanup operations
- ✅ **COMPLETE/FOUND** (green) - Success messages
- ❌ **ERROR** (red) - Error messages
- 🔒 **ERROR** (red) - Password/security errors
- 📊 **STATS** (cyan) - Statistics reports

### 3. Dynamic Download Stats ✨ NEW

Enhanced `Stats` class with comprehensive real-time tracking:
- **Bytes Downloaded** - Total network download traffic with real-time updates
- **Bytes Uploaded** - Total network upload traffic (placeholder for future use)
- **Files Packed** - Number of files written/saved
- **Files Unpacked** - Number of extraction operations
- **x86/x64 Tracking** - Count of 32-bit and 64-bit executables
- **Download Speed** - Instantaneous and smoothed average speed calculation
- **ETA Calculation** - Estimates time remaining based on progress
- **Current File Tracking** - Monitors the file currently being downloaded

### 4. Real-Time Progress Bars ✨ NEW
Downloads now feature beautiful rich progress bars showing:
- 🔄 **Spinner Animation** - Visual indicator of activity
- 📊 **Progress Bar** - Visual representation of download progress (40 chars wide)
- 📥 **Download Size** - Current/Total size (e.g., "12.5/25.0 MB")
- ⚡ **Transfer Speed** - Real-time speed (e.g., "2.5 MB/s")
- ⏱️ **Time Remaining** - ETA for current download

### 5. Speed Monitoring ✨ NEW
- **Instantaneous Speed**: Current download rate in B/s, KB/s, or MB/s
- **Average Speed**: Smoothed over last 10 samples to reduce fluctuations
- **Auto-formatting**: Intelligent unit selection based on speed

### 6. Automatic Stats Reporting
- Stats are automatically reported every 2 seconds during execution
- Thread-safe implementation using locks
- Shows real-time download speed, bytes, and pack/unpack counts
- Format: `⬇️ 125.45 MB @ 2.35 MB/s | 📦 Packed: 45 📂 Unpacked: 90 | 🖥️ x86: 30 💻 x64: 15`

### 7. Enhanced Statistics Display ✨ NEW
The `generate_live_table()` method creates rich formatted panels:
```
📊 Download Statistics
━━━━━━━━━━━━━━━━━━━━━━━━
📥 Downloaded:   125.45 MB
⚡ Speed:        2.35 MB/s
⏱️  Elapsed:     2m 15s
⏳ ETA:          3m 45s
📦 Packed:       45
📂 Unpacked:     90
🖥️  x86:         30
💻 x64:          15
📊 Progress:     45/100 (45.0%)
```

### 8. Final Stats Summary
At the end of execution, a comprehensive summary is displayed:
- Total MB downloaded
- Total MB uploaded
- Total files packed/unpacked
- Total x86/x64 files processed
- Overall execution time

## Features

### Thread Safety
- Used `threading.Lock()` to ensure stats are thread-safe
- Prevents race conditions if script is ever extended with parallel processing

### Rich Console Output
- Leverages the `rich` library for beautiful, colorful terminal output
- Better readability and visual organization of logs
- Consistent formatting across all messages

### Progress Tracking
- Clear visibility into what the script is doing at each step
- Percentage completion with file and total size tracking
- Real-time stats updates every 2 seconds

## Example Output

```
14:23:15 🌐 NETWORK Fetching pad links list...
14:23:16 ℹ️  INFO Found 250 links to process
14:23:16 🗑️  CLEANUP Removing 3 previous temporary directories...
14:23:17 🌐 NETWORK Fetching metadata: https://...
14:23:18 ⬇️  DOWNLOAD https://www.nirsoft.net/utils/nircmd.zip
14:23:19 📊 STATS ⬇️  2.34 MB ⬆️  0.00 MB | 📦 Packed: 5 📂 Unpacked: 4
14:23:19 💾 WRITE Writing 524288 bytes to h:\12345\67890.zip
14:23:20 💾 WRITE Wrote 524288 bytes
14:23:21 📈 PROGRESS [1/250] (0.40%) | File: 0.50 MB | Total: 0.50 MB
14:23:21 📂 UNPACK Extracting h:\12345\67890.zip...
14:23:22 🔍 SEARCH Looking for executable in ['nircmd.zip']
14:23:22 ℹ️  INFO Package name: nircmd
14:23:22 ✅ FOUND Executable: h:\12345\nircmd.exe
14:23:23 🔧 ANALYZE Machine type: 34404 (0x8664)
14:23:23 ℹ️  INFO Detected x64 architecture
14:23:24 📂 UNPACK Extracting to final destination: T:\!power-tools\NirLauncher\NirSoft\x64\nircmd
14:23:25 ✅ COMPLETE Successfully processed nircmd
14:23:27 📊 STATS ⬇️  4.67 MB ⬆️  0.00 MB | 📦 Packed: 10 📂 Unpacked: 9
...
15:45:30 📊 FINAL STATS
15:45:30 ⬇️  Total Downloaded: 125.67 MB
15:45:30 ⬆️  Total Uploaded: 0.00 MB
15:45:30 📦 Files Packed: 250
15:45:30 📂 Files Unpacked: 500
15:45:30 ✅ ALL DONE!
```

## Benefits

1. **Better Monitoring** - Easy to track what the script is doing at any moment
2. **Performance Metrics** - Real-time network and operation statistics
3. **Debugging** - Timestamps make it easier to identify bottlenecks
4. **Professional Output** - Clean, organized, and visually appealing logs

---

## Enhanced Operation Tracking (2026-02-13 Update)

### New Features

#### 📊 Comprehensive Statistics Tracking

**New Metrics Added:**
- `http_requests` - Total HTTP requests made
- `metadata_fetched` - Number of package metadata files downloaded
- `pe_analyzed` - PE (Portable Executable) files analyzed
- `temp_dirs_created` - Temporary directories created
- `temp_dirs_cleaned` - Temporary directories cleaned up
- `files_failed` - Files that failed to process
- `total_errors` - Total errors encountered
- `operations_history` - Last 20 operations with timing data

**Current Operation Tracking:**
- `current_operation` - Name of the operation being executed
- `current_package_name` - Package currently being processed
- `current_url` - URL being accessed
- `operation_start_time` - When the operation started (for duration calculation)

#### 🎨 Enhanced Visual Reports

**Formatted Status Panels:**
The script now displays rich formatted panels with:

```
╭─ 📊 STATUS REPORT - 12:35:00 ────────────────────╮
│          ═══ CURRENT OPERATION ═══                │
│  🔄 Operation:  Downloading package 15/245        │
│  📦 Package:    advancedrun                       │
│  🔗 URL:        https://www.nirsoft.net/utils...  │
│                                                   │
│          ═══ STATISTICS ═══                       │
│  📥 Downloaded:  45.23 MB @ 2.34 MB/s            │
│  📦 Packed:      15                               │
│  📂 Unpacked:    30                               │
│  🖥️ x86:         8                                │
│  💻 x64:         7                                │
│                                                   │
│          ═══ OPERATIONS ═══                       │
│  🌐 HTTP Requests:        31                      │
│  📋 Metadata:             15                      │
│  🔍 PE Analyzed:          15                      │
│  📁 Temp Created:         15                      │
│  🗑️ Temp Cleaned:         5                       │
│                                                   │
│          ═══ ERRORS ═══                           │
│  ❌ Failed:      2                                │
│  ⚠️ Errors:      3                                │
╰───────────────────────────────────────────────────╯
```

#### 🔍 Tracked Operations

Each of these operations is now tracked with start/end times and included in statistics:

1. **Fetch PAD links** - Initial network request for package list
2. **Cleanup** - Removing old temporary directories
3. **Fetch metadata** - Downloading package metadata (per package)
4. **Download package** - Downloading the actual package file
5. **Create temp directory** - Creating temporary extraction directory
6. **Write archive** - Writing downloaded content to disk
7. **Extract archive** - Initial extraction to temp directory
8. **Search executable** - Finding the main executable in extracted files
9. **Analyze PE** - Analyzing portable executable for architecture detection
10. **Final extraction** - Extracting to final destination directory

#### 📈 New Stat Methods

```python
# Error tracking
stats.add_failed()       # Failed file counter
stats.add_error()        # Total error counter

# Operation tracking
stats.add_http_request()      # HTTP request counter
stats.add_metadata_fetch()    # Metadata fetch counter
stats.add_pe_analysis()       # PE analysis counter
stats.add_temp_dir_created()  # Temp dir creation counter
stats.add_temp_dir_cleaned()  # Temp dir cleanup counter

# Operation context
stats.set_operation(operation, package_name, url)  # Set current operation
stats.complete_operation(operation)                # Complete and log duration
```

#### 🎯 Enhanced Error Handling

All error paths now:
- Increment appropriate error counters
- Complete the current operation tracking
- Log detailed error information with context
- Continue processing where appropriate

**Example:**
```python
except Exception as e:
    console.print(f"{get_timestamp()} [red]❌ ERROR[/red] Failed: {e}")
    stats.add_error()
    stats.add_failed()
    stats.complete_operation("Download package")
    continue
```

#### 💡 Real-Time Visibility

The enhanced logging provides:
- **What**: See exactly which operation is running
- **When**: Track how long each operation takes
- **Where**: Know which package and URL is being processed
- **How Much**: Comprehensive counters for all activities
- **How Well**: Error rates and success metrics

#### 🔧 Technical Implementation

**Stats Class Enhancements:**
- Thread-safe operation tracking with locks
- Operation history with automatic pruning (keeps last 20)
- Duration calculation for each operation
- Formatted panel generation with Rich library

**Report Format:**
- Auto-updates every 2 seconds during processing
- Sections clearly separated with visual dividers
- Icons for quick visual scanning
- Truncated URLs for readability
- Error section only shown when errors exist

### Usage Impact

**Before:**
```
14:23:19 📊 STATS ⬇️  2.34 MB | 📦 Packed: 5 📂 Unpacked: 4
```

**After:**
```
╭─ 📊 STATUS REPORT - 14:23:19 ───────────────────╮
│          ═══ CURRENT OPERATION ═══               │
│  🔄 Operation:  Downloading package 5/245        │
│  📦 Package:    nircmd                           │
│  🔗 URL:        https://www.nirsoft.net/...      │
│          ═══ STATISTICS ═══                      │
│  📥 Downloaded:  2.34 MB @ 1.2 MB/s             │
│  ...complete detailed breakdown...              │
╰──────────────────────────────────────────────────╯
```

### Benefits of Enhanced Logging

1. **Complete Transparency** - See every operation with full context
2. **Performance Monitoring** - Identify slow operations and bottlenecks
3. **Resource Tracking** - Monitor temp directory usage and cleanup
4. **Error Analysis** - Comprehensive error tracking with context
5. **Network Visibility** - Track all HTTP requests and bandwidth usage
6. **Progress Estimation** - Better ETAs based on actual operation timings

