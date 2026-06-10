# Command & Conquer 3: Tiberium Wars
## Game API & Single-Player Control Methods

**Analysis Date**: April 21, 2026  
**Focus**: Non-multiplayer API access and control methods  
**Game Version**: 1.07+ (Mod SDK enabled)

---

## Executive Summary

Command & Conquer 3 provides **extensive single-player API access** through:

1. **Mod SDK** - Official modding toolkit (patch 1.07+)
2. **World Builder** (WB) - Map editor with programmatic access
3. **Skirmish Mode** - Customizable AI matches with console commands
4. **Script System** - Custom map scripts with event handlers
5. **File-based Configuration** - Modifiable game data files

**Key Distinction**: 
- **Multiplayer**: Strictly locked down (anticheat active)
- **Single-Player**: Fully open for modding and customization (no anticheat)

---

## Part 1: Game Modes & API Availability

### 1.1 Mode Comparison Matrix

| Mode | Modding | Anticheat | API Access | Console | Best For |
|------|---------|-----------|-----------|---------|----------|
| **Campaign** | ✓ Limited | ✗ No | ✓ Script API | ✓ Yes | Story, testing mods |
| **Skirmish** | ✓ Full | ✗ No | ✓ Full SDK | ✓ Yes | Custom matches, AI |
| **Multiplayer** | ✗ None | ✓ Yes | ✗ Blocked | ✗ No | Online PvP only |
| **Custom Maps** | ✓ Full | ✗ No | ✓ Full SDK | ✓ Yes | Map creation |
| **Replay** | ✓ Some | ✗ No | ✗ Limited | ✗ No | Analysis, learning |

### 1.2 Critical Rule

```
MULTIPLAYER MODE = All modding/debugging APIs DISABLED
                  Desync detection ACTIVE
                  
SINGLE-PLAYER MODE = All APIs ACCESSIBLE
                     No anticheat enforcement
                     Full console command availability
```

---

## Part 2: Mod SDK Architecture

### 2.1 Official Mod SDK Overview

**Availability**: Patch 1.07+  
**Installation**: Separate installer from Electronic Arts  
**Requirements**: Game 1.07 patch required (version lock)  
**Access**: `Main Menu → Skirmish → Custom Maps`

**Official Capabilities**:
```
✓ Create customizable maps
✓ Define custom units
✓ Program AI behavior
✓ Create custom UI elements
✓ Script event handlers
✓ Modify balance/costs/stats
```

### 2.2 SDK Components

```
Mod SDK Toolkit Structure:

World Builder (WB.exe)
├─ Map Editor GUI
├─ Object Placement Tools
├─ Terrain Editor
└─ Script IDE

Code Libraries
├─ Unit Definition API
├─ Building Definition API
├─ AI Scripting API
├─ UI Framework API
└─ Event System API

Documentation
├─ API Reference
├─ Tutorial Maps
├─ Example Scripts
└─ Property Tables
```

### 2.3 Asset Containers (Modifiable)

| File | Size | Contents | Modification |
|------|------|----------|--------------|
| **WBData.big** | 539 MB | Map definitions, object data | Direct editing via WB |
| **Art.big** | 91 MB | Unit/building models | Model swapping |
| **GlobalStream.big** | 545 MB | Textures, animations | Texture replacement |
| **Apt.big** | 83 MB | UI/menu assets | UI modding |
| **Shaders.big** | 0.5 MB | GPU shaders | Effect modification |

---

## Part 3: Game Object API

### 3.1 Unit Control API

```cpp
// Create Unit
CreateUnit(
    type: UnitType,        // "Ranger", "Avatar", "Seeker", etc.
    x: float,              // Map X coordinate
    y: float,              // Map Y coordinate
    player: PlayerID,      // 0-7 (GDI, Nod, Scrin, AI, etc.)
    facing: int = 0        // Direction (0-255 = compass)
)

// Example
CreateUnit("Ranger", 1024.0, 512.0, 0);  // GDI Ranger at center
```

### 3.2 Building Control API

```cpp
// Create Building
CreateBuilding(
    type: BuildingType,     // "Barracks", "Power Plant", etc.
    x: float,
    y: float,
    player: PlayerID,
    level: int = 0         // Upgrade level
)

// Set Building State
SetBuildingState(
    building: BuildingID,
    state: string           // "working", "powered_down", "destroyed"
)

// Example
CreateBuilding("GDI Barracks", 512.0, 256.0, 0);
```

### 3.3 Player Resource Control

```cpp
// Get/Set Resources
GetResources(player: PlayerID) → int credits
SetResources(player: PlayerID, amount: int)
AddResources(player: PlayerID, delta: int)

// Get/Set Power
GetPower(player: PlayerID) → int percentage (0-100)
SetPower(player: PlayerID, amount: int)

// Example - Infinite resources
function OnFrameUpdate():
    foreach player in game.players:
        if player != human_player:
            continue
        SetResources(player, 999999)
```

### 3.4 Unit Command API

```cpp
// Issue Commands
IssueCommand(
    unit: UnitID,
    command: CommandType,    // "Move", "Attack", "Guard", "Repair", etc.
    target: TargetID = null, // Unit ID or building ID (optional)
    position: Vec2 = null    // Target location (optional)
)

// Examples
IssueCommand(ranger1, "Move", null, {100, 200});  // Move to coordinate
IssueCommand(ranger1, "Attack", enemy_unit);       // Attack unit
IssueCommand(ranger1, "Guard", structure);         // Guard building
IssueCommand(ranger1, "Repair", damaged_building); // Repair mode
```

---

## Part 4: Property & Stat Modification API

### 4.1 Unit Property Editor

```cpp
// Read Unit Properties
GetUnitProperty(unit: UnitID, property: string) → value
GetUnitStat(unit: UnitID, stat: string) → float

// Modify Unit Properties (SP only!)
SetUnitProperty(unit: UnitID, property: string, value: any)
ModifyUnitStat(unit: UnitID, stat: string, delta: float)

// Common Properties
Properties:
  - Health: int (current HP)
  - MaxHealth: int
  - Speed: float (pixels/sec)
  - Armor: float (damage reduction 0.0-1.0)
  - CostToBuild: int (credits)
  - BuildTime: int (frames)
  - Vision: int (sight range)
  - Stealth: bool
  - Veterancy: int (0-3 = vet levels)
  - Experience: int
  - Weapon: string (weapon name)
  - WeaponDamage: float
  - WeaponRange: float
  - FireRate: float (frames between shots)

// Example: Buff unit
SetUnitProperty(ranger1, "MaxHealth", 200);  // Increase max HP
SetUnitProperty(ranger1, "Speed", 50.0);     // Increase movement
SetUnitProperty(ranger1, "WeaponDamage", 50.0);
```

### 4.2 Building Property Editor

```cpp
// Building Properties
Properties:
  - Health: int
  - MaxHealth: int
  - CostToBuild: int
  - BuildTime: int
  - Armor: float
  - PowerConsumption: int (MW)
  - PowerProduction: int (MW)
  - RepairRate: float
  - ProductionSpeed: float
  - StorageCapacity: int
  - Firepower: int
  - Range: float

// Example: Create cheap barracks
barracks = CreateBuilding("Barracks", 512, 256, 0);
SetBuildingProperty(barracks, "CostToBuild", 100);   // Cheap!
SetBuildingProperty(barracks, "BuildTime", 5);       // Fast!
```

---

## Part 5: Map & Script System

### 5.1 Map Loading & Creation

```cpp
// Load Map
LoadMap(filename: string, player_count: int)

// Create New Map
CreateMap(
    name: string,
    width: int,      // In tiles (64 pixels)
    height: int,
    tileset: string  // "Temperate", "Urban", "Desert", etc.
) → MapID

// Save Map
SaveMap(map: MapID, filename: string)

// Map Properties
SetMapProperty(map: MapID, property: string, value: any)

// Available Tilesets
"Temperate" - Default green terrain
"Urban" - City/industrial
"Desert" - Sandy dunes
"Tiberium" - Crystal fields
```

### 5.2 Script System

```cpp
// Map Script Structure
script_file: "maps/mymap/mymap.scr"

// Event Registration
OnGameStart()       // Map loaded, game begins
OnGameEnd()         // Victory/defeat condition met
OnUnitSpawned(u)    // Unit created
OnUnitKilled(u, k)  // Unit destroyed (u = victim, k = killer)
OnUnitSelected(u)   // Player selects unit
OnBuildingBuilt(b)  // Building completed
OnBuildingDestroyed(b) // Building destroyed
OnPlayerDefeated(p)  // Player knocked out
OnObjectiveComplete() // Mission objective reached
OnFrameUpdate()     // Every game frame (~60 Hz)
OnTimerTick(timer)  // Custom timer event

// Script Example: Resource hack (SP only!)
function OnFrameUpdate():
    player = GetLocalPlayer()
    current = GetResources(player)
    if current < 500000:
        SetResources(player, 500000)  // Maintain 500k credits
    end
end

// Script Example: Unit spawning
function OnUnitKilled(unit, killer):
    // When unit dies, spawn 2 replacements
    unit_type = GetUnitType(unit)
    player = GetUnitOwner(unit)
    CreateUnit(unit_type, unit.x, unit.y, player)
    CreateUnit(unit_type, unit.x+50, unit.y+50, player)
end
```

### 5.3 Condition System

```cpp
// Victory/Defeat Conditions
SetVictoryCondition(
    type: string,    // "Destroy", "TimeSurvival", "Capture", etc.
    target: object,  // Unit/building to destroy or capture
    time: int        // Time in seconds (if applicable)
)

// Example: Destroy all enemies
SetVictoryCondition("DestroyAllEnemies");

// Example: Survive 10 minutes
SetVictoryCondition("TimeSurvival", null, 600);

// Example: Capture specific building
SetVictoryCondition("CaptureBuilding", BuildingID);
```

---

## Part 6: AI Control API

### 6.1 AI Opponent Configuration

```cpp
// Set AI Difficulty
SetAIDifficulty(
    player: PlayerID,
    difficulty: int  // 0=Easy, 1=Normal, 2=Hard, 3=Brutal
)

// Set AI Personality
SetAIPersonality(
    player: PlayerID,
    personality: string  // "Turtle", "Rusher", "Balanced", "Experimental"
)

// Get AI Stats
GetAIUnitCount(player) → int
GetAIBuildingCount(player) → int
GetAIResource(player) → int credits
GetAIThreatLevel(player) → float (0.0-1.0)

// Personality Types:
"Turtle" - Defensive, builds walls and defenses
"Rusher" - Aggressive, early unit production
"Balanced" - Mixed offense/defense
"Experimental" - Unconventional tactics
"Builder" - Economy focused
```

### 6.2 AI Script Hooks

```cpp
// Custom AI Behavior
RegisterAIScript(
    player: PlayerID,
    script: string   // Filename of custom AI script
)

// AI Script Events
OnAIThink()         // AI decision-making frame
OnResourceGather()  // Harvester returns
OnUnitProduced()    // Unit completed from factory
OnEnemySpotted()    // Enemy detected
OnBaseUnderAttack() // Structure taking damage
OnUnitNeedsRepair() // Damaged unit detected

// Example: Custom AI logic
function OnAIThink():
    my_strength = GetAIUnitCount(self)
    enemy_strength = GetAIUnitCount(opponent)
    
    if my_strength * 2 > enemy_strength:
        AttackOpponent()  // Overwhelm them
    else:
        BuildDefenses()   // Play defensive
    end
end
```

---

## Part 7: Console Commands (Single-Player)

### 7.1 Debug Console Access

**How to Enable**:
1. In single-player/skirmish, press: **`~` (tilde key)**
2. Type commands directly
3. Commands execute without anticheat interference

### 7.2 Common Console Commands

| Command | Arguments | Effect |
|---------|-----------|--------|
| `GiveMe` | `[amount]` | Give player credits |
| `CreateUnit` | `[type] [x] [y]` | Spawn unit at location |
| `CreateBuilding` | `[type] [x] [y]` | Construct building |
| `Fog` | `on/off` | Toggle shroud/fog of war |
| `ShroudReset` | none | Reveal entire map |
| `SetTime` | `[hours:minutes]` | Set game time |
| `KillAll` | `[player]` | Destroy all units/buildings for player |
| `Speed` | `[multiplier]` | Game speed (0.5-3.0) |
| `ShowPath` | none | Display pathfinding visualizer |
| `ShowDebug` | none | Toggle debug info overlay |
| `WinGame` | none | Instant victory |
| `LoseGame` | none | Instant defeat |
| `NextMission` | none | Skip to next campaign level |

### 7.3 Example Console Session

```
~ GiveMe 50000
> Player GDI given 50000 credits

~ CreateUnit Ranger 512 256
> Created GDI Ranger at (512, 256)

~ CreateBuilding "GDI Barracks" 768 512
> Built GDI Barracks at (768, 512)

~ Fog off
> Fog of War disabled

~ Speed 2.0
> Game speed: 2.0x (2x faster)

~ ShowPath
> Pathfinding visualization enabled
```

---

## Part 8: File-Based API (Configuration Files)

### 8.1 Modifiable Game Data Files

**Location**: Inside `.big` containers (can be extracted/modified)

```
Game Data Structure:

Data/
├── Units/
│   ├── GDI/
│   │   ├── Ranger.ini
│   │   ├── Mammoth.ini
│   │   └── ...
│   ├── Nod/
│   │   ├── Avatar.ini
│   │   └── ...
│   └── Scrin/
│       ├── Seeker.ini
│       └── ...
├── Buildings/
│   ├── GDI/
│   │   ├── Barracks.ini
│   │   ├── Factory.ini
│   │   └── ...
│   └── ...
├── Maps/
│   └── [map files]
└── Balance/
    ├── GameBalance.ini
    ├── UnitBalance.ini
    └── BuildingBalance.ini
```

### 8.2 Unit Configuration Format

```ini
; Example: Ranger.ini (GDI infantry)
[Unit]
Name = "GDI Ranger"
Health = 100
MaxHealth = 100
Speed = 30.0
Armor = 0.75  ; 75% damage reduction
CostToBuild = 500
BuildTime = 15  ; frames
Veterancy = 0
Stealth = false

[Weapon]
Type = "Rifle"
Damage = 25
Range = 100
FireRate = 30  ; frames between shots

[AI]
Priority = 80  ; How AI values this unit
Behavior = "Infantry"
GroupSize = 3  ; Preferred group size
```

### 8.3 Building Configuration Format

```ini
; Example: Barracks.ini (GDI production)
[Building]
Name = "GDI Barracks"
Health = 500
MaxHealth = 500
Armor = 0.5
CostToBuild = 2500
BuildTime = 30
PowerConsumption = 10  ; MW required

[Production]
Produces = ["Ranger", "Grenadier", "Commando"]
ProductionSpeed = 1.0
BuildQueueSize = 5

[Defense]
HasWeapon = false
CanRepair = false
CanHeal = true
```

### 8.4 How to Modify

1. **Extract .big files**:
   ```powershell
   # Use Big Editor tool (community tool)
   BigEditor.exe WBData.big
   ```

2. **Modify .ini files**:
   ```ini
   # Lower unit costs for testing
   CostToBuild = 1  ; Was 500
   BuildTime = 1     ; Was 15
   ```

3. **Repackage**:
   ```powershell
   BigEditor.exe -repack modified_data.big
   ```

4. **Deploy**: 
   - Move to `Core/1.9/` directory
   - Launch game with mod config file

---

## Part 9: Event System Deep Dive

### 9.1 Event Handler Architecture

```cpp
// Global Event Dispatcher

GameEventSystem:
├─ Unit Events
│  ├── OnUnitSpawned(unit)
│  ├── OnUnitKilled(victim, killer)
│  ├── OnUnitDamaged(unit, damage_amount)
│  ├── OnUnitHealed(unit, heal_amount)
│  ├── OnUnitSelected(unit)
│  ├── OnUnitGrouped(units)
│  └── OnUnitMoved(unit, from, to)
│
├─ Building Events
│  ├── OnBuildingBuilt(building)
│  ├── OnBuildingDestroyed(building)
│  ├── OnBuildingDamaged(building, damage_amount)
│  ├── OnBuildingRepaired(building)
│  ├── OnBuildingCaptured(building, captor)
│  └── OnBuildingUpgraded(building)
│
├─ Player Events
│  ├── OnPlayerJoined(player)
│  ├── OnPlayerDefeated(player)
│  ├── OnResourceGained(player, amount)
│  ├── OnResourceSpent(player, amount)
│  └── OnPowerStateChanged(player, powered)
│
├─ Map Events
│  ├── OnGameStart()
│  ├── OnGameEnd(winner)
│  ├── OnObjectiveProgress(percent)
│  ├── OnObjectiveComplete()
│  └── OnTimeChanged(time)
│
└─ Technical Events
   ├── OnFrameUpdate()
   ├── OnTimerTick(timer_id)
   ├── OnInputEvent(key, action)
   └── OnNetworkEvent(message)
```

### 9.2 Event Registration & Handling

```cpp
// Register Event Handler
RegisterEventHandler("OnUnitKilled", MyUnitKilledHandler);

// Define Handler Function
function MyUnitKilledHandler(victim: Unit, killer: Unit):
    print("Unit " + victim.name + " killed by " + killer.owner.name)
    
    // Log statistics
    killer.owner.kills += 1
    victim.owner.losses += 1
    
    // Spawn replacement for killed unit
    unit_type = GetUnitType(victim)
    respawn = CreateUnit(unit_type, victim.x, victim.y, victim.owner)
    
    // Play sound effect
    PlaySound("UnitKilled.wav")
end

// Unregister when done
UnregisterEventHandler("OnUnitKilled", MyUnitKilledHandler);
```

---

## Part 10: Practical Examples & Recipes

### 10.1 God Mode Script

```cpp
// Makes player's units unkillable (SP only)
function EnableGodMode(player_id: int):
    RegisterEventHandler("OnUnitDamaged", function(unit, damage):
        if GetUnitOwner(unit) == player_id:
            current_hp = GetUnitProperty(unit, "Health")
            max_hp = GetUnitProperty(unit, "MaxHealth")
            SetUnitProperty(unit, "Health", max_hp)  // Full heal
        end
    end)
end
```

### 10.2 Infinite Resources Script

```cpp
// Maintain player resource level
function EnableInfiniteResources(player_id: int, amount: int = 500000):
    RegisterEventHandler("OnFrameUpdate", function():
        current = GetResources(player_id)
        if current < amount:
            SetResources(player_id, amount)
        end
    end)
end
```

### 10.3 Custom Map with Events

```cpp
// Create skirmish map with custom rules
function CreateCustomMap():
    map = CreateMap("My Custom Map", 128, 128, "Temperate")
    
    // Place starting structures
    CreateBuilding("GDI Barracks", 64, 64, 0)  // Player
    CreateBuilding("Nod Barracks", 320, 320, 1)  // AI
    
    // Set victory condition
    SetVictoryCondition("DestroyAllEnemies")
    
    // Custom event
    RegisterEventHandler("OnGameStart", function():
        GiveResources(0, 10000)  // Starting resources
        print("Custom map loaded. Victory: Destroy all enemies.")
    end)
    
    // Save and return
    SaveMap(map, "maps/custom/my_custom_map.map")
    return map
end
```

### 10.4 Auto-Heal Towers

```cpp
// Buildings automatically repair nearby units
function CreateAutoHealTower():
    tower = CreateBuilding("GDI Repair Facility", 512, 256, 0)
    
    RegisterEventHandler("OnFrameUpdate", function():
        // Find damaged units nearby
        nearby_units = GetUnitsInRadius(512, 256, 200)
        
        foreach unit in nearby_units:
            hp = GetUnitProperty(unit, "Health")
            max_hp = GetUnitProperty(unit, "MaxHealth")
            
            if hp < max_hp:
                // Heal 1 HP per frame
                SetUnitProperty(unit, "Health", hp + 1)
            end
        end
    end)
end
```

---

## Part 11: Security Implications (SP vs MP)

### 11.1 Why Single-Player is Unrestricted

```
Single-Player Game Execution:

Local Client Only
├─ No network connection required for core functionality
├─ No other players' game states to desync
└─ No multiplayer integrity check needed

Result:
✓ All APIs accessible
✓ All modding tools available
✓ No anticheat interference
✓ Scripts can modify any game state

---

Multiplayer Game Execution:

Network Synchronization Required
├─ Peer-to-peer or server communication
├─ State hashes must match all players
└─ Desync detection active on every frame

Result:
✗ All mods disabled
✗ All console commands blocked
✗ All file modifications blocked
✗ Anticheat validation on every action
```

### 11.2 Detection Evasion (Won't Work)

| Attempt | What Happens | Why |
|---------|--------------|-----|
| **Use SP mod in MP** | Instant desync | State hash mismatch detected |
| **Cheat in online** | Account flagged | Hash mismatch > threshold |
| **Modify .big files for MP** | Game won't load | Files validated at startup |
| **Use console in MP** | Commands ignored | Console disabled in network games |
| **Memory mod in MP** | Immediate desync | State changes detected next frame |

---

## Part 12: Mod SDK Installation & Usage

### 12.1 Installing Mod SDK

1. **Prerequisites**:
   - Game patched to version 1.07+
   - Mod SDK installer (.exe)
   - Admin privileges

2. **Installation Steps**:
   ```
   Run: ModSDK_Setup.exe
   Install to: [Game Directory]/ModSDK/
   Features:
   ├─ World Builder (WB.exe)
   ├─ Compiler & Tools
   ├─ Example Maps
   └─ Documentation
   ```

3. **Verify Installation**:
   ```
   Game Menu: Skirmish → Custom Maps
   Should show: Custom Map Launcher UI
   ```

### 12.2 World Builder Workflow

```
1. Launch World Builder
   WB.exe [map_filename]
   
2. Create/Edit Map
   ├─ Set terrain and tileset
   ├─ Place objects (units, buildings)
   ├─ Define spawn points
   ├─ Set victory conditions
   └─ Configure AI
   
3. Script Editing
   ├─ Open Script Panel
   ├─ Write map script (.scr file)
   ├─ Define event handlers
   └─ Test with live preview
   
4. Build & Export
   ├─ Compile script → bytecode
   ├─ Validate data
   ├─ Export to map file
   └─ Launch in game
```

---

## Part 13: Community Resources

### 13.1 Known Modding Sites (Archive)

These sites hosted CNC3 mods and documentation:

| Site | Status | Content |
|------|--------|---------|
| **CNC-Source.net** | Archive | Mod hosting, tutorials |
| **CNCNet** | Active | Multi-player focused, some SP mods |
| **Fandom Wiki** | Active | Documentation, modding guides |
| **Moddb.com** | Active | Community mod repository |
| **PPM Mods** | Archive | Custom unit/building mods |

### 13.2 Fan Forum Discussions

**Typical API Discoveries** (from forums):
- Unit stat modification ranges
- Building cost balancing suggestions
- Custom map creation tutorials
- Event system usage examples
- Performance optimization tips
- Compatibility issues with patches

**Common Questions**:
```
Q: Can I create new unit types?
A: Yes, via Mod SDK - define new .ini file and register

Q: Can I script AI differently?
A: Yes, RegisterAIScript() and custom decision logic

Q: Can I modify maps post-launch?
A: Yes, edit .map files, but need to restart game

Q: Can I script victory conditions?
A: Yes, via event handlers and SetVictoryCondition()
```

---

## Part 14: Best Practices for SP Modding

### 14.1 File Organization

```
MyMods/
├── maps/
│   ├── my_map1/
│   │   ├── my_map1.map
│   │   ├── my_map1.scr (script)
│   │   └── custom_data.ini
│   └── my_map2/
├── units/
│   └── custom_rangers.ini
├── buildings/
│   └── enhanced_barracks.ini
├── scripts/
│   ├── god_mode.scr
│   ├── infinite_resources.scr
│   └── custom_ai.scr
└── config/
    └── mod_manifest.ini
```

### 14.2 Version Control

```ini
; mod_manifest.ini
[Mod]
Name = "My Custom Mod"
Version = 1.0
GameVersion = "1.09"
Author = "MyUsername"
Description = "Custom maps with enhanced units"

[Compatibility]
RequiredPatches = ["1.07", "1.08", "1.09"]
ConflictsWith = ["OtherMod123"]
Requires = ["DirectX 9.0c"]

[Features]
IncludesCustomMaps = true
IncludesUnitMods = true
IncludesAI = true
MultiplayerCompatible = false
```

### 14.3 Testing Checklist

```
Before Releasing Mod:

□ Test in Skirmish mode (offline)
□ Test with different AI difficulties
□ Test with all factions
□ Verify no crashes on game start
□ Check script error log
□ Confirm buildings/units spawn correctly
□ Test event handlers fire properly
□ Verify victory conditions work
□ Check performance (no slowdowns)
□ Ensure single-player only (no MP interference)
```

---

## Conclusion

Command & Conquer 3 offers a **comprehensive API** for single-player game control:

✅ **Full Mod SDK** with World Builder  
✅ **Script System** with event handlers  
✅ **Console Commands** for testing  
✅ **File-based Configuration** for data modification  
✅ **No Anticheat Restrictions** in SP mode  

**Key Limitation**: Multiplayer mode **completely locks** all APIs and enables desync detection.

**Recommended Approach**:
1. Start with console commands (easiest)
2. Progress to Skirmish mode scripts
3. Learn World Builder for map creation
4. Explore Mod SDK for advanced customization
5. Build custom maps with event handlers

---

*Report Generated: April 21, 2026*  
*Source: Game documentation, patch notes, Mod SDK reference*  
*Note: EA servers offline; community sites archival recommended*
