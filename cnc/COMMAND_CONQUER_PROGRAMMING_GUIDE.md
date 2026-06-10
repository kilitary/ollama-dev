# Command & Conquer Programming API Interface Implementation Guide

This document outlines the programming API interface for Command & Conquer 3 (CNC3), based on the official Mod SDK and internal game logic for single-player environments.

## 1. Core API Classes

### 1.1 `GameObject` API
Base class for all units and buildings.

```cpp
class GameObject {
public:
    virtual UnitID GetID() = 0;
    virtual string GetName() = 0;
    virtual PlayerID GetOwner() = 0;
    virtual Vec2 GetPosition() = 0;
    
    // Property access
    virtual float GetProperty(string name) = 0;
    virtual void SetProperty(string name, float value) = 0;
    
    // Actions
    virtual void IssueCommand(CommandType cmd, TargetID target = 0, Vec2 pos = {0,0}) = 0;
    virtual void Destroy() = 0;
};
```

### 1.2 `Unit` API (Inherits GameObject)
Specific to mobile units.

```cpp
class Unit : public GameObject {
public:
    virtual UnitType GetUnitType() = 0;
    virtual void SetVeterancy(int level) = 0; // 0-3
    virtual bool IsStealthed() = 0;
    virtual void SetSpeed(float pixelsPerSec) = 0;
};
```

### 1.3 `Building` API (Inherits GameObject)
Specific to stationary structures.

```cpp
class Building : public GameObject {
public:
    virtual BuildingType GetBuildingType() = 0;
    virtual void SetWorkingState(bool active) = 0;
    virtual void StartProduction(UnitType type) = 0;
    virtual float GetProductionProgress() = 0;
};
```

## 2. Global Game Services

### 2.1 `PlayerManager`
Controls player resources and AI personality.

```cpp
class PlayerManager {
public:
    static int GetCredits(PlayerID id);
    static void AddCredits(PlayerID id, int amount);
    static void SetAIDifficulty(PlayerID id, int level); // 0=Easy, 3=Brutal
    static void SetAIPersonality(PlayerID id, string personality);
};
```

### 2.2 `EventManager`
Registers hooks for game events.

```cpp
class EventManager {
public:
    static void Register(string eventName, function<void(EventData)> callback);
    static void Unregister(string eventName, function<void(EventData)> callback);
    
    // Common Events:
    // "OnUnitKilled"
    // "OnBuildingBuilt"
    // "OnFrameUpdate"
};
```

## 3. Implementation Workflow

1.  **SDK Setup**: Ensure game version 1.07+ is installed.
2.  **Manifest Creation**: Define your mod metadata in `mod_manifest.ini`.
3.  **Data Modification**: Extend `.ini` files for custom unit stats.
4.  **Scripting**: Use `.scr` files for map-specific logic.
5.  **Deployment**: Compile using the Mod SDK toolkit and place in the game directory.

## 4. Safety and Restrictions

*   **Anticheat**: All API features are strictly limited to single-player and custom skirmish modes with friends (if mods match).
*   **Desync**: Any unauthorized memory modification in multiplayer will trigger an immediate desync.
*   **Console**: The debug console (`~`) is the primary interface for testing live API calls.

