# Command & Conquer API Interface (CNC3)

This directory contains the programming interfaces and implementation guides for the Command & Conquer 3: Tiberium Wars API.

## Contents

- [**COMMAND_CONQUER_PROGRAMMING_GUIDE.md**](COMMAND_CONQUER_PROGRAMMING_GUIDE.md): The core API documentation, class definitions, and implementation workflow.
- [**examples/README.md**](./examples/README.md): Runnable custom unit + custom AI example pack with local validation script and SCR interface.
- [**Internal Analysis**: refer to `P:/ollama-dev/CNC3_GameAPI_SinglePlayer.md` for the underlying technical reference.]

## Quick Start

1.  Review the `Core API Classes` in the programming guide.
2.  Use the `EventManager` to register for game hooks.
3.  Implement custom logic using the `GameObject`, `Unit`, and `Building` interfaces.
4.  Test via the in-game debug console (`~`) before deployment.

## Notice

This interface is designed for **Single-Player/Skirmish** use only. Multiplayer usage will result in desynchronization and account penalties.
