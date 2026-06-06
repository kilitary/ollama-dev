"""
CNC3 SCR Interface - Python bindings for Command & Conquer 3 script files (.scr)

This module provides a Python interface for working with CNC3 map scripts and AI behavior.
Supports parsing, generating, and validating SCR files used in single-player/skirmish modes.
"""

import re
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import json


class SCRFunction:
    """Represents a single SCR function with name, parameters, and body."""

    def __init__(self, name: str, params: List[str] = None, body: str = ""):
        self.name = name
        self.params = params or []
        self.body = body.strip()

    def to_scr(self) -> str:
        """Convert to SCR format."""
        param_str = ", ".join(self.params) if self.params else ""
        return f"function {self.name}({param_str}):\n{self._indent_body()}\nend"

    def _indent_body(self) -> str:
        """Indent function body lines."""
        lines = self.body.split('\n')
        indented = []
        for line in lines:
            if line.strip():
                indented.append(f"    {line}")
            else:
                indented.append("")
        return '\n'.join(indented)


class SCRScript:
    """Represents a complete SCR script file."""

    def __init__(self):
        self.functions: Dict[str, SCRFunction] = {}
        self.global_vars: Dict[str, Any] = {}

    def add_function(self, func: SCRFunction):
        """Add a function to the script."""
        self.functions[func.name] = func

    def get_function(self, name: str) -> Optional[SCRFunction]:
        """Get a function by name."""
        return self.functions.get(name)

    def to_scr(self) -> str:
        """Convert entire script to SCR format."""
        lines = []
        for func in self.functions.values():
            lines.append(func.to_scr())
            lines.append("")  # Empty line between functions
        return '\n'.join(lines).strip()


class SCRInterface:
    """Main interface for working with SCR scripts."""

    # SCR syntax patterns
    FUNC_PATTERN = re.compile(r'function\s+(\w+)\s*\(([^)]*)\)\s*:\s*(.*?)\s*end', re.DOTALL)
    COMMENT_PATTERN = re.compile(r'//.*?$', re.MULTILINE)

    @classmethod
    def parse_scr(cls, scr_text: str) -> SCRScript:
        """Parse SCR text into SCRScript object."""
        # Remove comments
        clean_text = cls.COMMENT_PATTERN.sub('', scr_text)

        script = SCRScript()

        # Find all functions
        for match in cls.FUNC_PATTERN.finditer(clean_text):
            func_name = match.group(1)
            param_str = match.group(2).strip()
            body = match.group(3).strip()

            # Parse parameters
            params = [p.strip() for p in param_str.split(',') if p.strip()] if param_str else []

            # Clean up body indentation
            body_lines = body.split('\n')
            cleaned_body = []
            for line in body_lines:
                # Remove leading indentation (assuming 4 spaces)
                if line.startswith('    '):
                    cleaned_body.append(line[4:])
                else:
                    cleaned_body.append(line)

            func = SCRFunction(func_name, params, '\n'.join(cleaned_body))
            script.add_function(func)

        return script

    @classmethod
    def load_scr(cls, file_path: Union[str, Path]) -> SCRScript:
        """Load SCR script from file."""
        path = Path(file_path)
        with path.open('r', encoding='utf-8') as f:
            content = f.read()
        return cls.parse_scr(content)

    @classmethod
    def save_scr(cls, script: SCRScript, file_path: Union[str, Path]):
        """Save SCR script to file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as f:
            f.write(script.to_scr())

    @classmethod
    def generate_ai_script(cls, ai_profile: Dict[str, Any]) -> SCRScript:
        """Generate SCR script from AI profile JSON."""
        script = SCRScript()

        unit_name = ai_profile.get('unit', 'UnknownUnit')
        rules = ai_profile.get('rules', [])

        # Generate main AI think function
        think_func = cls._generate_ai_think_function(unit_name, rules)
        script.add_function(think_func)

        # Generate event hook functions
        for hook in ai_profile.get('hooks', []):
            if hook == 'OnAIThink':
                # Already added
                continue
            elif hook == 'OnEnemySpotted':
                spotted_func = cls._generate_enemy_spotted_function(unit_name)
                script.add_function(spotted_func)

        return script

    @classmethod
    def _generate_ai_think_function(cls, unit_name: str, rules: List[Dict]) -> SCRFunction:
        """Generate the main AI think function."""
        body_lines = []

        # Health ratio calculation
        body_lines.append("hp_ratio = GetUnitProperty(unit, \"Health\") / GetUnitProperty(unit, \"MaxHealth\")")
        body_lines.append("my_power = GetAIUnitCount(GetUnitOwner(unit)) * 10")
        body_lines.append("enemy_power = GetEnemyUnitCount(GetUnitOwner(unit)) * 10")
        body_lines.append("")

        # Generate rule conditions
        for rule in rules:
            condition = rule.get('condition', {})
            action = rule.get('action', 'HARASS')
            kind = condition.get('kind')

            if kind == 'health_below':
                threshold = condition.get('threshold', 0.5)
                body_lines.append(f"if hp_ratio < {threshold}:")
                body_lines.append(f"    IssueCommand(unit, \"Move\", null, GetSafeRetreatPoint(unit))")
                body_lines.append(f"    return \"{action}\"")
                body_lines.append("end")
                body_lines.append("")

            elif kind == 'power_ratio_at_least':
                ratio = condition.get('ratio', 1.0)
                body_lines.append(f"if enemy_power > 0 and (my_power / enemy_power) >= {ratio}:")
                body_lines.append(f"    target = GetHighestValueEnemyTarget(unit)")
                body_lines.append(f"    IssueCommand(unit, \"Attack\", target)")
                body_lines.append(f"    return \"{action}\"")
                body_lines.append("end")
                body_lines.append("")

            elif kind == 'always':
                body_lines.append("target = GetNearestHarassTarget(unit)")
                body_lines.append("IssueCommand(unit, \"Attack\", target)")
                body_lines.append(f"return \"{action}\"")
                break  # Always rule should be last

        body = '\n'.join(body_lines)
        return SCRFunction(f"{unit_name}OnAIThink", ["unit"], body)

    @classmethod
    def _generate_enemy_spotted_function(cls, unit_name: str) -> SCRFunction:
        """Generate OnEnemySpotted hook."""
        body_lines = [
            f"if GetUnitType(unit) == \"{unit_name}\":",
            f"    {unit_name}OnAIThink(unit)",
            "end"
        ]
        body = '\n'.join(body_lines)
        return SCRFunction("OnEnemySpotted", ["unit", "enemy"], body)

    @classmethod
    def generate_global_ai_think(cls, unit_name: str) -> SCRFunction:
        """Generate global OnAIThink function for all units of this type."""
        body_lines = [
            f"units = GetUnitsByType(\"{unit_name}\")",
            "foreach unit in units:",
            f"    {unit_name}OnAIThink(unit)",
            "end"
        ]
        body = '\n'.join(body_lines)
        return SCRFunction("OnAIThink", [], body)


def convert_ai_profile_to_scr(ai_profile_path: Union[str, Path], output_path: Union[str, Path]):
    """Convert AI profile JSON to SCR script file."""
    with open(ai_profile_path, 'r', encoding='utf-8') as f:
        ai_profile = json.load(f)

    script = SCRInterface.generate_ai_script(ai_profile)

    # Add global think function
    unit_name = ai_profile.get('unit', 'UnknownUnit')
    global_think = SCRInterface.generate_global_ai_think(unit_name)
    script.add_function(global_think)

    SCRInterface.save_scr(script, output_path)
    print(f"Generated SCR script: {output_path}")


# Example usage
if __name__ == "__main__":
    # Example: Convert the ShadowStriker AI profile to SCR
    ai_profile_path = Path(__file__).parent / "shadow_striker" / "ai_profile.json"
    output_path = Path(__file__).parent / "shadow_striker" / "generated" / "shadow_striker_ai_py.scr"

    if ai_profile_path.exists():
        convert_ai_profile_to_scr(ai_profile_path, output_path)
    else:
        print("AI profile not found. Run from examples/ directory.")

