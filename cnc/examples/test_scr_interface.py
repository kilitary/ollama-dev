#!/usr/bin/env python3
"""
Test script for SCR interface functionality.
Demonstrates parsing, generating, and converting SCR scripts.
"""

from scr_interface import SCRInterface, SCRFunction, SCRScript, convert_ai_profile_to_scr
from pathlib import Path
import json


def test_basic_parsing():
    """Test basic SCR parsing functionality."""
    print("Testing basic SCR parsing...")

    scr_text = """
function TestFunc(unit, target):
    hp = GetUnitProperty(unit, "Health")
    if hp < 100:
        IssueCommand(unit, "Move", null, {100, 200})
    end
end

function OnAIThink():
    units = GetUnitsByType("TestUnit")
    foreach unit in units:
        TestFunc(unit, null)
    end
end
"""

    script = SCRInterface.parse_scr(scr_text)

    assert len(script.functions) == 2
    assert "TestFunc" in script.functions
    assert "OnAIThink" in script.functions

    test_func = script.get_function("TestFunc")
    assert test_func.params == ["unit", "target"]
    assert "GetUnitProperty" in test_func.body

    print("✓ Basic parsing test passed")


def test_script_generation():
    """Test generating SCR from Python objects."""
    print("Testing SCR generation...")

    script = SCRScript()

    # Create a simple function
    func = SCRFunction("MyCustomAI", ["unit"], "IssueCommand(unit, \"Attack\", target)")
    script.add_function(func)

    # Generate SCR text
    scr_output = script.to_scr()

    # Verify it contains expected content
    assert "function MyCustomAI(unit):" in scr_output
    assert "IssueCommand(unit, \"Attack\", target)" in scr_output
    assert "end" in scr_output

    print("✓ Script generation test passed")


def test_ai_profile_conversion():
    """Test converting AI profile JSON to SCR."""
    print("Testing AI profile conversion...")

    # Create a test AI profile
    test_profile = {
        "unit": "TestStriker",
        "hooks": ["OnAIThink", "OnEnemySpotted"],
        "rules": [
            {
                "name": "retreat_low_health",
                "condition": {"kind": "health_below", "threshold": 0.3},
                "action": "RETREAT"
            },
            {
                "name": "rush_advantage",
                "condition": {"kind": "power_ratio_at_least", "ratio": 1.5},
                "action": "RUSH"
            },
            {
                "name": "default_harass",
                "condition": {"kind": "always"},
                "action": "HARASS"
            }
        ]
    }

    # Save test profile
    test_profile_path = Path("test_ai_profile.json")
    with test_profile_path.open('w') as f:
        json.dump(test_profile, f, indent=2)

    # Convert to SCR
    output_path = Path("test_generated.scr")
    convert_ai_profile_to_scr(test_profile_path, output_path)

    # Verify output exists and contains expected content
    assert output_path.exists()
    with output_path.open('r') as f:
        scr_content = f.read()

    assert "function TestStrikerOnAIThink(unit):" in scr_content
    assert "hp_ratio < 0.3" in scr_content
    assert "return \"RETREAT\"" in scr_content
    assert "return \"RUSH\"" in scr_content
    assert "return \"HARASS\"" in scr_content

    # Clean up
    test_profile_path.unlink()
    output_path.unlink()

    print("✓ AI profile conversion test passed")


def test_round_trip():
    """Test parsing generated SCR back into Python objects."""
    print("Testing round-trip parsing...")

    # Create a script
    script = SCRScript()
    func = SCRFunction("RoundTripTest", [], "print(\"Hello from SCR!\")")
    script.add_function(func)

    # Generate SCR
    scr_text = script.to_scr()

    # Parse it back
    parsed_script = SCRInterface.parse_scr(scr_text)

    # Verify
    assert len(parsed_script.functions) == 1
    parsed_func = parsed_script.get_function("RoundTripTest")
    assert parsed_func.name == "RoundTripTest"
    assert "Hello from SCR!" in parsed_func.body

    print("✓ Round-trip test passed")


def main():
    """Run all tests."""
    print("Running SCR Interface Tests...\n")

    try:
        test_basic_parsing()
        test_script_generation()
        test_ai_profile_conversion()
        test_round_trip()

        print("\n🎉 All SCR interface tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()
