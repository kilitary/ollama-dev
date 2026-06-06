import json
from pathlib import Path


BASE_DIR = Path(__file__).parent
UNIT_SPEC_PATH = BASE_DIR / "shadow_striker" / "unit_spec.json"
AI_PROFILE_PATH = BASE_DIR / "shadow_striker" / "ai_profile.json"
TEST_STATE_PATH = BASE_DIR / "shadow_striker" / "test_state.json"


REQUIRED_UNIT_FIELDS = ["unit_name", "faction", "base_unit", "properties"]
REQUIRED_PROPERTY_FIELDS = [
    "Health",
    "MaxHealth",
    "Speed",
    "Armor",
    "CostToBuild",
    "BuildTime",
    "WeaponDamage",
    "WeaponRange",
    "FireRate",
]
VALID_ACTIONS = {"RETREAT", "RUSH", "HARASS"}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_unit_spec(unit_spec: dict) -> list:
    errors = []
    for field in REQUIRED_UNIT_FIELDS:
        if field not in unit_spec:
            errors.append(f"unit_spec missing field: {field}")

    properties = unit_spec.get("properties", {})
    for field in REQUIRED_PROPERTY_FIELDS:
        if field not in properties:
            errors.append(f"unit_spec.properties missing field: {field}")

    health = properties.get("Health", 0)
    max_health = properties.get("MaxHealth", 0)
    if health <= 0 or max_health <= 0:
        errors.append("Health and MaxHealth must be > 0")
    if health > max_health:
        errors.append("Health must be <= MaxHealth")

    return errors


def validate_ai_profile(ai_profile: dict) -> list:
    errors = []
    rules = ai_profile.get("rules", [])
    if len(rules) < 1:
        errors.append("ai_profile.rules must include at least one rule")
        return errors

    for idx, rule in enumerate(rules):
        action = rule.get("action")
        if action not in VALID_ACTIONS:
            errors.append(f"rule[{idx}] action must be one of {sorted(VALID_ACTIONS)}")

        condition = rule.get("condition", {})
        kind = condition.get("kind")
        if kind not in {"health_below", "power_ratio_at_least", "always"}:
            errors.append(f"rule[{idx}] condition.kind invalid: {kind}")

    return errors


def select_action(ai_profile: dict, health_ratio: float, our_power: float, enemy_power: float) -> str:
    for rule in ai_profile.get("rules", []):
        condition = rule.get("condition", {})
        kind = condition.get("kind")

        if kind == "health_below":
            threshold = float(condition.get("threshold", 0))
            if health_ratio < threshold:
                return rule["action"]

        elif kind == "power_ratio_at_least":
            ratio = float(condition.get("ratio", 0))
            actual = float("inf") if enemy_power <= 0 else (our_power / enemy_power)
            if actual >= ratio:
                return rule["action"]

        elif kind == "always":
            return rule["action"]

    raise ValueError("No AI action matched; ensure there is an 'always' fallback rule")


def validate_test_scenarios(ai_profile: dict, test_state: dict) -> list:
    errors = []
    for scenario in test_state.get("scenarios", []):
        got = select_action(
            ai_profile,
            health_ratio=float(scenario["health_ratio"]),
            our_power=float(scenario["our_power"]),
            enemy_power=float(scenario["enemy_power"]),
        )
        expected = scenario["expected_action"]
        if got != expected:
            errors.append(f"scenario '{scenario['name']}' expected {expected} but got {got}")
    return errors


def main() -> int:
    unit_spec = read_json(UNIT_SPEC_PATH)
    ai_profile = read_json(AI_PROFILE_PATH)
    test_state = read_json(TEST_STATE_PATH)

    errors = []
    errors.extend(validate_unit_spec(unit_spec))
    errors.extend(validate_ai_profile(ai_profile))
    errors.extend(validate_test_scenarios(ai_profile, test_state))

    if errors:
        print("Validation FAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Validation PASSED")
    for scenario in test_state.get("scenarios", []):
        action = select_action(
            ai_profile,
            health_ratio=float(scenario["health_ratio"]),
            our_power=float(scenario["our_power"]),
            enemy_power=float(scenario["enemy_power"]),
        )
        print(f"- {scenario['name']}: {action}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

