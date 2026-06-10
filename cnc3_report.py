"""Generate focused real-time entry point report from cnc3_analysis.json"""
import json

data = json.load(open('cnc3_analysis.json'))

RT_KEYWORDS = [
    "Update","Tick","Logic","Game","Unit","Player","Object","World",
    "Script","Command","Event","Notify","Set","Get","Apply","Spawn",
    "Damage","Kill","Create","Destroy","Add","Remove","Change","Modify",
    "Attack","Move","Build","Upgrade","Power","Money","Cash","Resource",
    "Health","Armor","Speed","Weapon","Ability","Tech","Research",
    "Map","Camera","UI","Render","State","Load","Save","Net",
    "Sync","Frame","Timer","AI","Team","Win","Lose","Mission",
]

def is_rt(name):
    n = name.lower()
    return any(kw.lower() in n for kw in RT_KEYWORDS)

print("=" * 80)
print("C&C3 TIBERIUM WARS - SAGE ENGINE ENTRY POINT ANALYSIS")
print("=" * 80)

for binary, info in data.items():
    exps = info['exports']
    imps = info['imports']
    total_imp = sum(len(v) for v in imps.values())
    print(f"\n{'='*70}")
    print(f"BINARY: {binary}")
    print(f"  Exports: {len(exps)}   Imports: {total_imp} from {len(imps)} DLLs")

    rt_exps = [e for e in exps if is_rt(e['name'])]
    if rt_exps:
        print(f"\n  [REAL-TIME EXPORTS - {len(rt_exps)} hooks]")
        for e in rt_exps:
            print(f"    ord={e['ordinal']:4d}  rva={e['rva']:12s}  {e['name']}")

    rt_imps = []
    for dll, funcs in imps.items():
        for fn in funcs:
            if is_rt(fn):
                rt_imps.append((dll, fn))
    if rt_imps:
        print(f"\n  [REAL-TIME IMPORTS - {len(rt_imps)} hookable calls]")
        for dll, fn in rt_imps[:60]:
            print(f"    {dll:35s} {fn}")
        if len(rt_imps) > 60:
            print(f"    ... and {len(rt_imps)-60} more")

print("\n" + "=" * 80)
print("ALL EXPORTS FROM cnc3game.dat (SAGE Engine core):")
print("=" * 80)
for e in sorted(data.get('cnc3game.dat', {}).get('exports', []), key=lambda x: x['name']):
    marker = " <-- RT" if is_rt(e['name']) else ""
    print(f"  ord={e['ordinal']:4d}  rva={e['rva']:12s}  {e['name']}{marker}")

print("\n" + "=" * 80)
print("ALL EXPORTS FROM patchw32.dll:")
print("=" * 80)
for e in data.get('patchw32.dll', {}).get('exports', []):
    print(f"  ord={e['ordinal']:4d}  rva={e['rva']:12s}  {e['name']}")
