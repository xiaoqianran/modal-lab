"""Apply soft SAM3 import to traj_generate.py (for smoke without nav)."""
from pathlib import Path

p = Path("/opt/HY-World-2.0/hyworld2/worldgen/traj_generate.py")
t = p.read_text()
old = "from transformers import Sam3Processor, Sam3Model\n"
new = (
    "try:\n"
    "    from transformers import Sam3Processor, Sam3Model\n"
    "except Exception as _e:\n"
    "    print('[modal-lab] Sam3 import skipped:', _e)\n"
    "    Sam3Processor = Sam3Model = None\n"
)
if old in t and "Sam3 import skipped" not in t:
    p.write_text(t.replace(old, new, 1))
    print("patched sam3 import")
else:
    print("sam3 import already patched or missing")
