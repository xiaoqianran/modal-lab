"""Patch WorldMirror attention.py to allow SDPA without flash-attn."""
from pathlib import Path

p = Path("/opt/HY-World-2.0/hyworld2/worldrecon/hyworldmirror/models/layers/attention.py")
text = p.read_text()
old = """try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    _USE_FLASH_ATTN_V3 = True
except ImportError:
    from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2
    _USE_FLASH_ATTN_V3 = False
"""
new = """try:
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
    _USE_FLASH_ATTN_V3 = True
    _HAS_FLASH_ATTN = True
except ImportError:
    try:
        from flash_attn.flash_attn_interface import flash_attn_func as flash_attn_func_v2
        _USE_FLASH_ATTN_V3 = False
        _HAS_FLASH_ATTN = True
    except ImportError:
        flash_attn_func_v3 = None
        flash_attn_func_v2 = None
        _USE_FLASH_ATTN_V3 = False
        _HAS_FLASH_ATTN = False
"""
if old not in text:
    raise SystemExit("attention.py import block not found")
text = text.replace(old, new, 1)
old2 = "if q.dtype==torch.bfloat16 or q.dtype==torch.float16:"
new2 = "if _HAS_FLASH_ATTN and (q.dtype==torch.bfloat16 or q.dtype==torch.float16):"
if old2 not in text:
    raise SystemExit("attention.py dtype branch not found")
text = text.replace(old2, new2, 1)
p.write_text(text)
print("patched", p)
