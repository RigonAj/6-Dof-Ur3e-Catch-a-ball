"""Generate the left-hand (mirrored) robot USD variant.

Rotates the ``Hoop`` prim under ``wrist_3_link`` by 180 degrees about the
wrist Z axis, moving the racket from the right side (disk offset
``(-0.5, 0, 0)`` in ``wrist_3_link``) to the left side (``(+0.5, 0, 0)``).
The disk normal stays along wrist -Z, so pass-through semantics are
unchanged; only the hold side mirrors (yz-plane symmetry).

The edit is done at the Sdf layer level, so the remote ur3e payload is never
composed and the source file is left untouched.

Run with the Isaac Lab venv python:

    ~/env_isaaclab/bin/python scripts/make_left_hand_usd.py

If ``pxr`` is not importable, the script locates the USD libs bundled in the
``isaacsim`` pip package and re-executes itself with the right environment.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = REPO_ROOT / "USD_File" / "UR-with-gripper.usd"
DEFAULT_DST = REPO_ROOT / "USD_File" / "UR-with-gripper-left.usd"
HOOP_PRIM_PATH = "/World/ur3e/wrist_3_link/Hoop"


def _reexec_with_usd_env() -> None:
    """Find the isaacsim-bundled pxr module and re-exec with it on the path."""
    site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    usd_libs = sorted(glob.glob(str(site_packages / "isaacsim" / "extscache" / "omni.usd.libs-*")))
    kit_plugins = site_packages / "isaacsim" / "kit" / "kernel" / "plugins"
    if not usd_libs:
        sys.exit(
            "error: pxr not importable and no isaacsim omni.usd.libs extension found; "
            "run with the Isaac Lab venv python (~/env_isaaclab/bin/python)"
        )
    usd_lib = usd_libs[-1]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [usd_lib, env.get("PYTHONPATH", "")]))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, [f"{usd_lib}/bin", usd_lib, str(kit_plugins), env.get("LD_LIBRARY_PATH", "")])
    )
    env["_MAKE_LEFT_USD_REEXEC"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


try:
    from pxr import Gf, Sdf  # noqa: E402
except ImportError:
    if os.environ.get("_MAKE_LEFT_USD_REEXEC"):
        raise
    _reexec_with_usd_env()
    raise AssertionError("unreachable")


def rotate_hoop_180_about_z(src: Path, dst: Path) -> None:
    layer = Sdf.Layer.FindOrOpen(str(src))
    if layer is None:
        sys.exit(f"error: cannot open {src}")
    hoop = layer.GetPrimAtPath(HOOP_PRIM_PATH)
    if hoop is None:
        sys.exit(f"error: prim {HOOP_PRIM_PATH} not found in {src}")

    translate_attr = hoop.attributes.get("xformOp:translate")
    orient_attr = hoop.attributes.get("xformOp:orient")
    if translate_attr is None or orient_attr is None:
        sys.exit(f"error: {HOOP_PRIM_PATH} is missing xformOp:translate/orient")

    # The Hoop local transform decomposes as T * R * S (translate outermost in
    # xformOpOrder). Pre-multiplying by Rz(pi) about the wrist_3 origin gives
    # T' = Rz(pi) * t (negate x and y) and R' = Rz(pi) * R; scale ops are
    # unaffected.
    old_t = translate_attr.default
    new_t = type(old_t)(-old_t[0], -old_t[1], old_t[2])

    old_q = orient_attr.default
    quat_cls = type(old_q)
    rz_pi = quat_cls(0.0, 0.0, 0.0, 1.0)  # (real, i, j, k) = 180 deg about Z
    new_q = rz_pi * old_q

    print(f"Hoop translate: {tuple(old_t)} -> {tuple(new_t)}")
    print(
        f"Hoop orient:    ({old_q.GetReal()}, {tuple(old_q.GetImaginary())})"
        f" -> ({new_q.GetReal()}, {tuple(new_q.GetImaginary())})"
    )

    translate_attr.default = new_t
    orient_attr.default = new_q

    if not layer.Export(str(dst)):
        sys.exit(f"error: failed to export {dst}")
    print(f"Wrote {dst}")


def verify(dst: Path) -> None:
    layer = Sdf.Layer.FindOrOpen(str(dst))
    hoop = layer.GetPrimAtPath(HOOP_PRIM_PATH)
    t = hoop.attributes["xformOp:translate"].default
    q = hoop.attributes["xformOp:orient"].default
    assert t[0] > 0.49, f"expected hoop center near +0.5 on wrist_3 X, got {tuple(t)}"
    # Rz(pi) * Rx(pi/2) has zero real part and imaginary (0, s, s) with s=sqrt(2)/2.
    imag = q.GetImaginary()
    assert abs(q.GetReal()) < 1e-6 and abs(imag[1] - imag[2]) < 1e-6, f"unexpected orient {q}"
    # The disk normal (local +Z of the Disk mesh, -Z of wrist_3 in the right
    # variant) must still map to -Z of wrist_3: Rz(pi) leaves the Z axis fixed.
    rot = Gf.Rotation(Gf.Quatd(q.GetReal(), Gf.Vec3d(*imag)))
    hoop_z_in_wrist = rot.TransformDir(Gf.Vec3d(0, 0, 1))
    print(f"Verified: hoop center x = {t[0]:+.3f} m, hoop +Z in wrist_3 = {tuple(hoop_z_in_wrist)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC)
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST)
    args = parser.parse_args()
    rotate_hoop_180_about_z(args.src, args.dst)
    verify(args.dst)


if __name__ == "__main__":
    main()
