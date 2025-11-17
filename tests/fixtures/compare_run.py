from __future__ import annotations
import glob
import subprocess
import pprint
import os
from loguru import logger
from pathlib import Path
from typing import Any


def compare_run(
    expected_dir: str | os.PathLike[str],
    run_dir: str | os.PathLike[str],
    diff_is_error: bool = True,
    from_run_dir: bool = False,
) -> None:
    """Compare products from two runs."""
    expected_dir = str(expected_dir)
    run_dir = str(run_dir)
    dir1 = expected_dir
    dir2 = run_dir
    if from_run_dir:
        dir1 = run_dir
        dir2 = expected_dir
    else:
        dir1 = expected_dir
        dir2 = run_dir
    flist = glob.glob(f"{dir1}/*/Products/Products_L2*.nc")
    flist += glob.glob(f"{dir1}/*/Products/Lite_Products_*.nc")
    flist += glob.glob(f"{dir1}/*/Products/Products_Radiance*.nc")
    flist += glob.glob(f"{dir1}/*/Products/Products_Jacobian*.nc")
    flist += glob.glob(f"{dir1}/*/Products/Products_IRK.nc")
    for f in flist:
        f2 = f.replace(dir1, dir2)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
        if from_run_dir:
            cmd = f"ncdump -c {f} > {f}.struct"
            print(cmd, flush=True)
            subprocess.run(cmd, shell=True, check=diff_is_error)
            cmd = f"ncdump -c {f2} > {f}.struct.expected"
            print(cmd, flush=True)
            subprocess.run(cmd, shell=True, check=diff_is_error)
            cmd = f"diff -u {f}.struct {f}.struct.expected"
            print(cmd, flush=True)
            subprocess.run(cmd, shell=True, check=diff_is_error)
        else:
            cmd = f"ncdump -c {f} > {f2}.struct.expected"
            print(cmd, flush=True)
            subprocess.run(cmd, shell=True, check=diff_is_error)
            cmd = f"ncdump -c {f2} > {f2}.struct"
            print(cmd, flush=True)
            subprocess.run(cmd, shell=True, check=diff_is_error)
            cmd = f"diff -u {f2}.struct {f2}.struct.expected"
            print(cmd, flush=True)
            subprocess.run(cmd, shell=True, check=diff_is_error)
    print("")
    print("----------------------------------------------------")
    print(f"Update by '\cp {os.path.dirname(f2)}/*.nc {os.path.dirname(f)}/'")


def compare_muses_py_dict(
    our_dict: dict[str, Any],
    expected_dict: dict[str, Any],
    fname_base: str = "dict",
    check: bool = True,
) -> None:
    """It can be hard to compare the nested directory structures muses-py uses in
    various spots. As a simple test, we can pprint the directories, and do a
    simple diff. The diff can either just print results, or fail if there is differences,
    based on the value of "check".
    """
    f1 = Path(f"our_{fname_base}.txt").absolute()
    f2 = Path(f"expected_{fname_base}.txt").absolute()
    with open(f1, "w") as fh:
        pprint.pprint(our_dict, fh)
    with open(f2, "w") as fh:
        pprint.pprint(expected_dict, fh)
    logger.info(f"diff -u {f1} {f2}")
    subprocess.run(["diff", "-u", str(f1), str(f2)], check=check)
