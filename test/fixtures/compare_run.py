import glob
import subprocess


def compare_run(expected_dir, run_dir, diff_is_error=True, from_run_dir=False):
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
    for f in glob.glob(f"{dir1}/*/Products/Products_L2*.nc"):
        f2 = f.replace(dir1, dir2)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob(f"{dir1}/*/Products/Lite_Products_*.nc"):
        f2 = f.replace(dir1, dir2)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob(f"{dir1}/*/Products/Products_Radiance*.nc"):
        f2 = f.replace(dir1, dir2)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob(f"{dir1}/*/Products/Products_Jacobian*.nc"):
        f2 = f.replace(dir1, dir2)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob(f"{dir1}/*/Products/Products_IRK.nc"):
        f2 = f.replace(dir1, dir2)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
