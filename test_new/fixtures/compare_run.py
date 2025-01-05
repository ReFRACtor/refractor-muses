import glob
import subprocess


def compare_run(expected_dir, run_dir, diff_is_error=True):
    """Compare products from two runs."""
    expected_dir = str(expected_dir)
    run_dir = str(run_dir)
    for f in glob.glob(f"{expected_dir}/*/Products/Products_L2*.nc"):
        f2 = f.replace(expected_dir, run_dir)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob(f"{expected_dir}/*/Products/Lite_Products_*.nc"):
        f2 = f.replace(expected_dir, run_dir)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob(f"{expected_dir}/*/Products/Products_Radiance*.nc"):
        f2 = f.replace(expected_dir, run_dir)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob(f"{expected_dir}/*/Products/Products_Jacobian*.nc"):
        f2 = f.replace(expected_dir, run_dir)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
    for f in glob.glob(f"{expected_dir}/*/Products/Products_IRK.nc"):
        f2 = f.replace(expected_dir, run_dir)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)


def compare_irk(expected_dir, run_dir, diff_is_error=True):
    """Compare products from two runs, just the IRK part."""
    for f in glob.glob(f"{expected_dir}/*/Products/Products_IRK.nc"):
        f2 = f.replace(expected_dir, run_dir)
        cmd = f"h5diff --relative 1e-8 {f} {f2}"
        print(cmd, flush=True)
        subprocess.run(cmd, shell=True, check=diff_is_error)
