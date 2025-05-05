import glob
import subprocess
import os

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
