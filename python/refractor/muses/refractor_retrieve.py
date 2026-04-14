from __future__ import annotations
from .docopt_simple import docopt_simple, DocOptSimple
from .retrieval_strategy import RetrievalStrategy
from .input_file_helper import InputFileHelper
from loguru import logger
import sys
import glob
import os
import importlib.util
import socket
import subprocess
import warnings
import refractor.framework as rf  # type: ignore
import refractor.muses
from multiprocessing import Process
from typing import Any

version = "1.0.0"
usage = """Usage:
  refractor-retrieve stac [options] <retrieval_config> <strategy_file> <stac_file> [<output_dir>]
  refractor-retrieve -t <dirlist> [options] | --targets=<dirlist> [options]
  refractor-retrieve -h | --help
  refractor-retrieve -v | --version

Run a retrieval over a set of target/sounding directories, or on a stac file.

For the stac file, the directory structure tends to be difference so we directly
take the retrieval configuaration file, the strategy file, and the stac file. You
can optionally give an output directory, otherwise we use the same directory as the
stac file is in. Note that retrieval configuration and strategy file can either be
the old Table.asc format file, or separate YAML files. The YAML are easier to read
in my opinion, and I suggest that unless you need backwards comparability. The YAML
file also support retrieval steps other than Optimal Estimation (OE), e.g.,
Machine Learning (ML),  while the Table.asc only support OE.

Options:
  -h --help
      Print this message

  --debug
      Set logger level to DEBUG, which provides debugging log messages.

  --trace
      Set logger level to TRACE, which provides both debugging and tracing log
      messages.

  --gmao-dir=f
      Given the location of the GMAO path to use. If not specified, we default
      to the environment variable MUSES_GMAO_PATH, or the hardcoded path "../GMAO" if
      the environment variable isn't set.

  --mpi
      Process multiple targets in parallel using MPI.

  --osp-delta-dir=f
      Give the location of the OSP delta directory to use with the osp directory.
      If not given, we default to the environment variable MUSES_OSP_DELTA_PATH,
      or if the environment variable isn't set to not using any delta path.

  --osp-dir=f
      Give the location of the osp directory to used. If not specified, we default
      to the environment variable MUSES_OSP_PATH, or the hardcoded path "../OSP" if
      the environment variable isn't set.

  --refractor
      Ignored, we take this argument for backwards compatibility with py-retrieve.
      We always use ReFRACtor

  --refractor-config=f
      Use the given refractor python config file. If not supplied, we fall back to using
      the old py-retrieve forward models. Note that this is a different file
      than the retrieval_config.yaml or Table.asc format files. We could merge these
      together, but at least for now I think it makes sense to treat them as different.

  --plots
      Generate plots for a subset of the species

  -t --targets=dirlist
      Target directories

  -v --version
      Print program version

"""


def onerror(err: BaseException) -> None:
    logger.info("refractor-retrieve is done ...")
    logger.info("Exiting with code 1")
    sys.exit(1)


def retrieve_wrap(
    args: DocOptSimple,
    mpi_rank: int,
    rs: RetrievalStrategy,
    target_dir: str,
    in_process: bool = False,
) -> None:
    """Wrapper so we can run a sounding in a separate process, so any memory leaks
    don't grow."""
    try:
        # Move or add logging to local file
        if args.mpi:
            # If we aren't using MPI, then presumably we are running a terminal
            # where we want to see the output. So only remove stdout in mpi.
            logger.remove()
        loghid = logger.add(f"{target_dir}/refractor-retrieve.log")

        # Capture the names of input files read while the software is running into the target
        # directory along side the log file
        if args.debug:
            rs.input_file_helper.add_observer(
                refractor.muses.InputFileRecord(f"{target_dir}/input_file_list.log")
            )
        with logger.catch(reraise=True):
            # Forward C++ logging in framework to the python logger
            rf.PythonFpLogger.turn_on_logger(logger)
            rs.update_strategy_context(target_dir)
            # if True:
            if False:
                # Fake error, for use in testing handling of this
                if mpi_rank == 4:
                    raise RuntimeError("Fake retrieval error")
            rs.retrieval_ms()
            logger.info(f"Success: {os.path.basename(target_dir)}")
    except:  # noqa: E722
        logger.info(f"Failure: {os.path.basename(target_dir)}")
        if in_process:
            # Reraising the exception causes a printout at the top level we want
            # to avoid, so just exit
            sys.exit(1)
        else:
            # Otherwise, just reraise the error
            raise
    finally:
        # We ran into a weird seg fault on exit, presumably some lifetime issue.
        # Related somehow to the rf.PythonFpLogger (although not clear exactly what
        # is happening - we have rf.PythonFpLogger turned on and see the seg fault,
        # off and we don't).
        #
        # We only need this logger when we are doing a retrieval, so we just
        # manually turn this off after each target is run, and back on for
        # the next target. This isn't strictly necessary, we could probably
        # use sys.atexit or something similiar. But this is easy and sufficient
        # to fix the seg fault issue.
        rf.PythonFpLogger.turn_off_logger()
        logger.remove(loghid)
        if args.mpi:
            # Add back stdout, to capture any other logging not tied to target
            logger.add(sys.stdout)


def process_targets(
    args: DocOptSimple,
    rs: RetrievalStrategy,
    target_dir_list: list[str],
    mpi_rank: int,
    hostname: str,
    success_dir: str,
    error_dir: str,
) -> None:
    for target_dir in target_dir_list:
        # For MPI, skip anything already run. Just by convention, this isn't
        # done without MPI (the assumption being that this is just a small number of
        # target that we want to run, e.g., for testing).
        if args.mpi:
            indicator = f"{os.path.basename(target_dir)}-{hostname}_{mpi_rank}"
            if os.path.isfile(f"{success_dir}/{indicator}"):
                logger.info(
                    f"node: {hostname}, rank: {mpi_rank}, already successfully processed. Skipping: {os.path.basename(target_dir)}"
                )
                continue
            if os.path.isfile(f"{error_dir}/{indicator}"):
                logger.info(
                    f"node: {hostname}, rank: {mpi_rank}, errored out in a previous run. Skipping: {os.path.basename(target_dir)}"
                )
                continue
        # Special case of 1 target, run directly. This makes debugging an issue
        # easier - so you run the sounding you are interested in and everything
        # happens in the same process space.
        if len(target_dir_list) == 1:
            # For testing, force use of Process even with small test data set
            # if False:
            try:
                retrieve_wrap(args, mpi_rank, rs, target_dir)
                run_exitcode: int | None = 0
            except:  # noqa: E722
                run_exitcode = 1
        else:
            # Otherwise, run in a separate process space to avoid accumulation of
            # memory from small memory leaks in processing a sounding.
            p = Process(
                target=retrieve_wrap, args=(args, mpi_rank, rs, target_dir, True)
            )
            p.start()
            p.join()
            run_exitcode = p.exitcode
        if run_exitcode is not None and run_exitcode == 0:
            logger.info(f"Success: {os.path.basename(target_dir)}")
            if args.mpi:
                subprocess.run(["touch", f"{success_dir}/{indicator}"])
        else:
            logger.info(f"Failure: {os.path.basename(target_dir)}")
            if args.mpi:
                subprocess.run(["touch", f"{error_dir}/{indicator}"])


def process_stac() -> None:
    pass


def main() -> None:
    # Import other refractor package so we get any configuration/handles set up.
    # Ignore ruff warnings, we are importing this for side effects, we don't directly
    # use the imports
    import refractor.muses_py_fm  # noqa: F401
    import refractor.omi  # noqa: F401
    import refractor.tropomi  # noqa: F401
    import refractor.osr_ml  # noqa: F401

    args = docopt_simple(usage, version=version)
    # warnings to logger
    showwarning_ = warnings.showwarning

    def showwarning(message: Any, *args: Any, **kwargs: Any) -> None:
        logger.warning(message)
        if False:
            # Print warning as normal, if desired. Otherwise we just include
            # in log
            showwarning_(message, *args, **kwargs)

    warnings.showwarning = showwarning

    write_debug_output = False
    log_level = "INFO"
    if args.trace:
        log_level = "DEBUG"
        write_debug_output = True
    elif args.debug:
        log_level = "DEBUG"
        write_debug_output = True

    # log to stdout instead of stderr, this works better with the mpi log capture
    logger.remove()
    logger.add(sys.stdout, level=log_level)

    logger.info("refractor-retrieve is running ...")

    if args.refractor_config:
        logger.info(f"Loading refractor configuration file: {args.refractor_config}")
        spec = importlib.util.spec_from_file_location(
            "refractor_config", args.refractor_config
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Problem importing {args.refractor_config}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules["refractor_config"] = module
        rs: RetrievalStrategy = module.rs
    else:
        rs = RetrievalStrategy(
            filename=None, writeOutput=write_debug_output, writePlots=args.plots
        )
    rs.input_file_helper = InputFileHelper(
        osp_dir=args.osp_dir, osp_delta_dir=args.osp_delta_dir, gmao_dir=args.gmao_dir
    )

    # Set up for MPI run, if needed
    target_dir_full_list = glob.glob(os.path.expanduser(args.targets))
    if args.mpi:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        hostname = socket.gethostname()
        mpi_rank = comm.Get_rank()
        mpi_size = comm.Get_size()
        # Partition into ranges of close to equal sizes
        start = 0
        end = len(target_dir_full_list)
        count = (end - start) // mpi_size
        remainder = (end - start) % mpi_size
        i_start = start + mpi_rank * count + min([mpi_rank, remainder])
        i_stop = start + (mpi_rank + 1) * count + min([mpi_rank + 1, remainder])
        target_dir_list = target_dir_full_list[i_start:i_stop]

        # For MPI, we use a few fixed directories. By convention, we
        # don't use these for the non-MPI case.
        bpath = os.path.abspath(os.path.dirname(target_dir_full_list[0]))
        success_dir = f"{bpath}/success"
        error_dir = f"{bpath}/error"
        subprocess.run(["mkdir", "-p", success_dir])
        subprocess.run(["mkdir", "-p", error_dir])
    else:
        # We just process everything if we aren't using MPI.
        target_dir_list = target_dir_full_list
        mpi_rank=-1
        hostname="localhost"
        success_dir="not_used"
        error_dir="not_used"

    logger.info(f"Number of tasks: {len(target_dir_list)}")

    with logger.catch(onerror=onerror):
        # At least for now, we execute stac file directly. We might add
        # looping over this later, but for now just run directly
        if args.stac:
            process_stac()
        else:
            process_targets(
                args, rs, target_dir_list, mpi_rank, hostname, success_dir, error_dir
            )

    logger.info("refractor-retrieve is done ...")
    logger.info("Exiting with code 0")


# Don't export main, we might have other scripts with a main. We can
# always explicitly import this if needed.
__all__ = []
