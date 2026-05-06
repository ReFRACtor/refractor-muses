# Note sure if we will keep this or not, but there is a tool called click2cwl.
# There is not one for docopt. So we make a simple program that uses click
# to call refractor_retrieve.py. We may 1) Decide this is overkill and get
# rid of it 2) Just change refractor_retrieve.py to use click.

from __future__ import annotations
from .refractor_retrieve import main as main_retrieve
import click
try:
    # Only need this when creating the process.cwl file, so don't
    # require just to run.
    from click2cwl import dump
except ImportError:
    pass

@click.command(
    short_help="This runs refractor-retrieve",
    help="This runs refractor-retrieve",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.option("--retrieval-config",
              "retrieval_config",
              help="this is the retrieval configuration yaml file. Default is to use a fixed input file stored in the docker image.",
              default="/home/muses/cris_ml_test_in/ml_1/retrieval_config.yaml",
              multiple=False,
              required=False)
@click.option("--strategy-table",
              "strategy_table",
              help="this is the strategy table yaml file. Default is to use a fixed input file stored in the docker image.",
              default="/home/muses/cris_ml_test_in/ml_1/strategy.yaml",
              multiple=False,
              required=False)
@click.option("--stac-catalog-dir",
              "stac_catalog_dir",
              help="this is the directory with the STAC catalog for the input data, we look for catalog.json",
              type=click.Path(),
              multiple=False,
              required=True)
@click.option("--output-dir",
              "-o",
              "output_dir",
                help="this is the output directory. We generate a stac with date",
                type=click.Path(),
                multiple=False,
                required=True)
@click.pass_context
def stac(ctx, retrieval_config, strategy_table, stac_catalog_dir, output_dir) -> None:
    """Base tool"""
    if "--dump" in ctx.args:
        # If we are doing --dump ctx, just print that out. Otherwise, go on
        # to actually run the code
        dump(ctx)
    else:
        main_retrieve(["stac", retrieval_config, strategy_table, f"{stac_catalog_dir}/catalog.json", output_dir])


