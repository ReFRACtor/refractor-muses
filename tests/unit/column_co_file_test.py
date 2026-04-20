from refractor.muses import (
    MusesStrategyContext,
)
from refractor.osr_ml import ColumnCoFile


def test_cris_co_ml(cris_ml_test_in_dir, isolated_dir, ifile_hlp):
    ctx = MusesStrategyContext(cris_ml_test_in_dir, output_directory=isolated_dir)
    f = ColumnCoFile(
        "column_co.nc", ctx.retrieval_config["product_spec_path"] / "columns_CRIS_CO.nc"
    )
    f.write()
