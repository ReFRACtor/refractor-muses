from refractor.muses import (
    MusesStrategyContext,
)
from refractor.osr_ml import ColumnCoFile, read_l1b, features_l1b, prediction


def test_cris_co_ml(cris_ml_test_in_dir, isolated_dir, ifile_hlp):
    ctx = MusesStrategyContext(cris_ml_test_in_dir, output_directory=isolated_dir)
    l1b_file = []
    for lnk in ctx.stac_catalog.get_item_links():
        l1b_file.extend(
            [
                i.get_absolute_href()
                for i in lnk.resolve_stac_object()
                .target.get_assets(role="data")  # type:ignore[union-attr]
                .values()
            ]
        )
    l1b = read_l1b(files=l1b_file)
    features = features_l1b(
        l1b=l1b, prior=None, ml_model_path=ctx.retrieval_config["muses_ml_path"]
    )
    instrument = "CRIS-JPSS-1"
    species = "CO"
    pred = prediction(
        mdl_api="sequential",
        path=ctx.retrieval_config["muses_ml_path"],
        prefix=instrument + "_" + species + "_ret_col",
        # Until we get weights sorted out
        # suffix="keras-ANN",
        suffix="keras-ANN_new",
        features=features,
        batch_size_in=8192 * 2,
        evaluate=False,
        save_evaluate=False,
    )
    f = ColumnCoFile(
        pred,
        l1b_file,
        "column_co.nc",
        ctx.retrieval_config["product_spec_path"] / "columns_CRIS_CO.nc",
    )
    f.write()
