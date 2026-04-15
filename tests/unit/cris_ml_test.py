from refractor.muses import (
    MusesStrategyContext,
)
from refractor.osr_ml import features_l1b, prediction, read_l1b


def test_cris_co_ml(cris_ml_test_in_dir, cris_ml_dir, isolated_dir, ifile_hlp):
    """Basic test of using machine learning to predict CO. This is pulled from the
    troppy notebook https://github-fn.jpl.nasa.gov/fwerner/troppy/blob/main/troppy/notebooks/ml/CRIS/CO/09_cris_ml_predict_noprior_keras.ipynb"""
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
    features = features_l1b(l1b=l1b, prior=None, ml_model_path=cris_ml_dir)
    instrument = "CRIS-JPSS-1"
    species = "CO"
    pred = prediction(
        mdl_api="sequential",
        path=cris_ml_dir,
        prefix=instrument + "_" + species + "_ret_col",
        # Until we get weights sorted out
        # suffix="keras-ANN",
        suffix="keras-ANN_new",
        features=features,
        batch_size_in=8192 * 2,
        evaluate=False,
        save_evaluate=False,
    )
    print(pred.labels_pred)
    # This follow what align_l1b_l2muses does in troppy. Can check with Frank
    # to make sure this is correct ordering
    # latitude = np.reshape(l1b.latitude, (l1b.latitude.size))
