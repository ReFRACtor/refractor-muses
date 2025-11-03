from refractor.muses import (
    MusesRunDir,
    RetrievalConfiguration,
    MeasurementIdFile,
    FilterIdentifier,
    InstrumentIdentifier,
)
from refractor.osr_ml import read_l1b, features_l1b, prediction


def test_cris_co_ml(cris_test_in_dir, cris_ml_dir, isolated_dir, osp_dir, gmao_dir):
    """Basic test of using machine learning to predict CO. This is pulled from the
    troppy notebook https://github-fn.jpl.nasa.gov/fwerner/troppy/blob/main/troppy/notebooks/ml/CRIS/CO/09_cris_ml_predict_noprior_keras.ipynb"""
    r = MusesRunDir(cris_test_in_dir, osp_dir, gmao_dir)
    rconfig = RetrievalConfiguration.create_from_strategy_file(
        r.run_dir / "Table.asc", osp_dir=osp_dir
    )
    filter_list_dict = {
        InstrumentIdentifier("CRIS"): [
            FilterIdentifier("2B1"),
            FilterIdentifier("1B2"),
            FilterIdentifier("2A1"),
            FilterIdentifier("1A1"),
        ],
    }
    measurement_id = MeasurementIdFile(
        r.run_dir / "Measurement_ID.asc", rconfig, filter_list_dict
    )
    # This is a bit round about to get a single file name. But I imagine that
    # we will have some interface like this to get the file list.
    l1b_fname = measurement_id["CRIS_filename"]

    # This may get reworked, but for now use Frank's code
    l1b = read_l1b(
        files=[
            l1b_fname,
        ]
    )

    features = features_l1b(l1b=l1b, prior=None, ml_model_path=cris_ml_dir)
    instrument = "CRIS-JPSS-1"
    species = "CO"
    pred = prediction(
        mdl_api="sequential",
        path=cris_ml_dir,
        prefix=instrument + "_" + species + "_ret_col",
        # Until we get weights sorted out
        #suffix="keras-ANN",
        suffix="keras-ANN_new",
        features=features,
        batch_size_in=8192 * 2,
        evaluate=False,
        save_evaluate=False,
    )
    print(pred.labels_pred)
