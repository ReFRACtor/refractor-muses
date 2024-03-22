import pytest

def pytest_addoption(parser):
    parser.addoption("--run-long", action="store_true",
                     help="run long tests")
    parser.addoption("--run-capture", action="store_true",
                     help="run tests used to capture input arguments")
    parser.addoption("--run-capture-initial", action="store_true",
                     help="run tests used to capture input arguments, the initial version")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-long"):
        skip_long_test = pytest.mark.skip(reason="need --run-long option to run")
        for item in items:
            if "long_test" in item.keywords:
                item.add_marker(skip_long_test)
    if not config.getoption("--run-capture"):
        skip_capture_test = pytest.mark.skip(reason="input test capture test. Need --run-capture option to run")
        for item in items:
            if "capture_test" in item.keywords:
                item.add_marker(skip_capture_test)
    if not config.getoption("--run-capture-initial"):
        skip_capture_initial_test = pytest.mark.skip(reason="input test initial capture test. Need --run-initial-capture option to run")
        for item in items:
            if "capture_initial_test" in item.keywords:
                item.add_marker(skip_capture_initial_test)
