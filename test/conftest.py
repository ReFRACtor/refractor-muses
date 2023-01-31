import pytest

def pytest_addoption(parser):
    parser.addoption("--run-long", action="store_true",
                     help="run long tests")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-long"):
        skip_long_test = pytest.mark.skip(reason="need --run-long option to run")
        for item in items:
            if "long_test" in item.keywords:
                item.add_marker(skip_long_test)
