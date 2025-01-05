These are fixtures used by pytest.

Altough these files can be directly imported, normally you instead list them on
conftest.py as a pytest_plugins. This is just a pytest convention, fixtures listed there
are automatically included in all the tests without needing to explicitly import them.

We have a few fixtures to support testing against the old py_retrieve code, we don't
include this in our top level conftest.py to make it clear most tests don't use them. Instead,
we have a conftest.py in the old_py_retrieve directory that includes them there only.
This is just to keep things separate, and make it clear what code is used for the old
py-retrieve testing, and that may be removed over time when it becomes too expensive to
maintain them and not worth it.
