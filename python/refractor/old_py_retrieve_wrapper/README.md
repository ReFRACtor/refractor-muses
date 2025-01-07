As we did the initial development of refractor-muses, we did extensive testing against the
existing py-retrieve code. There are a number of classes developed that were useful during this
process, or that wrap py-retrieve code so we can test against it.

The package collects this old code. This should be considered depracated/dead code, however
we have kept it around for now because it is useful to be able to continue testing against
the py-retrieve code, and this can be useful to diagnose new bugs in refractor-muses as we
go along and run against a wider range of data.

The code in this package shouldn't in general be used, unless you are writing a unit test
against the old py-retrieve code.
