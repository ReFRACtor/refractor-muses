This is test support for comparing against the old py-retrieve.

The py-retrieve code is complicated to match and to test against. This
was very useful during initial development to make sure the ReFRACtor
code was working and generated similar results to py-retrieve.

At this point, the comparisons here are mostly done. In the future,
changes to ReFRACtor will be compared against the existing ReFRACtor
results, we will compare against py-retrieve less and less
frequently. At some point, the cost of maintaining this old code won't
be worth it - so these tests will likely disappear over time.

But for now, we keep this old functionality to support investigating
any old issue that pop up.
