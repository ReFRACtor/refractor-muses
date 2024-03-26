Introduction
------------

This has some notes I've made as writing this. We'll perhaps move this into some  documentation

Typing
------

There has been some recent moves to push optional static typing into python, through the 
use of typehints along with extra tools like Mypy. I looked into this a bit, and it looks
like this is *not* something we should be doing.

1. The tools don't really fully work - they don't catch all typing mistakes. A tool that 
   "sort of works" is actually worse than no tool.
2. People who have done this for particular projects spend a *lot* of time getting this right.
   For C++ sort of static typing, this only works if *everything* including library calls 
   are typed.
3. It tends to be fragile.
   
So putting this in fully is a whole lot of work, with actually no real pay off. This isn't
shocking, python was designed from the beginning to be dynamically typed and shoe horning 
C++ style static typing doesn't really work so well. Python also tends to have more complicated
types - so the typing is more like Haskell typing than C++. Haskell is very good at type 
inference, which none of the python tools have. And C++ has some level of type inference with
the "auto" keyword - there is no equivalent typing in python.

All that said, type *hints* are actually pretty useful to document what a function expects 
without needing to read the code in detail to see what it does with argument.

So in the ReFRACtor code we have typehints where these might be useful - as an additional way
to document functions. These should be read as exactly that, a "hint" at what type we want. We
don't have these everywhere, make no attempt to make them consistent (e.g., a function with
a particular type hint for an argument might pass that to another function with a different type
hint). It is not in general an error if an value passed as an argument doesn't match a type -
it just need to have the proper duck typing to supply whatever that function is expecting.

