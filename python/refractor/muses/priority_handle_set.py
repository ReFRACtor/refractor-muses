import collections
import collections.abc

class PriorityHandleSet(collections.abc.Set):
    '''This class was originally designed for a separate library
    (pynitf). Take a look at
    https://cartography-jpl.github.io/pynitf/design.html#priority-handle-set
    for a detailed description of this.

    We have a number of places where we want to have a
    chain-of-responsibility (see
    https://en.wikipedia.org/wiki/Chain-of-responsibility_pattern),
    with the addition of an ordering based on a priority order. We
    also want to be able to modify the handles, starting with a
    default set of handles.

    To support this, we have a class PriorityHandleSet. This is very
    similar to a priority queue except that we:

    1. Don’t want to actually pop from a queue, rather we iterate
       through the stored items.

    2. The items aren’t totally ordered. We iterate through items with
       the same priority in an arbitrary order.

    The PriorityHandleSet has members for adding and removing
    “Handles”. A handle is purposely vague, it is any object that the
    derived class wants to be. And in some cases the object is
    actually a class.

    The handles are called in priority order, and the first handle to
    say it can process a set of arguments get it (so chain of
    responsibility).

    It is currently considered an error for more than one handle of
    the same priority to say it can handle a set of arguments. We originally
    had just the first one we found handle this, but we ended up getting
    obscure non predictable errors where one handle was called in one
    run, and a second called in a different run. We can revisit this
    if needed. On the other hand, it is very common to have handles of
    different priority handle the same arguments - in that case the handle
    with the highest priority wins.
    '''
    def __init__(self):
        self.handle_set = collections.defaultdict(lambda : set())

    def __contains__(self, itm):
        return itm[0] in self.handle_set[itm[1]]

    def __len__(self):
        res = 0
        for t in self.handle_set.values():
            res += len(t)
        return res

    def __iter__(self):
        for k in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[k]:
                yield (h, k)

    @classmethod
    def _from_iterable(cls, it):
        obj = cls()
        for i in it:
            obj.add_handle(i[0],priority_order=i[1])
        return obj

    def __copy__(self):
        '''Copy the PrioritySet. This is a shallow copy, we have our own
        handle set but all the objects in it are the same as the original
        set.'''
        return self.__class__._from_iterable(iter(self))
    
    def add_handle(self, h, priority_order=0):
        '''Add a handler. The higher priority_order (larger number) items are
        tried first.'''
        self.handle_set[priority_order].add(h)

    def discard_handle(self, h):
        '''Discard the handle h. It is ok if h isn't actually in the set 
        of handles.'''
        for k in sorted(self.handle_set.keys()):
            self.handle_set[k].discard(h)

    def clear(self):
        '''Remove all handles in the set.'''
        self.handle_set.clear()

    @classmethod
    def default_handle_set(cls):
        '''Return the default set of handlers to use.'''
        if(not hasattr(cls, "_default_handle_set")):
            cls._default_handle_set = cls()
        return cls._default_handle_set
    
    @classmethod            
    def add_default_handle(cls, h, priority_order=0):
        '''Add the given handle to the default set of handlers.  The 
        higher priority_order (larger number) items are tried first.'''
        cls.default_handle_set().add_handle(h, priority_order)

    @classmethod            
    def discard_default_handle(cls, h):
        '''Discard the handle h from the default list. It is ok if h isn't 
        actually in the set of handles.'''
        cls.default_handle_set().discard_handle(h)

    def handle(self, *args, **keywords):
        '''Find the first handle that says it can process the given arguments,
        and return the results from that handle.'''
        could_handle = False
        res = None
        h_handle = None
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                c, r = self.handle_h(h, *args, **keywords)
                if(c and could_handle):
                    raise RuntimeError(f"Multiple handles of the same priority level {p} wanted to process the data. Handle {h_handle} and {h} both wanted to process the data. args={args}, keywords={keywords}.")
                if(c):
                    could_handle = True
                    h_handle = h
                    res = r
            if(could_handle):
                return res
        raise RuntimeError("No handle was found. args=%s, keywords=%s" % (args, keywords))

    def handle_h(self, h, *args, **keywords):
        raise NotImplementedError
        
        
__all__ = ["PriorityHandleSet",]        
