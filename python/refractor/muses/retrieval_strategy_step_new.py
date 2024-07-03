import logging
from .priority_handle_set import PriorityHandleSet

# This file should be temporary. We are reworking the RetrievalStrategyStep, and
# want a clean copy to tweak while leaving the other one in place. This will
# eventually just replace retrieval_strategy_step.py

logger = logging.getLogger("py-retrieve")

class RetrievalStrategyStepSetNew(PriorityHandleSet):
    '''This takes the retrieval_type and determines a RetrievalStrategyStep
    to handle this. It then does the retrieval step.

    Note RetrievalStrategyStep can assume that they are called for the same target, until
    notify_update_target is called. So if it makes sense, these objects can do internal
    caching for things that don't change when the target being retrieved is the same from
    one call to the next.'''
    def retrieval_step(self, retrieval_type : str, rs : 'RetrievalStrategy') -> None:
        self.handle(retrieval_type, rs)

    def notify_update_target(self, rs : 'RetrievalStrategy'):
        '''Clear any caching associated with assuming the target being retrieved is fixed'''
        for p in sorted(self.handle_set.keys(), reverse=True):
            for h in self.handle_set[p]:
                h.notify_update_target(rs)
        
    def handle_h(self, h : 'RetrievalStrategyStep', retrieval_type : str,
                 rs : 'RetrievalStrategy') -> (bool, None):
        return h.retrieval_step(retrieval_type, rs)
