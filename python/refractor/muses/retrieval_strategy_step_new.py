from .creator_handle import CreatorHandleSet, CreatorHandle
import logging
import abc


# This file should be temporary. We are reworking the RetrievalStrategyStep, and
# want a clean copy to tweak while leaving the other one in place. This will
# eventually just replace retrieval_strategy_step.py

logger = logging.getLogger("py-retrieve")

class RetrievalStrategyStepSetNew(CreatorHandleSet):
    '''This takes the retrieval_type and determines a RetrievalStrategyStep
    to handle this. It then does the retrieval step.
    '''
    def __init__(self):
        super().__init_("retrieval_step")
        
    def retrieval_step(self, retrieval_type : str, rs : 'RetrievalStrategy') -> None:
        self.handle(retrieval_type, rs)
