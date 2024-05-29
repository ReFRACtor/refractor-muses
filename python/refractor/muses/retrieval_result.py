import refractor.muses.muses_py as mpy

class PropagatedQA:
    '''There are a few parameters that get propagated from one step to the next. Not
    sure exactly what this gets looked for, it look just like flags copied from one
    step to the next. But pull this together into one place so we can track this.
    '''
    def __init__(self):
        self.propagated_qa = {'TATM' : 1, 'H2O' : 1, 'O3' : 1}

    @property
    def tatm_qa(self):
        return self.propagated_qa['TATM']
    
    @property
    def h2o_qa(self):
        return self.propagated_qa['H2O']

    @property
    def o3_qa(self):
        return self.propagated_qa['O3']

    def update(self, retrieval_state_element : 'list[str]', qa_flag : int):
        '''Update the QA flags for items that we retrieved.'''
        for state_element_name in retrieval_state_element:
            if(state_element_name in self.propagated_qa):
                self.propagated_qa[state_element_name] = qa_flag
               
class RetrievalResult:
    '''There are a few top level functions that work with a structure called
    retrieval_results. Pull all this together into an object so we can clearly
    see the interface and possibly change things.

    Unlike a number of things that we want to elevate to a class, this really does
    look like just a structure of various calculated things that then get reported in
    the output files - so I think this is probably little more than wrapping up stuff in
    one place.'''
    def __init__(self):
        pass

__all__ = ["PropagatedQA", "RetrievalResult"]    
