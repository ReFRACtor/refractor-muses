from .refractor_uip import RefractorUip
import refractor.framework as rf

class CostFuncCreator:
    '''This object creates a CostFunc that can be used with py-retrieve.
    This includes handling joint retrievals with multiple instruments.

    The object created is a NLLSMaxAPosteriori that wraps a
    MaxAPosterioriSqrtConstraint (a slight variation on the
    MaxAPosterioriStandard that we typically use in ReFRACtor).

    The defaults ForwardModel that does the actual calculations wrap
    the existing py-retrieve forward model functions. But the creator is
    designed to be modified by updating the dictionaries instrument_handler
    and state_vector_handler.
    '''
    def __init__(self, **kwargs):
        self.instrument_handler = {}
        self.state_vector_handler = {}
        self.creator_kwargs = kwargs

    def create_cost_func(self, rf_uip : RefractorUip):
        # Temp, we'll want to get logic in place to use
        # instrument_handler and state_vector_handler
        from refractor.tropomi import RefractorObjectCreator
        obj_creator = RefractorObjectCreator(rf_uip, state_vector_retrieval=True, **self.creator_kwargs)
        sv = obj_creator.state_vector
        fm = obj_creator.forward_model
        ret_info = rf_uip.ret_info
        mstand = rf.MaxAPosterioriSqrtConstraint(fm,
            obj_creator.observation, sv,
            ret_info["const_vec"], ret_info["sqrt_constraint"].transpose())
        mprob = rf.NLLSMaxAPosteriori(mstand)
        return mprob
        
        
        
        
