from glob import glob
import logging
import refractor.muses.muses_py as mpy
import os

logger = logging.getLogger("py-retrieve")

class RetrievalJacobianOutput:
    '''Observer of RetrievalStrategy, outputs the Products_Jacobian files.'''
    def notify_add(self, retrieval_strategy):
        self.retrieval_strategy = retrieval_strategy
        
    def notify_update(self, retrieval_strategy, location):
        self.retrieval_strategy = retrieval_strategy
        if(location != "retrieval step" or retrieval_strategy.results is None):
            return
        if len(glob(f"{self.out_fname}*")) == 0:
            # First argument isn't actually used in write_products_one_jacobian.
            # It is special_name, which doesn't actually apply to the jacobian file.
            os.makedirs(os.path.dirname(self.out_fname), exist_ok=True)
            # Code assumes we are in rundir
            with self.retrieval_strategy.chdir_run_dir():
                mpy.write_products_one_jacobian(None, self.out_fname,
                                                self.retrievalInfo,
                                                self.results,
                                                mpy.ObjectView(self.stateInfo),
                                                self.instruments, self.table_step)
        else:
            logger.info(f"Found a jacobian product file: {self.out_fname}")

    @property
    def retrievalInfo(self):
        return self.retrieval_strategy.retrievalInfo

    @property
    def strategy_table(self):
        return self.retrieval_strategy.strategy_table

    @property
    def special_tag(self):
        if self.retrieval_strategy.retrieval_type != 'default':
            return f"-{self.retrieval_strategy.retrieval_type}"
        return ""

    @property
    def out_fname(self):
        return f"{self.retrieval_strategy.output_directory}/Products/Products_Jacobian-{self.species_tag}{self.special_tag}"
    
    @property
    def species_tag(self):
        res = self.retrieval_strategy.step_name
        res = res.rstrip(', ')
        if 'EMIS' in res and res.index('EMIS') > 0:
            res = res.replace('EMIS', '')
        if res.endswith(',_OMI'):
            res = res.replace(',_OMI', '_OMI')  #  Change "H2O,O3,_OMI" to "H2O,O3_OMI"
        res = res.rstrip(', ')
        return res

    @property
    def table_step(self):
        return self.retrieval_strategy.table_step

    @property
    def results(self):
        return self.retrieval_strategy.results

    @property
    def stateInfo(self):
        return self.retrieval_strategy.stateInfo

    @property
    def instruments(self):
        return self.retrieval_strategy.instruments

__all__ = ["RetrievalJacobianOutput",]    
