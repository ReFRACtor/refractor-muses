from glob import glob
import logging
import refractor.muses.muses_py as mpy
import os
from collections import defaultdict
import copy

logger = logging.getLogger("py-retrieve")

class RetrievalOutput:
    '''Observer of RetrievalStrategy, common behavior for Products files.'''
    def notify_add(self, retrieval_strategy):
        self.retrieval_strategy = retrieval_strategy

    def notify_update(self, retrieval_strategy, location):
        raise NotImplementedError

    @property
    def strategy_table(self):
        return self.retrieval_strategy.strategy_table

    @property
    def step_dir(self):
        return self.strategy_table["stepDirectory"]

    @property
    def input_dir(self):
        return self.strategy_table["dirInput"]

    @property
    def analysis_dir(self):
        return self.strategy_table["dirAnalysis"]

    @property
    def elanor_dir(self):
        return self.strategy_table["dirELANOR"]
    
    @property
    def windows(self):
        return self.retrieval_strategy.windows

    @property
    def errorCurrent(self):
        return self.retrieval_strategy.errorCurrent
    
    @property
    def special_tag(self):
        if self.retrieval_strategy.retrieval_type != 'default':
            return f"-{self.retrieval_strategy.retrieval_type}"
        return ""

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
    def quality_name(self):
        return self.retrieval_strategy.quality_name
    
    @property
    def table_step(self):
        return self.retrieval_strategy.table_step

    @property
    def number_table_step(self):
        return self.retrieval_strategy.number_table_step

    @property
    def strategy_table(self):
        return self.retrieval_strategy.strategy_table
    
    @property
    def results(self):
        return self.retrieval_strategy.results

    @property
    def stateInfo(self):
        return mpy.ObjectView(self.retrieval_strategy.stateInfo)

    @property
    def radianceStep(self):
        return mpy.ObjectView(self.retrieval_strategy.radianceStep)

    @property
    def retrievalInfo(self):
        return self.retrieval_strategy.retrievalInfo
    
    @property
    def instruments(self):
        return self.retrieval_strategy.instruments

    @property
    def myobsrad(self):
        return self.retrieval_strategy.myobsrad
    
class RetrievalJacobianOutput(RetrievalOutput):
    '''Observer of RetrievalStrategy, outputs the Products_Jacobian files.'''
    def notify_update(self, retrieval_strategy, location):
        self.retrieval_strategy = retrieval_strategy
        if(location != "retrieval step" or self.results is None):
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
                                                self.stateInfo,
                                                self.instruments, self.table_step)
        else:
            logger.info(f"Found a jacobian product file: {self.out_fname}")

    @property
    def out_fname(self):
        return f"{self.retrieval_strategy.output_directory}/Products/Products_Jacobian-{self.species_tag}{self.special_tag}"

class RetrievalRadianceOutput(RetrievalOutput):
    '''Observer of RetrievalStrategy, outputs the Products_Radiance files.'''
    def notify_update(self, retrieval_strategy, location):
        self.retrieval_strategy = retrieval_strategy
        if(location != "retrieval step" or self.results is None):
            return
        if len(glob(f"{self.out_fname}*")) == 0:
            # First argument isn't actually used in write_products_one_jacobian.
            # It is special_name, which doesn't actually apply to the jacobian file.
            os.makedirs(os.path.dirname(self.out_fname), exist_ok=True)
            # Code assumes we are in rundir
            with self.retrieval_strategy.chdir_run_dir():
                mpy.write_products_one_radiance(None, self.out_fname,
                                                self.retrievalInfo,
                                                self.results,
                                                self.stateInfo,
                                                self.radianceStep,
                                                self.instruments, self.table_step,
                                                self.myobsrad)
        else:
            logger.info(f"Found a radiance product file: {self.out_fname}")

    @property
    def out_fname(self):
        return f"{self.retrieval_strategy.output_directory}/Products/Products_Radiance-{self.species_tag}{self.special_tag}"

class RetrievalL2Output(RetrievalOutput):
    '''Observer of RetrievalStrategy, outputs the Products_L2 files.'''
    @property
    def species_count(self):
        '''Dictionary that gives the index we should use for product file names.
        This is 0 if the species doesn't get retrieved in a following step, and
        the count of other times the species is retrieved. So for example if
        O3 is retrieved 4 times, the first time we retrieve it the file has
        a "O3-3" in the name, followed by "O3-2", "O3-1" and "O3-0"'''
        if(self._species_count is None):
            self._species_count = defaultdict(lambda: 0)
            tstep = self.table_step
            for i in range(self.table_step+1, self.number_table_step):
                for spc in mpy.table_get_entry(self.strategy_table, i, 'retrievalElements').split(","):
                    self._species_count[spc] += 1
        return self._species_count

    @property
    def species_list(self):
        '''List of species, partially ordered so TATM comes before H2O, H2O before HDO,
        and N2O before CH4.

        The ordering is because TATM, H2O and N2O are used in making the lite files
        of CH4, HDO and H2O lite files, so we need to data from these before we get
        to the lite files.'''
        if(self._species_list is None):
            self._species_list = self.retrievalInfo.species_names
            for spc in ('N2O', 'H2O', 'TATM'):
                if(spc in self._species_list):
                    self._species_list.remove(spc)
                    self._species_list.insert(0, spc)
        return self._species_list
            
    def notify_update(self, retrieval_strategy, location):
        self.retrieval_strategy = retrieval_strategy
        # Save these, used in later lite files. Note these actually get
        # saved between steps, so we initialize these for the first step but
        # then leave them alone
        if(location == "retrieval step" and self.table_step == 0):
            self.dataTATM = None
            self.dataH2O = None
            self.dataN2O = None
        if(location != "retrieval step" or self.results is None):
            return
        # Regenerate this for the current step
        self._species_count = None
        self._species_list = None
        for spcname in self.species_list:
            if(self.retrievalInfo.species_list_fm.count(spcname) <= 1 or
               spcname in ('CLOUDEXT', 'EMIS') or 
               spcname.startswith('OMI') or
               spcname.startswith('NIR')):
                continue
            out_fname = f"{self.retrieval_strategy.output_directory}/Products/Products_L2-{spcname}-{self.species_count[spcname]}.nc"
            os.makedirs(os.path.dirname(out_fname), exist_ok=True)
            # Not sure about the logic here, but this is what script_retrieval_ms does
            if(not os.path.exists(out_fname) or spcname in ('TATM', 'H2O', 'N2O')):
                # Code assumes we are in rundir
                with self.retrieval_strategy.chdir_run_dir():
                    _, dataInfo = mpy.write_products_one(spcname, out_fname,
                                                self.retrievalInfo.retrieval_info_obj,
                                                self.results,
                                                self.stateInfo,
                                                self.instruments,
                                                self.table_step)
                    if(spcname == "TATM"):
                        self.dataTATM = dataInfo
                    elif(spcname == "H2O"):
                        self.dataH2O = dataInfo
                    elif(spcname == "N2O"):
                        self.dataN2O = dataInfo
                    self.lite_file(spcname, dataInfo)

    def lite_file(self, spcname, dataInfo):
        '''Create lite file. We pull this out as a separate routine
        just to keep notify_update from getting too convoluted.
        '''
        if(spcname == "CH4"):
            if self.dataN2O is not None:
                data2 = self.dataN2O
            else:
                # Fake the data
                logger.warn("code has not been tested for species_name CH4 and dataN2O is None")
                data2 = copy.deepcopy(dataInfo)
                indn = self.stateInfo.species.index('N2O')
                value = self.stateInfo.initial['values'][indn, :]
                data2['SPECIES'][data2['SPECIES'] > 0] = copy.deepcopy(value)
                data2['INITIAL'][data2['SPECIES'] > 0] = copy.deepcopy(value)
                data2['CONSTRAINTVECTOR'][data2['SPECIES'] > 0] = copy.deepcopy(value)
                data2['AVERAGINGKERNEL'].fill(0.0)
                data2['OBSERVATIONERRORCOVARIANCE'].fill(0.0)
        elif(spcname == "HDO"):
            data2 = self.dataH2O
        else:
            data2 = None

        if(spcname == "H2O" and self.dataTATM is not None):
            out_fname = f"{self.retrieval_strategy.output_directory}/Products/Lite_Products_L2-RH-{self.species_count[spcname]}.nc"
            if("OCO2" not in self.instruments):
                liteDirectory = '../OSP/Lite/'
                # Code assumes we are in rundir
                with self.retrieval_strategy.chdir_run_dir():
                    mpy.make_lite_casper_script_retrieval(self.table_step,
                                  out_fname, self.quality_name, self.instruments,
                                  liteDirectory, dataInfo, self.dataTATM, "RH",
                                  step=self.species_count[spcname],
                                  times_species_retrieved=self.species_count[spcname])
                
        out_fname = f"{self.retrieval_strategy.output_directory}/Products/Lite_Products_L2-{spcname}-{self.species_count[spcname]}.nc"
        if 'OCO2' not in self.instruments:
            liteDirectory = '../OSP/Lite/'
            # Code assumes we are in rundir
            with self.retrieval_strategy.chdir_run_dir():
                data2 = mpy.make_lite_casper_script_retrieval(self.table_step,
                                  out_fname, self.quality_name, self.instruments,
                                  liteDirectory, dataInfo, data2, spcname,
                                  step=self.species_count[spcname],
                                  times_species_retrieved=self.species_count[spcname])
            
                
                    

__all__ = ["RetrievalJacobianOutput", "RetrievalL2Output", "RetrievalRadianceOutput"] 
