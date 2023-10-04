import refractor.muses.muses_py as mpy

class RefractorRetrievalInfo:
    '''Not sure if we'll keep this or not, but pull out RetrievalInfo stuff so
    we can figure out the interface and if we should replace this.

    A few functions seem sort of like member functions, we'll just make a list
    of these to sort out later but not try to get the full interface in place.

    update_state - I think this updates just doUpdateFM
    create_uip - This just reads values, mostly copying stuff over to the UIP
    systematic_jacobian - Pretty much just make the dummy retrieval_info_temp
    write_retrieval_input - Probably go away, since this is debug output and wouldn't
                            really apply if we change RetrievalInfo
    plot_results
    error_analysis_wrapper - Lots of stuff read here
    write_retrieval_summary
    write_products_one
    '''
    def __init__(self, retrieval_dict):
        self.retrieval_dict = retrieval_dict

    @property
    def retrieval_info_obj(self):
        return mpy.ObjectView(self.retrieval_dict)

    @property
    def initialGuessList(self):
        '''This is the initial guess for the state vector (not the full state)'''
        return self.retrieval_dict["initialGuessList"]

    @property
    def type(self):
        return self.retrieval_dict["type"]

    @property
    def apriori_cov(self):
        return self.retrieval_dict["Constraint"][0:self.n_totalParameters,0:self.n_totalParameters]

    @property
    def apriori(self):
        return self.retrieval_dict["constraintVector"][0:self.n_totalParameters]

    @property
    def species_names(self):
        return self.retrieval_dict["species"][0:self.retrieval_dict["n_species"]]

    @property
    def species_list_fm(self):
        return self.retrieval_dict["speciesListFM"][0:self.n_totalParametersFM]

    @property
    def pressure_list_fm(self):
        return self.retrieval_dict["pressureListFM"][0:self.n_totalParametersFM]

    # Synonyms used in the muses-py code.
    @property
    def speciesListFM(self):
        return self.species_list_fm

    @property
    def pressureListFM(self):
        return self.pressure_list_fm
    
    @property
    def n_species(self):
        return len(self.species_names)

    @property
    def minimumList(self):
        return self.retrieval_dict["minimumList"][0:self.n_totalParameters]

    @property
    def maximumList(self):
        return self.retrieval_dict["maximumList"][0:self.n_totalParameters]

    @property
    def maximumChangeList(self):
        return self.retrieval_dict["maximumChangeList"][0:self.n_totalParameters]
    
    @property
    def species_list_fm(self):
        return self.retrieval_dict["speciesListFM"][0:self.retrieval_dict["n_totalParametersFM"]]
    
    @property
    def n_totalParameters(self):
        # Might be a better place to get this, but start by getting from
        # initial guess
        return self.initialGuessList.shape[0]

    @property
    def n_totalParametersSys(self):
        return self.retrieval_dict["n_totalParametersSys"]

    @property
    def n_totalParametersFM(self):
        return self.retrieval_dict["n_totalParametersFM"]
    
    @property
    def n_speciesSys(self):
        return self.retrieval_dict["n_speciesSys"]
    
    @property
    def __doUpdateFM(self):
        return self.retrieval_dict["doUpdateFM"][0:self.n_totalParametersFM]

    @property
    def __initialGuessListFM(self):
        '''This is the initial guess for the FM state vector'''
        return self.retrieval_dict["initialGuessListFM"]

    @property
    def __species(self):
        # Not clear why these arrays are fixed size. Probably left over from IDL,
        # but go ahead and trim this
        return self.retrieval_dict["species"][0:self.retrieval_dict["n_species"]]

    @property
    def __n_species(self):
        return len(self.species)
        
    @property
    def __n_totalParametersFM(self):
        # Might be a better place to get this, but start by getting from
        # initial guess
        return self.initialGuessListFM.shape[0]
    
