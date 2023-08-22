
class Ant:
    
    def reset_colony(self):
        Colony.pheromone = np.ones((ns.shape[1], ns.shape[1]))
        Colony.traversed_nodes = np.zeros((ns.shape[1], ns.shape[1]))
        Colony.overall_ant_route = {}
        Colony.log = []
        Colony.fg = self.feature_choices.copy()


    def reset_fg(self, j: int | None=None):
        first_choice_of_ants = [x[0] for x in list(self.overall_ant_route.values())]
        str_of_first_choice = [self.feature_choices[l] for l in first_choice_of_ants[j:]]
        self.fg = self.feature_choices
        for y in str_of_first_choice: 
            if y in self.fg:
                self.fg.remove(y)
        for x in first_choice_of_ants:
            self.feature_choices[x]


    def initialization_alpha_beta(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta


    def reset_ant_route(self):
        self.ant_route = []


    def initialize_feature1(self):
        if not self.ant_route:
            self.feature1 = np.random.choice(self.feature_choices.copy())
        else:
            self.feature1 = self.feature_choices[self.ant_route[-1]]


    def generation(
        self, make_initialize: bool = None,
        number_of_ants_first_gen: int = None,
        number_of_ants_next_gen: int = None, 
        init_criteria: Union[int, str, None] = None,
        rate_decay: float = None, 
        phero_rate_decay: float = None,
        make_change_pheromone_init: bool = None,
        make_change_pheromone_gen: bool = None,
        with_core: bool = None,
        core_accuracy: float = None,
        q_in_init: float = None,
        q_in_gen: float = None,
        THRESHOLD: List[Union[int, str, None]] = None,
        alpha: Union[float, int, None] = None,
        beta: Union[float, int, None] = None,
        number_of_generations: int = None
        ) -> None:
        
        if self.colony_number == 0 or make_initialize:
            self.initialize_colony(
                number_of_ants_first_generation=number_of_ants_first_gen,
                init_criteria=init_criteria,
                q=q_in_init,
                phero_rate_decay=phero_rate_decay,
                make_change_pheromone=make_change_pheromone_init,
                with_core=with_core,
                core_accuracy=core_accuracy
                )
        for _ in range(number_of_generations):
            self.generate_next_ants(
                number_of_ants_next_generation=number_of_ants_next_gen,
                q=q_in_gen,
                phero_rate_decay=phero_rate_decay,
                THRESHOLD=THRESHOLD,
                alpha=alpha,
                beta=beta,
                with_core=with_core,
                change_pheromone=make_change_pheromone_gen
                )


    def change_pheromone(self, q, rho):
        minimum_ant_route_len = np.min([len(x) for x in list(Colonies.overall_ant_route.values())])
        for i in range(Colonies.delta.shape[0]):
            for j in range(Colonies.delta.shape[0]):
                if Colonies.traversed_nodes[i,j] == 1:
                    Colonies.delta[i,j] = q/minimum_ant_route_len
                    Colonies.pheromone[i,j] = (1 - self.rho) * Colonies.pheromone[i,j] + Colonies.delta[i,j]
                else:
                    Colonies.delta[i,j] = 0
                    Colonies.pheromone[i,j] = (1 - self.rho) * Colonies.pheromone[i,j] + Colonies.delta[i,j]


    def positive_region(self):
        df = self.dataframe
        partitions = [group for _, group in df.groupby(df.iloc[:-1])]
        positive_regions = [group[group.duplicated(df.columns[:-1], keep=False)] for group in partitions]
        return pd.concat(positive_regions)


    def probability_transition_rule(self, feature1, feature2, alpha, beta):
        col_index = ns.columns.tolist()
        i = col_index.index(feature1)
        j = col_index.index(feature2)
        l = col_index.copy()
        phrmn_ij = Colonies.pheromone[i,j]
        init = 0
        for k in range(len(l)):
            if k != l.index(feature2):
                phrmn_il = Colonies.pheromone[i,k]
                mic_il = Colonies.pheromone[i,k]
                init += (phrmn_il**alpha) * (mic_il**beta)
        if init == 0:
            print("init is zero!")
        pij = ((mic_ij**beta) * (phrmn_ij**alpha)) / init
        return pij


    def initialize_colony(
            self, number_of_ants_first_generation, init_criteria,
            q: float = None,
            phero_rate_decay: float = None,
            make_change_pheromone: bool = None,
            with_core: bool = None,
            core_accuracy: float = None 
            ) -> None:
        
        overall_ant_route_init = {}
        if with_core:
            self.find_core_feature(core_accuracy)
        first_choose = []
        for i in range(number_of_ants_first_generation):
            ant_instance = Ant(self)
            ant_instance.build_route_init(init_criteria)
            Colonies.overall_ant_route_init[i] = ant_instance.ant_route
            first_choose.append(ant_instance.ant_route[0])    
            overall_ant_route_init[f'ant_init:{i}'] = ant_instance.ant_route
        if make_change_pheromone:
            self.change_pheromone(q=q, rho=phero_rate_decay)
            self.phero[f'colony_number -> {self.colonies_instance.colony_number}| initialization_step'] = f'pheromone -> {Colonies.pheromone}'
        self.add_colony_number()
        self.overall_route[f'colony_number -> {self.colonies_instance.colony_number}| initialization_step'] = f'overall_ant_route_init -> {overall_ant_route_init}'


    def initialize_colony(
            self, number_of_ants_first_generation, init_criteria,
            q: float = None,
            phero_rate_decay: float = None,
            make_change_pheromone: bool = None,
            with_core: bool = None,
            core_accuracy: float = None 
            ) -> None:
        
        overall_ant_route_init = {}
        if with_core:
            self.find_core_feature(core_accuracy)
        first_choose = []
        for i in range(number_of_ants_first_generation):
            ant_instance = Ant(self)
            ant_instance.build_route_init(init_criteria)
            Colonies.overall_ant_route_init[i] = ant_instance.ant_route
            first_choose.append(ant_instance.ant_route[0])    
            overall_ant_route_init[f'ant_init:{i}'] = ant_instance.ant_route
        if make_change_pheromone:
            self.change_pheromone(q=q, rho=phero_rate_decay)
            self.phero[f'colony_number -> {self.colonies_instance.colony_number}| initialization_step'] = f'pheromone -> {Colonies.pheromone}'
        self.add_colony_number()
        self.overall_route[f'colony_number -> {self.colonies_instance.colony_number}| initialization_step'] = f'overall_ant_route_init -> {overall_ant_route_init}'


    def generate_next_ants(
            self, number_of_ants_next_generation,
            q: float = None, 
            phero_rate_decay: float = None,
            THRESHOLD: List[Union[int, str, None]] = None,
            alpha: Union[float, int, None] = None,
            beta: Union[float, int, None] = None,
            with_core: bool = None,
            change_pheromone: bool = None
            ) -> None:
        
        overall_ant_route_next = {}
        for i in range(number_of_ants_next_generation):
            ant_instance = Ant(self)
            ant_instance.build_route_next(q=q, phero_rate_decay=phero_rate_decay, THRESHOLD=THRESHOLD, alpha=alpha, beta=beta)
            overall_ant_route_next[f'ant_next:{i}'] = ant_instance.ant_route
        if change_pheromone:
            self.change_pheromone(q=q, rho=phero_rate_decay)
            self.phero[f'colony_number -> {self.colonies_instance.colony_number}| generation_step'] = f'pheromone -> {Colonies.pheromone}'
        self.add_colony_number()
        self.overall_route[f'colony_number -> {self.colonies_instance.colony_number}| generation_step'] = f'overall_ant_route_next -> {overall_ant_route_next}'
