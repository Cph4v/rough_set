import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_selection import mutual_info_classif

index = ["bkblk","bknwy","bkon8","bkona","bkspr","bkxbq","bkxcr","bkxwp","blxwp","bxqsq","cntxt","dsopp","dwipd",
 "hdchk","katri","mulch","qxmsq","r2ar8","reskd","reskr","rimmx","rkxwp","rxmsq","simpl","skach","skewr",
 "skrxp","spcop","stlmt","thrsk","wkcti","wkna8","wknck","wkovl","wkpos","wtoeg","win"]


data = pd.read_csv('/home/teslator/projects/rough_set/kr-vs-kp_csv.csv')
data.rename(columns = {'class':"win"}, inplace=True)

ns = pd.get_dummies(data[data.columns[:-1]], prefix=data.columns[:-1],drop_first=True, dtype=int)

ns_y = data['win'].copy()
Y = pd.get_dummies(ns_y,prefix='win',drop_first=True, dtype=int)


class colony:
    


    dataframe: object = ns
    pheromone = np.ones((ns.shape[1], ns.shape[1]))
    traversed_nodes = np.zeros((ns.shape[1],ns.shape[1]))
    ant_route: list[int] = []
    alpha: int = 0
    beta: int = 0
    feature_choices: list[str] = dataframe.columns.tolist()
    feature1: str
    log: list[str] = []
    fg: list[str] = feature_choices.copy()
    generation_number: int = 0
    # if not cls.ant_route:
    #     cls.feature1 = np.random.choice(cls.feature_choices)
    # else:
    #     cls.feature1 = cls.feature_choices[cls.ant_route[-1]]
    
    @classmethod
    def new_generation(cls):
        cls.generation_number += 1
        # cls.reset_colony() 
    
    @classmethod
    def get_log(cls):
        print(cls.log)
    
    @classmethod
    def reset_colony(cls):
        
        colony.pheromone = np.ones((ns.shape[1], ns.shape[1]))
        colony.traversed_nodes = np.zeros((ns.shape[1],ns.shape[1]))
        # colony.ant_route = []
        # colony.feature1 = ''
        colony.log = []
        colony.fg = cls.feature_choices.copy()
        
    @classmethod
    def initialization_alpha_beta(cls, alpha, beta):
        
        cls.alpha = alpha
        cls.beta = beta

    @classmethod
    def reset_ant_route(cls):
        cls.ant_route = []

    @classmethod
    def initialize_feature1(cls):
        ant_route = cls.ant_route
        if not ant_route:
            cls.feature1 = np.random.choice(cls.feature_choices)
        else:
            cls.feature1 = cls.feature_choices[cls.ant_route[-1]]
    
    
    @classmethod
    def ant(cls):
        # alpha = 0.5
        # beta = 0.5
        # traversed_nodes = np.zeros((ns.shape[1],ns.shape[1]))
        # if ant_route.empty() == False:
        #     ant_route = []
        # pair_nodes = {}
        cls.initialization_alpha_beta(0.5, 0.5)
        cls.initialize_feature1()
        feature1 = cls.feature1
        features = cls.dataframe.columns.tolist()
         
        cls.fg.remove(feature1)
        i = features.index(feature1)
        if i not in cls.ant_route:
            cls.ant_route.append(i)
        print(f'ant_route_1 -- {cls.ant_route}')

        ### below line cause loop doesnt choose from chosen features
        ### and nodes 
        ### we remove ant_route frome fg 
        # anty = cls.ant_route.copy()
        # if i in anty and anty[0] != i:
        #     anty.remove(i)
        #     print(f'antroute2 -- {anty}')
        #     for x in anty:
        #         fg.remove(features[x])
        ###

        
        
        # ant_route = []
        # ant_route.append(i)
        feat1_dist_prob = {}
        for feature2 in cls.fg:
            # if feature1 != feature2:
            pij = cls.__probability_transition_rule(cls.feature1, feature2)
            # i = features.index(feature1)
            j = features.index(feature2)
            feat1_dist_prob[j] = pij
            # ant_route.append(i)
        
            cls.log.append(f'{features.index(cls.feature1)}' + f'--{features.index(feature2)}--' + f'{pij}')
        
            # feat1_dist_prob.append(dist_prob)
    
        ant_next_target_index = [key for key, value in feat1_dist_prob.items() if value == max(feat1_dist_prob.values())][0]
        cls.traversed_nodes[i,ant_next_target_index] = 1
        # mitting criterion
        cls.ant_route.append(ant_next_target_index)
        fg_index = [key for key,value in feat1_dist_prob.items()]
        # return fg, fg_index, feat1_dist_prob
    
    @classmethod
    def ants(cls, make_initialize: bool | None, number_of_ants, 
             criteria: int | bool = cls.is_rough_set_criteria_met(cls.ant_route)):
        
        try:
            if make_initialize == True and criteria == int:
                cls.initialize_colony(number_of_ants_first_generation=number_of_ants,  init_criteria=criteria)
            elif (make_initialize == False or make_initialize == None) and criteria == bool:
                cls.generate_next_ants(number_of_ants_next_generation=number_of_ants,rough_set_criteria=criteria)
            else:
                error = Exception("enter valid 'make_initialize' and 'criteria'")
                raise error
        except:
            print(error)



    @classmethod
    def __probability_transition_rule(cls, feature1, feature2):
        col_index = ns.columns.tolist()
        i = col_index.index(feature1)
        j = col_index.index(feature2)
        l = col_index.copy()
        
        # l.remove(feature2)
        # mic_ij = mutual_info_classif(X=ns.iloc[:,[i,j]], y=Y['win_won'])
        feat1 = feature1
        feat2 = feature2
        # in formula below : [1] in last stage refer to feat2 mutual info regards to system consist of:
        # feat1 and Y as Target 
        mic_ij = mutual_info_classif(ns[[feat1, feat2]], y=Y['win_won'], random_state=0)[1]
        phrmn_ij = cls.pheromone[i,j]
        init = 0
        for k in range(0,len(l)):
            if k != l.index(feature2):
                phrmn_il = cls.pheromone[i,k]
                mic_il = cls.pheromone[i,k]
                init += (phrmn_il**cls.alpha) * (mic_il**cls.beta)
        pij = ((mic_ij**cls.beta) * (phrmn_ij**cls.alpha)) / init
        return pij

    @classmethod 
    def initialize_colony(cls, number_of_ants_first_generation, init_criteria):
        """
        first of all we run this method for initialize first generation of colony
        in this method we set manually number_of_ants_first_generation variable to a number
        As a CRITERIA just for initialize pheromone matrix.in next generations we set rough set feature selection CRITERIA and when selected features by each ant in a colony met this limit that ant stop exploration and next ant begins.
        CAUTIONS!: RUN THIS METHOD JUST ONE TIME IN EACH COLONY!
        """ 
        
        try:
            if cls.generation_number == 0:
                cls.new_generation()
                cls.reset_colony()
                i = 0
                while i <= init_criteria: 
                # for j in range(number_of_ants_first_generation):
                    cls.ant()
                    i += 1
                    if (cls.ant_route()) <= init_criteria:
                        cls.reset_ant_route()
                        continue
            else:
                error = Exception("This is NOT first generation!")
                raise error
        except:
            print(error)

    @classmethod
    def is_rough_set_criteria_met(cls, selected_feature: list[int]) -> bool:
        pass
        
    @classmethod 
    def generate_next_ants(cls, number_of_ants_next_generation: int,
                           rough_set_criteria: bool):
        
        try:
            if cls.generation_number > 0:

                i = 0
                while cls.is_rough_set_criteria_met(cls.ant_route): 
                # for j in range(number_of_ants_first_generation):
                    cls.ant()
                    i += 1
                    if (cls.ant_route()) <= init_criteria:
                        cls.reset_ant_route()
                        continue
                
                for i in range(number_of_ants_next_generation):
                    colony.ant()
                    i += 1
                    if cls.is_rough_set_criteria_met(colony.ant_route()):
                        cls.reset_ant_route()
                        continue
            else:
                error = Exception("Colony generation doesnt initialized first!")
                raise error
        except:
            print(error)
        
      
# colony.reset_colony()
# colony.ant()
# colony.ant()
# colony.ant()
# colony.feature1
# colony.ant_route
