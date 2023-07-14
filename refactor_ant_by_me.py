import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path
from frlearn.classifiers import FRNN
from sklearn.model_selection import train_test_split
from frlearn.base import probabilities_from_scores
from sklearn.metrics import roc_auc_score


index = ["bkblk","bknwy","bkon8","bkona","bkspr","bkxbq","bkxcr","bkxwp","blxwp","bxqsq","cntxt","dsopp","dwipd",
         "hdchk","katri","mulch","qxmsq","r2ar8","reskd","reskr","rimmx","rkxwp","rxmsq","simpl","skach","skewr",
         "skrxp","spcop","stlmt","thrsk","wkcti","wkna8","wknck","wkovl","wkpos","wtoeg","win"]


data = pd.read_csv('/home/hamid/hamash_amir/research/rough_set/kr-vs-kp_csv.csv')
data.rename(columns = {'class':"win"}, inplace=True)


ns = pd.get_dummies(data[data.columns[:-1]], prefix=data.columns[:-1],drop_first=True, dtype=int)

ns_y = data['win'].copy()
Y = pd.get_dummies(ns_y,prefix='win',drop_first=True, dtype=int)

class Ant:
    
    def __init__(self, colony):
        self.colony = colony
        self.fg = self.colony.feature_choices.copy() 
        self.ant_route = []

    def choose_feature(self):
        ant_route = self.ant_route
        features = self.colony.dataframe.columns.tolist()
        if ant_route == []:
            feature1 = np.random.choice(self.colony.feature_choices.copy())
            i = features.index(feature1)
            self.ant_route.append(i)
        else:
            feature1 = self.colony.feature_choices[self.ant_route[-1]]
        # feature_index = np.random.choice(range(len(self.fg)))
        # self.ant_route.append()
        if feature1 in self.fg:
            self.fg.remove(feature1)
        i = features.index(feature1)
        if i not in self.ant_route:
            self.ant_route.append(i)
        feat1_dist_prob = {}
        for feature2 in self.fg:
           
            pij = self.colony.probability_transition_rule(feature1, feature2, alpha=0.9, beta=0.3)
      
            j = features.index(feature2)
            feat1_dist_prob[j] = pij
        


        keys_with_max_value = [key for key,value in feat1_dist_prob.items() if value == max(feat1_dist_prob.values())]
        print(f'cls.fg in ant class: {[self.colony.feature_choices.index(value) for value in self.fg]}')

        if keys_with_max_value:
            ant_next_target_index = keys_with_max_value[0]
            self.colony.traversed_nodes[i,ant_next_target_index] = 1

            self.ant_route.append(ant_next_target_index)    
        return self.ant_route

    def build_route(self, route_length):
        if self.colony.colony_number == 0:
            for _ in range(route_length):
                feature = self.choose_feature()
                # i = self.colony.feature_choices.copy()
                # j = i.index(feature)
                # self.ant_route.append(j)
                
    def build_next_gen_route(self, THRESHOLD: list[int,str,None]):
        k = 0
        while True:
            feature = self.choose_feature()
            i = self.colony.feature_choices.copy()
            # j = i.index(feature)
            # self.ant_route.append(j)
            if THRESHOLD[1] == 'accuracy':
                is_criteria_met, criteria = self.colony.is_rough_set_criteria_met(self.ant_route ,THRESHOLD[0])
            if THRESHOLD[1] == 'landa':
                is_criteria_met, criteria = self.colony.is_landa_met(self.ant_route , THRESHOLD[2], THRESHOLD[0])
                
            if is_criteria_met:
                print(f'stop with this criteria {criteria}\n'
                        f'with this ant_route{self.ant_route}\n')
                print(f'self.colony.fg: {[self.colony.feature_choices.index(x) for x in self.colony.fg]}')
                # self.ant_route = []
                break
            elif not is_criteria_met and self.colony.fg == []:
                print(f'ant couldnt find route with {criteria} accuracy')
                # self.ant_route = []
                break
            k += 1



class Colony:
    


    dataframe: object = ns
    pheromone = np.ones((ns.shape[1], ns.shape[1]))
    traversed_nodes = np.zeros((ns.shape[1],ns.shape[1]))
    ant_route: list[int] = []
    alpha: float | int 
    beta: float | int 
    feature_choices = dataframe.columns.tolist()
    feature1: str
    log: dict = {}
    fg = dataframe.columns.tolist() 
    acc_criteria: float
    rho: int | float
    delta = np.zeros((ns.shape[1],ns.shape[1]))
    colony_number: int = 0
    overall_ant_route: dict = {}
    

    
    
    @classmethod
    def set_stopping_criteria(cls, criteria):
        cls.acc_criteria = criteria
    
    @classmethod
    def set_rate_decay(cls, rate_decay):
        cls.rho = rate_decay
    
    @classmethod
    def add_generation(cls):
        cls.colony_number += 1

    
    @classmethod
    def get_log(cls):
        print(cls.log)
    
    @classmethod
    def reset_colony(cls):
        
        Colony.pheromone = np.ones((ns.shape[1], ns.shape[1]))
        Colony.traversed_nodes = np.zeros((ns.shape[1], ns.shape[1]))

        Colony.overall_ant_route = {}
        Colony.log = []
        Colony.fg = cls.feature_choices.copy()
    
    @classmethod
    def reset_fg(cls, j: int | None=None):
        """
        this function reset fg that is responsible for feature space possible for each ant
        and remove first choice of each ant from this space to avoid selecting same features 
        by next ant as beginner feature.AND AND AND
        
        remove first choice of each ant from fg to prevent next ants choosing that
        and each ant beging from different node as begining.
        """
        
        first_choice_of_ants = [x[0] for x in list(cls.overall_ant_route.values())]
        print(f'first_choice_of_ants: {first_choice_of_ants}')
        str_of_first_choice = [cls.feature_choices[l] for l in first_choice_of_ants[j:]]
        print(f'str_of_first_choice: {str_of_first_choice}')
        cls.fg = cls.feature_choices
        for y in str_of_first_choice: 
            if y in cls.fg:
                cls.fg.remove(y)
        for x in first_choice_of_ants:
            cls.feature_choices[x]
        
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
        if ant_route == []:
            cls.feature1 = np.random.choice(cls.feature_choices.copy())
        else:
            cls.feature1 = cls.feature_choices[cls.ant_route[-1]]
    
    @classmethod
    def ant(cls):
        ant_instance = Ant(cls)
        ant_instance.build_route(init_criteria)
        cls.ant_route.append(ant_instance.ant_route)

    
    @classmethod
    def ants(
        cls, make_initialize: bool | None=None,
            number_of_ants_first_gen: int | None=None,
            number_of_ants_next_gen: int | None=None, 
            init_criteria: int | str | None=None,
            rate_decay: float | None=None, 
            phero_rate_decay: float | None=None):

        if cls.colony_number == 0 or make_initialize == True:
            cls.initialize_colony(
                number_of_ants_first_generation=number_of_ants_first_gen,
                                init_criteria=init_criteria)
            cls.generate_next_ants(
                number_of_ants_next_generation=number_of_ants_next_gen,
                rate_decay=rate_decay, phero_rate_decay=phero_rate_decay,
                criteria_func=cls.is_rough_set_criteria_met)
            

        
    @classmethod
    def change_pheromone(cls, q, rho):
        """
        After each generation of several ants or after each generation 
        it update pheromone matrix according to following formula:
         
        phromone[i,j](t+1) = (1-rho)*phromone[i,j](t) + Delta(ij)(t)
        
        rho is pheromone decay coefficient along time
        
        Delta(ij)(t) = |_| = Delta(ij)(t) = 
                       |_|   q/sigma(landa_prime(ant_route)/len(ant_route)) ,if traversed_node[i,j] = 1
                       |_| = 0                                            ,if traversed_node[i,j] = 0
        landa_prime is rough_set_measure           
        """
        cls.set_rate_decay(rho)
        minimum_ant_route_len = np.min([len(x) for x in list(cls.overall_ant_route.values())])
        for i in range(cls.delta.shape[0]):
            for j in range(cls.delta.shape[0]):
                if cls.traversed_nodes[i,j] == 1:
                    cls.delta[i,j] = q/minimum_ant_route_len
                    cls.pheromone[i,j] = (1 - cls.rho)*cls.pheromone[i,j] + cls.delta[i,j]
                elif cls.traversed_nodes[i,j] == 0:
                    cls.delta[i,j] = 0
                    cls.pheromone[i,j] = (1 - cls.rho)*cls.pheromone[i,j] + cls.delta[i,j]
        

    @classmethod
    def positive_region(cls):

        df = cls.dataframe
        partitions = [group for _, group in df.groupby(df.iloc[:-1])]
    
        # find positive region for each partition
        positive_regions = [group[group.duplicated(df.columns[:-1], keep=False)] for group in partitions]
    
        # return union of all positive regions
        return pd.concat(positive_regions)


    
    @classmethod
    def probability_transition_rule(cls, feature1, feature2, alpha, beta):
        col_index = ns.columns.tolist()
        i = col_index.index(feature1)
        j = col_index.index(feature2)
        l = col_index.copy()
        

        feat1 = feature1
        feat2 = feature2

        mic_ij = mutual_info_classif(ns[[feat1, feat2]], y=Y['win_won'], random_state=0)[1]
        phrmn_ij = cls.pheromone[i,j]
        init = 0
        for k in range(0,len(l)):
            if k != l.index(feature2):
                phrmn_il = cls.pheromone[i,k]
                mic_il = cls.pheromone[i,k]
                init += (phrmn_il**alpha) * (mic_il**beta)
        if init == 0:
            print("init is zero!")
        pij = ((mic_ij**beta) * (phrmn_ij**alpha)) / init
        return pij

    @classmethod
    def initialize_colony(cls, number_of_ants_first_generation, init_criteria):
        cls.fg = cls.feature_choices.copy()
        cls.overall_ant_route = {}
        first_choose = []
        for i in range(number_of_ants_first_generation):
            ant_instance = Ant(cls)
            ant_instance.build_route(init_criteria)
            cls.overall_ant_route[i] = ant_instance.ant_route
            first_choose.append(ant_instance.ant_route[0])    
            cls.log[i] = ant_instance.ant_route
            print(f'ant_route: {ant_instance.ant_route}')
            cls.fg = ns.columns.tolist()
            for i in first_choose:
                if cls.feature_choices[i] in cls.fg:
                    cls.fg.remove(cls.feature_choices[i])
    
           
            cls.reset_ant_route()
            print(f'cls.fg: {[cls.feature_choices.index(x) for x in cls.fg]}')
            # j += 1
        # cls.reset_ant_route()                                                                              
        cls.fg = ns.columns.tolist()
        cls.add_generation()
        print(f'initiall final: {[cls.feature_choices.index(x) for x in cls.fg]}')
        print('#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#')

    @classmethod
    def is_rough_set_criteria_met(cls, selected_feature: list[int], acc_criteria: float | int):

        cls.set_stopping_criteria(acc_criteria)
        if selected_feature != [] and len(selected_feature) > 1:
            data = ns.iloc[:, selected_feature]
            X_train, X_test, y_train, y_test = train_test_split(data.values, Y.values.squeeze(), test_size=0.33, random_state=42)
            
            clf = FRNN()
            model = clf(X_train, y_train)


            scores = model(X_test)

            probabilities = probabilities_from_scores(scores)
            auroc = roc_auc_score(y_test, probabilities[:,-1])
            
            
            if auroc >= cls.acc_criteria:
                cls.log['acc'] = auroc
                return True, auroc
            else:
                return False, auroc
        else:
            return False, 0.0

        
    @classmethod
    def is_landa_met(
        cls, selected_feature: list[int], sim_threshold = 0.5, landa_criteria = 0.8):
        """
        sim_threshold: similarity of each pair of instances that more than that
        these pairs trace as Similar together
        landa_criteria: stop one ant route exploration if met.more landa equall less redundant.  
        """
        if selected_feature != [] and len(selected_feature) > 1:

            X = ns.iloc[:, selected_feature].values
            y = Y.values.squeeze()
            R_a = np.minimum(np.maximum(1 - np.abs(X[:, None, :] - X), 0), 
                             y[:, None, None] != y[:, None])

            # Calculate the differences between pairs of instances
            differences = []
            all = []
            for i in range(len(X)):
                for j in range(i+1, len(X)):
                    difference = np.count_nonzero(R_a[i, j])
                    similarity_percent = 1 - (difference/X.shape[1])
                    if similarity_percent > sim_threshold and i != j:
                        differences.append(difference)
                    all.append(difference)

            # Calculate the average difference
            criteria = len(differences) / ((X.shape[0]**2)/2)

            landa = criteria
            cls.log['landa'] = landa
            if landa >= landa_criteria:
                return True, landa
            elif landa < landa_criteria:
                return False , landa
        else:
            return False, 0.0

    @classmethod 
    def generate_next_ants(
        cls, number_of_ants_next_generation: int,
        rate_decay,
        phero_rate_decay,
        THRESHOLD: list[int,str,None]
        ):
        """
        THRESHOLD[0]: criteria or limit for accuracy or landa
        THRESHOLD[1]: 'accuracy' or 'landa' 
        THRESHOLD[2]: similarity between two rows that when touch it rows
                      known as ROUGH. 
        
        THRESHOLD[0]: criteria limit
        THRESHOLD[1]: if it was accuracy then criteria limit = [0,1] or 
        if it was equall landa then criteria limit = [0,1] AND THRESHOLD[2]
        must be enter
        """

        if cls.colony_number > 0:
            j = 0
            cls.fg = cls.feature_choices.copy()
            first_choose = []
            while j < number_of_ants_next_generation:

                next_ant = Ant(cls)
                next_ant.build_next_gen_route(THRESHOLD)
                cls.overall_ant_route[j] = next_ant.ant_route
                cls.log[j] = next_ant.ant_route
                print(f'ant_route: {next_ant.ant_route}')
                
                first_choose.append(next_ant.ant_route[0])
                cls.fg = ns.columns.tolist()
                for i in first_choose:
                    if i in cls.fg:
                        cls.fg.remove(i)   
                j += 1
            cls.change_pheromone(q=rate_decay, rho=phero_rate_decay)
        else:
            print("Colony generation doesnt initialized first!")

        
Colony.initialize_colony(number_of_ants_first_generation=5,init_criteria=4)
Colony.generate_next_ants(number_of_ants_next_generation=5,rate_decay=0.5,phero_rate_decay=0.3, THRESHOLD=[0.92,'accuracy',0.7])

print(Colony.overall_ant_route)