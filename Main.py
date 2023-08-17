import networkx as nx
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path
from frlearn.classifiers import FRNN
from sklearn.model_selection import train_test_split
from frlearn.base import probabilities_from_scores
from sklearn.metrics import roc_auc_score
from frlearn.neighbours.instance_preprocessors import FRPS
import time
from frlearn.array_functions import soft_min
from frlearn.weights import ReciprocallyLinearWeights


index = ["bkblk","bknwy","bkon8","bkona","bkspr","bkxbq","bkxcr","bkxwp","blxwp","bxqsq","cntxt","dsopp","dwipd",
         "hdchk","katri","mulch","qxmsq","r2ar8","reskd","reskr","rimmx","rkxwp","rxmsq","simpl","skach","skewr",
         "skrxp","spcop","stlmt","thrsk","wkcti","wkna8","wknck","wkovl","wkpos","wtoeg","win"]


data = pd.read_csv('/home/hamid/hamash_amir/research/rough_set/kr-vs-kp_csv.csv')
data.rename(columns = {'class':"win"}, inplace=True)


ns = pd.get_dummies(data[data.columns[:-1]], prefix=data.columns[:-1],drop_first=True, dtype=int)

ns_y = data['win'].copy()
Y = pd.get_dummies(ns_y,prefix='win',drop_first=True, dtype=int)


class Ant:

    def __init__(self, colony) -> None:
        self.ant_route = []
        self.colony = colony
        self.fg = colony.feature_choices.copy()
        self.refrences = self.colony.feature_choices.copy()
        self.fg_core = self.colony.feature_choices.copy()
        self.first_choose_as_core_features = Colonies.core_feature
        self.colonies_instance = Colonies()
        # self.fg_core = colony.fg_without_core 

    def choose_feature(self) -> None:
        feature_value = np.random.choice(self.fg)
        feature_index = self.refrences.index(feature_value)
        if self.refrences[feature_index] in self.fg:
            self.fg.remove(self.refrences[feature_index])
        self.ant_route.append(feature_index)
         
        
    # def choose_feature_init_with_core(self):
    #     if self.colony.colony_number == 0:
    #         for _ in range(route_length):
    #             self.choose_feature()
        
    def choose_feature_gen_with_core(self, alpha, beta):
        """
        This function force ants to just start from core features and then 
        continue the path from choosing from 'fg' 
        """

        # self.fg_core = [value for value in self.refrences if self.refrences.index(value) not in Colonies.core_feature]
        # self.choose_feature_gen(alpha, beta)
        features = self.refrences
        if self.ant_route == []:
            # feature_value = np.random.choice(Colonies.core_feature)
            feature_index = np.random.choice(self.first_choose_as_core_features)
            self.first_choose_as_core_features = [x for x in Colonies.core_feature if x != feature_index]
            # feature_index = self.refrences.index(feature_value)
            # output_feature = feature_value
            
            # if self.refrences[feature_index] in self.fg:
            #     self.fg.remove(self.refrences[feature_index])
                
            self.ant_route.append(feature_index)

        else:
            feature_index = self.ant_route[-1]
            feature1 = self.refrences[feature_index]
            feat1_dist_prob = {}
            for feature2 in self.fg_core:    
                pij = self.colony.probability_transition_rule(feature1, feature2, alpha, beta)
                j = features.index(feature2)
                feat1_dist_prob[j] = pij
            keys_with_max_value = [key for key,value in feat1_dist_prob.items() if value == max(feat1_dist_prob.values())]
            print(f'self.fg_core in ant class: {[self.refrences.index(value) for value in self.fg_core]}')
            if keys_with_max_value:
                ant_next_target_index = keys_with_max_value[0]
                Colonies.traversed_nodes[feature_index, ant_next_target_index] = 1
                output_feature = ant_next_target_index
                
                if self.refrences[ant_next_target_index] in self.fg_core:
                    self.fg_core.remove(self.refrences[ant_next_target_index])
                
                # self.fg.remove(self.refrences[ant_next_target_index])
                self.ant_route.append(ant_next_target_index)
        # return output_feature
    
    def choose_feature_gen(self, alpha, beta) -> str:
        features = self.refrences
        if self.ant_route == []:
            feature_value = np.random.choice(self.fg)
            feature_index = self.refrences.index(feature_value)
            output_feature = feature_index
            
            if self.refrences[feature_index] in self.fg:
                self.fg.remove(self.refrences[feature_index])
                
            self.ant_route.append(feature_index)

        else:
            feature_index = self.ant_route[-1]
            feature1 = self.refrences[feature_index]
            feat1_dist_prob = {}
            for feature2 in self.fg:    
                pij = self.colony.probability_transition_rule(feature1, feature2, alpha, beta)
                j = features.index(feature2)
                feat1_dist_prob[j] = pij
            keys_with_max_value = [key for key,value in feat1_dist_prob.items() if value == max(feat1_dist_prob.values())]
            print(f'self.fg in ant class: {[self.refrences.index(value) for value in self.fg]}')
            if keys_with_max_value:
                ant_next_target_index = keys_with_max_value[0]
                Colonies.traversed_nodes[feature_index, ant_next_target_index] = 1
                output_feature = ant_next_target_index
                
                if self.refrences[ant_next_target_index] in self.fg:
                    self.fg.remove(self.refrences[ant_next_target_index])
                
                # self.fg.remove(self.refrences[ant_next_target_index])
                self.ant_route.append(ant_next_target_index)
        return output_feature

    def build_route_init(self, route_length: int,
                        #  with_core: bool | None=None
                         ) -> None:
        if self.colony.colony_number == 0:
            for _ in range(route_length):
                # if with_core == True:
                #     self.choose_feature_init_with_core()
                # else:
                self.choose_feature()

    def build_route_next_gen(self, THRESHOLD: list[int,str,None], alpha, beta, with_core: bool | None=None) -> None:
        
        if with_core == True:
            for i in Colonies.core_feature:
                if self.refrences[i] in self.fg_core:
                    self.fg_core.remove(self.refrences[i])
        timebase = 1000
        start = time.time()
        k = 0
        while True:
            if with_core == True:
                self.choose_feature_gen_with_core(alpha, beta)
            else:
                self.choose_feature_gen(alpha, beta)
                
            i = self.colony.feature_choices.copy()
            if THRESHOLD[1] == 'accuracy':
                is_criteria_met, criteria = self.colony.is_rough_set_criteria_met(self.ant_route ,THRESHOLD[0], with_core)
            if THRESHOLD[1] == 'landa':
                is_criteria_met, criteria = self.colony.is_landa_met(self.ant_route , THRESHOLD[0], with_core)
            stop = time.time()
            thresh = abs(stop - start)
            if is_criteria_met:
                print(f'Ant-gen: {k} stop with this criteria {criteria}\n'
                        f'with this ant_route{self.ant_route}\n')
                print(f'self.colony.fg: {[self.refrences.index(x) for x in self.fg]}')
                # self.ant_route = []
                break
            # elif (not is_criteria_met and self.fg == []) or (thresh > 350):
            elif (self.fg_core == [] or self.fg == []) or (thresh > 350):
                print(f'ant couldnt find route with {criteria} accuracy')
                print(f'time_per_thesh: {thresh}')
                # self.ant_route = []
                break
            k += 1
        stop = time.time()
        thresh = abs(stop - start)
        time_per_iteration = thresh/timebase
        # print(f'time_per_iteration: {time_per_iteration}')
        
class Colonies:
    
    def __init__(self):
        self.log_pheromone = {}
        self.log_traversed_nodes = {}
        self.log_overall_ant_route = {}
    
    pheromone = np.ones((ns.shape[1], ns.shape[1]))
    traversed_nodes = np.zeros((ns.shape[1],ns.shape[1]))
    overall_ant_route: dict = {}
    overall_ant_route_init: dict = {}
    delta = np.zeros((ns.shape[1],ns.shape[1]))
    core_feature: list[int] | None=None
    
    
    
class Colony:
    
    # dataframe: object = ns
    # alpha: float | int 
    # beta: float | int 
    # feature_choices = dataframe.columns.tolist()
    # feature1: str
    # log: dict = {}
    # fg = dataframe.columns.tolist() 
    # acc_criteria: float
    # rho: int | float
    # colony_number: int = 0
    # fg_without_core: list[str] | None=None
    
    def __init__(self):
        self.dataframe: object = ns
        self.alpha: float | int 
        self.beta: float | int 
        self.feature_choices = self.dataframe.columns.tolist()
        self.feature1: str
        self.log: dict = {}
        self.fg = self.dataframe.columns.tolist() 
        self.acc_criteria: float
        self.rho: int | float
        self.colony_number: int = 0
        self.fg_without_core: list[str] | None=None
        self.colonies_instance = Colonies()
        
    

    def set_stopping_criteria(self, criteria):
        self.acc_criteria = criteria
    

    def set_rate_decay(self, rate_decay):
        self.rho = rate_decay
    

    def add_generation(self):
        self.colony_number += 1

    

    def get_log(self):
        if Colonies.overall_ant_route:
            self.colonies_instance.log_overall_ant_route[f"{self.__class__.__name__}-gen"] = Colonies.overall_ant_route
        if Colonies.overall_ant_route_init:
            self.colonies_instance.log_overall_ant_route[f"{self.__class__.__name__}-init"] = Colonies.overall_ant_route_init
        # if Colonies.pheromone.any() != 1:
        self.colonies_instance.log_pheromone[f"{self.__class__.__name__}-pheromone"] = Colonies.pheromone
        if Colonies.traversed_nodes.any() == 1:
            self.colonies_instance.log_traversed_nodes[f"{self.__class__.__name__}-node"] = Colonies.traversed_nodes
        self.log[f'{self.colony_number} - overall_ant_route_init'] = Colonies.overall_ant_route_init
        self.log[f'{self.colony_number} - overall_ant_route'] = Colonies.overall_ant_route
        self.log[f'{self.colony_number} pheromone'] = Colonies.pheromone
        self.log[f'{self.colony_number} - traversed_nodes'] = Colonies.traversed_nodes
        # print(self.log)
    

    def reset_colony(self):
        Colony.pheromone = np.ones((ns.shape[1], ns.shape[1]))
        Colony.traversed_nodes = np.zeros((ns.shape[1], ns.shape[1]))
        Colony.overall_ant_route = {}
        Colony.log = []
        Colony.fg = self.feature_choices.copy()
    

    def reset_fg(self, j: int | None=None):
        """
        this function reset fg that is responsible for feature space possible for each ant
        and remove first choice of each ant from this space to avoid selecting same features 
        by next ant as beginner feature.AND AND AND
        
        remove first choice of each ant from fg to prevent next ants choosing that
        and each ant beging from different node as begining.
        """
        
        first_choice_of_ants = [x[0] for x in list(self.overall_ant_route.values())]
        print(f'first_choice_of_ants: {first_choice_of_ants}')
        str_of_first_choice = [self.feature_choices[l] for l in first_choice_of_ants[j:]]
        print(f'str_of_first_choice: {str_of_first_choice}')
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
        ant_route = self.ant_route
        if ant_route == []:
            self.feature1 = np.random.choice(self.feature_choices.copy())
        else:
            self.feature1 = self.feature_choices[self.ant_route[-1]]


    

    def ants(
        self, make_initialize: bool | None=None,
            number_of_ants_first_gen: int | None=None,
            number_of_ants_next_gen: int | None=None, 
            init_criteria: int | str | None=None,
            rate_decay: float | None=None, 
            phero_rate_decay: float | None=None):

        if self.colony_number == 0 or make_initialize == True:
            self.initialize_colony(
                number_of_ants_first_generation=number_of_ants_first_gen,
                                init_criteria=init_criteria)
            self.generate_next_ants(
                number_of_ants_next_generation=number_of_ants_next_gen,
                rate_decay=rate_decay, phero_rate_decay=phero_rate_decay,
                criteria_func=self.is_rough_set_criteria_met)
            

        

    def change_pheromone(self, q, rho):
        """
        After each generation of several ants or after each generation 
        it update pheromone matrix according to following formula:
         
        phromone[i,j](t+1) = (1-rho)*phromone[i,j](t) + Delta(ij)(t)
        
        :ivar rho is pheromone decay coefficient along time
        :ivar q is just an coefficient

        Delta(ij)(t) = |_| = Delta(ij)(t) = 
                       |_|   q/sigma(landa_prime(ant_route)/len(ant_route)) ,if traversed_node[i,j] = 1
                       |_| = 0                                            ,if traversed_node[i,j] = 0
        landa_prime is rough_set_measure           
        """
        self.set_rate_decay(rho)
        minimum_ant_route_len = np.min([len(x) for x in list(Colonies.overall_ant_route.values())])
        for i in range(Colonies.delta.shape[0]):
            for j in range(Colonies.delta.shape[0]):
                if Colonies.traversed_nodes[i,j] == 1:
                    Colonies.delta[i,j] = q/minimum_ant_route_len
                    Colonies.pheromone[i,j] = (1 - self.rho)*Colonies.pheromone[i,j] + Colonies.delta[i,j]
                elif Colonies.traversed_nodes[i,j] == 0:
                    Colonies.delta[i,j] = 0
                    Colonies.pheromone[i,j] = (1 - self.rho)*Colonies.pheromone[i,j] + Colonies.delta[i,j]
        


    def positive_region(self):

        df = self.dataframe
        partitions = [group for _, group in df.groupby(df.iloc[:-1])]
    
        # find positive region for each partition
        positive_regions = [group[group.duplicated(df.columns[:-1], keep=False)] for group in partitions]
    
        # return union of all positive regions
        return pd.concat(positive_regions)
    

    def probability_transition_rule(self, feature1, feature2, alpha, beta):
        col_index = ns.columns.tolist()
        i = col_index.index(feature1)
        j = col_index.index(feature2)
        l = col_index.copy()
        feat1 = feature1
        feat2 = feature2
        mic_ij = mutual_info_classif(ns[[feat1, feat2]], y=Y['win_won'], random_state=0)[1]
        phrmn_ij = Colonies.pheromone[i,j]
        init = 0
        for k in range(0,len(l)):
            if k != l.index(feature2):
                phrmn_il = Colonies.pheromone[i,k]
                mic_il = Colonies.pheromone[i,k]
                init += (phrmn_il**alpha) * (mic_il**beta)
        if init == 0:
            print("init is zero!")
        pij = ((mic_ij**beta) * (phrmn_ij**alpha)) / init
        return pij


    def initialize_colony(self,
                          number_of_ants_first_generation,
                          init_criteria,
                          q: float | None=None,
                          phero_rate_decay: float | None=None,
                          make_change_pheromone: bool | None=None,
                          with_core: bool | None=None,
                          core_accuracy: float | int | None=None
                          ):
        
        # self.overall_ant_route_init = {}
        # if with_core == True:
        #     self.find_core_feature()
        if with_core == True:
            self.find_core_feature(core_accuracy)
        first_choose = []
        for i in range(number_of_ants_first_generation):
            ant_instance = Ant(self)
            # if with_core == True:
            #     ant_instance.choose_feature_init_with_core()
            # else:    
            ant_instance.build_route_init(init_criteria)
            Colonies.overall_ant_route_init[i] = ant_instance.ant_route
            first_choose.append(ant_instance.ant_route[0])    
            self.log[i] = ant_instance.ant_route
           
            print(f'ant_route: {ant_instance.ant_route}')
            print(f'ant_instance.fg: {[self.feature_choices.index(x) for x in ant_instance.fg]}')
        if make_change_pheromone == True:
            self.change_pheromone(q=q, rho=phero_rate_decay)
        self.add_generation()
        self.get_log()
        print(f'initiall final: {[self.feature_choices.index(x) for x in ant_instance.fg]}')
        print('#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#')


    def is_rough_set_criteria_met(self, selected_feature: list[int],
                                  acc_criteria: float | int,
                                  with_core: bool | None=None):
        
        self.set_stopping_criteria(acc_criteria)
        # this [1:] is just for avoid conclude core feature in metrics because
        # core feature overfit metrics and pass them easily.and core feature
        # located at first index of ant_route.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if with_core:
            selected_feat = selected_feature[1:]
        else:
            selected_feat = selected_feature
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if selected_feat != [] and len(selected_feat) > 1:
            data = ns.iloc[:, selected_feat]
            X_train, X_test, y_train, y_test = train_test_split(data.values, Y.values.squeeze(), test_size=0.33, random_state=42)
            clf = FRNN()
            model = clf(X_train, y_train)
            scores = model(X_test)
            probabilities = probabilities_from_scores(scores)
            auroc = roc_auc_score(y_test, probabilities[:,-1])
            
            if auroc >= self.acc_criteria:
                self.log['acc'] = auroc
                return True, auroc
            else:
                return False, auroc
        else:
            return False, 0.0
    
    
    

    def lower(self, Cs, co_Cs):
        owa_weights: ReciprocallyLinearWeights()
        aggr_R = np.mean
        return np.concatenate([soft_min(
            aggr_R(np.abs(C[:, None, :] - co_C), axis=-1),
            owa_weights, k=None, axis=-1
        ) for C, co_C in zip(Cs, co_Cs)], axis=0)
        

    def is_landa_met(
        self, selected_feature: list[int], landa_criteria, with_core):
        """
        sim_threshold: similarity of each pair of instances that more than that
        these pairs trace as Similar together.
        landa_criteria: stop one ant route exploration if met.more landa equall less redundant.  
        """
        
        # this [1:] is just for dont conclude core feature in metrics because
        # core feature overfit metrics and pass them easily.and core feature
        # located at first index of ant_route
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if with_core:
            selected_feat = selected_feature[1:]
        else:
            selected_feat = selected_feature
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
        selected_feat = selected_feature   
        if selected_feat != []:
            X = ns.iloc[:, selected_feature].values
            y = Y.values.squeeze()
            ## use knn as classifier 
            #~~~~~~~~~~~~~~~~~~~~~~~~~
            # classes = np.unique(y)
            # Cs = [X[np.where(y == c)] for c in classes]
            # X_unscaled = np.concatenate(Cs, axis=0)
            # scale = np.amax(X_unscaled, axis=0) - np.amin(X_unscaled, axis=0)
            # scale = np.where(scale == 0, 1, scale)
            # X = X_unscaled/scale
            # Cs = [C/scale for C in Cs]
            # co_Cs = [X[np.where(y != c)] for c in classes]
            # Q = self.lower(Cs, co_Cs)
            # lower_approximation = Q
            # data = lower_approximation
            # Q1 = np.percentile(data, 25)
            # Q3 = np.percentile(data, 75)
            # in_range = ((data >= Q1) & (data <= Q3))
            # count = np.count_nonzero(in_range)
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            frps_instance = FRPS()
            X_new, y_new = frps_instance(X, y)
            count = len(X_new)

            criteria = count/len(ns)
            landa = criteria
            self.log[f'landa: {selected_feature[-1]}'] = landa
            if landa >= landa_criteria:
                return True, landa
            elif landa < landa_criteria:
                return False, landa
        else:
            landa = 0.0
            return False, landa
        

    def find_core_feature(self, accurate_more_than = 0.75):
        """
        :ivar accurate_more_than: features that results in accuracy more than 
        this variable will save as core features that have huge impact on 
        accuracy of an ant_route and if we remove it from route choices of each ant
        and force ants to choose one of them as their first choice we can look 
        more clearly at other features.
        """
        Colonies.core_feature = []
        y = Y.values.squeeze()  
        clf = FRPS()  
        for feature_index in range(len(self.feature_choices)):
            ## FRPS() choose instances that are not ROUGH!
            ## instances with quality measure more than maximum tau
            X = ns.iloc[:, [feature_index]].values
            selected_dataset = clf(X, y)
            if (len(selected_dataset[0])/len(X)) >= accurate_more_than:
                Colonies.core_feature.append(feature_index)
                print(feature_index)
        print(Colonies.core_feature)
        
        # for i in Colonies.core_feature:
        #     if self.feature_choices[i]:
        #         copy = self.feature_choices
        #         Colonies.fg_without_core = copy.remove(self.feature_choices[i])

        return Colonies.core_feature

 
    def generate_next_ants(
        self, number_of_ants_next_generation: int,
        q,
        phero_rate_decay,
        THRESHOLD: list[int,str,None],
        alpha,
        beta,
        with_core: bool | None=None,
        change_pheromone: bool = True
        ):
        """
        :ivar THRESHOLD[0]: criteria or limit for accuracy or landa
        :ivar THRESHOLD[1]: 'accuracy' or 'landa' 
        :ivar THRESHOLD[2]: similarity between two rows that when touch it rows
        known as ROUGH. 
        :ivar beta is exploITATION coefficient --> mutual_information
        :ivar alpha is exploRATION coefficient --> pheromone
        :ivar rate_decay
        :ivar phero_rate_decay

        THRESHOLD[0]: criteria limit
        THRESHOLD[1]: if it was accuracy then criteria limit = [0,1] or 
        if it was equall landa then criteria limit = [0,1] AND THRESHOLD[2]
        must be enter
        """
        self.add_generation()
        if self.colony_number > 0:
            j = 0
            first_choose = []
            # if with_core == True:
            #     self.find_core_feature()
            while j < number_of_ants_next_generation:
                next_ant = Ant(self)
                # if with_core == True:
                #     next_ant.choose_feature_gen_with_core(alpha, beta)
                # else:
                next_ant.build_route_next_gen(THRESHOLD, alpha, beta, with_core)
                Colonies.overall_ant_route[f'Colony Number:{self.colony_number} ant-num:{j}'] = next_ant.ant_route
                self.log[j] = next_ant.ant_route
                print(f'ant_route: {next_ant.ant_route}')
                j += 1
            if change_pheromone:
                self.change_pheromone(q=q, rho=phero_rate_decay)
            self.get_log()
        else:
            print("Colony generation doesnt initialized first!") 



# Colony.initialize_colony(number_of_ants_first_generation=5,
#                          init_criteria=3,
#                          q=0.9,
#                          phero_rate_decay=0.5,
#                          make_change_pheromone=True,
#                          with_core=True)
    
# Colony.generate_next_ants(number_of_ants_next_generation=4,
#                           q=1,
#                           phero_rate_decay=0.5,
#                           THRESHOLD=[0.97, 'landa', 0.80],
#                           alpha=0.8,
#                           beta=0.2,

#                           with_core=True)
# print(Colonies.overall_ant_route)