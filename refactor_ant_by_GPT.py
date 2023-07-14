import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from frlearn.classifiers import FRNN
from sklearn.model_selection import train_test_split
from frlearn.base import probabilities_from_scores
from sklearn.metrics import roc_auc_score
from typing import List, Tuple, Union


class Ant:
    def __init__(self, colony):
        self.colony = colony
        self.fg = self.colony.feature_choices.copy()  # Copy feature list for each Ant instance
        self.ant_route = []

    def choose_feature(self) -> str:
        feature_index = np.random.choice(range(len(self.fg)))
        chosen_feature = self.fg.pop(feature_index)
        return chosen_feature

    def build_route(self, route_length: int) -> None:
        if self.colony.colony_number == 0:
            for _ in range(route_length):
                feature = self.choose_feature()
                self.ant_route.append(self.colony.feature_choices.index(feature))

    def build_next_gen_route(self, THRESHOLD: List[Union[int,str,None]]) -> None:
        k = 0
        while True:
            feature = self.choose_feature()
            self.ant_route.append(self.colony.feature_choices.index(feature))

            if THRESHOLD[1] == 'accuracy':
                is_criteria_met, criteria = self.colony.is_rough_set_criteria_met(self.ant_route ,THRESHOLD[0])
            elif THRESHOLD[1] == 'landa':
                is_criteria_met, criteria = self.colony.is_landa_met(self.ant_route , THRESHOLD[2], THRESHOLD[0])
                
            if is_criteria_met:
                print(f'stop with this criteria {criteria}\n'
                      f'with this ant_route {self.ant_route}\n'
                      f'self.colony.fg: {self.colony.feature_choices.index(x) for x in self.colony.fg}')
                break
            elif not is_criteria_met and not self.colony.fg:
                print(f'ant could not find route with {criteria} accuracy')
                break
            k += 1


class Colony:
    dataframe: pd.DataFrame = None
    pheromone: np.ndarray = None
    traversed_nodes: np.ndarray = None
    ant_route: List[int] = []
    alpha: Union[float, int] = 0.5
    beta: Union[float, int] = 0.5
    feature_choices: List[str] = []
    feature1: str = ""
    log: dict = {}
    fg: List[str] = [] 
    acc_criteria: Union[float, int] = 0.8
    rho: Union[int, float] = 0.5
    delta: np.ndarray = None
    colony_number: int = 0
    overall_ant_route: dict = {}

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
        self.feature_choices = dataframe.columns.tolist()
        self.fg = dataframe.columns.tolist() 
        self.pheromone = np.ones((dataframe.shape[1], dataframe.shape[1]))
        self.traversed_nodes = np.zeros((dataframe.shape[1], dataframe.shape[1]))
        self.delta = np.zeros((dataframe.shape[1], dataframe.shape[1]))

    def set_stopping_criteria(self, criteria: Union[float, int]) -> None:
        self.acc_criteria = criteria

    def set_rate_decay(self, rate_decay: float) -> None:
        self.rho = rate_decay

    def add_generation(self) -> None:
        self.colony_number += 1

    def get_log(self) -> dict:
        return self.log

    def reset_colony(self) -> None:
        self.pheromone = np.ones((self.dataframe.shape[1], self.dataframe.shape[1]))
        self.traversed_nodes = np.zeros((self.dataframe.shape[1], self.dataframe.shape[1]))
        self.overall_ant_route = {}
        self.log = {}
        self.fg = self.feature_choices.copy()

    def reset_fg(self, j: Union[int, None] = None) -> None:
        first_choice_of_ants = [x[0] for x in list(self.overall_ant_route.values())]
        str_of_first_choice = [self.feature_choices[l] for l in first_choice_of_ants[j:]]
        self.fg = self.feature_choices
        for y in str_of_first_choice: 
            if y in self.fg:
                self.fg.remove(y)

    def reset_ant_route(self) -> None:
        self.ant_route = []

    def initialize_feature1(self) -> None:
        if not self.ant_route:
            self.feature1 = np.random.choice(self.feature_choices.copy())
        else:
            self.feature1 = self.feature_choices[self.ant_route[-1]]

    def ant(self) -> None:
        ant_instance = Ant(self)
        ant_instance.build_route(init_criteria)
        self.ant_route.append(ant_instance.ant_route)

    def positive_region(self) -> pd.DataFrame:
        partitions = [group for _, group in self.dataframe.groupby(self.dataframe.iloc[:-1])]
        positive_regions = [group[group.duplicated(self.dataframe.columns[:-1], keep=False)] for group in partitions]
        return pd.concat(positive_regions)

    def __probability_transition_rule(self, feature1: str, feature2: str) -> float:
        col_index = self.dataframe.columns.tolist()
        i = col_index.index(feature1)
        j = col_index.index(feature2)
        l = col_index.copy()

        mic_ij = mutual_info_classif(self.dataframe[[feature1, feature2]], y=self.dataframe['win'], random_state=0)[1]
        phrmn_ij = self.pheromone[i, j]
        init = 0
        for k in range(0,len(l)):
            if k != l.index(feature2):
                phrmn_il = self.pheromone[i, k]
                mic_il = self.pheromone[i, k]
                init += (phrmn_il**self.alpha) * (mic_il**self.beta)
        if init == 0:
            print("init is zero!")
        pij = ((mic_ij**self.beta) * (phrmn_ij**self.alpha)) / init
        return pij

    def initialize_colony(self, number_of_ants_first_generation: int, init_criteria: int) -> None:
        self.fg = self.feature_choices.copy()
        self.overall_ant_route = {}
        first_choose = []
        for i in range(number_of_ants_first_generation):
            ant_instance = Ant(self)
            ant_instance.build_route(init_criteria)
            self.overall_ant_route[i] = ant_instance.ant_route
            first_choose.append(ant_instance.ant_route[0])    
            self.log[i] = ant_instance.ant_route
            print(f'ant_route: {ant_instance.ant_route}')
            self.fg = self.dataframe.columns.tolist()
            for i in first_choose:
                if self.feature_choices[i] in self.fg:
                    self.fg.remove(self.feature_choices[i])
            self.reset_ant_route()
            print(f'self.fg: {[self.feature_choices.index(x) for x in self.fg]}')
        self.reset_ant_route()                                                                              
        self.fg = self.dataframe.columns.tolist()
        self.add_generation()
        print(f'initiall final: {[self.feature_choices.index(x) for x in self.fg]}')

    def is_rough_set_criteria_met(self, selected_feature: list[int], acc_criteria: Union[float, int]) -> Tuple[bool, float]:
        self.set_stopping_criteria(acc_criteria)
        if selected_feature != [] and len(selected_feature) > 1:
            data = self.dataframe.iloc[:, selected_feature]
            X_train, X_test, y_train, y_test = train_test_split(data.values, self.dataframe['win'].values.squeeze(), test_size=0.33, random_state=42)
            
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

    def is_landa_met(self, selected_feature: list[int], sim_threshold = 0.5, landa_criteria = 0.8) -> Tuple[bool, float]:
        if selected_feature != [] and len(selected_feature) > 1:
            X = self.dataframe.iloc[:, selected_feature].values
            y = self.dataframe['win'].values.squeeze()
            R_a = np.minimum(np.maximum(1 - np.abs(X[:, None, :] - X), 0), y[:, None, None] != y[:, None])
            differences = []
            all = []
            for i in range(len(X)):
                for j in range(i+1, len(X)):
                    difference = np.count_nonzero(R_a[i, j])
                    similarity_percent = 1 - (difference/X.shape[1])
                    if similarity_percent > sim_threshold and i != j:
                        differences.append(difference)
                    all.append(difference)

            criteria = len(differences) / ((X.shape[0]**2)/2)
            landa = criteria
            self.log['landa'] = landa
            if landa >= landa_criteria:
                return True, landa
            elif landa < landa_criteria:
                return False , landa
        else:
            return False, 0.0

    def generate_next_ants(self, number_of_ants_next_generation: int, rate_decay: float, phero_rate_decay: float, THRESHOLD: List[Union[int,str,None]]) -> None:
        if self.colony_number > 0:
            j = 0
            self.fg = self.feature_choices.copy()
            first_choose = []
            while j < number_of_ants_next_generation:
                next_ant = Ant(self)
                next_ant.build_next_gen_route(THRESHOLD)
                self.overall_ant_route[j] = next_ant.ant_route
                self.log[j] = next_ant.ant_route
                print(f'ant_route: {next_ant.ant_route}')
                first_choose.append(next_ant.ant_route[0])
                self.fg = self.dataframe.columns.tolist()
                for i in first_choose:
                    if i in self.fg:
                        self.fg.remove(i)
                j += 1
            self.change_pheromone(q=rate_decay, rho=phero_rate_decay)
        else:
            print("Colony generation hasn't been initialized first!")

Colony.initialize_colony(number_of_ants_first_generation=5,init_criteria=4)
Colony.generate_next_ants(number_of_ants_next_generation=5,rate_decay=0.5,phero_rate_decay=0.3, THRESHOLD=[0.6,'landa',0.7])

print(Colony.overall_ant_route)