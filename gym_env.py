import gymnasium as gym
import numpy as np
from gymnasium.spaces import Discrete

from data import get_transition_matrix

class ArmyRetentionEnv(gym.Env):

    def __init__(self,args) -> None:
        super().__init__()

        self.n_grades = args.n_grades    # Number of Grades
        self.T = args.T                  # Maximum Time in Service before complete retirement
        self.beta = args.beta            # Discount Factor
        self.tau = args.tau              # Life Expectancy
        self.eta = args.eta              # Joining Age in Army
        self.delta = args.delta          # Ratio of Civilian Pay and Military Pay for each grade
        self.eps = args.eps              # Expected Pay Rise in Civilian Sector
        self.alpha =args.alpha           # weight for retirement pay
        self.min_TiS = args.min_TiS      # Minimum Service Year for retirement Pay

        # Random Pay (If not initialised)
        if (len(args.military_pay) != self.n_grades or len(args.compensation_pay) != self.n_grades) :
            self.military_pay = np.sort(np.random.randint(20000,50000,size=self.n_grades))
            self.compensation_pay = np.sort(np.random.randint(2000,5000,size=self.n_grades))
        else:
            self.military_pay = args.military_pay
            self.compensation_pay = args.compensation_pay

        if len(self.delta) != self.n_grades:
            self.delta = np.sort(np.random.random(self.n_grades) + 1)

        # Obseravation and Action Space Initialization
        self.total_states = self.n_grades*self.T + 1
        self.total_actions = 2
        self.action_space = Discrete(2)
        self.observation_space = Discrete(self.total_states) # Army State + Loss State

        # Starting and Terminating State
        self._valid_starting_cells: list[tuple[int, int]] = [(0,0)]
        self.terminal_state = self.n_grades*self.T

        self.action_dir = {
            0 : "Leave",
            1 : "Stay"
        }

        # (Row,Col) = (Grade, TiS) for all 1<=i<=n_grades  
        self.map = np.zeros((self.n_grades,self.T),dtype=int) # Army States

        # Reward for army states
        for grade in range(self.map.shape[0]):
            self.map[grade] = self.military_pay[grade] + self.compensation_pay[grade]

        # Precalculate transition probabilities
        self.trans_matrix = get_transition_matrix(args.dataroot,self.n_grades,self.T)
        self.P: dict[int, dict[int, list[tuple[float, int, float, bool]]]] = {}

        for row in range(self.map.shape[0]):
            for col in range(self.map.shape[1]):
                state = self._cell_to_state((row, col))
                self.P[state] = {
                    action: self._calculate_transitions((row, col), action) for action in range(self.action_space.n)
                }
        
        self.P[self.terminal_state] = {
            action : [(1.0,self.terminal_state,0,True)] for action in range(self.action_space.n)
        }

    def _weight(self,t):

        if t < self.min_TiS:
            return 0
        else:
            return self.alpha + (1-self.alpha) * (t-self.min_TiS)/(self.T - self.min_TiS)
    
    def _delta(self,state):

        
        return self.delta[state[0]]
    
    def _terminal_state_reward(self,army_state):

        grade,t = army_state

        sum_r_df = np.sum(np.geomspace(1,np.power(self.beta,self.tau-self.eta-t),num=self.tau-self.eta-t+1))
        sum_c_df = (1+self.eps)*np.sum(np.geomspace(1,np.power(self.beta*(1+self.eps),self.T-t),num=self.T-t+1))

        retire_pay = self._weight(t) * self.military_pay[grade] * sum_r_df
        civilian_pay = self._delta(army_state) * self.military_pay[grade] * sum_c_df

        return retire_pay + civilian_pay

    def _cell_to_state(self,cell):

        return (cell[0] * self.T + cell[1])
    def _state_to_cell(self,state):

        return (state // self.T, state % self.T )

    # Finds the next state given the current state and action, along with bounds checking
    # Returns a list with single tuple (prob, nextState, reward, terminal)
    def _calculate_transitions(self, current_cell: tuple[int, int], action: int) -> list[tuple[float, int, float, bool]]:
        
        if action == 0:
            return [(1,self.total_states-1,self._terminal_state_reward(current_cell),True)]
        
        
        state = self._cell_to_state(current_cell)

        transitions = []
        for s in range(self.n_grades*self.T):

            cell = s//self.T,s%self.T

            if self.trans_matrix[state][s] != 0:
                transitions.append((self.trans_matrix[state][s],s,self.map[cell],False))
        

        if self.trans_matrix[state][self.terminal_state] != 0:

            transitions.append((self.trans_matrix[state][self.terminal_state],self.terminal_state,self._terminal_state_reward(current_cell),True))

        return transitions


    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[int, dict]:
        super().reset(seed=seed, options=options)

        self.path = np.zeros((self.n_grades,self.T),dtype=float) # Movement of the Agent
        self.past_cell = (0,0)

        start_cell = self._valid_starting_cells[np.random.randint(len(self._valid_starting_cells))]
        self.state = self._cell_to_state(start_cell)
        return self.state, {}
    
    def _get_transition(self,transitions: np.ndarray):
        
        prob = [transition[0] for transition in transitions]

        idx = np.random.choice(len(transitions),1,p=prob)[0]
        return transitions[idx]

    def step(self, action: any) -> tuple[int, float, bool, bool, dict]:
        assert self.action_space.contains(action)

        transition = self._get_transition(self.P[self.state][action])
        self.state = transition[1]

        return transition[1], transition[2], transition[3], False, {}

    def render(self):
        current_cell = self._state_to_cell(self.state)
        
        self.path[self.past_cell] = 0.5
        if self.state != self.terminal_state:
            self.path[current_cell] = 1
            self.past_cell = current_cell

        return self.path