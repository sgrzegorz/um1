from __future__ import annotations
import collections
import random

import numpy as np
import sklearn.preprocessing as skl_preprocessing

from problem import Action, available_actions, Corner, Driver, Experiment, Environment, State, DRAWING_FREQUENCY

ALMOST_INFINITE_STEP = 100000
MAX_LEARNING_STEPS = 500


class RandomDriver(Driver):
    def __init__(self):
        self.current_step: int = 0

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        return random.choice(available_actions(state))

    def control(self, state: State, last_reward: int) -> Action:
        self.current_step += 1
        return random.choice(available_actions(state))

    def finished_learning(self) -> bool:
        return self.current_step > MAX_LEARNING_STEPS


class OffPolicyNStepSarsaDriver(Driver):
    def __init__(self, step_size: float, step_no: int, experiment_rate: float, discount_factor: float) -> None:
        self.step_size: float = step_size
        self.step_no: int = step_no
        self.experiment_rate: float = experiment_rate #alpha = 0.05
        self.discount_factor: float = discount_factor # gamma wolimy mieć nagrodę teraz niż później
        self.q: dict[tuple[State, Action], float] = collections.defaultdict(float) # Nasze przypuszczenia dotyczące prawdziwych wartości akcji q. Zauważ że te wartości są zachowywane dla kolejnych epizodów
        self.current_step: int = 0
        self.final_step: int = ALMOST_INFINITE_STEP
        self.finished: bool = False
        self.states: dict[int, State] = dict()
        self.actions: dict[int, Action] = dict()
        self.rewards: dict[int, int] = dict()

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        self.states[self._access_index(self.current_step)] = state
        action = self._select_action(self.epsilon_greedy_policy(state, available_actions(state)))
        self.actions[self._access_index(self.current_step)] = action
        self.final_step = ALMOST_INFINITE_STEP
        self.finished = False
        return action

    def control(self, state: State, last_reward: int) -> Action:
        if self.current_step < self.final_step:
            self.rewards[self._access_index(self.current_step + 1)] = last_reward
            self.states[self._access_index(self.current_step + 1)] = state
            if self.final_step == ALMOST_INFINITE_STEP and (
                    last_reward == 0 or self.current_step == MAX_LEARNING_STEPS
            ):
                self.final_step = self.current_step
            action = self._select_action(self.epsilon_greedy_policy(state, available_actions(state)))
            self.actions[self._access_index(self.current_step + 1)] = action
        else:
            action = Action(0, 0) #akcje to przyspieszenia

        update_step = self.current_step - self.step_no + 1
        if update_step >= 0: #tau
            return_value_weight = self._return_value_weight(update_step) #p
            return_value = self._return_value(update_step)  #G

            state_t = self.states[self._access_index(update_step)]
            action_t = self.actions[self._access_index(update_step)]

            self.q[state_t, action_t] = self.q[state_t, action_t] + self.step_size*return_value_weight*(return_value - self.q[state_t, action_t]) # TODO: Tutaj trzeba zaktualizować tablicę wartościującą akcje Q

        if update_step == self.final_step - 1:
            self.finished = True

        self.current_step += 1
        return action

    def _return_value(self, update_step)-> float: # TODO: Tutaj trzeba policzyć zwrot G
        return_value = 0.0
        i=update_step+1
        while(i<=min(update_step+self.step_no-1,self.final_step-1)):
            return_value+=pow(self.discount_factor,i-update_step-1)*self.rewards[self._access_index(i)]
            i+=1

        if(update_step+self.step_no)<self.final_step:
            state_t = self.states[self._access_index(update_step+self.step_no)]
            action_t = self.actions[self._access_index(update_step+self.step_no)]
            return_value +=  pow(self.discount_factor,self.step_no)* self.q[state_t, action_t]

        return return_value

    def _return_value_weight(self, update_step) -> float: #p
        return_value_weight = 1.0
        # TODO: Tutaj trzeba policzyć korektę na różne prawdopodobieństwa ρ (ponieważ uczymy poza-polityką)
        i = update_step + 1
        while (i <= min(update_step + self.step_no-1, self.final_step-1)):
            state_i = self.states[self._access_index(i)]
            action_i = self.actions[self._access_index(i)]

            # p_pi = self.greedy_policy(state_i, [action_i])[action_i]
            # p_b =self.epsilon_greedy_policy(state_i, [action_i])[action_i]

            p_pi = self.greedy_policy(state_i, available_actions(state_i))[action_i]
            p_b =self.epsilon_greedy_policy(state_i, available_actions(state_i))[action_i]
            return_value_weight *=p_pi/p_b
            i += 1
        return return_value_weight

    def finished_learning(self) -> bool:
        return self.finished

    def _access_index(self, index: int) -> int:
        return index % (self.step_no + 1)

    @staticmethod
    def _select_action(actions_distribution: dict[Action, float]) -> Action:
        actions = list(actions_distribution.keys())
        probabilities = list(actions_distribution.values())
        i = np.random.choice(list(range(len(actions))), p=probabilities)
        return actions[i]

    def epsilon_greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        # TODO: tutaj trzeba ustalic prawdopodobieństwa wyboru akcji według polityki ε-zachłannej
        #str 32 RLBook 2020
        rand= random.random()
        # if(rand< self.experiment_rate): #probability epsilon # rand <0.05
        #     probabilities = self._random_probabilities(actions)
        #
        # else:  #probability 1 − epsilon  # rand >= 0.05
        #     probabilities = self._greedy_probabilities(state, actions)
        probabilities = (1.0 - self.experiment_rate) * self._greedy_probabilities(state, actions) + self.experiment_rate * self._random_probabilities(actions)


        return {action: probability for action, probability in zip(actions, probabilities)}

    def greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]: # polityka każdej akcji Action przypisuje p-stwo jej wykonania jeśli działamy zgodnie z tą polityką
        probabilities = self._greedy_probabilities(state, actions)
        return {action: probability for action, probability in zip(actions, probabilities)}

    def _greedy_probabilities(self, state: State, actions: list[Action]) -> np.ndarray:
        values = [self.q[state, action] for action in actions]
        maximal_spots = (values == np.max(values)).astype(float)
        return self._normalise(maximal_spots)

    @staticmethod
    def _random_probabilities(actions: list[Action]) -> np.ndarray:
        maximal_spots = np.array([1.0 for _ in actions])
        return OffPolicyNStepSarsaDriver._normalise(maximal_spots)

    @staticmethod
    def _normalise(probabilities: np.ndarray) -> np.ndarray:
        return skl_preprocessing.normalize(probabilities.reshape(1, -1), norm='l1')[0]


def main() -> None:

    experiment = Experiment(
        environment=Environment(
            corner=Corner(
                name='corner_c'
            ),
            steering_fail_chance=0.01,
        ),
        driver=OffPolicyNStepSarsaDriver(
            step_no=5,
            step_size=0.3,
            experiment_rate=0.3, #alpha
            discount_factor=1.00, ##gamma
        ),
        number_of_episodes=10000,
    )

    experiment.run()


if __name__ == '__main__':
    main()
