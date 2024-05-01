import random
import numpy as np


class Bandit:
    def __init__(self, payoff_probs):
        self.actions = range(len(payoff_probs))
        self.pay_offs = payoff_probs

    def sample(self, action):
        selector = random.random()
        return 1 if selector <= self.pay_offs[action] else 0


def explore_greedy_agent(bandit, iterations, initial_rounds=10, random_agent=False):
    
    pay_offs = dict()
    best_action = -1
    actions = []
    rewards = []

    if random_agent:
        initial_rounds = iterations

    for t in range(iterations):
        # for the initial rounds pick a random action
        if t < initial_rounds:
            a = random.choice(bandit.actions) #odabire random radnju
            r = bandit.sample(a) #daje nagradu od 1 ili 0, za random odabranu akciju 

            # update rewards
            if a in pay_offs:
                pay_offs[a].append(r)
            else:
                pay_offs[a] = [r]
        # otherwise choose the best action so far
        else:
            if (best_action == -1):
                # check for the action with the best average payoff
                mean_dict = {}
                for key,val in pay_offs.items():

                    mean_dict[key] = np.mean(val) 
                best_action = max(mean_dict, key=mean_dict.get)
            a = best_action

            r = bandit.sample(a)
        
        actions.append(a)
        rewards.append(r)

    return actions, rewards


def random_agent(bandit, iterations):

    return explore_greedy_agent(bandit, iterations, random_agent=True)
    

def epsilon_greedy_agent(bandit, iterations, epsilon=0.2,
 initial_rounds=10, decay=1, optimistic=False):
    
    pay_offs = dict()
    actions = []
    rewards = []

    if optimistic:
        pay_offs = {a:[1] for a in bandit.actions}

    for t in range(iterations):
        # sometimes randomly pick an action to explore
        if random.random() < epsilon or t < initial_rounds:
            a = random.choice(bandit.actions)
        # otherwise choose the best one thus far
        else:
            new_dict = {}
            for key,val in pay_offs.items():
                new_dict[key] = np.mean(val) 
            a = max(new_dict, key=new_dict.get)

        r = bandit.sample(a)

        # update rewards
        if a in pay_offs:
            pay_offs[a].append(r)
        else:
            pay_offs[a] = [r]
        
        epsilon *= decay

        actions.append(a)
        rewards.append(r)

    return actions, rewards


def decaying_epsilon_greedy_agent(bandit, iterations,
  epsilon=0.2, initial_rounds=1, decay=0.99):
    
    return epsilon_greedy_agent(bandit, iterations, epsilon=epsilon,
     initial_rounds=initial_rounds, decay=decay)


def optimistic_greedy_agent(bandit, iterations, epsilon=0.2,
  initial_rounds=1, decay=0.99):
    
    return epsilon_greedy_agent(bandit, iterations, epsilon=epsilon,
     initial_rounds=initial_rounds, decay=decay, optimistic=True)


def optimal_agent(bandit, iterations):
    
    a = np.argmax(bandit.pay_offs)
    actions = []
    rewards = []

    for _ in range(iterations):
        r = bandit.sample(a)
        actions.append(a)
        rewards.append(r)

    return actions, rewards


def ucb_agent(bandit, iterations, c=0.3):
    
    pay_offs = dict()
    no_of_times_action_chosen = {a:0 for a in bandit.actions}
    q_values = {a:0 for a in bandit.actions}
    actions = []
    rewards = []

    
    for a in bandit.actions:
        for _ in range(2):
            r = bandit.sample(a)
            if a in pay_offs:
                pay_offs[a].append(r)
            else:
                pay_offs[a] = [r]
            actions.append(a)
            rewards.append(r)
            no_of_times_action_chosen[a] += 1
            q_values[a] = sum(pay_offs[a])/no_of_times_action_chosen[a]

    for t in range(2*len(bandit.actions), iterations):
        values = [q_values[a] + c*np.sqrt(np.log(t)/no_of_times_action_chosen[a]) for a in bandit.actions]
        a = np.argmax(values)
        r = bandit.sample(a)
        pay_offs[a].append(r)
        actions.append(a)
        rewards.append(r)
        
        # update q values and no_of_times_per_action
        no_of_times_action_chosen[a] += 1
        q_values[a] = sum(pay_offs[a])/no_of_times_action_chosen[a]
    
    return actions, rewards


def thompson_agent(bandit, iterations):
    
    pay_offs = dict()
    actions = []
    rewards = []

    alphas, betas = {a:1 for a in bandit.actions}, {a:1 for a in bandit.actions}

    for t in range(iterations):
        a = np.argmax([np.random.beta(alphas[a], betas[a]) for a in bandit.actions])
        r = bandit.sample(a)
        if a in pay_offs:
            pay_offs[a].append(r)
        else:
            pay_offs[a] = [r]

        actions.append(a)
        rewards.append(r)

        alphas[a] += r
        betas[a] += (1-r)

    return actions, rewards
