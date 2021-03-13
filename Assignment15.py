from typing import Sequence, Tuple, Mapping
import numpy as np

S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
CountFunc = Mapping[S, Mapping[S, int]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]

def count_to_prob(counts : CountFunc) -> ProbFunc:
    """a"""
    probs = {s:counts[s] for s in counts}
    for s,c in probs.items():
        tot = sum(c.values())
        probs[s] = {z: c[z]/tot for z in c}
    return probs

def get_state_return_samples(
    data: DataType
) -> Sequence[Tuple[S, float]]:
    """
    prepare sequence of (state, return) pairs.
    Note: (state, return) pairs is not same as (state, reward) pairs.
    """
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]


def get_mc_value_function(
    state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    """
    Implement tabular MC Value Function compatible with the interface defined above.
    """
    counts_map : Mapping[S, int] = {}
    V : ValueFunc = {}
    for state_return in state_return_samples:
        st,Gt = state_return
        counts_map[st] = counts_map.get(st, 0.) + 1
        α_n = 1/counts_map[st] #weight for the new sample to add to the moving average
        V[st] = V.get(st, 0.) + (Gt- V.get(st, 0.)) * α_n
    return V

def get_state_reward_next_state_samples(
    data: DataType
) -> Sequence[Tuple[S, float, S]]:
    """
    prepare sequence of (state, reward, next_state) triples.
    """
    return [(s, r, l[i+1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    """
    Implement code that produces the probability transitions and the
    reward function compatible with the interface defined above.
    """
    counts : CountFunc = {}
    rewards : RewardFunc = {}
    state_counts : Mapping[S, int] = {}
    for srs in srs_samples:
        s,r,s_next = srs
        state_counts[s] = state_counts.get(s, 0.) + 1
        α_n = 1/state_counts[s]
        rewards[s] = rewards.get(s, 0.)*(1 - α_n) + r * α_n
        if s not in counts:
            counts[s] = {}
        counts[s][s_next] = counts[s].get(s_next, 0) + 1
    probs = count_to_prob(counts)
    return (probs,rewards)

def get_mrp_value_function(
    prob_func: ProbFunc,
    reward_func: RewardFunc
) -> ValueFunc:
    """
    Implement code that calculates the MRP Value Function from the probability
    transitions and reward function, compatible with the interface defined above.
    Hint: Use the MRP Bellman Equation and simple linear algebra
    """
    Ss = sorted(list(reward_func.keys()))
    m = len(Ss)

    P = np.zeros((m,m))
    R = np.array([reward_func[s] for s in Ss]).reshape(-1,1)
    I = np.eye(m)
    for i,s in enumerate(Ss):
        for j,s_next in enumerate(Ss):
            P[i,j] = prob_func[s].get(s_next, 0.)
    V = (np.linalg.inv(I - P)).dot(R).flatten()

    return {s:V[i] for i,s in enumerate(Ss)}

def get_td_value_function(
    srs_samples: Sequence[Tuple[S, float, S]],
    num_updates: int = 300000,
    learning_rate: float = 0.3,
    learning_rate_decay: int = 30
) -> ValueFunc:
    """
    Implement tabular TD(0) (with experience replay) Value Function compatible
    with the interface defined above. Let the step size (alpha) be:
    learning_rate * (updates / learning_rate_decay + 1) ** -0.5
    so that Robbins-Monro condition is satisfied for the sequence of step sizes.
    """
    def get_alpha(n : int):
        return learning_rate * (n / learning_rate_decay + 1) ** -0.5
    updates = 1
    V : ValueFunc = {}
    while updates<num_updates:
        i = updates%len(srs_samples)
        s,r,s_next = srs_samples[i]
        alpha = get_alpha(updates)
        V[s] = V.get(s, 0.)  + alpha * ( r + V.get(s_next, 0.) - V.get(s, 0.))
        updates+=1
    return V

def get_lstd_value_function(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    """
    Implement LSTD Value Function compatible with the interface defined above.
    Hint: Tabular is a special case of linear function approx where each feature
    is an indicator variables for a corresponding state and each parameter is
    the value function for the corresponding state.
    """
    Ss = sorted(list(set(srs[0] for srs in srs_samples)))
    m = len(Ss)
    
    def make_phi(s):
        return np.array([float(s == state) for state in Ss]).reshape(-1,1)

    A = np.zeros((m,m))
    b = np.zeros((m,1))
    for s,r,s_next in srs_samples:
        phi1, phi2 = make_phi(s), make_phi(s_next)
        A += np.outer(phi1, phi1-phi2)
        b += phi1*r
    w = np.linalg.inv(A).dot(b).flatten()
    V : ValueFunc = {s:w[i] for i,s in enumerate(Ss)}
    return V

if __name__ == '__main__':
    given_data: DataType = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    sr_samps = get_state_return_samples(given_data)

    print("------------- MONTE CARLO VALUE FUNCTION --------------")
    print(get_mc_value_function(sr_samps))

    srs_samps = get_state_reward_next_state_samples(given_data)

    pfunc, rfunc = get_probability_and_reward_functions(srs_samps)
    print("-------------- MRP VALUE FUNCTION ----------")
    print(get_mrp_value_function(pfunc, rfunc))

    print("------------- TD VALUE FUNCTION --------------")
    print(get_td_value_function(srs_samps))

    print("------------- LSTD VALUE FUNCTION --------------")
    print(get_lstd_value_function(srs_samps))