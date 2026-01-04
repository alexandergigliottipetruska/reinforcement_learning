# Expected SARSA
Expected SARSA is a temporal difference algorithm that computes its update by taking the expectation over all possible actions at the next state. It is off-policy as it takes the expected value for the target policy. If it were greedy with respect to the target policy, it would be equivalent to Q-learning. The update rule is as follows:

$$Q(s, a) \gets Q(s, a) + \alpha (r + \gamma \sum_{a'} \pi (a'|s')Q(s', a') - Q(s, a))$$

The environment used is Taxi and is deterministic, involving a taxi picking up customers at randomly initialized locations and taking them to one of four possible destination points. A reward of 20 is provided for successfully dropping off the passenger, -10 for illegal drop-off and pick-up actions, and -1 for step unless reward is triggered. If it collides against a wall it gets -1.
