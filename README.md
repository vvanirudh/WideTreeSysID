# System Identification for WideTree environment

In this project, we implement both mle based model learning and value moment based model learning approaches for SysID of WideTree environment.

The Widetree environment can be described as below:


```
                                            Root = 1
                                        /           \
                                       /             \
                                      /               \
                                left \in {2, 3}      right = 5 - left
                                    |                   |   
                                    |                   |
                        left_parent \in {4, 5}       right_parent = 9 - left_parent
                        / / / / / / \ \ \ \ \ \             / / / / / / \ \ \ \ \ \
                       / / / / / /   \ \ \ \ \ \           / / / / / /   \ \ \ \ \ \
                    N/4 leaves          N/4 leaves         N/4 leaves     N/4 leaves
```

where the MDP has N + 5 states, with N terminal states. At each state, we have two actions 0 (left) or 1 (right). In the left and right states, both actions end up in the same successor.

The cost function for any state-action pair (s, a) is as follows:

```
c(s, a) = 0 if s != 2 else 1
```

The real world uses `left = 2, right = 3, left_parent = 4, right_parent = 5` and the leaves are ordered left-to-right from `6 to N+5`. 

The model class contains "good" models and a "bad" model. 

The "good" models are constructed by assigning `left = 2, right = 3, left_parent = 5, right_parent = 4` and the leaves are a random permutation of `6 to N+5`. Note that the "good" models and real world are only guaranteed to match dynamics at the root.

The "bad" model is constructed by assigning `left = 3, right = 2, left_parent = 4, right_parent = 5` and the leaves are ordered left-to-right from `6 to N+5`. Note that the only difference between "bad" model and real world is the dynamics at the root.