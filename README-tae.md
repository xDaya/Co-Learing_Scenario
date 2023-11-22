## Dependencies

- python3.8
- pip install matrx==2.3.0
- pip install typedb-client
- typedb-server=2.18.0

## Questions

- basic behavior (full MDP)
  - States are represented as a tuple [A, B, C]
    - What are they again?
      - progress: 1, 2, 3, 4
        - relative progress of the number of rocks being removed
      - contribution: equal, human, robot
      - human-standing-bool: true, false
    - There are in total of 24 states.
  - There are five actions for the agent: (1) move back and forth; (2) stand stil; (3) pick up; and (4) break, (5) drop
    - These are macro actions
    - They are full MDP, which includes p(s'| s, a) and R(s, a)
- CPs (Contextual bandit)
  - have starting and end conditions. These conditions are always one of the 24 states? But how do you match them?
    - No these have different states.
  - Why is this contextual bandit? I don't think learning is happening here.
    - q(s, a) = E[R_{t+1} | S_t=s, A_t =a]
    - Episodes are one step
    - actions do not affect state transitions -> no long-term consequences.
  - Why does it matter to know what humans did? Is this used as feedback / suggestion?
    - No. It just expects humans to follow them.
  - So from the agent's perspective, is a CP then actually an action to take given a state?
    - yes
  - "What we do In this situation, we did this:" How can the human and agent understand this as one of the four actions?
    - It's not one of the five actions.