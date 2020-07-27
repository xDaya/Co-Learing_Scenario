import numpy as np

from matrx.goals.goals import WorldGoal


class USAR_Goal(WorldGoal):

    def __init__(self):
        super().__init__()
        self.counter = 0
        self.reward_agent = None

    def goal_reached(self, grid_world):
        # Add here code to retrieve state from environment
        if self.reward_agent is None:
            for obj in grid_world.registered_agents.keys():
                if "AgentBrain" in grid_world.registered_agents[obj].properties['class_inheritance']:
                    if "RewardGod" in grid_world.registered_agents[obj].properties['class_inheritance']:
                        self.reward_agent = obj

        #if grid_world.environment_objects['rewardobj'].properties['goalreached']:
            # wait one tick before stopping, such that an update is sent to the frontend that shows a completion screen
        #    self.is_done = True

        return self.is_done