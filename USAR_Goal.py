import numpy as np

from matrx.goals.goals import WorldGoal


class USAR_Goal(WorldGoal):

    def __init__(self):
        super().__init__()
        self.counter = 0
        self.reward_agent = None
        self.end_obj = None
        self.done_counter = 0

    def goal_reached(self, grid_world):
        self.counter += 1
        self.is_done = False

        # Add here code to retrieve state from environment
        if self.reward_agent is None:
            for obj in grid_world.registered_agents.keys():
                if "AgentBrain" in grid_world.registered_agents[obj].properties['class_inheritance']:
                    if "RewardGod" in grid_world.registered_agents[obj].properties['class_inheritance']:
                        self.reward_agent = obj

        if self.end_obj == None:
            for obj in grid_world.environment_objects:
                if "goal_reached_img" in obj:
                    self.end_obj = grid_world.environment_objects[obj]

        try:
            if grid_world.registered_agents[self.reward_agent].properties['goal_reached']:
                # Code for delaying reset with a few seconds
                self.done_counter += 1
                if self.done_counter >= 200:
                    self.is_done = True
        except KeyError:
            self.reward_agent = None
            self.counter = 0
            self.done_counter = 0
            self.end_obj = None

        return self.is_done
