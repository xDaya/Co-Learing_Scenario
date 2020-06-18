import numpy as np

from matrx.goals.goals import WorldGoal


class USAR_Goal(WorldGoal):

    def goal_reached(self, grid_world):
        # Add here code to retrieve state from environment
        print(grid_world.environment_objects['rewardobj'])
        if grid_world.environment_objects['rewardobj'].properties['goalreached']:
            # wait one tick before stopping, such that an update is sent to the frontend that shows a completion screen
            self.is_done = True

        return self.is_done