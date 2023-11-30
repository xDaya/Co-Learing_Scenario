from matrx.logger.logger import GridWorldLogger, GridWorldLoggerV2


class LogDuration(GridWorldLogger):
    """ Log the number of ticks the Gridworld was running on completion """

    def __init__(self, save_path="", file_name_prefix="", file_extension=".csv", delimeter=";"):
        super().__init__(save_path=save_path, file_name=file_name_prefix, file_extension=file_extension,
                         delimiter=delimeter, log_strategy=self.LOG_ON_LAST_TICK)

    def log(self, grid_world, agent_data):
        log_statement = {
            "tick": grid_world.current_nr_ticks
        }

        return log_statement

class LogDurationV2(GridWorldLoggerV2):
    """ Log the number of ticks the Gridworld was running on completion """

    def __init__(self, save_path="", file_name_prefix="", file_extension=".csv", delimeter=";"):
        super().__init__(save_path=save_path, file_name=file_name_prefix, file_extension=file_extension,
                         delimiter=delimeter, log_strategy=self.LOG_ON_LAST_TICK)

    def log(self, world_state, agent_data, grid_world):
        distance = None
        for agent_id, agent_body in grid_world.registered_agents.items():
            if 'reward' in agent_id:
                distance = agent_id['distance']
        log_statement = {
            "tick": grid_world.current_nr_ticks,
            "remaining_distance": "test"
        }

        return log_statement
