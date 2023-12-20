from matrx.logger.logger import GridWorldLogger, GridWorldLoggerV2


class LogDuration(GridWorldLogger):
    """ Log the number of ticks the Gridworld was running on completion """

    def __init__(self, save_path="", file_name_prefix="", file_extension=".csv", delimeter=";"):
        super().__init__(save_path=save_path, file_name=file_name_prefix, file_extension=file_extension,
                         delimiter=delimeter, log_strategy=self.LOG_ON_LAST_TICK)

    def log(self, grid_world, agent_data):
        distance = None
        victim_harm = None
        idle_time_robot = None
        idle_time_human = None
        q_table_cps = None
        q_table_cps_runs = None
        q_table_basic = None
        for agent_id, agent_body in grid_world.registered_agents.items():
            if 'reward' in agent_id:
                distance = agent_body.properties['distance']
            if 'victim' in agent_id:
                victim_harm = 800-agent_body.properties['harm']
            if 'robot' in agent_id:
                idle_time_robot = agent_body.properties['idle_time']
                q_table_cps = agent_body.properties['q_table_cps']
                q_table_cps_runs = agent_body.properties['q_table_cps_runs']
                q_table_basic = agent_body.properties['q_table_basic']
        log_statement = {
            "tick": grid_world.current_nr_ticks,
            "corrected_tick": grid_world.current_nr_ticks - 399,
            "remaining_distance": distance,
            "victim_harm": victim_harm,
            "idle_time_robot": idle_time_robot,
            "q_table_cps": q_table_cps,
            "q_table_cps_runs": q_table_cps_runs,
            "q_table_basic": q_table_basic
        }

        return log_statement

class LogDurationV2(GridWorldLoggerV2):
    """ Log the number of ticks the Gridworld was running on completion """

    def __init__(self, save_path="", file_name_prefix="", file_extension=".csv", delimeter=";"):
        super().__init__(save_path=save_path, file_name=file_name_prefix, file_extension=file_extension,
                         delimiter=delimeter, log_strategy=self.LOG_ON_LAST_TICK)

    def log(self, world_state, agent_data, grid_world):
        distance = None
        victim_harm = None
        idle_time_robot = None
        idle_time_human = None
        q_table_cps = None
        q_table_basic = None
        for agent_id, agent_body in grid_world.registered_agents.items():
            if 'reward' in agent_id:
                distance = agent_id['distance']
            if 'victim' in agent_id:
                victim_harm = agent_id['harm']
        log_statement = {
            "tick": grid_world.current_nr_ticks,
            "remaining_distance": "test"
        }

        return log_statement
