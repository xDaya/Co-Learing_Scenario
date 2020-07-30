from matrx.logger.logger import GridWorldLogger
from matrx.grid_world import GridWorld


class LearningLogger(GridWorldLogger):

    def __init__(self, save_path="", file_name_prefix="", file_extension=".csv", delimiter=";"):
        super().__init__(save_path=save_path, file_name=file_name_prefix, file_extension=file_extension,
                         delimiter=delimiter, log_strategy=100)

    def log(self, grid_world, agent_data):
        log_data = {}
        for agent_id, agent_body in grid_world.registered_agents.items():
            if 'robot selector' in agent_id:
                #print(agent_body.custom_properties)
                log_data[agent_id + '_q-table'] = agent_body.custom_properties['q_table']

        return log_data
