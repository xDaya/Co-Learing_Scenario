from matrx.agents.agent_types.human_agent import *
from custom_actions import *
from matrx.messages.message import Message
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
import csv
import pickle
from typedb.client import *


class OntologyGod(AgentBrain):

    def __init__(self):
        super().__init__()
        self.state_tracker = None

    def initialize(self):
        self.state_tracker = StateTracker(agent_id=self.agent_id)

    def decide_on_action(self, state):
        action_kwargs = {}
        action = None


        return action, action_kwargs
