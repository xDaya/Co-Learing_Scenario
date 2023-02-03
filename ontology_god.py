from matrx.agents.agent_types.human_agent import *
from custom_actions import *
from matrx.messages.message import Message
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from custom_objects import PartLargeObject
import csv
import pickle
import numpy as np
from typedb.client import *


class OntologyGod(AgentBrain):

    def __init__(self):
        super().__init__()
        self.state_tracker = None

        self.human = None

        self.query_batch = [] # The list of queries

        self.cp_list = [] # Maintain a list of CP names that were previously instantiated

    def initialize(self):
        self.state_tracker = StateTracker(agent_id=self.agent_id)

        start_conditions = []

        # At initialization, check if there are new CPs that weren't yet shown in the GUI. Retrieve them and store
        with TypeDB.core_client("localhost:1729") as client:
            with client.session("CP_ontology", SessionType.DATA) as session:
                # Session is opened, now specify that it's a read session
                with session.transaction(TransactionType.READ) as read_transaction:
                    answer_iterator = read_transaction.query().match(
                        "match $x isa collaboration_pattern, has name $name; get $name;")

                    for answer in answer_iterator:
                        cp_retrieved = answer.get('name')._value
                        if cp_retrieved not in self.cp_list:
                            self.cp_list.append(cp_retrieved)

        print('Ontology God Initialized')

    def filter_observations(self, state):
        self.state_tracker.update(state)

        # Filter the irrelevant state stuff out here (god agents, world, helper objects)
        # Only the rocks and agents should be left out

        # List to store everything that should be removed
        remove_list = []

        # Remove world from state
        remove_list.append("World")

        # Remove self
        remove_list.append(self.agent_id)

        # Remove all god agents
        remove_list.append('gravitygod')
        #remove_list.append('rewardgod')

        # Remove helper objects (avatar agents, goal reached image)
        remove_list.append('human')
        remove_list.append('machine')
        remove_list.append('goal_reached_img')

        # Find all parts of large objects and remove from state directly
        rock_parts = state.get_of_type(PartLargeObject.__name__)

        [state.remove(rockpart["obj_id"]) for rockpart in rock_parts]

        # Remove everything that was put on the remove list
        [state.remove(obj) for obj in remove_list]

        #TODO Translate object locations to what range they are in, then add that information

        # State now only contains rocks (without parts), selector agents and victim
        return state

    def first_tick(self, state):
        # Code that runs only during the first tick

        # At the first tick, check if there are new CPs that weren't yet shown in the GUI. Retrieve them and store
        with TypeDB.core_client("localhost:1729") as client:
            with client.session("CP_ontology", SessionType.DATA) as session:
                # Session is opened, now specify that it's a read session
                with session.transaction(TransactionType.READ) as read_transaction:
                    answer_iterator = read_transaction.query().match(
                        "match $x isa collaboration_pattern, has name $name; get $name;")

                    for answer in answer_iterator:
                        cp_retrieved = answer.get('name')._value
                        if cp_retrieved not in self.cp_list:
                            self.cp_list.append(cp_retrieved)

        return

    def decide_on_action(self, state):
        action_kwargs = {}
        action = None

        cp_name = None
        cp_situation = None
        cp_actionsA = None
        cp_actionsB = None
        cp_postsitu = None

        # If there are messages, deal with them, and send to TypeDB
        if self.received_messages:
            cp_name, cp_situation, cp_actionsA, cp_actionsB, cp_postsitu = self.message_handling()

            if cp_name in self.cp_list:
                # This means that it is an existing CP that gets sent, move to edit functions
                print('Existing CP name...')
                self.edit_cp_data(cp_name, cp_situation, cp_actionsA, cp_actionsB, cp_postsitu)
            elif cp_name:
                # If we end up here, the CP name is new, so we should create a new entry
                self.send_cp_data(cp_name, cp_situation, cp_actionsA, cp_actionsB, cp_postsitu)
                self.cp_list.append(cp_name)

        return action, action_kwargs

    # Function that translates x and y coordinates into the predefined location ranges
    def determine_location(self):
        # Top of rock pile

        # Above rock pile

        # Bottom of rock pile

        # Left/Right side of rock pile

        # Left/Right side of field

        # On top of [object/actor/location]

        return

    def determine_objects(self, obj):
        obj_type = None
        obj_size = None
        obj_color = None

        if 'rock' in obj:
            obj_type = 'rock'

        if 'Small' in obj:
            obj_size = 'small'
        elif 'Large' in obj:
            obj_size = 'large'

        if 'Brown' in obj:
            obj_color = 'brown'
            obj_size = 'large'
        else:
            obj_color = 'grey'

        return obj_type, obj_size, obj_color

    # Deals with incoming messages, focus on translating CP input into a workable format
    def message_handling(self):
        # Describe how to deal with messages

        cp_name = None  # Data type of the label is just a string
        cp_situation = []   # Data type of the situation description is a list of dictionaries; each dictionary contains an item (actor, object) and related attributes (location)
        cp_actionsA = []     # Data type of the actions is a list of dictionaries; each dictionary contains an actor, action, and related attributes (object, location)
        cp_actionsB = []
        cp_postsitu = []

        # Go through all received messages
        for message in self.received_messages:
            # If a received message is an updated/new CP
            if isinstance(message, dict) and 'actionA' in message:
                # Pull info out of the message
                cp_name = message['name']
                cp_situation_input = message['situation']
                cp_actions_a_input = message['actionA']
                cp_actions_b_input = message['actionB']
                cp_postsitu_input = message['postsitu']

                # Then start processing the input into workable

                # Situation split: the input consists of a list with the input from each box
                for situation_condition in cp_situation_input:
                    # For each action in this list, we have to identify all items
                    cp_situation.append(self.identify_items(situation_condition))

                # Actions Agent 1 split: the action input consists of a list with the input from each box
                for action in cp_actions_a_input:
                    # For each action in this list, we have to identify all items
                    cp_actionsA.append(self.identify_items(action))

                # Actions Agent 2 split: the action input consists of a list with the input from each box
                for action in cp_actions_b_input:
                    # For each action in this list, we have to identify all items
                    cp_actionsB.append(self.identify_items(action))

                # Situation after CP split: the input consists of a list with the input from each box
                for situation_condition in cp_postsitu_input:
                    # For each action in this list, we have to identify all items
                    cp_postsitu.append(self.identify_items(situation_condition))

            # After dealing with each message, remove it
            self.received_messages.remove(message)

            if cp_name:
                print("Name: ")
                print(cp_name)
                print("Start Situation: ")
                print(cp_situation)
                print("Actions Human: ")
                print(cp_actionsA)
                print("Actions Agent: ")
                print(cp_actionsB)
                print("End Situation: ")
                print(cp_postsitu)

        return cp_name, cp_situation, cp_actionsA, cp_actionsB, cp_postsitu

    # Function to translate html content into dictionaries (data prep for sending to ontology)
    def identify_items(self, html_input):
        # When given an html string, dissect it into it's items and return that as a dictionary
        input_dict = {}

        # While the input still contains valuable information
        while len(html_input)>20:
            # Find what kind of item we have first
            type_start = html_input.find('class="item ')
            type_end = html_input.find('" clonable=')

            # If there is no item type, we must be done, so return directly
            if type_start < 0:
                return input_dict

            # Find the name of the item
            word_start = html_input.find('<p>')
            word_end = html_input.find('</p>')

            # Find where this item ends
            item_end = html_input.find('</div')

            # Store the item type and the name of the item
            item_type = html_input[type_start + 12:type_end]
            word = html_input[word_start + 3:word_end]

            # Store the item type and the name of the item in the right format
            input_dict[item_type] = word

            # Remove the first item from the input
            html_input = html_input[item_end + 4:]

        return input_dict

    # Function that actually sends new data instances to the ontology
    def send_cp_data(self, name, situation, action_a, action_b, postsitu):

        # Create CP data entry query
        cp_instantiation = f'''insert $cp isa collaboration_pattern, has name '{name}';'''
        self.query_batch.append(cp_instantiation)

        # -----------------------Precondition seqs----------------------------------------------------------
        # For each condition, create a condition, is present in and context (including objects etc.)
        for condition in situation:
            # Add all of the important info together in one query that can be appended to the batch
            if len(condition) > 0:
                self.condition_translation(condition, name, "start", situation.index(condition))

        # ------------------------Action A seqs-------------------------------------------------------------
        # Dealing with action sequences of agent A (human)
        for single_action in action_a:
            # Add all of the important info together in one query that can be appended to the batch
            if len(single_action) > 0:
                self.action_translation(single_action, name, "human", action_a.index(single_action))

        # ------------------------Action B seqs-------------------------------------------------------------
        # Dealing with action sequences of agent B (robot)
        for single_action in action_b:
            # Add all of the important info together in one query that can be appended to the batch
            if len(single_action) > 0:
                self.action_translation(single_action, name, "robot", action_b.index(single_action))

        # -----------------------Postcondition seqs---------------------------------------------------------
        for condition in postsitu:
            # Add all of the important info together in one query that can be appended to the batch
            if len(condition) > 0:
                self.condition_translation(condition, name, "end", postsitu.index(condition))

        with TypeDB.core_client("localhost:1729") as client:
            with client.session("CP_ontology", SessionType.DATA) as session:
                # Create a write transaction
                self.write_batch(session, self.query_batch)

        return

    def edit_cp_data(self, name, situation, action_a, action_b, postsitu):
        # Write code for editing CP
        return

    def condition_translation(self, condition, name, pre_post, index):
        # Check if we're talking about a start- or a postcondition
        pre_post_cond = None
        if pre_post == 'start':
            pre_post_cond = 'startcondition'
        elif pre_post == 'end':
            pre_post_cond = 'endcondition'

        # Create unique context id
        ts = datetime.timestamp(datetime.now())  # for timestamp
        context_id = 'context_' + pre_post + '_' + str(index) + '_' + str(ts)

        # Create unique condition id
        condition_id = 'condition_' + pre_post + '_' + str(index) + '_' + str(ts)

        # Create the condition instance
        condition_instantiation = f'''insert $condition isa condition, has condition_type '{pre_post_cond}', has condition_id '{condition_id}'; '''

        # Create a context instance
        condition_instantiation = condition_instantiation + f'''$context isa context, has context_id '{context_id}'; '''

        # Append to query batch and empty
        self.query_batch.append(condition_instantiation)
        condition_instantiation = None

        # If the entry contains an object
        if 'object' in condition.keys():
            # Dissect object into object itself and its properties
            obj_type, obj_size, obj_color = self.determine_objects(condition['object'])

            # Create instance of the object
            if obj_color == 'grey':
                condition_instantiation = f'''match $object isa resource, has obj_type '{obj_type}', has size '{obj_size}', has color '{obj_color}';
                                        $context isa context, has context_id '{context_id}'; '''
            else:
                condition_instantiation = f'''match $object isa object, has obj_type '{obj_type}', has size '{obj_size}', has color '{obj_color}';
                                        $context isa context, has context_id '{context_id}'; '''

            # Connect object to context
            condition_instantiation = condition_instantiation + f'''insert $contains (whole: $context, part: $object) isa contains; '''

            # Append to query batch and empty
            self.query_batch.append(condition_instantiation)
            condition_instantiation = None

        # If the entry contains an actor
        if 'actor' in condition.keys():
            # Find instance of the actor at hand
            actor_type = None

            if 'human' in condition['actor']:
                actor_type = 'human'
            elif 'robot' in condition['actor']:
                actor_type = 'robot'
            elif 'Victim' in condition['actor']:
                actor_type = 'victim'

            condition_instantiation = f'''match $actor isa actor, has actor_type '{actor_type}';
                                    $context isa context, has context_id '{context_id}'; '''

            # Connect object to context
            condition_instantiation = condition_instantiation + f'''insert $contains (whole: $context, part: $actor) isa contains; '''

            # Append to query batch and empty
            self.query_batch.append(condition_instantiation)
            condition_instantiation = None

        # If the entry contains a location
        if 'location' in condition.keys():
            # Create instance of the location
            condition_instantiation = f'''match $location isa location, has range '{condition['location']}';
                                    $context isa context, has context_id "{context_id}"; '''

            # Connect object to context
            condition_instantiation = condition_instantiation + f'''insert $contains (whole: $context, part: $location) isa contains; '''

            # Append to query batch and empty
            self.query_batch.append(condition_instantiation)
            condition_instantiation = None

            # Create relation between object and location or actor, depending on which one exists
            if 'object' in condition.keys():
                # Dissect object into object itself and its properties
                obj_type, obj_size, obj_color = self.determine_objects(condition['object'])

                # Create instance of the object
                if obj_color == 'grey':
                    condition_instantiation = f'''match $object isa resource, has obj_type '{obj_type}', has size '{obj_size}', has color '{obj_color}';
                                                                        $location isa location, has range '{condition['location']}'; '''
                else:
                    condition_instantiation = f'''match $object isa object, has obj_type '{obj_type}', has size '{obj_size}', has color '{obj_color}';
                                                                        $location isa location, has range '{condition['location']}'; '''

                condition_instantiation = condition_instantiation + f'''insert $position (item: $object, location: $location) isa positioned_at; '''

                # Append to query batch and empty
                self.query_batch.append(condition_instantiation)
                condition_instantiation = None

            if 'actor' in condition.keys():
                # Find instance of the actor at hand
                actor_type = None

                if 'human' in condition['actor']:
                    actor_type = 'human'
                elif 'robot' in condition['actor']:
                    actor_type = 'robot'
                elif 'victim' in condition['actor']:
                    actor_type = 'victim'

                condition_instantiation = f'''match $actor isa actor, has actor_type '{actor_type}';
                                                                    $location isa location, has range '{condition['location']}'; '''

                condition_instantiation = condition_instantiation + f'''insert $position (item: $object, location: $location) isa positioned_at; '''

                # Append to query batch and empty
                self.query_batch.append(condition_instantiation)
                condition_instantiation = None

        # Connect condition to CP via existing starts when relation
        if pre_post_cond == 'startcondition':
            condition_instantiation = f'''match $condition isa condition, has condition_id '{condition_id}';
                                    $cp isa collaboration_pattern, has name '{name}';
                                    insert $start (condition: $condition, cp: $cp) isa starts_when;'''

            # Append to query batch and empty
            self.query_batch.append(condition_instantiation)
            condition_instantiation = None

            condition_instantiation = f'''match $condition isa condition, has condition_id '{condition_id}';
                                    $context isa context, has context_id '{context_id}';
                                    insert $present_in (condition: $condition, situation: $context) isa is_present_in; '''

            # Append to query batch and empty
            self.query_batch.append(condition_instantiation)
            condition_instantiation = None


        elif pre_post_cond == 'endcondition':
            condition_instantiation = f'''match $condition isa condition, has condition_id '{condition_id}';
                                                $cp isa collaboration_pattern, has name '{name}';
                                                insert $start (condition: $condition, cp: $cp) isa ends_when;'''

            # Append to query batch and empty
            self.query_batch.append(condition_instantiation)
            condition_instantiation = None

            condition_instantiation = f'''match $condition isa condition, has condition_id '{condition_id}';
                                                $context isa context, has context_id '{context_id}';
                                                insert $present_in (condition: $condition, situation: $context) isa is_present_in; '''

            # Append to query batch and empty
            self.query_batch.append(condition_instantiation)
            condition_instantiation = None

        return

    def action_translation(self, single_action, name, actor, index):

        action_instantiation = None

        # First, check if the entry actually contains some task
        if 'task' in single_action.keys():
            # Create unique task id
            ts = datetime.timestamp(datetime.now())  # for timestamp
            task_id = 'task_' + actor + '_' + str(index) + '_' + str(ts)

            # Create instance of the task
            action_instantiation = f'''insert $task isa task, has task_name "{single_action['task']}", has task_id "{task_id}", has order_value "{index}"; '''

            # Append to query batch and empty
            self.query_batch.append(action_instantiation)
            action_instantiation = None

            # Find task and cp for relation
            action_instantiation = f'''match $task isa task, has task_name "{single_action['task']}", has task_id "{task_id}";
                                    $cp isa collaboration_pattern, has name "{name}"; '''

            # Create the instance of the relation between this task and the CP we're working on
            action_instantiation = action_instantiation + f'''insert $prescription (cp: $cp, task: $task) isa is_part_of, has action_type "individual"; '''

            # Append to query batch and empty
            self.query_batch.append(action_instantiation)
            action_instantiation = None

            # Create the instance of the agent doing it
            action_instantiation = f'''match $actor isa actor, has actor_type "{actor}";
                                    $task isa task, has task_name "{single_action['task']}", has task_id "{task_id}"; '''

            # Create the instance of the relation between the agent and the task
            action_instantiation = action_instantiation + f'''insert $performing (actor: $actor, action: $task) isa performed_by; '''

            # Append to query batch and empty
            self.query_batch.append(action_instantiation)
            action_instantiation = None

            # Check if the task contains an object
            if 'object' in single_action.keys():
                # Dissect object into object itself and its properties
                obj_type, obj_size, obj_color = self.determine_objects(single_action['object'])

                # Create instance of the object
                if obj_color == 'grey':
                    action_instantiation = f'''match $object isa resource, has obj_type "{obj_type}", has size "{obj_size}", has color "{obj_color}";
                                            $task isa task, has task_name "{single_action['task']}", has task_id "{task_id}"; '''

                    # Also create the relevant relations
                    action_instantiation = action_instantiation + f'''insert $using (resource: $object, action: $task) isa uses;
                                                                   $affording (object: $object, action: $task) isa affords; '''

                    # Append to query batch and empty
                    self.query_batch.append(action_instantiation)
                    action_instantiation = None

                else:
                    action_instantiation = f'''match $object isa object, has obj_type "{obj_type}", has size "{obj_size}", has color "{obj_color}";
                                            $task isa task, has task_name "{single_action['task']}", has task_id "{task_id}"; '''

                    # Also create the relevant relations
                    action_instantiation = action_instantiation + f'''insert $using (resource: $object, action: $task) isa uses;
                                                                    $affording (object: $object, action: $task) isa affords; '''

                    # Append to query batch and empty
                    self.query_batch.append(action_instantiation)
                    action_instantiation = None

            # Check if the task contains a location
            if 'location' in single_action.keys():
                # Create instance of the location
                action_instantiation = f'''match $location isa location, has range "{single_action['location']}";
                                        $task isa task, has task_name "{single_action['task']}", has task_id "{task_id}"; '''

                # Also create the relevant relation
                action_instantiation = action_instantiation + f'''insert $takingplace (location: $location, action: $task) isa takes_place_at; '''

                # Append to query batch and empty
                self.query_batch.append(action_instantiation)
                action_instantiation = None

        return

    def write_batch(self, session, batch):
        # Code grabbed from example at
        # https://github.com/typedb-osi/typedb-bio/blob/6c22ccefcb29f5bebbfde5c00a8b6563f8bca5e3/Migrators/Helpers/batchLoader.py
        with session.transaction(TransactionType.WRITE) as tx:
            for query in batch:
                print("Query: ")
                print(query)
                tx.query().insert(query)
            tx.commit()
            self.query_batch = []

        return
