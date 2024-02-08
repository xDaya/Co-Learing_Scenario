from matrx.agents.agent_types.human_agent import *
from custom_actions import *
from matrx.messages.message import Message
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.move_actions import *
import csv
import pickle
import deepdiff
import numpy as np
import pandas as pd



class RobotPartner(AgentBrain):

    def __init__(self, move_speed=5):
        super().__init__()
        self.state_tracker = None
        self.move_speed = move_speed
        self.navigator = None

        self.actionlist = None
        self.action_history = None
        self.q_table = {}
        self.initial_heights = None
        self.human_location = []
        self.previous_action = None
        self.previous_exec = None
        self.previous_phase = None
        self.final_update = False
        self.alpha = 0.5
        self.gamma = 0.6
        self.run_number = 0

        #with open('qtable_backup.pkl', 'rb') as f:
        #    self.q_table = pickle.load(f)

        # Ontology related variables
        self.cp_list = []
        self.start_conditions = []
        self.end_conditions = []
        self.executing_cp = False

        self.executing_action = False

        self.cp_actions = [] # Keeps track of the actions still left in the CP

        self.current_human_action = None
        self.current_robot_action = None
        self.past_human_actions = []

        # Global variables for learning algorithms
        self.q_table_cps = pd.DataFrame()
        self.q_table_cps_runs = pd.DataFrame()
        self.nr_chosen_cps = 0
        self.q_table_basic = pd.DataFrame(columns=['Move back and forth', 'Stand Still', 'Pick up', 'Drop', 'Break'])
        self.visited_states = []
        self.starting_state = []
        self.starting_state_distance = 0
        self.first_tick_distance = 0

        # Global progress variables
        self.nr_ticks = 0
        self.nr_move_actions = 0
        self.nr_productive_actions = 0
        self.victim_harm = 0
        self.idle_ticks = 0
        self.robot_contribution = 0
        self.human_standing_still = False

        # Helper variables
        self.previous_objs = []
        self.previous_locs = []
        self.field_locations = []
        for x in range(5, 15):
            for y in range(0, 11):
                self.field_locations.append((x, y))

        self.condition = 2

        self.exp_condition = 'communication'

        self.database_name = None

        # Code that ensures backed up q-tables are retrieved in case of crash
        print("Retrieving backed up q-tables...")
        try:
            with open('qtable_cps_backup.pkl', 'rb') as f:
                self.q_table_cps = pickle.load(f)
            print("Backed up CPs q-table stored in variable.")
        except:
            print("No backed up q-table available for the CPs.")

        try:
            with open('qtable_cps_runs_backup.pkl', 'rb') as f:
                self.q_table_cps_runs = pickle.load(f)
            print("Backed up CPs q-table stored in variable.")
        except:
            print("No backed up q-table available for the CPs.")

        try:
            with open('qtable_basic_backup.pkl', 'rb') as f:
                self.q_table_basic = pickle.load(f)
            print("Backed up basic behavior q-table stored in variable.")
        except:
            print("No backed up q-table available for the basic behavior.")

    def initialize(self):
        self.state_tracker = StateTracker(agent_id=self.agent_id)

        self.navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set, algorithm=Navigator.A_STAR_ALGORITHM)

        self.actionlist = [[], []]      # Array that stores which actual actions must be executed (and their arguments)
        self.action_history = []  # Array that stores actions that have been taken in a certain state
        self.previous_phase = None
        self.final_update = False
        self.run_number = self.run_number +1
        if self.run_number > 3:
            self.alpha = self.alpha / 2
        self.received_messages = []

        self.database_name = "CP_ontology_" + str(self.agent_properties['participant_nr'])
        self.database_name = "CP_ontology"

        if self.exp_condition == 'ontology':
            # Initialize existing CP's and their conditions
            print("Robot Initializing CPs")
            self.store_cp_conditions(self.start_conditions)
            self.store_cp_conditions(self.end_conditions)
            print("Start conditions:")
            print(self.start_conditions)
            print("End conditions:")
            print(self.end_conditions)

        # Check if q-tables for the CPs are filled. If not, initialize. If yes, check if it matches the current CP set
        # and correct where necessary.
            if self.q_table_cps.empty:
                for cp in self.cp_list:
                    self.q_table_cps[cp] = 0
                    self.q_table_cps_runs[cp] = 0
            else:
                for cp in self.cp_list:
                    if cp not in self.q_table_cps.columns:
                        self.q_table_cps[cp] = 0
                        self.q_table_cps_runs[cp] = 0
                for column in self.q_table_cps.columns:
                    if column not in self.cp_list:
                        self.q_table_cps.drop(column, axis=1, inplace=True)
                        self.q_table_cps_runs.drop(column, axis=1, inplace=True)

        # Remove columns with None as name for testing phase
        #self.q_table_cps.drop('None', axis=1, inplace=True)

        # Start with some wait actions
        #self.wait_action(None)
        #self.wait_action(None)
        #self.wait_action(None)
        #self.wait_action(None)

    def filter_observations(self, state):
        self.state_tracker.update(state)
        return state

    def filter_observations_learning(self, state):
        self.state_tracker.update(state)

        phase2 = False
        phase3 = False
        phase4 = False
        goal_phase = False
        # Define the state for learning here
        # Get all perceived objects
        object_ids = list(state.keys())
        # Remove world from state
        object_ids.remove("World")
        # Remove self
        object_ids.remove(self.agent_id)
        # Store and remove all (human)agents
        human_id = [obj_id for obj_id in object_ids if "CustomHumanAgentBrain" in state[obj_id]['class_inheritance']]
        object_ids = [obj_id for obj_id in object_ids if "AgentBrain" not in state[obj_id]['class_inheritance'] and
                      "AgentBody" not in state[obj_id]['class_inheritance']]

        # Location of human (abstracted)
        human_location = state[human_id[0]]['location']

        # Location of robot (abstracted)
        robot_location = state[self.agent_id]['location']

        # Objects carrying
        currently_carrying = len(state[self.agent_id]['is_carrying'])

        # Rubble locations
        lower_bound = 11
        left_bound = 5
        right_bound = 15
        upper_bound = 1
        empty_rubble_locations = []
        for y_loc in range(upper_bound, lower_bound):
            for x_loc in range(left_bound, right_bound):
                empty_rubble_locations.append((x_loc, y_loc))

        # Number of free columns and spots
        empty_columns = list(range(5,15))
        for object_id in object_ids:
            object_loc = state[object_id]['location']
            object_loc_x = object_loc[0]
            if object_loc_x in empty_columns:
                empty_columns.remove(object_loc_x)
            if object_loc in empty_rubble_locations:
                empty_rubble_locations.remove(object_loc)

        nr_empty_columns = len(empty_columns)

        # Ratio of small/large blocks (or number of large blocks) (now for all blocks, change to blocks in field?)
        nr_large_blocks = 0
        nr_small_blocks = 0
        for object_id in object_ids:
            if "large" in state[object_id]:
                nr_large_blocks += 1
            elif "bound_to" in state[object_id]:
                continue
            else:
                nr_small_blocks += 1

        if nr_large_blocks > 0:
            ratio_small_large = round(nr_small_blocks/nr_large_blocks)
        else:
            ratio_small_large = nr_small_blocks

        # Height differences between columns (looks at all blocks, not just field)
        column_heights = []
        for object_id in object_ids:
            if state[object_id]['location'][0] >= 15 or state[object_id]['location'][0] < 5:
                continue
            object_loc = state[object_id]['location']
            object_loc_x = object_loc[0]
            object_loc_y = object_loc[1]
            if column_heights and object_loc_x in np.asarray(column_heights)[:,0]:
                index = list(np.asarray(column_heights)[:,0]).index(object_loc_x)
                if 11- object_loc_y > column_heights[index][1]:
                    column_heights[index][1] = 11 - object_loc_y
            else:
                column_heights.append([object_loc_x, 11 - object_loc_y])

        np_column_heights = np.asarray(column_heights)
        try:
            np_column_heights = np_column_heights[np.argsort(np_column_heights[:,0])]
        except IndexError:
            np_column_heights = np_column_heights

        try:
            column_sum =  np.sum(np_column_heights[:,1], axis=0)
            column_sum_diff = np.sum(np.abs(np.diff(np_column_heights[:,1])))
        except IndexError:
            column_sum = 0
            column_sum_diff = 0


        # Check if phase 2 is reached
        if self.initial_heights is None:
            self.initial_heights = column_sum

        if self.initial_heights is not None and self.initial_heights - column_sum >= 8:
            phase2 = True

        # Check if phase 3 is reached
        if self.initial_heights is not None and self.initial_heights - column_sum >= 20:
            phase3 = True

        # Check if phase 4 is reached
        # TODO:(now checks for empty columns, needs to be adapted for bridge scenarios)

        # If all rubble is gone from the victim itself
        if {(8,9), (8,10), (9,9), (9,10), (10,9), (10,10), (11,9), (11,10)}.issubset(set(empty_rubble_locations)):
            phase4 = True

        # If there is a free path from the left side
        if {(5,7), (5,8), (5,9), (5,10), (6,7), (6,8), (6,9), (6,10), (7,7), (7,8), (7,9), (7,10)}.issubset(set(empty_rubble_locations)):
            phase4 = True

        # If there is a free path from the right side
        if {(12,7), (12,8), (12,9), (12,10), (13,7), (13,8), (13,9), (13,10), (14,7), (14,8), (14,9), (14,10)}.issubset(set(empty_rubble_locations)):
            phase4 = True

        # Check if goal phase is reached
        # TODO:(now checks for empty columns, needs to be adapted for bridge scenarios)

        # If all rubble is gone from the victim and there is a free path from the left side
        if {(8, 9), (8, 10), (9, 9), (9, 10), (10, 9), (10, 10), (11, 9), (11, 10), (5,7), (5,8), (5,9), (5,10), (6,7),
                (6,8), (6,9), (6,10), (7,7), (7,8), (7,9), (7,10)}.issubset(set(empty_rubble_locations)):
            goal_phase = True

        # If all rubble is gone from the victim and there is a free path from the right side
        if {(8, 9), (8, 10), (9, 9), (9, 10), (10, 9), (10, 10), (11, 9), (11, 10), (12,7), (12,8), (12,9), (12,10),
                (13,7), (13,8), (13,9), (13,10), (14,7), (14,8), (14,9), (14,10)}.issubset(set(empty_rubble_locations)):
            goal_phase = True

        filtered_state = {}
        #filtered_state['empty_columns'] = nr_empty_columns
        #filtered_state['ratio_small_large'] = ratio_small_large
        #filtered_state['column_sum'] = column_sum
        #filtered_state['column_sum_diff'] = column_sum_diff
        #filtered_state['currently_carrying'] = currently_carrying
        #filtered_state['large_blocks'] = nr_large_blocks
        filtered_state['Phase 2'] = phase2
        filtered_state['Phase 3'] = phase3
        filtered_state['Phase 4'] = phase4
        filtered_state['Goal Phase'] = goal_phase

        return filtered_state

    def update_q_table(self, current_state, chosen_action, done_action, done_state, reward):
        gamma = 0.6
        # Update the expected reward for a specific state-action pair
        if frozenset(current_state.items()) in self.q_table:
            # State already exists in q-table, so we update
            max_q = self.q_table[frozenset(current_state.items())][chosen_action]
            #print(max_q)
            self.q_table[frozenset(done_state.items())][done_action] = self.q_table[frozenset(done_state.items())][done_action] + self.alpha * (reward + gamma * max_q - self.q_table[frozenset(done_state.items())][done_action])
        else:
            # State does not exist in q-table, so we create a new entry with all zero's
            self.q_table[frozenset(current_state.items())] = [0] * 3
            # And then update
            Max_Q = self.q_table[frozenset(current_state.items())][chosen_action]
            self.q_table[frozenset(current_state.items())][chosen_action] = reward + gamma * Max_Q
        self.agent_properties["q_table"] = str(self.q_table)

        with open('qtable_backup.csv', 'w', newline='') as f:
            w = csv.writer(f, delimiter=';')
            for key in self.q_table.keys():
                w.writerow((list(key),self.q_table[key]))

        with open('qtable_backup.pkl', 'wb') as f:
            pickle.dump(self.q_table, f, pickle.HIGHEST_PROTOCOL)

        return


# The functions below directly point to actual actions. They are created to ensure the right action arguments
    def pickup_action(self, object_ids, state):
        small_obj = []
        y_loc_list = []
        for object_id in object_ids:
            object_id = object_id['obj_id']
            if "large" in self.state[object_id]:
                continue
            if "bound_to" in self.state[object_id] and self.state[object_id]['bound_to'] is not None:
                continue
            if self.state[object_id]['location'][0] >= 15 or self.state[object_id]['location'][0] < 5:
                continue

            y_loc_list.append(self.state[object_id]['location'][1])
            small_obj.append(object_id)

        if not y_loc_list:
            return

        chosen_object = small_obj[y_loc_list.index(min(y_loc_list))]    # Pick object with smallest y
        object_loc = self.state[chosen_object]['location']

        # Add move action to action list
        self.navigator.add_waypoint(object_loc)
        route_actions = list(self.navigator._Navigator__get_route(self.state_tracker).values())
        for action in route_actions:
            self.actionlist[0].append(action)
            self.actionlist[1].append({})

        # Add pick up action to action list (define arguments here as well)
        pickup_kwargs = {}
        pickup_kwargs['object_id'] = chosen_object
        pickup_kwargs['grab_range'] = 1
        pickup_kwargs['max_objects'] = 5
        self.actionlist[0].append(GrabObject.__name__)
        self.actionlist[1].append(pickup_kwargs)
        return

    def pickup_large_action(self, object_ids, state, location):
        large_obj = []
        parts_list = []
        y_loc_list = []
        dist_list = []
        chosen_part = None
        object_ids = object_ids
        if isinstance(object_ids, dict):
            object_ids = [object_ids] + self.state[{'bound_to'}]
        else:
            object_ids = object_ids + self.state[{'bound_to'}]
        for object_id in object_ids:
            if "obstruction" in object_id:
                continue
            if object_id['location'][0] >= 15 or object_id['location'][0] < 5:
                continue
            if "large" in object_id:
                y_loc_list.append(object_id['location'][1])
                large_obj.append(object_id['obj_id'])
                if location is not None:
                    dist = int(np.ceil(np.linalg.norm(np.array(object_id['location'])
                                                  - np.array(location))))
                    dist_list.append(dist)
            if "bound_to" in object_id:
                parts_list.append(object_id['obj_id'])

        if not y_loc_list:
            return

        if location is not None:
            chosen_part = large_obj[dist_list.index(min(dist_list))]
        else:
            chosen_part = large_obj[y_loc_list.index(min(y_loc_list))]

        if chosen_part is None:
            return

        large_name = self.state[chosen_part]['name']
        object_loc = self.state[chosen_part]['location']
        large_obj = [chosen_part]
        for part in parts_list:
            if self.state[part]['bound_to'] == large_name:
                large_obj.append(part)

        if object_loc is None:
            return

        # Add move action to action list
        self.navigator.add_waypoint(object_loc)
        route_actions = list(self.navigator._Navigator__get_route(self.state_tracker).values())
        for action in route_actions:
            self.actionlist[0].append(action)
            self.actionlist[1].append({})

        # Add pick up action to action list (define arguments here as well)
        pickup_kwargs = {}
        pickup_kwargs['object_id'] = large_obj
        pickup_kwargs['grab_range'] = 1
        pickup_kwargs['max_objects'] = 5
        self.actionlist[0].append(GrabLargeObject.__name__)
        self.actionlist[1].append(pickup_kwargs)
        return

    def drop_action(self, state, location):
        drop_action = None
        repetition = 1
        obj_type = None
        chosen_loc = location
        # Check if the agent is actually carrying something
        if self.state[self.agent_id]['is_carrying']:
            carrying_obj = self.state[self.agent_id]['is_carrying'][0]
            if "large" in carrying_obj:
                drop_action = DropLargeObject.__name__
            elif "bound_to" in carrying_obj and carrying_obj['bound_to'] is not None:
                drop_action = DropLargeObject.__name__
            else:
                drop_action = DropObject.__name__
                repetition = len(self.state[self.agent_id]['is_carrying'])

            if "vert" in carrying_obj:
                obj_type = 'vert'
            if "long" in carrying_obj:
                obj_type = 'long'

            # Choose location for dropping
            if chosen_loc is None:
                # Retrieve where the agent is now
                agent_loc = self.agent_properties['location']
                # If a good location for dropping, choose this x and a bit higher y
                if agent_loc[0] < 3 or agent_loc[0] > 15:
                    chosen_loc = (agent_loc[0], agent_loc[1]-2)
                # Otherwise, choose location outside of field near agent
                elif agent_loc[0] <= 10:
                    possible_xloc = list(range(0, 2))
                    x_loc = random.choice(possible_xloc)
                    chosen_loc = (x_loc, agent_loc[1]-2)
                else:
                    possible_xloc = list(range(16,19))
                    x_loc = random.choice(possible_xloc)
                    chosen_loc = (x_loc, agent_loc[1]-2)

            # Add move action to action list
            self.navigator.add_waypoint(chosen_loc)         # Add some code that searches for an empty spot out of the field
            route_actions = list(self.navigator._Navigator__get_route(self.state_tracker).values())
            for action in route_actions:
                self.actionlist[0].append(action)
                self.actionlist[1].append({})

            # Add pick up action to action list (define arguments here as well)
            pickup_kwargs = {}
            pickup_kwargs['obj_type'] = obj_type
            for i in range(repetition):
                self.actionlist[0].append(drop_action)
                self.actionlist[1].append(pickup_kwargs)
        return

    def break_action(self, object_ids, state, location):
        large_obj = []
        parts_list = []
        y_loc_list = []
        dist_list = []
        chosen_part = None
        object_ids = object_ids + self.state[{'bound_to'}]
        print(object_ids)
        for object_id in object_ids:
            object_id = object_id['obj_id']
            if "obstruction" in self.state[object_id]:
                continue
            if "large" in self.state[object_id] and self.state[object_id]['location'][0] >= 5 and self.state[object_id]['location'][0] < 15:
                y_loc_list.append(self.state[object_id]['location'][1])
                large_obj.append(object_id)
                dist = int(np.ceil(np.linalg.norm(np.array(self.state[object_id]['location'])
                                                  - np.array(self.state[self.agent_id]['location']))))
                dist_list.append(dist)
            if "bound_to" in self.state[object_id]:
                parts_list.append(object_id)

        if not y_loc_list:
            return

        if location is not None:
            chosen_part = large_obj[dist_list.index(min(dist_list))]
        else:
            chosen_part = large_obj[y_loc_list.index(min(y_loc_list))]
        large_name = self.state[chosen_part]['name']
        object_loc = self.state[chosen_part]['location']
        large_obj = [chosen_part]
        for part in parts_list:
            if self.state[part]['bound_to'] == large_name:
                large_obj.append(part)

        # Add move action to action list
        self.navigator.add_waypoint(object_loc)
        route_actions = list(self.navigator._Navigator__get_route(self.state_tracker).values())
        for action in route_actions:
            self.actionlist[0].append(action)
            self.actionlist[1].append({})

        # Add pick up action to action list (define arguments here as well)
        pickup_kwargs = {}
        pickup_kwargs['object_id'] = large_obj
        pickup_kwargs['grab_range'] = 1
        pickup_kwargs['max_objects'] = 5
        self.actionlist[0].append(BreakObject.__name__)
        self.actionlist[1].append(pickup_kwargs)
        return

    def wait_action(self, location):
        # Check if there is a specific location in which we should wait
        if location is not None:
            # Then add move actions first
            self.navigator.add_waypoint(location)
            route_actions = list(self.navigator._Navigator__get_route(self.state_tracker).values())
            for action in route_actions:
                self.actionlist[0].append(action)
                self.actionlist[1].append({})

        pickup_kwargs = {}
        for i in range(5):
            self.actionlist[0].append(Idle.__name__)
            self.actionlist[1].append(pickup_kwargs)
        return

    def move_back_forth_action(self, location):
        # Check if there is a specific location in which we should move
        if location is not None:
            # Then add move actions to this location first
            self.navigator.add_waypoint(location)
            route_actions = list(self.navigator._Navigator__get_route(self.state_tracker).values())
            for action in route_actions:
                self.actionlist[0].append(action)
                self.actionlist[1].append({})

        # Define the back and forth move actions
        back_forth_moves = [MoveEast.__name__, MoveEast.__name__, MoveWest.__name__, MoveWest.__name__]
        # Add those actions to the action list
        for i in range(3):
            for move_action in back_forth_moves:
                self.actionlist[0].append(move_action)
                self.actionlist[1].append({})

        return

# Policies 1, 2 and 3 are the high level actions
    def policy1(self, object_ids, state):
        # If carrying less than 5 objects: look at the top small and large objects and choose one to pick up
        if self.previous_exec == "drop" and len(state[self.agent_id]['is_carrying']) > 0:
            self.drop_action(state, None)
            self.previous_exec = "drop"
        elif len(state[self.agent_id]['is_carrying']) < 5:
            obj = []
            y_loc_list = []
            for object_id in object_ids:
                # If object is outside of field, skip
                if state[object_id]['location'][0] >= 15 or state[object_id]['location'][0] < 5:
                    continue
                # If agent is already carrying something and the object is large, skip
                if len(state[self.agent_id]['is_carrying']) > 0 and "large" in state[object_id]:
                    continue
                # If agent is already carrying something and the object is bound_to (thus large), skip
                if len(state[self.agent_id]['is_carrying']) > 0 and "bound_to" in state[object_id]:
                    # Added check to make sure bound_to = None objects are not skipped
                    if state[object_id]["bound_to"] is not None:
                        continue
                # If object is brown, skip
                if 'brown' in state[object_id]['name']:
                    continue

                y_loc_list.append(state[object_id]['location'][1])
                obj.append(object_id)

            # If there are no objects left
            if len(obj) < 1:
                # And the agnet is carrying something
                if len(state[self.agent_id]['is_carrying']) > 0:
                    self.drop_action(state, None)
                    self.previous_exec = "drop"
                else:
                    self.wait_action(None)
            # If there are objects left
            else:
                chosen_object = obj[y_loc_list.index(min(y_loc_list))]  # Pick object with smallest y
                if "large" in state[chosen_object]:
                    self.pickup_large_action(object_ids, state, None)
                    self.previous_exec = "pickup"
                elif "bound_to" in state[chosen_object]:
                    if state[chosen_object]['bound_to'] is not None:
                        self.pickup_large_action(object_ids, state, None)
                        self.previous_exec = "pickup"
                    else:
                        self.pickup_action(object_ids, state)
                        self.previous_exec = "pickup"
                else:
                    self.pickup_action(object_ids, state)
                    self.previous_exec = "pickup"
        # If carrying 5 objects (or a large one) choose a spot outside of field and drop the object there
        else:
            self.drop_action(state, None)
            self.previous_exec = "drop"
        return

    def policy2(self, object_ids, state):
        objects = list(state.keys())
        human_ids = []
        object_locs = []
        for obj_id in objects:
            if 'class_inheritance' in state[obj_id] and "CustomHumanAgentBrain" in state[obj_id]['class_inheritance']:
                human_ids.append(obj_id)

        human_id = human_ids[0]
        human_loc = state[human_id]['location']

        # Record location of human and how long they have been there
        if self.human_location:
            if self.human_location[0] == human_loc:
                self.human_location[1] = self.human_location[1] + 1
            else:
                self.human_location[0] = human_loc
                self.human_location[1] = 0
        else:
            self.human_location.append(human_loc)
            self.human_location.append(0)

        # If the human lingers
        if self.human_location[1] >= 3:
            # Check if the human is in an empty spot or on a large block
            # Create list with object locations first
            for object_id in object_ids:
                if "GoalReachedObject" in state[object_id]['class_inheritance']:
                    continue
                object_loc = state[object_id]['location']
                object_locs.append(object_loc)
            # Then check if the human's location is in the object locations list
            if self.human_location[0] not in object_locs:
                # Apparently human lingers around an empty spot, so we need to decide how big the spot is
                if len(state[self.agent_id]['is_carrying']) > 0:
                    self.drop_action(state, self.human_location[0])
                else:
                    self.pickup_large_action(object_ids, state, None)
            else:
                # Apparently the human lingers around blocks, so we need to decide which block that is and pick it up (if it is a large block?)
                # But first check if the agent is still carrying something, and drop that
                if len(state[self.agent_id]['is_carrying']) > 0:
                    self.drop_action(state, None)
                else:
                    self.pickup_large_action(object_ids, state, self.human_location[0])
        else:
            # If the human is moving around, wait for them to act
            self.wait_action(None)
        return

    def policy3(self, object_ids, state):
        objects = list(state.keys())
        human_ids = []
        for obj_id in objects:
            if 'class_inheritance' in state[obj_id] and "CustomHumanAgentBrain" in state[obj_id]['class_inheritance']:
                human_ids.append(obj_id)

        human_id = human_ids[0]
        human_loc = state[human_id]['location']

        # Record location of human and how long they have been there
        if self.human_location:
            if self.human_location[0] == human_loc:
                self.human_location[1] = self.human_location[1] + 1
            else:
                self.human_location[0] = human_loc
                self.human_location[1] = 0
        else:
            self.human_location.append(human_loc)
            self.human_location.append(0)

        # When human lingers/stands still, break rocks around them
        if self.human_location[1] >= 3:
            self.break_action(object_ids, state, self.human_location[0])
            self.human_location[1] = 0
        else:
            # If they move, follow the human (add a condition here)
            self.navigator.add_waypoint(human_loc)
            route_actions = list(self.navigator._Navigator__get_route(self.state_tracker).values())
            for action in route_actions:
                self.actionlist[0].append(action)
                self.actionlist[1].append({})
        return

# Here is the decide on action as you can see!
    def decide_on_action(self, state):
        action_kwargs = {}
        action = None
        # List with all objects
        # Get all perceived objects
        object_ids = list(state.keys())
        # Remove world from state
        object_ids.remove("World")
        # Remove self
        object_ids.remove(self.agent_id)
        # Remove all (human)agents
        object_ids = [obj_id for obj_id in object_ids if "AgentBrain" not in state[obj_id]['class_inheritance'] and
                      "AgentBody" not in state[obj_id]['class_inheritance']]

        object_ids = [obj_id for obj_id in object_ids if "obstruction" not in state[obj_id]]

        if self.first_tick_distance == 0:
            self.first_tick_distance = self.distance_goal_state()
        self.human_standing_still = self.human_standstill()
        # ----------------------------Goal reached check-----------------------------------------------------------
        reward_agent = self.state[{'class_inheritance': "RewardGod"}]
        if reward_agent['goal_reached']:
            return None, None

        # -----------------------------Image management for carrying----------------------------------------
        if self.executing_cp:
            if state[self.agent_id]['is_carrying']:
                if len(state[self.agent_id]['is_carrying']) == 1:
                    self.agent_properties["img_name"] = "/images/robot_hand_small_1_cp.png"
                elif len(state[self.agent_id]['is_carrying']) == 2:
                    self.agent_properties["img_name"] = "/images/robot_hand_small_2_cp.png"
                elif len(state[self.agent_id]['is_carrying']) == 3:
                    self.agent_properties["img_name"] = "/images/robot_hand_small_3_cp.png"
                elif len(state[self.agent_id]['is_carrying']) == 4:
                    self.agent_properties["img_name"] = "/images/robot_hand_small_4_cp.png"
                else:
                    if 'large' in state[self.agent_id]['is_carrying'][0]:
                        self.agent_properties["img_name"] = "/images/robot_hand_large_cp.png"
                    else:
                        self.agent_properties["img_name"] = "/images/robot_hand_small_5_cp.png"
            else:
                self.agent_properties["img_name"] = "/images/robot_hand_cp.png"
            self.agent_properties['executing_cp'] = self.executing_cp
        else:
            if state[self.agent_id]['is_carrying']:
                if len(state[self.agent_id]['is_carrying']) == 1:
                    self.agent_properties["img_name"] = "/images/robot_hand_small_1.png"
                elif len(state[self.agent_id]['is_carrying']) == 2:
                    self.agent_properties["img_name"] = "/images/robot_hand_small_2.png"
                elif len(state[self.agent_id]['is_carrying']) == 3:
                    self.agent_properties["img_name"] = "/images/robot_hand_small_3.png"
                elif len(state[self.agent_id]['is_carrying']) == 4:
                    self.agent_properties["img_name"] = "/images/robot_hand_small_4.png"
                else:
                    if 'large' in state[self.agent_id]['is_carrying'][0]:
                        self.agent_properties["img_name"] = "/images/robot_hand_large.png"
                    else:
                        self.agent_properties["img_name"] = "/images/robot_hand_small_5.png"
            else:
                self.agent_properties["img_name"] = "/images/robot_hand.png"
            self.agent_properties['executing_cp'] = False


        # ----------------------------Do message handling---------------------------------------------
        if self.received_messages:
            self.message_handling()

        # --------------------------New Main Action Planning Loop--------------------------------------------
        # TODO Add some code that checks if there are new CPs (or edits). Can come from messages

        # Record task progress
        self.record_progress(False)

        #print('ACTIONLIST')
        #print(self.actionlist)

        # If the actionlist ends up empty, that means we're done executing an action.
        # If that was an action from the basic behavior, we should do the reward update here
        if len(self.actionlist[0]) == 0 and self.executing_action:
            self.reward_update_basic()
            self.record_progress(True)
            self.executing_action = False
            print('Actionlist empty after basic behavior action')
        elif len(self.actionlist[0]) == 1 and self.condition > 0:
            # There is only one action left in the list, therefore, communicate about this action
            self.communicate_actions()

        # Check the conditions of stored CPs
        #self.check_cp_conditions(self.start_conditions)
        # Start by checking if the agent is currently executing a CP
        if self.executing_cp:
            #print("Agent is executing a CP:")
            #print(self.executing_cp)
            # Check if the endconditions for this CP hold (of if the startconditions no longer hold)
            # Also check if the CP wasn't deleted in the meantime
            if self.executing_cp in self.check_cp_conditions(self.end_conditions) or self.executing_cp not in self.check_cp_conditions(self.start_conditions) or self.executing_cp not in self.cp_list:
                # If yes, finish, process reward and restart loop
                print("The endconditions for this CP hold, so we'll stop executing it.")
                self.send_message(Message(content=f"I will stop following the Collaboration Pattern {self.executing_cp}", from_id=self.agent_id, to_id=None))
                if self.executing_cp in self.cp_list:
                    self.reward_update_cps()
                self.executing_cp = False
                self.actionlist = [[], []]
                self.navigator.reset_full()     # Reset navigator to make sure there are no remaining waypoints
                self.cp_actions = []
                # Reset progress variables
                self.record_progress(True)
            else:
                # If no, continue current CP
                # Original Action Planning Code
                # If an action has already been translated, continue the accompanying atomic actions
                #print('Continue current CP')
                if len(self.actionlist[0]) != 0 and self.current_robot_action:
                    # This means that an action is still being executed
                    action = self.actionlist[0].pop(0)
                    action_kwargs = self.actionlist[1].pop(0)
                    action_kwargs['action_duration'] = self.move_speed

                    # If the actionlist ends up empty here, that means we're done executing an action.
                    # The consequence should be that the self.current_robot_action is reset, and removed from the list
                    if len(self.actionlist[0]) == 0:
                        self.cp_actions.remove(self.current_robot_action)
                        self.current_robot_action = None

                        # If the CP actions list ends up empty here, we should do a reward update
                        if len(self.cp_actions) == 0 and self.executing_cp in self.cp_list:
                            self.reward_update_cps()

                        print('Actionlist empty after CP action')

                    #return action, action_kwargs  # Returned here, so code underneath is then not executed
                # If no action was translated, look at the CP to see what should be the next action to be translated
                else:
                    self.execute_cp(self.executing_cp, state)
        else:
            # Not currently executing a CP
            print("Not working with a CP currently!")

            # Check if the start conditions for any existing CPs hold
            if self.exp_condition == 'ontology':
                cps_hold = self.check_cp_conditions(self.start_conditions)
            else:
                cps_hold = []
            if len(cps_hold) > 0:
                # This means there are CPs that are applicable and that should be executed.
                # If we were still executing a Basic Behavior Action, we should now end that, reset action list and do reward update
                if self.executing_action:
                    self.actionlist = [[], []]
                    self.navigator.reset_full()  # Reset navigator to make sure there are no remaining waypoints
                    self.reward_update_basic()
                    self.record_progress(True)
                    self.executing_action = False
                # Check how many CPs hold.
                if len(cps_hold) == 1:
                    # Only one CP is applicable, so we can directly start executing it
                    print("Only one CP holds: " + cps_hold[0])
                    msg = f"I will now follow the Collaboration Pattern {cps_hold[0]}."
                    self.send_message(Message(content=msg, from_id=self.agent_id, to_id=None))
                    self.executing_cp = cps_hold[0]
                    self.starting_state = self.translate_state()
                    self.execute_cp(self.executing_cp, state)
                else:
                    # Several CPs hold. We need a method to choose between them.
                    print("Choose an appropriate CP:")
                    chosen_cp = self.choose_cp_from_list(cps_hold)
                    msg = f"I will now follow the Collaboration Pattern {chosen_cp}."
                    self.send_message(Message(content=msg, from_id=self.agent_id, to_id=None))
                    self.executing_cp = chosen_cp
                    print(self.executing_cp)
                    self.execute_cp(self.executing_cp, state)
            else:
                # This means that there are no CPs that are applicable to the current situation
                #print("No applicable CPs, do as normal.")
                # Check if there are still actions in the action list
                if len(self.actionlist[0]) != 0:
                    # This means that an action is still being executed
                    action = self.actionlist[0].pop(0)
                    action_kwargs = self.actionlist[1].pop(0)
                    action_kwargs['action_duration'] = self.move_speed

                    #return action, action_kwargs  # Returned here, so code underneath is then not executed
                else:
                    # If not, choose a new action
                    self.basic_behavior()

        # Record some progress variables
        if action:
            if 'Move' in action:
                self.nr_move_actions = self.nr_move_actions + 1
            elif 'Idle' in action:
                # Intentional idle
                self.idle_ticks = self.idle_ticks + 1
            else:
                self.nr_productive_actions = self.nr_productive_actions + 1
                if 'GrabObject' in action:
                    self.robot_contribution = self.robot_contribution + 1
                elif 'GrabLargeObject' in action:
                    self.robot_contribution = self.robot_contribution + 4
        else:
            # Unintentional idle
            self.idle_ticks = self.idle_ticks + 1

        self.agent_properties['idle_time'] = self.idle_ticks

        return action, action_kwargs

# Functions that deal with the ontology stuff
    def store_cp_conditions(self, start_end):
        # The variable start_end should be either self.start_conditions or self.end_conditions, depending on which
        # you want to fill.
        # Store all conditions of existing CPs in a list; join duplicates, but store which CP they belong to
        self.cp_list = []
        start_end.clear()

        # Look at the list of CPs
        with TypeDB.core_client('localhost:1729') as client:
            with client.session(self.database_name, SessionType.DATA) as session:
                with session.transaction(TransactionType.READ) as read_transaction:
                    answer_iterator = read_transaction.query().match("match $cp isa collaboration_pattern, has name $name; get $name;")

                    for answer in answer_iterator:
                        cp_retrieved = answer.get('name')._value
                        if cp_retrieved not in self.cp_list:
                            self.cp_list.append(cp_retrieved)

                    # For each CP, look up all conditions
                    for cp in self.cp_list:
                        condition_list = []

                        # First, find all conditions related to the CP at hand
                        if start_end == self.start_conditions:
                            answer_iterator = read_transaction.query().match(
                                f'''match $cp isa collaboration_pattern, has name '{cp}'; 
                                $starts (condition: $condition, cp: $cp) isa starts_when; 
                                $condition isa condition, has condition_id $id; get $id;''')
                        elif start_end == self.end_conditions:
                            answer_iterator = read_transaction.query().match(
                                f'''match $cp isa collaboration_pattern, has name '{cp}'; 
                                $ends (condition: $condition, cp: $cp) isa ends_when; 
                                $condition isa condition, has condition_id $id; get $id;''')

                        # Save the conditions in a list
                        for answer in answer_iterator:
                            condition_retrieved = answer.get('id')._value
                            condition_list.append(condition_retrieved)

                        # For each condition, find the accompanying context
                        for condition in condition_list:
                            context_list = []
                            context_athand = None
                            answer_iterator = read_transaction.query().match(
                                f'''match $condition isa condition, has condition_id '{condition}'; 
                                $present (condition: $condition, situation: $context) isa is_present_in; 
                                $context isa context, has context_id $id; get $id;''')

                            for answer in answer_iterator:
                                context_retrieved = answer.get('id')._value
                                context_list.append(context_retrieved)

                            if len(context_list) > 1:
                                print("More than one context found...")
                                print(context_list)
                            else:
                                context_athand = context_list[0]

                            # Now that we have the context, search for all objects that are contained by this context
                            items_contained = {}
                            answer_iterator = read_transaction.query().match(
                                f'''match $context isa context, has context_id '{context_athand}'; 
                                $contains (whole: $context, part: $item) isa contains; $item has $attr; 
                                get $item, $attr;''')

                            for answer in answer_iterator:
                                # Store the entity concepts and it's attributes
                                item_retrieved = answer.get('item').as_entity().get_type().get_label().name()
                                attribute_value = answer.get('attr')._value
                                attribute_type = answer.get('attr').as_attribute().get_type().get_label().name()
                                if item_retrieved in items_contained.keys():
                                    items_contained[item_retrieved][attribute_type] = attribute_value
                                else:
                                    items_contained[item_retrieved] = {attribute_type: attribute_value}

                            # Store that as a single condition if it is not yet in the overall condition list
                            #print('ITEMS')
                            #print(items_contained)
                            #print('START END')
                            #print([val[0] for val in start_end])
                            conditions_np = np.array([val[0] for val in start_end])
                            if len(start_end) > 0 and items_contained in conditions_np:
                                index = np.where(conditions_np == items_contained)[0][0]
                                start_end[index][1].append(cp)
                            else:
                                start_end.append([items_contained, [cp]])
        return

    def check_cp_conditions(self, start_end):
        # Check all conditions of existing CPs and store which ones currently hold (how to do this efficiently??)
        conditions_hold = []
        cps_hold = []

        # For each condition, check if it holds
        for condition in start_end:
            object = None
            location = None
            object_type = None

            relevant_objects = None

            # Store the items in the condition
            if 'object' in condition[0]:
                object = condition[0]['object']
            elif 'resource' in condition[0]:
                object = condition[0]['resource']

            if 'location' in condition[0]:
                location = condition[0]['location']

            if object is not None:
                # Check what type of object we're dealing with, small, large or brown
                if 'brown' in object['color']:
                    object_type = 'brown'
                    relevant_objects = self.state[{"obstruction": True}]
                elif 'large' in object['size']:
                    object_type = 'large'
                    relevant_objects = self.state[{'large': True, 'is_movable': True}]
                elif 'small' in object['size']:
                    object_type = 'small'
                    relevant_objects = self.state[{'name': 'rock1'}]

            # Check where this type of object is located, and whether that is the same as the location in the condition
            if relevant_objects:
                # First check if location is None, if so, condition holds
                if location is None:
                    # Check if the objects that were found aren't outside of the field TODO
                    if condition not in conditions_hold:
                        conditions_hold.append(condition)
                    break
                else:
                    # It exists! Translate and check locations
                    if isinstance(relevant_objects, dict):
                        # There is just one such object, check it's location
                        if location['range'] in self.translate_location(relevant_objects['obj_id'], object_type):
                            # It is the same, condition holds!
                            if condition not in conditions_hold:
                                conditions_hold.append(condition)
                            break
                    elif isinstance(relevant_objects, list):
                        # It is a list, we'll need to loop through
                        for object in relevant_objects:
                            # Translate the location and check whether it is the one in the condition
                            if location['range'] in self.translate_location(object['obj_id'], object_type):
                                # It is the same, condition holds! We can break the for loop
                                if condition not in conditions_hold:
                                    conditions_hold.append(condition)
                                break
            #else:
                # There are no such objects, we can stop here
                #print("Condition doesn't hold")

        # Then check if there is any CP for which each start condition holds (or if the end condition of the current holds)

        # For each condition that holds
        for condition in conditions_hold:
            # Check to which CP this condition is tied
            bound_cps = condition[1]

            for cp in bound_cps:
                # Add a check, if the CP at hand is already in the CPs_hold list, we can skip
                if cp not in cps_hold:
                    # For each CP, check if there are other conditions for this CP
                    other_conditions = [i for i, x in enumerate(start_end) if cp in x[1]]

                    if len(other_conditions) > 1:
                        # This means there are other conditions. Check if all of them are in the conditions_hold list
                        all_conditions = True
                        for index in other_conditions:
                            if start_end[index][0] not in np.asarray([val[0] for val in conditions_hold]):
                                all_conditions = False
                        # If all of them are, CP is valid
                        if all_conditions:
                            cps_hold.append(cp)
                    else:
                        # This means that there are no other conditions. Therefore, this CP holds!
                        cps_hold.append(cp)

        return cps_hold

    def execute_cp(self, cp, state):
        # Retrieve the actions from the CP

        # Check if there are actions left in the action list for the cp
        if len(self.cp_actions) > 0:
            # Yes, there are actions already determined
            # Check if there are already determined currently to-be-executed actions by the human and/or robot
            if self.current_robot_action:
                # The robot is supposed to do something, continue executing
                # The robot needs to translate the current_robot_action to an actual action
                #print('We need to translate the current action to an actual action.')
                self.translate_action(self.current_robot_action, state)
            elif self.current_human_action:
                # If the robot is not doing anything, but the human is supposed to do something, check if they did it yet
                #print("Check if the human did their task")
                if len(self.past_human_actions) > 0:
                    if self.current_human_action['task']['task_name'] in np.array([val[0] for val in self.past_human_actions]):
                        # This means that the action we're looking for is in the past 5 actions of the human.
                        # Now we need to check if the location is also present
                        location_present = False
                        human_action_indices = np.where(np.array([val[0] for val in self.past_human_actions]) ==
                                                        self.current_human_action['task']['task_name'])[0]
                        for index in human_action_indices:
                            if self.current_human_action['location']['range'] in self.past_human_actions[index][1]:
                                location_present = True
                                break

                        if location_present:
                            # The human did the action, so we can remove it from the action list and continue
                            self.cp_actions.remove(self.current_human_action)
                            self.current_human_action = None
                            # Also empty the past human actions list as we're moving to a new cycle
                            self.past_human_actions = []

                            # If the CP actions list ends up empty here, we should do a reward update
                            if len(self.cp_actions) == 0 and self.executing_cp in self.cp_list:
                                self.reward_update_cps()
                # In the meantime, the robot should idle and wait for the human to finish their task
                #self.wait_action(None)
            else:
                print("Find the next actions")
                # If none of the agents have something to do, check for the next tasks
                order_values = []
                # Store and/or retrieve what position in the CP we're at (which action)
                for action in self.cp_actions:
                    order_values.append(int(action['task']['order_value']))

                current_action_indices = list(filter(lambda x: order_values[x] == min(order_values), range(len(order_values))))
                current_actions = list(map(self.cp_actions.__getitem__, current_action_indices))

                for action in current_actions:
                    if action['actor']['actor_type'] == 'robot':
                        # This is an action done by the robot. Store and execute
                        self.current_robot_action = action
                    elif action['actor']['actor_type'] == 'human':
                        # This is an action done by the human. Store such that it can be checked
                        self.current_human_action = action

        else:
            # This means there are no actions, so we need to retrieve them
            print("Retrieve actions...")
            # Start TypeDB session and retrieve information about the current CP
            with TypeDB.core_client('localhost:1729') as client:
                with client.session(self.database_name, SessionType.DATA) as session:
                    with session.transaction(TransactionType.READ) as read_transaction:
                        # First, find all tasks related to the CP at hand
                        answer_iterator = read_transaction.query().match(
                            f'''match $cp isa collaboration_pattern, has name '{cp}';
                            $part_of (cp: $cp, task: $task) isa is_part_of;
                            $task isa task, has task_id $id, has task_name $name, has order_value $value; get $id, $name, $value;''')

                        # Save the task data in the list
                        for answer in answer_iterator:
                            task_id_retrieved = answer.get('id')._value
                            task_name_retrieved = answer.get('name')._value
                            order_value_retrieved = answer.get('value')._value
                            # Check for the task name if there is an extra space at the end, remove if this is the case
                            # TODO this is a quick fix, find the real problem and fix there
                            if task_name_retrieved[-1] == ' ':
                                task_name_retrieved = task_name_retrieved[:-1]
                            self.cp_actions.append({'task': {'task_id': task_id_retrieved,
                                                             'task_name': task_name_retrieved,
                                                             'order_value': order_value_retrieved}})

                        # Find the location, actor and resource info
                        for task in self.cp_actions:
                            # Find location info
                            answer_iterator = read_transaction.query().match(
                                f'''match $task isa task, has task_id '{task['task']['task_id']}';
                                $takes_place (action: $task, location: $location) isa takes_place_at; 
                                $location has $attr; get $location, $attr;''')

                            for answer in answer_iterator:
                                # Store location info
                                item_retrieved = answer.get('location').as_entity().get_type().get_label().name()
                                attribute_value = answer.get('attr')._value
                                attribute_type = answer.get('attr').as_attribute().get_type().get_label().name()
                                if item_retrieved in task.keys():
                                    task[item_retrieved][attribute_type] = attribute_value
                                else:
                                    task[item_retrieved] = {attribute_type: attribute_value}

                            # Find actor info
                            answer_iterator = read_transaction.query().match(
                                f'''match $task isa task, has task_id '{task['task']['task_id']}';
                                $done_by (action: $task, actor: $actor) isa performed_by; 
                                $actor has $attr; get $actor, $attr;''')

                            for answer in answer_iterator:
                                # Store actor info
                                item_retrieved = answer.get('actor').as_entity().get_type().get_label().name()
                                attribute_value = answer.get('attr')._value
                                attribute_type = answer.get('attr').as_attribute().get_type().get_label().name()
                                if item_retrieved in task.keys():
                                    task[item_retrieved][attribute_type] = attribute_value
                                else:
                                    task[item_retrieved] = {attribute_type: attribute_value}

                            # Find resource info
                            answer_iterator = read_transaction.query().match(
                                f'''match $task isa task, has task_id '{task['task']['task_id']}';
                                $uses (action: $task, resource: $resource) isa uses; 
                                $resource has $attr; get $resource, $attr;''')

                            for answer in answer_iterator:
                                # Store resource info
                                item_retrieved = answer.get('resource').as_entity().get_type().get_label().name()
                                attribute_value = answer.get('attr')._value
                                attribute_type = answer.get('attr').as_attribute().get_type().get_label().name()
                                if item_retrieved in task.keys():
                                    task[item_retrieved][attribute_type] = attribute_value
                                else:
                                    task[item_retrieved] = {attribute_type: attribute_value}
                        print(self.cp_actions)
            # If we have done all this and still we have no action, we should break out of this CP
            if len(self.cp_actions) < 1:
                # This CP doesn't contain any actions, so we should break out of it and give a negative reward
                self.send_message(
                    Message(content=f"I will stop following the Collaboration Pattern {self.executing_cp}",
                            from_id=self.agent_id, to_id=None))
                if self.executing_cp in self.cp_list:
                    self.reward_update_cps()
                self.executing_cp = False

        return

    def translate_location(self, object_id, object_type):
        # This function checks in which location ranges an object is located

        object_location = self.state[object_id]['location']
        object_loc_x = object_location[0]
        object_loc_y = object_location[1]

        # List that contains all the high level locations an object is in (e.g. left side of pile and top of pile)
        locations = []

        # Identify how many nr of rows we should look at
        nr_rows = 1
        nr_vert_rows = 1
        if object_type == 'large' or object_type == 'brown':
            # Determine what kind of large rock it is/what the orientation is; that determines
            rock_name = object_id
            if 'vert' in rock_name:
                nr_rows = 1
                nr_vert_rows = 4
            elif 'long' in rock_name:
                nr_rows = 4
                nr_vert_rows = 1
            elif 'large' in rock_name:
                nr_rows = 2
                nr_vert_rows = 2

        # Top of rock pile (= no rocks on top of this object)
        top_check = True
        for x in range (0, nr_rows):
            for i in range(0, object_loc_y):
                loc_to_check = [object_loc_x + x, i]
                objects_found = self.state[{"location": loc_to_check, 'is_movable': True}]
                if objects_found is not None:
                    # An object was found, meaning that the area above the rock isn't empty TODO create exception for agents
                    top_check = False

        if top_check == True:
            locations.append('Top of rock pile')

        # Bottom of rock pile (= no rocks below this object)
        bottom_check = True
        for x in range (0, nr_rows):
            for i in range(object_loc_y + nr_vert_rows, 11):
                loc_to_check = [object_loc_x + x, i]
                objects_found = self.state[{"location": loc_to_check, 'is_movable': True}]
                if objects_found is not None:
                    # An object was found, meaning that the area below the rock isn't empty TODO create exception for agents
                    bottom_check = False

        if bottom_check == True:
            locations.append('Bottom of rock pile')

        # Do on top of check
        for x in range (0, nr_rows):
            loc_to_check = [object_loc_x + x, object_loc_y + nr_vert_rows]
            objects_found = self.state[{"location": loc_to_check, 'is_movable': True}]
            if objects_found is not None:
                # Now we can also do the 'On top of' check for objects
                # Check what objects are found
                if isinstance(objects_found, list):
                    for obj in objects_found:
                        if obj['name'] == 'rock1' or ('bound_to' in obj.keys() and obj['bound_to'] is None):
                            if 'On top of Small rock' in locations:
                                continue
                            else:
                                locations.append('On top of Small rock')
                        elif 'brown' in obj['name'] and 'bound_to' in obj.keys():
                            if 'On top of Brown rock' in locations:
                                continue
                            else:
                                locations.append('On top of Brown rock')
                        elif 'bound_to' in obj.keys() and obj['bound_to'] is not None:
                            if 'On top of Large rock' in locations:
                                continue
                            else:
                                locations.append('On top of Large rock')

                else:
                    if objects_found['name'] == 'rock1' or ('bound_to' in objects_found.keys() and objects_found['bound_to'] is None):
                        if 'On top of Small rock' in locations:
                            continue
                        else:
                            locations.append('On top of Small rock')
                    elif 'brown' in objects_found['name'] and 'bound_to' in objects_found.keys():
                        if 'On top of Brown rock' in locations:
                            continue
                        else:
                            locations.append('On top of Brown rock')
                    elif 'bound_to' in objects_found.keys() and objects_found['bound_to'] is not None:
                        if 'On top of Large rock' in locations:
                            continue
                        else:
                            locations.append('On top of Large rock')

        # Check if the object is on top of victim separately
        for x in range(0, nr_rows):
            for y in range(0, nr_vert_rows):
                loc_to_check = (object_loc_x + x, object_loc_y + y)
                if loc_to_check in [(8, 9), (8, 10), (9, 9), (9, 10), (10, 9), (10, 10), (11, 9), (11, 10)]:
                    locations.append('On top of Victim')
                    break

        # Left/Right side of rock pile (= within the bounds of the pile, left or right half)
        if object_loc_x >= 5 and object_loc_x <= 9:
            locations.append('Left side of rock pile')
        elif object_loc_x >= 10 and object_loc_x <= 15:
            locations.append('Right side of rock pile')
        # Left/Right side of field (= outside the bounds of the pile, left or right)
        elif object_loc_x < 5:
            locations = ['Left side of field']
        elif object_loc_x > 15:
            locations = ['Right side of field']

        # Above rock pile (not relevant for rocks, only for agents)
        #print(locations)
        return locations

    def translate_action(self, action, state):
        task_name = action['task']['task_name']
        task_location = None

        # Check if there is a location specified
        if 'location' in action.keys():
            task_location = action['location']['range']

        carried_objects = self.state[{'carried_by': self.agent_id}]
        print(carried_objects)

        if 'Pick up' in task_name:
            # This is a pick up action!
            # If hands are full, drop first to make space
            if len(state[self.agent_id]['is_carrying']) > 4:
                self.drop_action(state, None)
                return
            # Check if we're dealing with a large or a small rock
            object_size = action['resource']['size']
            if 'large' in object_size:
                # We have to pick up a large rock
                # Find all relevant objects first, according to size
                relevant_objects = self.state[{'large': True, 'is_movable': True}]
                if relevant_objects:
                    # If hands are too full, drop first to make space
                    if len(state[self.agent_id]['is_carrying']) > 0:
                        self.drop_action(state, None)
                        return
                    # It exists! Translate and check locations
                    if isinstance(relevant_objects, dict):
                        # There is just one such object, check it's location; if it is correct, pick up this one
                        if task_location is None:
                            # There is no task location specified, just pick up the found object
                            self.pickup_large_action([relevant_objects], state, None)
                        elif task_location in self.translate_location(relevant_objects['obj_id'], object_size):
                            self.pickup_large_action([relevant_objects], state, None)
                        else:
                            # There is no such object, can't perform this action
                            print("Can't perform this action, object doesn't exist.")
                    elif isinstance(relevant_objects, list):
                        objects_right_location= []
                        # It is a list, we'll need to loop through
                        for object in relevant_objects:
                            # Translate the location and check whether it is the one in the condition
                            if task_location is None:
                                # No location specified, which means that we can choose any object found
                                objects_right_location.append(object)
                            elif task_location in self.translate_location(object['obj_id'], object_size):
                                # It is the same, this is an object we can choose!
                                objects_right_location.append(object)
                        self.pickup_large_action(objects_right_location, state, None)
                else:
                    # There is no such object, can't perform this action
                    print("Can't perform this action, object doesn't exist.")
                return
            elif 'small' in object_size:
                # We have to pick up a small rock
                # Find all relevant objects first, according to size
                relevant_objects = self.state[{'name': 'rock1'}]
                if relevant_objects:
                    # It exists! Translate and check locations
                    if isinstance(relevant_objects, dict):
                        # There is just one such object, check it's location; if it is correct, pick up this one
                        if task_location is None:
                            # There is no task location specified, just pick up the found object
                            self.pickup_large_action([relevant_objects], state, None)
                        elif task_location in self.translate_location(relevant_objects['obj_id'], object_size):
                            self.pickup_action([relevant_objects], state)
                        else:
                            # There is no such object, can't perform this action
                            print("Can't perform this action, object doesn't exist.")
                    elif isinstance(relevant_objects, list):
                        objects_right_location= []
                        # It is a list, we'll need to loop through
                        for object in relevant_objects:
                            # Translate the location and check whether it is the one in the condition
                            if task_location is None:
                                # No location specified, which means that we can choose any object found
                                objects_right_location.append(object)
                            elif task_location in self.translate_location(object['obj_id'], object_size):
                                # It is the same, this is an object we can choose!
                                objects_right_location.append(object)
                        self.pickup_action(objects_right_location, state)
                else:
                    # There is no such object, can't perform this action
                    print("Can't perform this action, object doesn't exist.")
                return
        elif 'Stand still' in task_name:
            # We should move to the location specified and stand still there
            self.wait_action(self.translate_loc_backwards(task_location))
            return
        elif 'Drop' in task_name:
            # We have to drop a rock
            self.drop_action(state, self.translate_loc_backwards(task_location))
            return
        elif 'Break' in task_name:
            # We have to break a rock
            # Find all relevant objects first, according to size
            relevant_objects = self.state[{'large': True, 'is_movable': True}] #+ self.state[{'bound_to'}]
            if relevant_objects:
                # It exists! Translate and check locations
                if isinstance(relevant_objects, dict):
                    # There is just one such object, check its location; if it is correct, pick up this one
                    if task_location in self.translate_location(relevant_objects['obj_id'], 'large'):
                        self.break_action([relevant_objects], state, None)
                    else:
                        # There is no such object, can't perform this action
                        print("Can't perform this action, object doesn't exist.")
                elif isinstance(relevant_objects, list):
                    objects_right_location = []
                    # It is a list, we'll need to loop through
                    for object in relevant_objects:
                        # Translate the location and check whether it is the one in the condition
                        if task_location in self.translate_location(object['obj_id'], 'large'):
                            # It is the same, this is an object we can choose!
                            objects_right_location.append(object)
                    self.break_action(objects_right_location, state, None)
            else:
                # There is no such object, can't perform this action
                print("Can't perform this action, object doesn't exist.")
            return
        elif 'Move back and forth' in task_name:
            # Add the move back and forth action to the actionlist
            self.move_back_forth_action(self.translate_loc_backwards(task_location))
            return

    def translate_state(self):
        # A nested dictionary to store all locations with the objects that they currently entail
        # The nestedness ensures that we can use the state rep at different abstraction levels for later tweaking
        # E.g. fully, with the amount of each object, or simply with the object types present
        # Data format: {'location1': {'large rock': [obj1, ...], 'small rock': [obj1, ...]}, ...}
        '''obj_loc_dict = {}

        brown_rocks = self.state[{"obstruction": True}]
        large_rocks = self.state[{'large': True, 'is_movable': True}]
        small_rocks = self.state[{'name': 'rock1'}]

        if isinstance(brown_rocks, dict):
            brown_rocks = [brown_rocks]
        if isinstance(large_rocks, dict):
            large_rocks = [large_rocks]
        if isinstance(large_rocks, dict):
            small_rocks = [small_rocks]

        # For all rock objects, check at what locations they are
        if brown_rocks:
            for rock in brown_rocks:
                locations = self.translate_location(rock['obj_id'], 'brown')
                # Check if that location is already in the dict. If yes, add under the right object type
                for location in locations:
                    if location in obj_loc_dict:
                        obj_loc_dict[location]['brown rock'].append(rock)
                    # If no, add the location to the dict first, then add object under the right object type
                    else:
                        obj_loc_dict[location] = {'small rock': [], 'large rock': [], 'brown rock': []}
                        obj_loc_dict[location]['brown rock'].append(rock)

        # For all rock objects, check at what location they are
        if large_rocks:
            for rock in large_rocks:
                locations = self.translate_location(rock['obj_id'], 'large')
                # Check if that location is already in the dict. If yes, add under the right object type
                for location in locations:
                    if location in obj_loc_dict:
                        obj_loc_dict[location]['large rock'].append(rock)
                    # If no, add the location to the dict first, then add object under the right object type
                    else:
                        obj_loc_dict[location] = {'small rock': [], 'large rock': [], 'brown rock': []}
                        obj_loc_dict[location]['large rock'].append(rock)

        # For all rock objects, check at what location they are
        if small_rocks:
            for rock in small_rocks:
                locations = self.translate_location(rock['obj_id'], 'small')
                # Check if that location is already in the dict. If yes, add under the right object type
                for location in locations:
                    if location in obj_loc_dict:
                        obj_loc_dict[location]['small rock'].append(rock)
                    # If no, add the location to the dict first, then add object under the right object type
                    else:
                        obj_loc_dict[location] = {'small rock': [], 'large rock': [], 'brown rock': []}
                        obj_loc_dict[location]['small rock'].append(rock)

        # Translation for simpler state
        state_array = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]] # State with no rocks

        if 'Top of rock pile' in obj_loc_dict and len(obj_loc_dict['Top of rock pile']['small rock']) > 0:
            state_array[0][0] = 1
        if 'Top of rock pile' in obj_loc_dict and len(obj_loc_dict['Top of rock pile']['large rock']) > 0:
            state_array[0][1] = 1
        if 'Top of rock pile' in obj_loc_dict and len(obj_loc_dict['Top of rock pile']['brown rock']) > 0:
            state_array[0][2] = 1
        if 'Bottom of rock pile' in obj_loc_dict and len(obj_loc_dict['Bottom of rock pile']['small rock']) > 0:
            state_array[1][0] = 1
        if 'Bottom of rock pile' in obj_loc_dict and len(obj_loc_dict['Bottom of rock pile']['large rock']) > 0:
            state_array[1][1] = 1
        if 'Bottom of rock pile' in obj_loc_dict and len(obj_loc_dict['Bottom of rock pile']['brown rock']) > 0:
            state_array[1][2] = 1
        if 'Left side of rock pile' in obj_loc_dict and len(obj_loc_dict['Left side of rock pile']['small rock']) > 0:
            state_array[2][0] = 1
        if 'Left side of rock pile' in obj_loc_dict and len(obj_loc_dict['Left side of rock pile']['large rock']) > 0:
            state_array[2][1] = 1
        if 'Left side of rock pile' in obj_loc_dict and len(obj_loc_dict['Left side of rock pile']['brown rock']) > 0:
            state_array[2][2] = 1
        if 'Right side of rock pile' in obj_loc_dict and len(obj_loc_dict['Right side of rock pile']['small rock']) > 0:
            state_array[3][0] = 1
        if 'Right side of rock pile' in obj_loc_dict and len(obj_loc_dict['Right side of rock pile']['large rock']) > 0:
            state_array[3][1] = 1
        if 'Right side of rock pile' in obj_loc_dict and len(obj_loc_dict['Right side of rock pile']['brown rock']) > 0:
            state_array[3][2] = 1
        if 'Left side of field' in obj_loc_dict and len(obj_loc_dict['Left side of field']['small rock']) > 0:
            state_array[4][0] = 1
        if 'Left side of field' in obj_loc_dict and len(obj_loc_dict['Left side of field']['large rock']) > 0:
            state_array[4][1] = 1
        if 'Left side of field' in obj_loc_dict and len(obj_loc_dict['Left side of field']['brown rock']) > 0:
            state_array[4][2] = 1
        if 'Right side of field' in obj_loc_dict and len(obj_loc_dict['Right side of field']['small rock']) > 0:
            state_array[5][0] = 1
        if 'Right side of field' in obj_loc_dict and len(obj_loc_dict['Right side of field']['large rock']) > 0:
            state_array[5][1] = 1
        if 'Right side of field' in obj_loc_dict and len(obj_loc_dict['Right side of field']['brown rock']) > 0:
            state_array[5][2] = 1'''

        # Strongly simplified state (total of 24 states)
        # Position 1: distance to goal state (4, 3, 2, 1)
        # Position 2: contribution team members (robot, human, equal)
        # Position 3: human stand still (yes, no)
        state_array = [0, 0, 0]

        # Calculate position 1: distance to goal state (4, 3, 2, 1)
        distance_metric = self.first_tick_distance/4
        current_distance = self.distance_goal_state()
        if current_distance > distance_metric*3:
            state_array[0] = 4
        elif current_distance > distance_metric*2:
            state_array[0] = 3
        elif current_distance > distance_metric:
            state_array[0] = 2
        else:
            state_array[0] = 1

        # Calculate position 2: contribution team members (robot, human, equal)
        progress = self.first_tick_distance - current_distance
        if (progress / 2)+1 >= self.robot_contribution >= (progress / 2)-1 or progress < 3:
            state_array[1] = 'equal'
        elif self.robot_contribution < (progress/2)-1:
            state_array[1] = 'human'
        elif self.robot_contribution > (progress/2)+1:
            state_array[1] = 'robot'

        # Calculate position 3: human stand still (yes, no)
        if self.human_standing_still:
            state_array[2] = True
        else:
            state_array[2] = False

        return state_array

    def translate_loc_backwards(self, location):
        coordinates = ()

        if location == 'Top of rock pile':
            coordinates = ()
            poss_coordinates = []
            for x in list(range(5, 14)):
                for y in list(range(0, 10)):
                    if self.state[{"location": (x, y)}] is not None:
                        poss_coordinates.append((x, y))
                        break
            coordinates = random.choice(poss_coordinates)
        elif location == 'Above rock pile':
            coordinates = ()
            poss_coordinates = []
            for x in list(range(5, 14)):
                for y in list(range(0, 10)):
                    if self.state[{"location": (x, y)}] is not None:
                        break
                    else:
                        poss_coordinates.append((x, y))
            coordinates = random.choice(poss_coordinates)
        elif location == 'Bottom of rock pile':
            poss_coordinates = [(5, 10), (6, 10), (7, 10), (8, 10), (9, 10), (10, 10), (11, 10), (12, 10), (13, 10), (14, 10)]
            for loc in poss_coordinates:
                if self.state[{"location": loc}] is None:
                    poss_coordinates.remove(loc)
            coordinates = random.choice(poss_coordinates)
        elif location == 'Left side of rock pile':
            coordinates = ()
            poss_coordinates = []
            poss_xloc = list(range(5, 9))
            poss_yloc = list(range(0, 10))
            for x in poss_xloc:
                for y in poss_yloc:
                    if self.state[{"location": (x, y)}] is not None:
                        poss_coordinates.append((x, y))
            coordinates = random.choice(poss_coordinates)
        elif location == 'Right side of rock pile':
            coordinates = ()
            poss_coordinates = []
            poss_xloc = list(range(5, 9))
            poss_yloc = list(range(0, 10))
            for x in poss_xloc:
                for y in poss_yloc:
                    if self.state[{"location": (x, y)}] is not None:
                        poss_coordinates.append((x, y))
            coordinates = random.choice(poss_coordinates)
        elif location == 'Left side of field':
            poss_xloc = list(range(0, 4))
            poss_yloc = list(range(0, 10))
            coordinates = (random.choice(poss_xloc), random.choice(poss_yloc))
        elif location == 'Right side of field':
            poss_xloc = list(range(15, 20))
            poss_yloc = list(range(0, 9))
            coordinates = (random.choice(poss_xloc), random.choice(poss_yloc))
        elif 'On top of' in location:
            # First check whether it is victim, large rock, small rock or brown rock
            if 'Victim' in location:
                poss_locations = [(8,9), (8,10), (9,9), (9,10), (10,9), (10,10), (11,9), (11,10)]
                coordinates = random.choice(poss_locations)
            elif 'Large' in location:
                poss_locations = []
                large_objs = self.state[{'large': True, 'is_movable': True}]
                if large_objs is not None:
                    # There are large objects to be on top of
                    for obj in large_objs:
                        obj_location = obj['location']
                        y_loc_above = obj_location[1] - 1
                        if self.state[{"location": (obj_location[0], y_loc_above)}] is None:
                            poss_locations.append((obj_location[0], y_loc_above))
                    coordinates = random.choice(poss_locations)
            elif 'Small' in location:
                poss_locations = []
                small_objs = self.state[{'name': 'rock1'}] + self.state[{'bound_to': None}]
                if small_objs is not None:
                    # There are small objects to be on top of
                    for obj in small_objs:
                        obj_location = obj['location']
                        y_loc_above = obj_location[1] - 1
                        if self.state[{"location": (obj_location[0], y_loc_above)}] is None:
                            poss_locations.append((obj_location[0], y_loc_above))
                    coordinates = random.choice(poss_locations)
            elif 'Brown' in location:
                poss_locations = []
                brown_objs = self.state[{"obstruction": True}]
                if brown_objs is not None:
                    # There are brown objects to be on top of
                    for obj in brown_objs:
                        obj_location = obj['location']
                        y_loc_above = obj_location[1] - 1
                        if self.state[{"location": (obj_location[0], y_loc_above)}] is None:
                            poss_locations.append((obj_location[0], y_loc_above))
                    coordinates = random.choice(poss_locations)

        # If the coordinates are empty here, fill them with the current location of the agent
        if len(coordinates) < 1:
            coordinates = self.agent_id['location']

        return coordinates

# Functions that deal with learning
    def choose_cp_from_list(self, cp_list):
        # Check if this state has been visited before
        current_state = self.translate_state()
        # Store current state as the starting state for this decision
        self.starting_state = current_state
        self.starting_state_distance = self.distance_goal_state()
        if str(current_state) in self.q_table_cps.index:
            # If state was visited before, check how often it was visited TODO
            # Choose CP based on expected reward (with exploration rate based on uncertainty?), limit to applicable CPs
            q_values = self.q_table_cps.loc[str(current_state)]/self.q_table_cps_runs.loc[str(current_state)] + \
                       np.sqrt(2 * np.log(self.nr_chosen_cps + 1) / (self.q_table_cps_runs + 1e-6))
            q_values = q_values.loc[str(current_state)]
            q_values = q_values.fillna(0)
            chosen_cp = q_values.idxmax()
        else:
            # If state was not visited before, find the nearest state that was visited for which these CPs also hold
            nearest_state = self.nearest_visited_state()
            if nearest_state:
                # Choose CP based on expected reward in that state (this is like an educated guess based on similarity)
                q_values = self.q_table_cps.loc[str(nearest_state)] / \
                           self.q_table_cps_runs.loc[str(nearest_state)] + \
                           np.sqrt(2 * np.log(self.nr_chosen_cps + 1) / (self.q_table_cps_runs + 1e-6))
                chosen_cp = q_values.idxmax(axis=1)
            else:
                # If no nearest state, choose randomly and initialize Q-values for this state
                self.q_table_cps.loc[len(self.q_table_cps.index)] = 0
                self.q_table_cps.rename(index={len(self.q_table_cps.index) - 1: str(current_state)}, inplace=True)
                self.q_table_cps_runs.loc[len(self.q_table_cps_runs.index)] = 0
                self.q_table_cps_runs.rename(index={len(self.q_table_cps_runs.index) - 1: str(current_state)}, inplace=True)
                self.visited_states.append(current_state)
                chosen_cp = random.choice(cp_list)
                print(self.q_table_cps)

        # Update the total number of times a CP was chosen
        self.nr_chosen_cps = self.nr_chosen_cps + 1
        return chosen_cp

    def basic_behavior(self):
        actions = ['Move back and forth', 'Stand Still', 'Pick up', 'Drop', 'Break']

        action_to_exclude = []
        large_objs_field = []
        for object in self.state[{'large': True, 'is_movable': True}]:
            if object['location'][0] >= 15 or object['location'][0] < 5:
                continue
            else:
                large_objs_field.append(object)

        if len(self.state[self.agent_id]['is_carrying']) >= 5:
            # Hands are full, now it shouldn't be possible to pick up
            action_to_exclude.append('Pick up')
        elif self.state[{'name': 'rock1'}] is None and self.state[{'bound_to': None}] is None:
            # There are no objects to pickup
            action_to_exclude.append('Pick up')
        if len(self.state[self.agent_id]['is_carrying']) == 0:
            # Hands are empty, now it shouldn't be possible to drop
            action_to_exclude.append('Drop')
        if self.state[{'large': True, 'is_movable': True}] is None or len(large_objs_field) < 1:
            # No large objects available, so we cannot break
            action_to_exclude.append('Break')

        # Check if this state has been visited before
        current_state = self.translate_state()
        # Store current state as the starting state for this decision
        self.starting_state = current_state
        self.starting_state_distance = self.distance_goal_state()
        if str(current_state) in self.q_table_basic.index:
            # If state was visited before, check how often it was visited TODO
            # Choose action based on expected reward (with exploration rate based on uncertainty?)
            q_values = self.q_table_basic.loc[str(current_state)]
            if len(action_to_exclude) > 0:
                for action in action_to_exclude:
                    q_values = q_values.drop(action)
            print(q_values)
            chosen_action = q_values.idxmax()
        else:
            # If state was not visited before, find the nearest state that was visited
            nearest_state = self.nearest_visited_state()
            if str(nearest_state) in self.q_table_basic.index:
                # Choose action based on expected reward in that state (this is like an educated guess based on similarity)
                q_values = self.q_table_basic.loc[str(nearest_state)]
                if action_to_exclude is not None:
                    q_values = q_values.drop(action_to_exclude)
                chosen_action = q_values.idxmax()

                # Also initialize q-values for this state and add state to visited states
                self.q_table_basic.loc[len(self.q_table_basic.index)] = 0
                self.q_table_basic.rename(index={len(self.q_table_basic.index) - 1: str(current_state)}, inplace=True)
                self.visited_states.append(current_state)
            else:
                # If no nearest state, choose randomly and initialize Q-values for this state
                self.q_table_basic.loc[len(self.q_table_basic.index)] = 0
                self.q_table_basic.rename(index={len(self.q_table_basic.index) - 1:str(current_state)}, inplace=True)
                self.visited_states.append(current_state)
                chosen_action = random.choice(actions)
                print(self.q_table_basic)

        print(chosen_action)

        # Now we have chosen an action, translate and move to executing that action
        if chosen_action == "Stand Still":
            self.wait_action(None)
        elif chosen_action == "Pick up":
            # If the human is standing still, pick up a large rock around them (if available)
            if self.human_standstill():
                if self.state[{'large': True}] is not None and len(self.state[self.agent_id]['is_carrying']) == 0 and len(large_objs_field) > 0:
                    self.pickup_large_action(self.state[{'large': True}], self.state, self.human_location[0])
                else:
                    self.pickup_action(self.state[{'name': 'rock1'}] + self.state[{'bound_to': None}], self.state)
            else:
                self.pickup_action(self.state[{'name': 'rock1'}] + self.state[{'bound_to': None}], self.state)
        elif chosen_action == "Drop":
            self.drop_action(self.state, None)
        elif chosen_action == "Break":
            break_objects = self.state[{'large': True, 'is_movable': True}]
            if isinstance(break_objects, dict):
                break_objects = [break_objects]
            try:
                break_objects = break_objects + self.state[{'img_name': "/images/transparent.png"}]
                if self.human_standstill():
                    self.break_action(break_objects, self.state, self.human_location[0])
                else:
                    self.break_action(break_objects, self.state, None)
            except:
                print("No objects left to break")
        elif chosen_action == "Move back and forth":
            self.move_back_forth_action(None)

        self.executing_action = chosen_action

        return

    def reward_update_cps(self):
        # Do the reward updating for the CP that we just executed
        # Reward based on three factors:
        # 1. Decrease in distance to goal state (compute when starting, compute when finishing)
        # 2. Discounted by (combined) idle time
        # 3. Discounted by victim harm

        # Decrease in distance to goal state
        distance_decrease = self.starting_state_distance - self.distance_goal_state()

        # Idle time of the agent is idle ticks minus move and productive ticks
        idle_time = self.idle_ticks - self.nr_move_actions - self.nr_productive_actions

        # Number of times victim was harmed multiplied by a severance factor
        victim_harm = self.victim_harm * 5

        total_reward = distance_decrease - victim_harm - idle_time
        print("Starting state")
        print(self.starting_state)
        # If the state is already stored in the q-table, the reward is added
        if str(self.starting_state) in self.q_table_cps.index:
            # Update reward: rewards are stored cumulatively
            self.q_table_cps.at[str(self.starting_state), self.executing_cp] = \
                self.q_table_cps.at[str(self.starting_state), self.executing_cp] + total_reward

            # Also update how many times this CP was chosen in this state by 1
            self.q_table_cps_runs.at[str(self.starting_state), self.executing_cp] = \
                self.q_table_cps_runs.at[str(self.starting_state), self.executing_cp] + 1
        # If the state is not yet stored, create the initial value
        else:
            # Update reward: rewards are stored cumulatively
            self.q_table_cps.at[str(self.starting_state), self.executing_cp] = 0
            self.q_table_cps.at[str(self.starting_state), self.executing_cp] = total_reward
            # Also update how many times this CP was chosen in this state by 1
            self.q_table_cps_runs.at[str(self.starting_state), self.executing_cp] = 0
            self.q_table_cps_runs.at[str(self.starting_state), self.executing_cp] = 1

        print(self.q_table_cps)
        with open('qtable_cps_backup.pkl', 'wb') as f:
            pickle.dump(self.q_table_cps, f, pickle.HIGHEST_PROTOCOL)

        with open('qtable_cps_runs_backup.pkl', 'wb') as f:
            pickle.dump(self.q_table_cps_runs, f, pickle.HIGHEST_PROTOCOL)

        self.agent_properties["q_table_cps"] = self.q_table_cps.to_string()
        self.agent_properties["q_table_cps_runs"] = self.q_table_cps_runs.to_string()

        return

    def reward_update_basic(self):
        # Do the reward updating for the action that we just executed
        # Reward based on two factors:
        # 1. Did we actually decrease the distance to the goal state?
        # 2. Discounted by victim
        # 3. Discounted by the time it took to execute the action

        basic_reward = 0
        current_state = self.translate_state()
        if self.starting_state_distance > self.distance_goal_state():
            # If the state is not the same, we have a state transition, which means we get a positive reward
            basic_reward = 5
        else:
            basic_reward = -1

        victim_harm = self.victim_harm * 5
        total_reward = basic_reward - victim_harm - self.nr_ticks

        # Determine Max Q (expected utility of next state-action pair). If value doesn't exist, default to 0
        try:
            q_values = self.q_table_basic.loc[str(current_state)].astype('int')
            expected_action = q_values.idxmax()
            max_q = self.q_table_basic.at[current_state, expected_action]
        except:
            max_q = 0

        if str(self.starting_state) in self.q_table_basic.index:
            # If the starting state is already in the q-table, update q-value
            self.q_table_basic.at[str(self.starting_state), self.executing_action] = \
                self.q_table_basic.at[str(self.starting_state), self.executing_action] + self.alpha * \
                (total_reward + self.gamma * max_q -
                 self.q_table_basic.at[str(self.starting_state), self.executing_action])
        else:
            # If the starting state is not yet in the q-table, initialize and update q-value
            self.q_table_basic.at[str(self.starting_state), self.executing_action] = 0
            self.q_table_basic.at[str(self.starting_state), self.executing_action] = \
                self.q_table_basic.at[str(self.starting_state), self.executing_action] + self.alpha * \
                (total_reward + self.gamma * max_q -
                 self.q_table_basic.at[str(self.starting_state), self.executing_action])

        #print(self.q_table_basic)
        print(total_reward)
        with open('qtable_basic_backup.pkl', 'wb') as f:
            pickle.dump(self.q_table_basic, f, pickle.HIGHEST_PROTOCOL)

        self.agent_properties["q_table_basic"] = self.q_table_basic.to_string()

        return

    def nearest_visited_state(self):
        # Given the current state, find the nearest visited state
        # If talking about a CP choice, find the nearest visited state in which the current CPs also hold
        # (just remove all states in which the conditions do not hold and calculate distance after)

        nearest_state = None
        similarities = []

        current_state = self.translate_state()
        flattened_state = np.array(current_state).flatten()

        for state in self.visited_states:
            similarity = np.sum(flattened_state == np.array(state).flatten())
            similarities.append(similarity)

        if len(similarities) > 0:
            nearest_state = self.visited_states[similarities.index(max(similarities))]

        return nearest_state

    def record_progress(self, reset):
        # Function to record changes in the environment that indicate progress

        all_rocks = self.state[{'is_movable': True}]
        object_ids = []
        if isinstance(all_rocks, list):
            for rock in all_rocks:
                object_ids.append(rock['obj_id'])

        if reset:
            # Reset all variables
            self.nr_ticks = 0
            self.nr_move_actions = 0
            self.nr_productive_actions = 0
            self.victim_harm = 0
            self.idle_ticks = 0
        else:
            # Increment relevant variables
            self.nr_ticks = self.nr_ticks + 1
            self.victim_harm = self.victim_harm + self.victim_crash(object_ids)

        return

# Helper functions
    def message_handling(self):

        for message in self.received_messages:

            # Old message handling code, TODO will need to be adapted
            if isinstance(message, dict) and 'cp_new' in message:
                # A new CP was created!
                cp_name = message['cp_new']
                # Double check if it was already in the list
                if not cp_name in self.cp_list:
                    # If not, retrieve info about this CP and add to all the lists
                    self.store_cp_conditions(self.start_conditions)
                    self.store_cp_conditions(self.end_conditions)
                    # Also add to the q-tables
                    if cp_name not in self.q_table_cps.columns:
                        self.q_table_cps[cp_name] = 0
                        self.q_table_cps_runs[cp_name] = 0
            elif isinstance(message, dict) and 'cp_delete' in message:
                # An existing CP was deleted!
                cp_name = message['cp_delete']
                # Delete the name from the CP list, and the accompanying conditions from the conditions lists
                self.cp_list.remove(cp_name)
                self.store_cp_conditions(self.start_conditions)
                self.store_cp_conditions(self.end_conditions)
                print("New start conditions:")
                print(self.start_conditions)
                print("New end conditions:")
                print(self.end_conditions)
                # Also delete from q-tables
                if cp_name not in self.cp_list:
                    self.q_table_cps.drop(cp_name, axis=1, inplace=True)
                    self.q_table_cps_runs.drop(cp_name, axis=1, inplace=True)
            elif isinstance(message, dict) and 'cp_edit' in message:
                # An existing CP was edited, double check that it was already in the list
                cp_name = message['cp_edit']
                if cp_name in self.cp_list:
                    self.store_cp_conditions(self.start_conditions)
                    self.store_cp_conditions(self.end_conditions)
            elif message == 'FAIL' and not self.final_update:
                print("FINAL Q UPDATE")
                last_message = float(self.received_messages[-2])
                done_action = self.action_history[-1][1]
                done_state = self.action_history[-1][0]
                print(self.q_table)
                self.update_q_table(done_state, done_action, done_action, done_state, last_message)
                print(self.q_table)
                self.final_update = True
            elif not self.final_update:
                try:
                    last_message = float(message)
                except:
                    # Make sure we store only 5 past human actions max
                    if len(self.past_human_actions) > 4:
                        self.past_human_actions.pop(0)
                    self.past_human_actions.append(message)

            # After dealing with each message, remove it
            self.received_messages.remove(message)

        return

    def victim_crash(self, object_ids):
        hits = 0
        object_locs = []
        for object_id in object_ids:
            loc = self.state[object_id]['location']
            object_locs.append(loc)
            # Check if the object existed in the field before
            if object_id in self.previous_objs:
                # Check if the object changed location
                if loc is not self.previous_locs[self.previous_objs.index(object_id)]:
                    # Check if the new location is part of the victim
                    victim_locs = [(8,9), (8,10), (9,9), (9,10), (10,9), (10,10), (11,9), (11,10)]
                    if loc in victim_locs:
                        hits = hits + 1

        self.previous_objs = object_ids
        self.previous_locs = object_locs
        return hits

    def distance_goal_state(self):
        # Calculating a distance metric to the goal state, purely based on the amount of grid locations that still need
        # to be emptied before the task is done.
        distance_type = "all"

        distance = 0

        distance_1 = 0

        distance_2 = 0

        goal_state_base = [(8, 9), (8, 10), (9, 9), (9, 10), (10, 9), (10, 10), (11, 9), (11, 10)]

        goal_state_1 = [(5, 7), (5, 8), (5, 9), (5, 10), (6, 7), (6, 8), (6, 9), (6, 10), (7, 7), (7, 8), (7, 9), (7, 10)]

        goal_state_2 = [(12, 7), (12, 8), (12, 9), (12, 10), (13, 7), (13, 8), (13, 9), (13, 10), (14, 7), (14, 8), (14, 9), (14, 10)]

        if distance_type == "all":
            for loc_to_check in self.field_locations:
                objects_found = self.state[{"location": loc_to_check}]
                if objects_found is not None:
                    distance = distance + 1
        else:
            for loc_to_check in goal_state_base:
                objects_found = self.state[{"location": loc_to_check}]
                if objects_found is not None:
                    distance = distance + 1

            for loc_to_check in goal_state_1:
                objects_found = self.state[{"location": loc_to_check}]
                if objects_found is not None:
                    distance_1 = distance_1 + 1

            for loc_to_check in goal_state_2:
                objects_found = self.state[{"location": loc_to_check}]
                if objects_found is not None:
                    distance_2 = distance_2 + 1

            if distance_1 < distance_2:
                distance = distance + distance_1
            else:
                distance = distance + distance_2

        return distance

    def communicate_actions(self):
        # Function that deals with communication about the action done
        current_action = None
        current_location = None
        msg = None
        obj_tograb = None

        # First, figure out what the actual action is
        if self.executing_action:
            # We are executing a basic behavior action
            current_action = self.executing_action
            msg = f"Now executing {current_action}"
        elif self.current_robot_action:
            current_action = self.current_robot_action
            # Check what the actual MATRX action is
            actual_action = self.actionlist[0][0]
            # If the MATRX action is not part of the CP, don't communicate (this only happens for drop when pick up)
            if current_action['task']['task_name'] == 'Pick up' and 'Drop' in actual_action:
                print("Don't communicate")
            else:
                if current_action['resource'] and 'location' in current_action.keys():
                    obj_tograb = current_action['resource']['size']
                    msg = f"Now executing {current_action['task']['task_name']} a {obj_tograb} rock at {current_action['location']['range']}"
                elif 'location' in current_action.keys():
                    msg = f"Now executing {current_action['task']['task_name']} at {current_action['location']['range']}"
                elif current_action['resource']:
                    obj_tograb = current_action['resource']['size']
                    msg = f"Now executing {current_action['task']['task_name']} a {obj_tograb} rock"
                else:
                    msg = f"Now executing {current_action['task']['task_name']}"
        else:
            print("No action??")

        # Communicate if it is a pick up, break or drop action
        self.send_message(Message(content= msg, from_id=self.agent_id, to_id=None))

        return

    def human_standstill(self):
        standstill = False
        # Retrieve location of the human
        human = self.state[{'class_inheritance': "CustomHumanAgentBrain"}]
        # Check if the human is standing still
        # First, check if a previous location is stored
        if self.human_location:
            # Check if the current location is the same as the previous
            if self.human_location[0] == human['location']:
                self.human_location[1] = self.human_location[1] + 1
            else:
                self.human_location[0] = human['location']
                self.human_location[1] = 0
        else:
            self.human_location.append(human['location'])
            self.human_location.append(0)

        # If the human lingers
        if self.human_location[1] >= 3:
            standstill = True

        return standstill

