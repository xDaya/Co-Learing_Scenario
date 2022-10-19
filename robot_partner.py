from matrx.agents.agent_types.human_agent import *
from custom_actions import *
from matrx.messages.message import Message
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
import csv
import pickle


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
        self.run_number = 0

        #with open('qtable_backup.pkl', 'rb') as f:
        #    self.q_table = pickle.load(f)

        # Ontology related variables
        self.cp_list = []
        self.start_conditions = []
        self.executing_cp = False

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

        # Initialize existing CP's and their conditions
        print("Robot Initializing CPs")
        self.store_cp_conditions()

        # Start with some wait actions
        self.wait_action()
        self.wait_action()
        self.wait_action()
        self.wait_action()

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
            if "large" in state[object_id]:
                continue
            if "bound_to" in state[object_id] and state[object_id]['bound_to'] is not None:
                continue
            if state[object_id]['location'][0] >= 15 or state[object_id]['location'][0] < 5:
                continue

            y_loc_list.append(state[object_id]['location'][1])
            small_obj.append(object_id)

        chosen_object = small_obj[y_loc_list.index(min(y_loc_list))]    # Pick object with smallest y
        object_loc = state[chosen_object]['location']

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
        for object_id in object_ids:
            if "obstruction" in state[object_id]:
                continue
            if state[object_id]['location'][0] >= 15 or state[object_id]['location'][0] < 5:
                continue
            if "large" in state[object_id]:
                y_loc_list.append(state[object_id]['location'][1])
                large_obj.append(object_id)
                if location is not None:
                    dist = int(np.ceil(np.linalg.norm(np.array(state[object_id]['location'])
                                                  - np.array(location))))
                    dist_list.append(dist)
            if "bound_to" in state[object_id]:
                parts_list.append(object_id)

        if not y_loc_list:
            return

        if location is not None:
            chosen_part = large_obj[dist_list.index(min(dist_list))]
        else:
            chosen_part = large_obj[y_loc_list.index(min(y_loc_list))]
        large_name = state[chosen_part]['name']
        object_loc = state[chosen_part]['location']
        large_obj = [chosen_part]
        for part in parts_list:
            if state[part]['bound_to'] == large_name:
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
        self.actionlist[0].append(GrabLargeObject.__name__)
        self.actionlist[1].append(pickup_kwargs)
        return

    def drop_action(self, state, location):
        drop_action = None
        obj_type = None
        chosen_loc = location
        # Check if the agent is actually carrying something
        if state[self.agent_id]['is_carrying']:
            carrying_obj = state[self.agent_id]['is_carrying'][0]
            if "large" in carrying_obj:
                drop_action = DropLargeObject.__name__
            elif "bound_to" in carrying_obj and carrying_obj['bound_to'] is not None:
                drop_action = DropLargeObject.__name__
            else:
                drop_action = DropObject.__name__

            if "vert" in carrying_obj:
                obj_type = 'vert'
            if "long" in carrying_obj:
                obj_type = 'long'

            # Choose location for dropping
            possible_xloc = list(range(0,2)) + list(range(16,19))
            x_loc = random.choice(possible_xloc)

            if chosen_loc is None:
                chosen_loc = (x_loc, 1)

            # Add move action to action list
            self.navigator.add_waypoint(chosen_loc)         # Add some code that searches for an empty spot out of the field
            route_actions = list(self.navigator._Navigator__get_route(self.state_tracker).values())
            for action in route_actions:
                self.actionlist[0].append(action)
                self.actionlist[1].append({})

            # Add pick up action to action list (define arguments here as well)
            pickup_kwargs = {}
            pickup_kwargs['obj_type'] = obj_type
            self.actionlist[0].append(drop_action)
            self.actionlist[1].append(pickup_kwargs)
        return

    def break_action(self, object_ids, state, location):
        large_obj = []
        parts_list = []
        y_loc_list = []
        dist_list = []
        chosen_part = None
        for object_id in object_ids:
            if "obstruction" in state[object_id]:
                continue
            if "large" in state[object_id]:
                y_loc_list.append(state[object_id]['location'][1])
                large_obj.append(object_id)
                dist = int(np.ceil(np.linalg.norm(np.array(state[object_id]['location'])
                                                  - np.array(state[self.agent_id]['location']))))
                dist_list.append(dist)
            if "bound_to" in state[object_id]:
                parts_list.append(object_id)

        if not y_loc_list:
            return

        if location is not None:
            chosen_part = large_obj[dist_list.index(min(dist_list))]
        else:
            chosen_part = large_obj[y_loc_list.index(min(y_loc_list))]
        large_name = state[chosen_part]['name']
        object_loc = state[chosen_part]['location']
        large_obj = [chosen_part]
        for part in parts_list:
            if state[part]['bound_to'] == large_name:
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

    def wait_action(self):
        pickup_kwargs = {}
        self.actionlist[0].append(Idle.__name__)
        self.actionlist[1].append(pickup_kwargs)
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
                    self.wait_action()
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
            self.wait_action()
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

        if state[self.agent_id]['is_carrying']:
            self.agent_properties["img_name"] = "/images/selector_holding2.png"
        else:
            self.agent_properties["img_name"] = "/images/selector2.png"

        self.check_cp_conditions()

        # ----------------------------Get reward from the messages---------------------------------------------
        last_message = None
        if self.received_messages:
            #print(self.received_messages)
            if self.received_messages[-1] == 'FAIL' and not self.final_update:
                print("FINAL Q UPDATE")
                last_message = float(self.received_messages[-2])
                done_action = self.action_history[-1][1]
                done_state = self.action_history[-1][0]
                print(self.q_table)
                self.update_q_table(done_state, done_action, done_action, done_state, last_message)
                print(self.q_table)
                self.final_update = True
            elif not self.final_update:
                last_message = float(self.received_messages[-1])

        # --------------------------New Main Action Planning Loop--------------------------------------------
        # TODO Add some code that checks if there are new CPs (or edits). Can come from messages

        # Start by checking if the agent is currently executing a CP
        if self.executing_cp:
            print("Agent is executing a CP!")
            # Check if the endconditions for this CP hold
            if self.check_cp_endconditions():
                self.executing_cp = False
                # If yes, finish and restart loop
            else:
                # If no, continue current CP
                self.execute_cp(self.executing_cp)
        else:
            # Not currently executing a CP
            print("Not working with a CP currently!")

            # Check if the start conditions for any existing CPs hold
            cps_hold = self.check_cp_conditions()
            if len(cps_hold) > 0:
                # This means there are CPs that are applicable. Check how many.
                if len(cps_hold) == 1:
                    # Only one CP is applicable, so we can directly start executing it
                    print("Only one CP holds: " + cps_hold[0])
                    self.executing_cp = cps_hold[0]
                    self.execute_cp(self.executing_cp)
                else:
                    # Several CPs hold. We need a method to choose between them.
                    print("Choose the appropriate CP.")
                    chosen_cp = self.choose_cp_from_list(cps_hold)
                    self.executing_cp = chosen_cp
                    self.execute_cp(self.executing_cp)
            else:
                # This means that there are no CPs that are applicable to the current situation
                print("No applicable CPs, do as normal.")

        # ----------------------------Original Action Planning--------------------------------------------
        if self.actionlist[0]:
            # This means that an action is still being executed
            action = self.actionlist[0].pop(0)
            action_kwargs = self.actionlist[1].pop(0)
            action_kwargs['action_duration'] = self.move_speed
            return action, action_kwargs        # Returned here, so code underneath is then not executed
        else:
            # This means we are done with our previous action, so we update Q-table
            current_state = self.filter_observations_learning(state)
            #print(current_state)
            done_action = 0
            done_state = current_state
            if self.action_history:
                done_action = self.action_history[-1][1]
                done_state = self.action_history[-1][0]
            if frozenset(current_state.items()) not in self.q_table:
                # State does not exist in q-table, so we create a new entry with all zero's
                self.q_table[frozenset(current_state.items())] = [0] * 3
            # Pick a new action based on the state we're currently in
            q_values = self.q_table[frozenset(current_state.items())]

            if self.previous_phase is None:                                         # If this is the start of the game
                self.previous_phase = self.filter_observations_learning(state)
                chosen_action = np.argmax(q_values)
            elif self.previous_phase == self.filter_observations_learning(state):   # If there is no phase change
                chosen_action = self.action_history[-1][1]
            else:                                                                   # If the phase changed
                print("NEW PHASE")
                chosen_action = np.argmax(q_values)
                self.update_q_table(current_state, chosen_action, done_action, done_state, last_message)
                self.previous_phase = self.filter_observations_learning(state)

            if chosen_action == 0:
                self.policy1(object_ids, state)
                print("P1")
            elif chosen_action == 1:
                self.policy2(object_ids, state)
                print("P2")
            elif chosen_action == 2:
                self.policy3(object_ids, state)
                print("P3")

            # Safe into the action history
            self.action_history.append([current_state, chosen_action])

        self.wait_action()
        print(self.q_table)

        return action, action_kwargs

# Functions that deal with the ontology stuff
    def store_cp_conditions(self):
        # Store all start conditions of existing CPs in a list; join duplicates, but store which CP they belong to
        self.cp_list = []
        self.start_conditions = []

        # Look at the list of CPs
        with TypeDB.core_client('localhost:1729') as client:
            with client.session("CP_ontology", SessionType.DATA) as session:
                with session.transaction(TransactionType.READ) as read_transaction:
                    answer_iterator = read_transaction.query().match("match $cp isa collaboration_pattern, has name $name; get $name;")

                    for answer in answer_iterator:
                        cp_retrieved = answer.get('name')._value
                        if cp_retrieved not in self.cp_list:
                            self.cp_list.append(cp_retrieved)

                    # For each CP, look up all start and end conditions
                    for cp in self.cp_list:
                        condition_list = []

                        # First, find all conditions related to the CP at hand
                        answer_iterator = read_transaction.query().match(
                            f'''match $cp isa collaboration_pattern, has name '{cp}'; 
                            $starts (condition: $condition, cp: $cp) isa starts_when; 
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
                            start_conditions_np = np.array(self.start_conditions)
                            if len(self.start_conditions) > 0 and items_contained in start_conditions_np[:, 0]:
                                index = np.where(start_conditions_np == items_contained)[0][0]
                                self.start_conditions[index][1].append(cp)
                            else:
                                self.start_conditions.append([items_contained, [cp]])

        print(self.start_conditions)
        return

    def check_cp_conditions(self):
        # Check all conditions of existing CPs and store which ones currently hold (how to do this efficiently??)
        conditions_hold = []
        cps_hold = []

        # For each condition, check if it holds
        for condition in self.start_conditions:
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
                # It exists! Translate and check locations
                if isinstance(relevant_objects, dict):
                    # There is just one such object, check it's location
                    if location['range'] in self.translate_location(relevant_objects['obj_id'], object_type):
                        # It is the same, condition holds!
                        if condition not in conditions_hold:
                            conditions_hold.append(condition)
                elif isinstance(relevant_objects, list):
                    # It is a list, we'll need to loop through
                    for object in relevant_objects:
                        # Translate the location and check whether it is the one in the condition
                        if location['range'] in self.translate_location(object['obj_id'], object_type):
                            # It is the same, condition holds! We can break the for loop
                            if condition not in conditions_hold:
                                conditions_hold.append(condition)
                            break
            else:
                # There are no such objects, we can stop here
                print("Condition doesn't hold")


        # Then check if there is any CP for which each start condition holds (or if the end condition of the current holds)

        # For each condition that holds
        for condition in conditions_hold:
            # Check to which CP this condition is tied
            bound_cps = condition[1]

            for cp in bound_cps:
                # Add a check, if the CP at hand is already in the CPs_hold list, we can skip
                if cp not in cps_hold:
                    # For each CP, check if there are other conditions for this CP
                    other_conditions = [i for i, x in enumerate(self.start_conditions) if cp in x[1]]

                    if len(other_conditions) > 1:
                        # This means there are other conditions. Check if all of them are in the conditions_hold list
                        all_conditions = True
                        for index in other_conditions:
                            if self.start_conditions[index][0] not in np.asarray(conditions_hold)[:,0]:
                                all_conditions = False
                        # If all of them are, CP is valid
                        if all_conditions:
                            cps_hold.append(cp)
                    else:
                        # This means that there are no other conditions. Therefore, this CP holds!
                        cps_hold.append(cp)

        return cps_hold

    def execute_cp(self, cp):
        # Retrieve the actions from the CP

        # Store and/or retrieve what position in the CP we're at (which action)

        # Check whether it's time to move to the next action

        # Execute action
        return

    def check_cp_endconditions(self):
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
            for i in range(0, object_loc_y-1):
                loc_to_check = [object_loc_x + x, i]
                objects_found = self.state[{"location": loc_to_check}]
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
                objects_found = self.state[{"location": loc_to_check}]
                if objects_found is not None:
                    # An object was found, meaning that the area below the rock isn't empty TODO create exception for agents
                    bottom_check = False

        if bottom_check == True:
            locations.append('Bottom of rock pile')


        # Left/Right side of rock pile (= within the bounds of the pile, left or right half)
        if object_loc_x >= 5 and object_loc_x <= 9:
            locations.append('Left side of rock pile')
        elif object_loc_x >= 10 and object_loc_x <= 15:
            locations.append('Right side of rock pile')
        # Left/Right side of field (= outside the bounds of the pile, left or right)
        elif object_loc_x < 5:
            locations.append('Left side of field')
        elif object_loc_x > 15:
            locations.append('Right side of field')

        # On top of [object/actor/location] (to be implemented later)

        # Above rock pile (not relevant for rocks, only for agents)

        return locations

# Functions that deal with learning
    def choose_cp_from_list(self, cp_list):
        # Some algorithm that is able to deal with several CPs that hold, possibly RL
        chosen_cp = cp_list[0]
        return chosen_cp

