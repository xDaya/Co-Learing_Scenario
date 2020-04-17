import numpy as np
from matrx.actions.action import Action, ActionResult
from matrx.actions.object_actions import _is_drop_poss, _act_drop, _possible_drop, _find_drop_loc
from matrx.world_builder import RandomProperty

rock_imgs = ['/images/rock1.png', '/images/rock2.png', '/images/rock3.png']
rock_img_property = RandomProperty(values=rock_imgs)

class BreakObject(Action):

    def __init__(self, duration_in_ticks=1):
        super().__init__(duration_in_ticks)

    # The is_possible function here is currently stolen from the grab_large, should be adjusted later
    def is_possible(self, grid_world, agent_id, **kwargs):
        # Set default values check

        object_id = None if 'object_id' not in kwargs else kwargs['object_id']
        grab_range = np.inf if 'grab_range' not in kwargs else kwargs['grab_range']
        max_objects = np.inf if 'max_objects' not in kwargs else kwargs['max_objects']

        return _is_possible_grab_large(grid_world, agent_id=agent_id, object_id=object_id, grab_range=grab_range,
                                 max_objects=max_objects)

    def mutate(self, grid_world, agent_id, **kwargs):
        # Additional check
        assert 'object_id' in kwargs.keys()
        assert 'grab_range' in kwargs.keys()

        object_ids = kwargs['object_id']     # So this is the list with large object and parts that is in range

        # Remove first item from the list and environment, which is the large object itself
        succeeded = grid_world.remove_from_grid(object_ids[0])
        object_ids.pop(0)

        # For loop that loops through list of parts to change
        for object_id in object_ids:
            env_obj = grid_world.environment_objects[object_id]  # Environment object

            # Change property 'bound_to' to None
            env_obj.release_bound()
            env_obj.change_property('img_name', rock_img_property)


        if not succeeded:
            return BreakObjectResult(
                BreakObjectResult.FAILED_TO_REMOVE_OBJECT_FROM_WORLD.replace("{OBJECT_ID}",
                                                                                     env_obj.obj_id), False)

        return BreakObjectResult(BreakObjectResult.RESULT_SUCCESS, True)


class BreakObjectResult(ActionResult):

    """ Result when the object can be successfully grabbed. """
    RESULT_SUCCESS = 'Grab action success'

    """ Result when the specified object is not within range. """
    NOT_IN_RANGE = 'Object not in range'

    """ Result when the specified object is an agent. """
    RESULT_AGENT = 'This is an agent, cannot be picked up'

    """ Result when no object was specified. """
    RESULT_NO_OBJECT = 'No Object specified'

    """ Result when the specified object does not exist in the :class:`matrxs.grid_world.GridWorld` """
    RESULT_UNKNOWN_OBJECT_TYPE = 'obj_id is no Agent and no Object, unknown what to do'

    """ Result when the specified object is not movable. """
    RESULT_OBJECT_UNMOVABLE = 'Object is not movable'

    def __init__(self, result, succeeded):
        super().__init__(result, succeeded)


class GrabLargeObject(Action):
    def __init__(self, duration_in_ticks=1):
        super().__init__(duration_in_ticks)

    def is_possible(self, grid_world, agent_id, **kwargs):
        # Set default values check

        object_id = None if 'object_id' not in kwargs else kwargs['object_id']
        grab_range = np.inf if 'grab_range' not in kwargs else kwargs['grab_range']
        max_objects = np.inf if 'max_objects' not in kwargs else kwargs['max_objects']

        return _is_possible_grab_large(grid_world, agent_id=agent_id, object_id=object_id, grab_range=grab_range,
                                 max_objects=max_objects)

    def mutate(self, grid_world, agent_id, **kwargs):
        # Additional check
        assert 'object_id' in kwargs.keys()
        assert 'grab_range' in kwargs.keys()
        assert 'max_objects' in kwargs.keys()

        # if possible:
        object_ids = kwargs['object_id']  # assign

        # Loading properties
        reg_ag = grid_world.registered_agents[agent_id]  # Registered Agent
        # For loop that loops through list of large object and its parts
        for object_id in object_ids:
            env_obj = grid_world.environment_objects[object_id]  # Environment object

            # Updating properties
            env_obj.carried_by.append(agent_id)
            reg_ag.is_carrying.append(env_obj)  # we add the entire object!

            # Remove it from the grid world (it is now stored in the is_carrying list of the AgentAvatar
            succeeded = grid_world.remove_from_grid(object_id=env_obj.obj_id, remove_from_carrier=False)

            if not succeeded:
                return GrabLargeObjectResult(GrabLargeObjectResult.FAILED_TO_REMOVE_OBJECT_FROM_WORLD.replace("{OBJECT_ID}",
                                                                                                        env_obj.obj_id), False)

            # Updating Location (done after removing from grid, or the grid will search the object on the wrong location)
            env_obj.location = reg_ag.location

        return GrabLargeObjectResult(GrabLargeObjectResult.RESULT_SUCCESS, True)


class GrabLargeObjectResult(ActionResult):
    """ Result when the object can be successfully grabbed. """
    RESULT_SUCCESS = 'Grab action success'

    """ Result when the grabbed object cannot be removed from the :class:`matrxs.grid_world.GridWorld`. """
    FAILED_TO_REMOVE_OBJECT_FROM_WORLD = 'Grab action failed; could not remove object with id {OBJECT_ID} from grid.'

    """ Result when the specified object is not within range. """
    NOT_IN_RANGE = 'Object not in range'

    """ Result when the specified object is an agent. """
    RESULT_AGENT = 'This is an agent, cannot be picked up'

    """ Result when no object was specified. """
    RESULT_NO_OBJECT = 'No Object specified'

    """ Result when the agent is at its maximum carrying capacity. """
    RESULT_CARRIES_OBJECT = 'Agent already carries the maximum amount of objects'

    """ Result when the specified object is already carried by another agent. """
    RESULT_OBJECT_CARRIED = 'Object is already carried by {AGENT_ID}'

    """ Result when the specified object does not exist in the :class:`matrxs.grid_world.GridWorld` """
    RESULT_UNKNOWN_OBJECT_TYPE = 'obj_id is no Agent and no Object, unknown what to do'

    """ Result when the specified object is not movable. """
    RESULT_OBJECT_UNMOVABLE = 'Object is not movable'

    def __init__(self, result, succeeded):
        super().__init__(result, succeeded)


class DropLargeObject(Action):
    def __init__(self, duration_in_ticks=1):
        super().__init__(duration_in_ticks)

    def is_possible(self, grid_world, agent_id, **kwargs):
        reg_ag = grid_world.registered_agents[agent_id]

        drop_range = 1 if 'drop_range' not in kwargs else kwargs['drop_range']

        # If no object id is given, the last item is dropped
        if 'object_id' in kwargs:
            obj_id = kwargs['object_id']
        elif len(reg_ag.is_carrying) > 0:
            obj_id = reg_ag.is_carrying[-1] # Not edited for now; eventually we will need to make sure there is space for all blocks contained by the large block, now checking for 1 is enough
        else:
            return DropLargeObjectResult(DropLargeObjectResult.RESULT_NO_OBJECT, False)

        return _possible_drop(grid_world, agent_id=agent_id, obj_id=obj_id, drop_range=drop_range)

    def mutate(self, grid_world, agent_id, **kwargs):
        reg_ag = grid_world.registered_agents[agent_id]

        # fetch range from kwargs
        drop_range = 1 if 'drop_range' not in kwargs else kwargs['drop_range']

        parts_obj = []

        # If no object id is given, the last item is dropped
        if 'object_id' in kwargs:
            env_obj = kwargs['object_id']
        elif len(reg_ag.is_carrying) > 0:
            env_obj = reg_ag.is_carrying[-1]
            parts_obj = reg_ag.is_carrying[::-1]    # Assuming here that agent only carries 1 object which is large. Reversed so the parts come first, then the large object itself
        else:
            return DropLargeObjectResult(DropLargeObjectResult.RESULT_NO_OBJECT_CARRIED, False)

        # check that it is even possible to drop this object somewhere
        if not env_obj.is_traversable and not reg_ag.is_traversable and drop_range == 0:
            raise Exception(
                f"Intraversable agent {reg_ag.obj_id} can only drop the intraversable object {env_obj.obj_id} at its "
                f"own location (drop_range = 0), but this is impossible. Enlarge the drop_range for the DropAction to "
                f"atleast 1")

        # check if we can drop it at our current location
        curr_loc_drop_poss = _is_drop_poss(grid_world, env_obj, reg_ag.location, agent_id)

        # drop it on the agent location if possible
        if curr_loc_drop_poss:
            return _act_drop_large(grid_world, agent=reg_ag, parts_obj=parts_obj, drop_loc=reg_ag.location)       # We need to make this loop over the different objects

        # if the agent location was the only within range, return a negative action result
        elif not curr_loc_drop_poss and drop_range == 0:
            return DropLargeObjectResult(DropLargeObjectResult.RESULT_OBJECT, False)

        # Try finding other drop locations from close to further away around the agent
        drop_loc = _find_drop_loc(grid_world, reg_ag, env_obj, drop_range, reg_ag.location)

        # If we didn't find a valid drop location within range, return a negative action result
        if not drop_loc:
            return DropLargeObjectResult(DropLargeObjectResult.RESULT_OBJECT, False)

        return _act_drop(grid_world, agent=reg_ag, env_obj=env_obj, drop_loc=drop_loc)


class DropLargeObjectResult(ActionResult):
    """ Result when dropping the object succeeded. """
    RESULT_SUCCESS = 'Drop action success'

    """ Result when there is not object in the agent's inventory. """
    RESULT_NO_OBJECT = 'The item is not carried'

    """ Result when the specified object is not in the agent's inventory. """
    RESULT_NONE_GIVEN = "'None' used as input id"

    """ Result when the specified object should be dropped on an agent. """
    RESULT_AGENT = 'Cannot drop item on an agent'

    """ Result when the specified object should be dropped on an intraversable object."""
    RESULT_OBJECT = 'Cannot drop item on another intraversable object'

    """ Result when the specified object does not exist (anymore). """
    RESULT_UNKNOWN_OBJECT_TYPE = 'Cannot drop item on an unknown object'

    """ Result when the agent is not carrying anything. """
    RESULT_NO_OBJECT_CARRIED = 'Cannot drop object when none carried'

    def __init__(self, result, succeeded, obj_id=None):
        super().__init__(result, succeeded)
        self.obj_id = obj_id


class Fall(Action):
    def __init__(self, duration_in_ticks=1):
        super().__init__(duration_in_ticks)

    def is_possible(self, grid_world, agent_id, **kwargs):
        # Maybe do a check to see if the empty location is really and still empty?
        return FallResult(FallResult.RESULT_SUCCESS, True)

    def mutate(self, grid_world, agent_id, **kwargs):
        # Make sure this can deal with a list of objects and lets them fall down (in an order that makes sense?)
        # Additional check
        assert 'object_list' in kwargs.keys()

        # if possible:
        falling_objs = kwargs['object_list']  # assign

        for object_id in falling_objs:
            env_obj = grid_world.environment_objects[object_id]  # Environment object
            object_loc = env_obj.location
            object_loc_x = object_loc[0]
            object_loc_y = object_loc[1]

            # Update y value
            new_y = object_loc_y + 1
            new_loc = (object_loc_x, new_y)

            # Actually update location
            env_obj.location = new_loc

        return FallResult(FallResult.RESULT_SUCCESS, True)


class FallResult(ActionResult):
    """ Result when falling succeeded. """
    RESULT_SUCCESS = 'Falling action successful'

    """ Result when the emptied space was not actually empty. """
    RESULT_NOT_EMPTY = 'There was no empty space for the objects to fall in'

    """ Result when the emptied space was not actually empty. """
    RESULT_FAILED = 'Failed to let object fall'

    def __init__(self, result, succeeded):
        super().__init__(result, succeeded)

def _is_possible_grab_large(grid_world, agent_id, object_id, grab_range, max_objects):
    reg_ag = grid_world.registered_agents[agent_id]  # Registered Agent
    loc_agent = reg_ag.location  # Agent location

    # There is no large object specified
    if not object_id:
        return GrabLargeObjectResult(GrabLargeObjectResult.RESULT_NO_OBJECT, False)

    # Already carries an/too many object(s)
    if len(reg_ag.is_carrying) + len(object_id) > max_objects:
        return GrabLargeObjectResult(GrabLargeObjectResult.RESULT_CARRIES_OBJECT, False)

    # Go through all objects at the desired location

    # Set random object in range

    # Check if object is in range

    # Check if object_id is the id of an agent

    # Check if it is an object

    else:
        return GrabLargeObjectResult(GrabLargeObjectResult.RESULT_SUCCESS, True)


def _act_drop_large(grid_world, agent, parts_obj, drop_loc):
    x_drop_loc = drop_loc[0]
    y_drop_loc = drop_loc[1]

    locations = [drop_loc, (x_drop_loc+1, y_drop_loc), (x_drop_loc, y_drop_loc+1), (x_drop_loc+1, y_drop_loc+1), drop_loc]

    for env_obj in parts_obj:
        # Updating properties
        agent.is_carrying.remove(env_obj)
        env_obj.carried_by.remove(agent.obj_id)

        # We return the object to the grid location we are standing at
        env_obj.location = locations[parts_obj.index(env_obj)]
        grid_world._register_env_object(env_obj)