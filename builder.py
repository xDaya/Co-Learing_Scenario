from matrx.agents.agent_types.human_agent import HumanAgentBrain
from matrx.agents.agent_types.patrolling_agent import PatrollingAgentBrain
from matrx.logger.log_agent_actions import LogActions
from matrx.world_builder import WorldBuilder
from matrx.actions.move_actions import *
from matrx.actions.object_actions import *
from matrx.actions.door_actions import *
from matrx.world_builder import RandomProperty
from matrx.objects.standard_objects import *
from custom_objects import *
from custom_agents import *
from custom_actions import *
import random

block_size = 1
block_colors = ['#DCDDDC', '#D3D2D2', '#A9A9A9']
rock_imgs = ['/images/rock1.png', '/images/rock2.png', '/images/rock3.png']
lower_bound = 11
left_bound = 5
right_bound = 15
upper_bound = 1

rock_img_property = RandomProperty(values=rock_imgs)

rubble_locations = []
for y_loc in range(upper_bound, lower_bound):
    for x_loc in range(left_bound, right_bound):
        rubble_locations.append((x_loc, y_loc))


def create_builder():
    factory = WorldBuilder(shape=[20, 12], run_matrx_visualizer=True, visualization_bg_clr="#ffffff",
                           visualization_bg_img='/images/background.png', tick_duration=0.1)

    # Link agent names to agent brains
    human_agent = CustomHumanAgentBrain(max_carry_objects=5, grab_range=1)
    autonomous_agent = PatrollingAgentBrain(waypoints=[(0, 0), (0, 7)])

    human_img = HumanAgentBrain()
    machine_img = HumanAgentBrain()
    victim_img = HumanAgentBrain()
    gravity_god = GravityGod()

    key_action_map = {
        'w': MoveNorth.__name__,
        'd': MoveEast.__name__,
        's': MoveSouth.__name__,
        'a': MoveWest.__name__,
        'p': GrabObject.__name__,
        'n': DropObject.__name__,
        'r': RemoveObject.__name__,
        'o': OpenDoorAction.__name__,
        'l': GrabLargeObject.__name__,
        'm': DropLargeObject.__name__,
        'b': BreakObject.__name__
    }

    # Add the selector agent that allows humans to interact
    factory.add_human_agent([3, 4], human_agent, name="Human Selector", key_action_map=key_action_map,
                            visualize_shape='img', img_name="/images/selector.png", visualize_size=1)

    # Add agents that are static and mostly just show the image of the 'actual' agent
    factory.add_human_agent([1, 7], human_img, name="Human", visualize_shape='img',
                            img_name="/images/human_square.png", visualize_size=4)
    factory.add_human_agent([15, 7], machine_img, name="Machine", visualize_shape='img',
                            img_name="/images/machine_square.png", visualize_size=4)
    factory.add_human_agent([8, 7], victim_img, name="Victim", visualize_shape='img',
                            img_name="/images/victim_square.png", visualize_size=4)

    # Add Gravity by adding the GravityGod agent
    factory.add_agent((0, 0), gravity_god, name="GravityGod", visualize_shape='img', img_name="/images/transparent.png")

    # factory.add_agent([0,2], autonomous_agent, name="Robot", visualize_shape='img',
    #                   img_name="/images/machine_square.png", visualize_size=2)

    generate_rubble_pile(name="test_pile", locations=rubble_locations, world=factory)

    return factory


def create_large_object(name, location, world):
    x_loc = location[0]
    y_loc = location[1]

    img_part = "/images/transparent.png"
    world.add_object(location, name="Part_tl_" + name, img_name=img_part, callable_class=PartLargeObject,
                     bound_to=name)
    world.add_object((x_loc+1, y_loc), name="Part_tr_" + name, img_name=img_part, callable_class=PartLargeObject,
                     bound_to=name)
    world.add_object((x_loc, y_loc+1), name="Part_bl_" + name, img_name=img_part, callable_class=PartLargeObject,
                     bound_to=name)
    world.add_object((x_loc+1, y_loc+1), name="Part_br_" + name, img_name=img_part, callable_class=PartLargeObject,
                     bound_to=name)

    world.add_object(location, name=name, img_name=rock_img_property, callable_class=LargeObject)


def generate_rubble_pile(name, locations, world):
    possible_locations = locations[:]
    random.shuffle(possible_locations)
    for items in range(0, 6):
        initial_loc = possible_locations[0]
        ix_loc = initial_loc[0]
        iy_loc = initial_loc[1]
        loc_to_check = {initial_loc, (ix_loc+1, iy_loc), (ix_loc, iy_loc+1), (ix_loc+1, iy_loc+1)}
        if loc_to_check <= set(possible_locations):
            create_large_object(name="large_"+str(items), location=initial_loc, world=world)
        for locs in loc_to_check:
            if locs in possible_locations:
                possible_locations.remove(locs)
    for items in range(0, 40):
        loc = possible_locations.pop()
        world.add_object(loc, name=name, visualize_shape='img', img_name=rock_img_property,
                         is_traversable=True, is_movable=True)
