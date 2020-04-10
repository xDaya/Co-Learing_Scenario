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

block_size = 1
block_colors = ['#DCDDDC', '#D3D2D2', '#A9A9A9']
rock_imgs = ['/images/rock1.png', '/images/rock2.png', '/images/rock3.png']

rock_img_property = RandomProperty(values=rock_imgs)


def create_builder():
    factory = WorldBuilder(shape=[20, 12], run_matrx_visualizer=True, visualization_bg_clr="#685E58", visualization_bg_img='/images/background.png')

    loc = (5, 5)
    rubble_st_size = 1
    rubble_lrg_size = 2

    factory.add_object((5, 5), name="1st_block", visualize_shape='img', img_name=rock_img_property, visualize_size=rubble_st_size, is_movable=True, is_traversable=True)
    factory.add_object((5, 6), name="2nd_block", visualize_shape='img', img_name=rock_img_property, visualize_size=rubble_st_size)
    factory.add_object((6, 5), name="3rd_block", visualize_shape='img', img_name=rock_img_property, visualize_size=rubble_st_size)

    # Random generation of several objects
    for i in range(0, 8):
        factory.add_object((i+10, i+1), name="randomgen_" + str(i), visualize_size=rubble_st_size, visualize_shape='img', img_name=rock_img_property)

    create_large_object(name="test2_large", location=(6, 3), world=factory)

    # Link agent names to agent brains
    human_agent = CustomHumanAgentBrain(max_carry_objects=5, grab_range=1)
    autonomous_agent = PatrollingAgentBrain(waypoints=[(0, 0), (0, 7)])

    human_img = HumanAgentBrain()
    machine_img = HumanAgentBrain()
    victim_img = HumanAgentBrain()

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

    factory.add_human_agent([3, 4], human_agent, name="Human Selector", key_action_map=key_action_map, visualize_shape='img', img_name="/images/selector.png", visualize_size=1)

    factory.add_human_agent([1, 7], human_img, name="Human", visualize_shape='img', img_name="/images/human_square.png", visualize_size=4)
    factory.add_human_agent([15, 7], machine_img, name="Machine", visualize_shape='img',
                            img_name="/images/machine_square.png", visualize_size=4)
    factory.add_human_agent([8, 7], victim_img, name="Victim", visualize_shape='img',
                            img_name="/images/victim_square.png", visualize_size=4)

    # factory.add_agent([0,2], autonomous_agent, name="Robot", visualize_shape='img', img_name="/images/machine_square.png", visualize_size=2)

    return factory


def create_large_object(name, location, world):
    x_loc = location[0]
    y_loc = location[1]

    world.add_object(location, name="Part_tl_" + name, img_name=rock_img_property, callable_class=PartLargeObject, bound_to=name)
    world.add_object((x_loc+1, y_loc), name="Part_tr_" + name, img_name=rock_img_property, callable_class=PartLargeObject, bound_to=name)
    world.add_object((x_loc, y_loc+1), name="Part_bl_" + name, img_name=rock_img_property, callable_class=PartLargeObject, bound_to=name)
    world.add_object((x_loc+1, y_loc+1), name="Part_br_" + name, img_name=rock_img_property, callable_class=PartLargeObject, bound_to=name)

    world.add_object(location, name=name, img_name=rock_img_property, callable_class=LargeObject)


def generate_rubble_pile(name, location, dimensions, world):
    pass