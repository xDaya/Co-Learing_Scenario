import os, requests

# New version for development

from builder import create_builder
from custom_visualizer import visualization_server
import pandas



if __name__ == "__main__":
    # Variable that indicates start scenario
    start_scenario = None
    # Variable that indicates participant number (to change port nr etc.)
    participant_nr = 3000
    level_order = [1, 2, 3, 4, 5, 6, 7, 8]

    level_order_list = pandas.read_csv('latin_square_list.csv', sep=';')

    # Start overarching MATRX scripts and threads, such as the api and/or visualizer if requested. Here we also link our
    # own media resource folder with MATRX.

    media_folder = os.path.dirname(os.path.realpath(__file__))

    #builder.startup(media_folder=media_folder)

    while not start_scenario == 'exit':
        # Ask for start scenario
        print("\n\nWhich scenario do you want to start with? ")
        start_scenario = int(input())
        print("\n\nWhich participant number do you want to use? ")
        participant_nr = int(input())

        if participant_nr in level_order_list['participantnr'].values:
            level_order = level_order_list.loc[level_order_list['participantnr'] == participant_nr].values.flatten().tolist()
            level_order = level_order[1:]

        # start the custom visualizer
        print("Starting custom visualizer")
        vis_thread = visualization_server.run_matrx_visualizer(verbose=False, media_folder=media_folder, port_nr=participant_nr)

        for round in range(start_scenario, 9):
            level = 0
            if round > 0:
                level = level_order[round - 1]

            # Create our world builder
            builder = create_builder(round, level, participant_nr)
            builder.startup(media_folder=media_folder)

            print("Started world...")

            # run the world
            world = builder.get_world()
            world.run(builder.api_info)

    # stop the custom visualizer
    print("Shutting down custom visualizer")
    r = requests.get("http://localhost:" + str(visualization_server.port) + "/shutdown_visualizer")
    vis_thread.join()
