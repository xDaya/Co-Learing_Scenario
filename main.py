import os, requests

# New version for development

from builder import create_builder
from custom_visualizer import visualization_server



if __name__ == "__main__":


    # Start overarching MATRX scripts and threads, such as the api and/or visualizer if requested. Here we also link our

    # own media resource folder with MATRX.

    media_folder = os.path.dirname(os.path.join(os.path.realpath("C:/Users/zoelenev/PycharmProjects/Co-Learing_Scenario"), "media"))

    #builder.startup(media_folder=media_folder)

    # start the custom visualizer
    print("Starting custom visualizer")
    vis_thread = visualization_server.run_matrx_visualizer(verbose=False, media_folder=media_folder)

    for level in range(0, 8):
        # Create our world builder
        builder = create_builder(level)
        builder.startup(media_folder=media_folder)

        print("Started world...")

        # run the world
        world = builder.get_world()
        world.run(builder.api_info)

    # stop the custom visualizer
    print("Shutting down custom visualizer")
    r = requests.get("http://localhost:" + str(visualization_server.port) + "/shutdown_visualizer")
    vis_thread.join()
