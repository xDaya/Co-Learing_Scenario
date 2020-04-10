from matrx.objects.env_object import EnvObject


class FallingObject(EnvObject):

    def __init__(self, location, name="FallingObject"):

        self.location = location
        current_location = self.location

        # Some code

        super().__init__(location=current_location, name=name, is_traversable=True, is_movable=True, class_callable=FallingObject)


    def fall_object(self):

        bottom_y = 19

        if self.location[1] == bottom_y:
            return

        # Some check on location below current location and whether it is empty

        # If this is the case, come up with new location

        # Update location property

        self.location = self.new_location

class PartLargeObject(EnvObject):
    def __init__(self, location, name="PartLargeObject", img_name="/images/rock.png", bound_to=None, **kwargs):
        """
        A simple object with the added property 'bound_to' to create objects that can be bound to a larger object.
        """

        self.bound_to = bound_to
        super().__init__(name=name, location=location, visualize_shape='img', img_name=img_name, visualize_size=1, class_callable=PartLargeObject, is_traversable=True, is_movable=True, bound_to=bound_to, **kwargs)

    def release_bound(self):
        """
        In case of a break action, this releases the bound of the objects and makes them regular blocks.
        """

        # Actually release the bound
        self.bound_to = None



class LargeObject(EnvObject):
    def __init__(self, location, name="LargeObject", large=True, **kwargs):
        self.large = large

        super().__init__(name=name, location=location, visualize_shape='img', visualize_size=2, class_callable=LargeObject, is_traversable=True, is_movable=True, large=True, **kwargs)