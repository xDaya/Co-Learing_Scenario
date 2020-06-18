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

        super().__init__(name=name, location=location, visualize_shape='img', visualize_size=1, class_callable=PartLargeObject, is_traversable=True, is_movable=True, **kwargs)

        self.bound_to = bound_to
        self.add_property('bound_to', self.bound_to)

        self.img_name = img_name
        self.add_property('img_name', self.img_name)



class LargeObject(EnvObject):
    def __init__(self, location, name="LargeObject", visualize_size = 2, large=True, **kwargs):
        self.large = large

        super().__init__(name=name, location=location, visualize_shape='img', visualize_size=visualize_size, class_callable=LargeObject, is_traversable=True, is_movable=True, large=True, **kwargs)


class ObstructionObject(EnvObject):
    def __init__(self, location, name="ObstructionObject", visualize_size = 4, obstruction=True, large=True, **kwargs):
        self.obstruction = obstruction
        self.large = large

        super().__init__(name=name, location=location, visualize_shape='img', visualize_size=visualize_size, class_callable=ObstructionObject, is_traversable=True, is_movable=False, obstruction=self.obstruction, large=True, **kwargs)


class RewardObject(EnvObject):
    def __init__(self, location, name="rewardobj", goalreached=False, **kwargs):

        super().__init__(name=name, location=location, visualize_shape='img', is_traversable=True, is_movable=False, class_callable=RewardObject, goalreached=goalreached)
