from matrx.objects.env_object import EnvObject


class PartLargeObject(EnvObject):
    def __init__(self, location, name="PartLargeObject", img_name="/images/rock.png", bound_to=None, **kwargs):
        """
        A simple object with the added property 'bound_to' to create objects that can be bound to a larger object.
        """

        super().__init__(name=name, location=location, visualize_shape='img', visualize_size=1, class_callable=PartLargeObject, is_traversable=True, is_movable=True, visualize_from_center=False, **kwargs)

        self.bound_to = bound_to
        self.add_property('bound_to', self.bound_to)

        self.img_name = img_name
        self.add_property('img_name', self.img_name)



class LargeObject(EnvObject):
    def __init__(self, location, name="LargeObject", visualize_size = 2, large=True, **kwargs):
        self.large = large

        super().__init__(name=name, location=location, visualize_shape='img', visualize_size=visualize_size, class_callable=LargeObject, is_traversable=True, is_movable=True, large=True, visualize_from_center=False, **kwargs)


class ObstructionObject(EnvObject):
    def __init__(self, location, name="ObstructionObject", visualize_size = 4, obstruction=True, large=True, **kwargs):
        self.obstruction = obstruction
        self.large = large

        super().__init__(name=name, location=location, visualize_shape='img', visualize_size=visualize_size, class_callable=ObstructionObject, is_traversable=True, is_movable=False, obstruction=self.obstruction, large=True, visualize_from_center=False, **kwargs)


class GoalReachedObject(EnvObject):
    def __init__(self, location, name="GoalReachedObject", img_name="/images/transparent.png", visualize_size=12, **kwargs):
        super().__init__(name=name, location=location, visualize_shape='img', class_callable=GoalReachedObject, is_traversable=True, visualize_from_center=False, **kwargs)
        self.img_name = img_name
        self.add_property('img_name', self.img_name)
        self.visualize_size = visualize_size
        self.add_property('visualize_size', self.visualize_size)
