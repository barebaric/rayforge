class Modifier:
    """
    Modifies a Cairo surface.
    """
    def __init__(self, **kwargs):
        self.label = self.__class__.__name__

    def to_dict(self):
        """
        Serializes the modifier to a dictionary.
        """
        return {'name': self.__class__.__name__}

    @classmethod
    def from_dict(cls, data: dict):
        """
        Factory to create a modifier instance from a dictionary.

        This method looks up the correct modifier class from the 'name'
        key in the dictionary and delegates the actual instantiation.
        """
        # Local import to avoid a circular dependency at module-load time.
        # The modifier_by_name map is built in the package's __init__.py,
        # which imports this module.
        from . import modifier_by_name

        modifier_name = data.get('name')
        if not modifier_name:
            raise ValueError("Dictionary must contain a 'name' key.")

        modifier_class = modifier_by_name.get(modifier_name)
        if not modifier_class:
            raise ValueError(f"Unknown modifier name: '{modifier_name}'")

        # Instantiate the class with parameters from the dictionary.
        # This allows for future producers to have configurable state.
        params = data.get('params', {})
        return modifier_class(**params)

    def run(self, surface):
        """
        - step: the Step that the process is a part of
        - surface: an input surface. Can be manipulated in-place,
          or alternatively a new surface may be returned.
        - pixels_per_mm: tuple: pixels_per_mm_x, pixels_per_mm_y
        - ymax: machine max in y direction
        """
        pass
