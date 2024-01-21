import carla
import weakref

class ObstacleDetectionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.distance = None
        self._event_count = 0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.obstacle')
        bp.set_attribute('distance','50')
        bp.set_attribute('hit_radius','5')
        bp.set_attribute('debug_linetrace','True')
        bp.set_attribute('only_dynamics', 'True')
        bp.set_attribute('sensor_tick', '0.1')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: ObstacleDetectionSensor._on_obstacle(weak_self, event))

    @staticmethod
    def _on_obstacle(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.distance = event.distance
        self._event_count += 1
        print ("Event %s, in line of sight with %s at distance %u" % (self._event_count, event.other_actor.type_id, event.distance))