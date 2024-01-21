import math
import weakref
import collections
import carla
import random
import time
import spade
import asyncio
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour
from spade import wait_until_finished
import numpy as np
import threading

DISTANCE_VALUE = 50
TARGETS = []
temp_targets = []
BATTERY_LEVEL = 60
battery_stop_event = threading.Event()
PICKED_UP = False


def battery_control():
    global BATTERY_LEVEL
    while not battery_stop_event.is_set() and BATTERY_LEVEL > 0:
        time.sleep(1)
        BATTERY_LEVEL -= 1
        print(f"Battery Level: {BATTERY_LEVEL}")

class PIDController:
    def __init__(self, kp, ki, kd, setpoint):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0
        self.previous_error = 0

    def update(self, current_value):
        error = self.setpoint - current_value
        self.integral += error
        derivative = error - self.previous_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output


class TargetLocation:
    def __init__(self, carla_location, location_type):
        self.location = carla_location
        self.location_type = location_type
        self.is_arrived = False

    def set_arrived(self, is_arrived):
        self.is_arrived = is_arrived



def find_suitable_location(targets):
    for target in targets:
        if BATTERY_LEVEL < 50 and target.location_type == "charge":
            print("Battery level is low, going to charge")
            return target
        if not target.is_arrived and PICKED_UP and target.location_type == "drop":
            return target
        if not target.is_arrived and not PICKED_UP and target.location_type == "take":
            return target
    return None




def draw_markers(world):
    global temp_targets
    for target in temp_targets:
        debug = world.debug
        waypoint = world.get_map().get_waypoint(target.location)
        if debug is not None:
            if target.location_type == "take" and not target.is_arrived:
                debug.draw_box(carla.BoundingBox(waypoint.transform.location, carla.Vector3D(0.5, 0.5, 0.5)),
                               waypoint.transform.rotation, 0.05, carla.Color(255, 0, 0), 1.0)
            elif target.location_type == "drop" and target.is_arrived:
                debug.draw_box(carla.BoundingBox(waypoint.transform.location, carla.Vector3D(0.5, 0.5, 0.5)),
                               waypoint.transform.rotation, 0.05, carla.Color(255, 0, 0), 1.0)
            elif target.location_type == "charge":
                debug.draw_box(carla.BoundingBox(waypoint.transform.location, carla.Vector3D(3.5, 3.5, 3.5)),
                               waypoint.transform.rotation, 0.05, carla.Color(255, 0, 255), 1.0)

            if not target.is_arrived:
                arrow_start = carla.Location(waypoint.transform.location.x, waypoint.transform.location.y,
                                             waypoint.transform.location.z + 5)  # Start 2 meters above the ground
                arrow_end = carla.Location(waypoint.transform.location.x, waypoint.transform.location.y,
                                           waypoint.transform.location.z + 2)  # End at the target location
                arrow_color = carla.Color(0, 0, 255)  # Blue color for the arrow
                debug.draw_arrow(arrow_start, arrow_end, thickness=0.2, arrow_size=0.2, color=arrow_color, life_time=1.0)



def spawn_walkers(client, number_of_walkers):
    world = client.get_world()
    walker_bp = random.choice(world.get_blueprint_library().filter("walker.pedestrian.*"))
    walker_list = world.get_blueprint_library().filter("walker.pedestrian.*")
    spawn_points = world.get_map().get_spawn_points()
    walkers = []
    for _ in range(number_of_walkers):
        spawn_point = random.choice(spawn_points)
        walker = world.try_spawn_actor(walker_bp, spawn_point)
        if walker is not None:
            walkers.append(walker)
    return walkers


def spawn_vehicles(client, number_of_vehicles):
    world = client.get_world()
    vehicle_bp = world.get_blueprint_library().filter("vehicle.*")[7]
    spawn_points = world.get_map().get_spawn_points()
    vehicles = []
    for _ in range(number_of_vehicles):
        # spawn_point = carla.Transform(carla.Location(x=57.9, y=-89.6, z=0.275307), carla.Rotation(pitch=0.000000, yaw=180.000000, roll=0.000000))
        #spawn_point = carla.Transform(carla.Location(x=-148.7, y=2.2, z=0.275307))
        spawn_point = carla.Transform(carla.Location(x=100.0, y=2.5, z=0.275307))
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is not None:
            vehicles.append(vehicle)
    return vehicles


def spawn_camera_sensor(vehicle):
    world = vehicle.get_world()
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

    # Set camera parameters (modify these as needed)
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')

    # Adjust camera transform relative to the vehicle (modify as needed)
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))

    # Spawn camera and attach to vehicle
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera_sensor


def spawn_collision_sensor(vehicle):
    world = vehicle.get_world()
    bp = world.get_blueprint_library().find('sensor.other.collision')
    sensor = world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
    return sensor


def on_collision(agent, event):
    collision_actor = event.other_actor
    impulse = event.normal_impulse
    intensity = (impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2) ** 0.5

    # Set the collision flag for the agent
    agent.has_collided = True


class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z

        self.velocity_range = 7.5  # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(10))
        bp.set_attribute('vertical_fov', str(10))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x, z=bound_z, y=bound_y),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._radar_callback(weak_self, radar_data))

    @staticmethod
    def _radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            # get distance
            global DISTANCE_VALUE
            DISTANCE_VALUE = detect.depth


class ObstacleDetectionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.distance = None
        self._event_count = 0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.obstacle')
        bp.set_attribute('distance', '50')
        bp.set_attribute('hit_radius', '5')
        bp.set_attribute('debug_linetrace', 'True')
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
        if event.distance < 20:
            print("Event %s, in line of sight with %s at distance %u" % (
            self._event_count, event.other_actor.type_id, event.distance))
        global DISTANCE_VALUE
        DISTANCE_VALUE = event.distance


# Define a VehicleAgent class for controlling vehicles
class VehicleAgent(Agent):
    def __init__(self, jid, password, vehicle):
        super().__init__(jid, password)
        self.vehicle = vehicle
        self.has_collided = False  # Flag to indicate collision
        self.pid_controller = PIDController(0.1, 0.01, 0.5, 10)
        self.location = None

    class ControlVehicleBehaviour(CyclicBehaviour):
        async def run(self):
            global PICKED_UP, BATTERY_LEVEL
            print("Current location: ", self.agent.vehicle.get_location())
            draw_markers(self.agent.vehicle.get_world())
            next_target = find_suitable_location(TARGETS)
            if BATTERY_LEVEL < 30:
                for elem in TARGETS:
                    if elem.location_type == "charge":
                        next_target = elem
                        break
            if next_target:
                print("Next target: ", next_target.location)
            if self.agent.has_collided:
                # Stop the vehicle in case of collision
                control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
                print("Vehicle has collided!")

            elif next_target and self.arrivedAtTarget(next_target.location):
                # Stop the vehicle in case of collision
                control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
                print("Vehicle has arrived at target!")
                if next_target.location_type == "charge":
                    BATTERY_LEVEL = 60
                    print("Battery level is full")
                    #stop vehicle
                    control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=100.0)
                    self.agent.vehicle.apply_control(control)
                    time.sleep(5)
                    await asyncio.sleep(500)
                else:
                    PICKED_UP = True
                next_target.set_arrived(True)

            elif DISTANCE_VALUE < 15:
                pid_output = self.agent.pid_controller.update(DISTANCE_VALUE)
                print("Distance: ", DISTANCE_VALUE)
                print("PID output: ", pid_output)
                # Use PID output for steering control
                control = carla.VehicleControl()
                control.steer = max(-1.0, min(1.0, pid_output*2))  # Adjust steering
                control.throttle = 0.5  # Set a constant throttle value
                control.brake = 0.0  # Adjust brake if needed

            else:
                if next_target:
                    turn_angle = self.calculateTurnAngle(next_target.location)
                    tabs = math.fabs(turn_angle)
                    print("Yaw: ", self.agent.vehicle.get_transform().rotation.yaw)
                    print("Turn Angle: ", self.calculateTurnAngle(next_target.location))
                else:
                    turn_angle = 0
                    tabs = 0

                control = carla.VehicleControl()
                if tabs > 1:
                    control.steer = turn_angle / 100
                    control.throttle = 0.5
                    control.brake = 0.0
                else:
                    control.steer = 0.0
                    control.throttle = 0.5
                    control.brake = 0.0

            self.agent.vehicle.apply_control(control)

            await asyncio.sleep(0.1)

        def calculateTurnAngle(self, target):
            # Calculate the angle between the target and the vehicle
            location = self.agent.vehicle.get_location()
            print("z: ", location.z)
            angle = math.atan2(target.y - location.y, target.x - location.x)
            angle = math.degrees(angle)
            angle -= self.agent.vehicle.get_transform().rotation.yaw
            angle = math.fmod(angle + 180, 360) - 180

            return angle

        def arrivedAtTarget(self, target):
            # Calculate the angle between the target and the vehicle
            location = self.agent.vehicle.get_location()
            y_diff = math.fabs(target.y - location.y)
            x_diff = math.fabs(target.x - location.x)
            distance = math.sqrt(y_diff * y_diff + x_diff * x_diff)
            if distance < 2:
                return True
            else:
                return False

    async def setup(self):
        self.add_behaviour(self.ControlVehicleBehaviour())


# Define a WalkerAgent class for controlling walkers
class WalkerAgent(Agent):
    def __init__(self, jid, password, walker):
        super().__init__(jid, password)
        self.walker = walker

    class ControlWalkerBehaviour(CyclicBehaviour):
        async def run(self):
            # Example control: walk in a direction
            walker_control = carla.WalkerControl()
            walker_control.speed = 1.2
            walker_control.direction = carla.Vector3D(1, 0, 0)
            self.agent.walker.apply_control(walker_control)
            await asyncio.sleep(1)  # Control frequency

    async def setup(self):
        self.add_behaviour(self.ControlWalkerBehaviour())


# Main function to spawn entities and start agents
async def main():
    global TARGETS, temp_targets
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    client.load_world('Town05')

    battery_thread = threading.Thread(target=battery_control)
    battery_thread.start()

    # TARGET = TargetLocation(carla.Location(x=-124.6, y=-24.9), "take")
    # TARGETS.append(TARGET)
    # #TARGET = TargetLocation(carla.Location(x=-126.3, y=-61.6, z=5), "drop")
    # TARGET = TargetLocation(carla.Location(x=-125.2, y=33.2), "drop")
    # TARGETS.append(TARGET)
    # #TARGET = TargetLocation(carla.Location(x=-83.5, y=2.1), "charge")
    # TARGET = TargetLocation(carla.Location(x=-123.8, y=53.8), "charge")
    # TARGETS.append(TARGET)
    # temp_targets = TARGETS.copy()
    #-95.8 -89.8
    walkers = spawn_walkers(client, 0)
    vehicles = spawn_vehicles(client, 1)
    sensors = []

    # Create and start vehicle agents
    vehicle_agents = []
    for vehicle in vehicles:
        agent = VehicleAgent("vehicle@localhost", "vehicle", vehicle)
        sensor = spawn_collision_sensor(vehicle)
        sensor.listen(lambda event: on_collision(agent, event))
        sensors.append(sensor)

        camera_sensor = spawn_camera_sensor(vehicles[0])
        sensors.append(camera_sensor)

        # obstacle_sensor = ObstacleDetectionSensor(vehicle)
        # sensors.append(obstacle_sensor)

        radar_sensor = RadarSensor(vehicle)
        sensors.append(radar_sensor)
        await agent.start()
        vehicle_agents.append(agent)

    if vehicles:
        # Get the world's spectator object
        spectator = client.get_world().get_spectator()

        # Attach the spectator to the first vehicle
        transform = vehicles[0].get_transform()
        spectator.set_transform(transform)
        print("Camera attached to vehicle")

    # Create and start walker agents
    walker_agents = []
    for walker in walkers:
        agent = WalkerAgent("walker@localhost", "walker", walker)
        await agent.start()
        walker_agents.append(agent)

    try:
        for agent in vehicle_agents:
            await wait_until_finished(agent)
    finally:
        print("Keyboard interrupt received. Exiting...")
        # Stop agents and cleanu

        battery_stop_event.set()
        battery_thread.join()

        for sensor in sensors:
            try:
                print("Destroying sensors.")
                sensor.destroy()
            except Exception as e:
                sensor.sensor.destroy()
        for vehicle in vehicles:
            print("Destroying vehicle {}".format(vehicle.id))
            vehicle.destroy()
        for walker in walkers:
            print("Destroying walker {}".format(walker.id))
            walker.destroy()
        for agent in vehicle_agents:
            print("Stopping agent {}".format(agent.jid))
            await agent.stop()
        for agent in walker_agents:
            print("Stopping agent {}".format(agent.jid))
            await agent.stop()

        print("Simulation finished!")


if __name__ == '__main__':
    spade.run(main())
