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

DISTANCE_VALUE = 50

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
        spawn_point = carla.Transform(carla.Location(x=57.9, y=-89.6, z=0.275307), carla.Rotation(pitch=0.000000, yaw=180.000000, roll=0.000000))

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
        bound_x = self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = self._parent.bounding_box.extent.z

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(5))
        bp.set_attribute('vertical_fov', str(5))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05, y=bound_y),
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

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))
            #get distance
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
        if event.distance < 20:
            print ("Event %s, in line of sight with %s at distance %u" % (self._event_count, event.other_actor.type_id, event.distance))
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
            if self.agent.has_collided:
                # Stop the vehicle in case of collision
                control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
                print("Vehicle has collided!")
            elif DISTANCE_VALUE < 20:
                pid_output = self.agent.pid_controller.update(DISTANCE_VALUE)
                print("Distance: ", DISTANCE_VALUE)
                print("PID output: ", pid_output)
                # Use PID output for steering control
                control = carla.VehicleControl()
                control.steer = max(-1.0, min(1.0, pid_output))  # Adjust steering
                control.throttle = 0.5  # Set a constant throttle value
                control.brake = 0.0  # Adjust brake if needed

            else:
                print("STRAIGHT Distance: ", DISTANCE_VALUE)
                control = carla.VehicleControl(throttle=0.5, steer=0.0, brake=0.0)
            self.agent.vehicle.apply_control(control)
            print("Location: ", self.agent.vehicle.get_location())
            await asyncio.sleep(0.1)

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
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    client.load_world('Town05')
    print(client.get_world().get_map())
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
    # Stop agents and cleanup
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
