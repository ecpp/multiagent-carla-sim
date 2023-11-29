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
    vehicle_bp = random.choice(world.get_blueprint_library().filter("vehicle.*"))
    spawn_points = world.get_map().get_spawn_points()
    vehicles = []
    for _ in range(number_of_vehicles):
        spawn_point = random.choice(spawn_points)
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

class ObstacleDetectionSensor(object):
    def init(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.distance = None
        self._event_count = 0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.obstacle')
        bp.set_attribute('distance','50')
        bp.set_attribute('hit_radius','5')
        bp.set_attribute('debug_linetrace','true')
        bp.set_attribute('only_dynamics', 'true')
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


# Define a VehicleAgent class for controlling vehicles
class VehicleAgent(Agent):
    def __init__(self, jid, password, vehicle):
        super().__init__(jid, password)
        self.vehicle = vehicle
        self.has_collided = False  # Flag to indicate collision

    class ControlVehicleBehaviour(CyclicBehaviour):
        async def run(self):
            if self.agent.has_collided:
                # Stop the vehicle in case of collision
                control = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
                print("Vehicle has collided!")
            else:
                vehicle_state = self.agent.vehicle.get_velocity()
                target_velocity = 50  # You can adjust the target velocity
                velocity_error = target_velocity - vehicle_state.x
                control = carla.VehicleControl()
                if velocity_error > 0:
                    control.throttle = 2.5  # You can adjust the throttle value
                    control.brake = 0.0
                else:
                    control.throttle = 0.0
                    control.brake = 0.2  # You can adjust the brake value

                    # Apply steering (optional)
                control.steer = 0.0  # You can adjust the steering angle if needed

            # Apply the control to the vehicle
            self.agent.vehicle.apply_control(control)

            await asyncio.sleep(1)  # Control frequency Control frequency

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

    walkers = spawn_walkers(client, 1)
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
        obstacle_sensor = ObstacleDetectionSensor()
        obstacle_sensor.init(vehicle)
        sensors.append(obstacle_sensor)
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

    for agent in vehicle_agents:
        await wait_until_finished(agent)

    try:
        # Run simulation for a while
        await asyncio.sleep(15)  # Adjust as needed
    finally:
        print("Keyboard interrupt received. Exiting...")
    # Stop agents and cleanup
        for agent in vehicle_agents:
            print("Stopping agent {}".format(agent.jid))
            await agent.stop()
        for agent in walker_agents:
            print("Stopping agent {}".format(agent.jid))
            await agent.stop()
        for vehicle in vehicles:
            print("Destroying vehicle {}".format(vehicle.id))
            vehicle.destroy()
        for walker in walkers:
            print("Destroying walker {}".format(walker.id))
            walker.destroy()
        for sensor in sensors:
            print("Destroying sensor {}".format(sensor.id))
            sensor.destroy()

    print("Simulation finished!")


if __name__ == '__main__':
    spade.run(main())
