import struct
import glob
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from sklearn.neighbors import KDTree

from zone import *

NODES_GLOB = './nodes/nodes*.dat'

class StructStream:
    def __init__(self, data):
        self.data = data
        self.ptr = 0

    def read_next(self, fmt, length):
        # TODO: use calcsize for length
        v = struct.unpack(fmt, self.data[self.ptr:self.ptr+length])[0]
        self.ptr += length
        return v

    def read_next_bytes(self, length):
        v = self.data[self.ptr:self.ptr+length]
        self.ptr += length
        return v

    def skip(self, length):
        self.ptr += length

class BitStream:
    def __init__(self, v):
        self.value = v

    def read_next_int(self, num_bits):
        v = self.value & ((1 << num_bits) - 1)
        self.value >>= num_bits
        return v

    def read_next_bool(self):
        v = self.value & 1
        self.value >>= 1
        return True if v == 1 else False

    def skip(self, num_bits):
        self.value >>= num_bits

class VehicleNode:
    SIZEOF = 28
    def __init__(self, data):
        assert len(data) == VehicleNode.SIZEOF

        ss = StructStream(data)

        self.mem_address = ss.read_next('<L', 4)
        always_zero = ss.read_next('<L', 4)
        assert(always_zero == 0)

        self.x = ss.read_next('<h', 2) / 8
        self.y = ss.read_next('<h', 2) / 8
        self.z = ss.read_next('<h', 2) / 8
        self.heuristic_cost = ss.read_next('<h', 2)
        assert(self.heuristic_cost == 0x7FFE)

        self.link_id = ss.read_next('<H', 2)
        self.area_id = ss.read_next('<H', 2)
        self.node_id = ss.read_next('<H', 2)
        self.path_width = ss.read_next('<B', 1)
        self.flood_fill = ss.read_next('<B', 1)

        flags_stream = BitStream(ss.read_next('<L', 4))

        self.link_count = flags_stream.read_next_int(4)
        self.dead_end = flags_stream.read_next_bool()
        self.is_disabled = flags_stream.read_next_bool()
        self.roadblocks = flags_stream.read_next_bool()
        self.boats = flags_stream.read_next_bool()
        self.emergency_vehicles_only = flags_stream.read_next_bool()
        zero = flags_stream.read_next_int(1)
        assert(zero == 0)

        self.dont_wander = flags_stream.read_next_bool()

        zero = flags_stream.read_next_int(1)
        assert(zero == 0)

        self.is_not_highway = flags_stream.read_next_bool()
        self.is_highway = flags_stream.read_next_bool()

        zero = flags_stream.read_next_int(2)
        assert(zero == 0)

        self.spawn_probability = flags_stream.read_next_int(4)

        unknown = flags_stream.read_next_int(4)

class NaviNode:
    SIZEOF = 14
    def __init__(self, data):
        assert len(data) == NaviNode.SIZEOF

        ss = StructStream(data)

        self.x = ss.read_next('<h', 2) / 8
        self.y = ss.read_next('<h', 2) / 8

        self.area_id = ss.read_next('<H', 2)
        self.node_id = ss.read_next('<H', 2)

        self.dir_x = ss.read_next('<B', 1)
        self.dir_y = ss.read_next('<B', 1)

        flags_stream = BitStream(ss.read_next('<L', 4))

        self.path_node_width = flags_stream.read_next_int(8)
        self.num_left_lanes = flags_stream.read_next_int(3)
        self.num_right_lanes = flags_stream.read_next_int(3)
        self.same_direction_as_traffic_lights = flags_stream.read_next_bool()
        zero = flags_stream.read_next_int(1)
        assert zero == 0

        self.traffic_light_behaviour = flags_stream.read_next_int(2)

        self.train_crossing = flags_stream.read_next_bool()

class NodeLink:
    SIZEOF = 4
    def __init__(self, data):
        assert len(data) == NodeLink.SIZEOF

        ss = StructStream(data)

        self.area_id = ss.read_next('<H', 2)
        self.node_id = ss.read_next('<H', 2)

class World:
    def __init__(self):
        self.vehicle_nodes = [[] for i in range(64)]
        self.navi_nodes = []
        self.node_links = []
        self.link_lengths = []

        for filename in glob.glob(NODES_GLOB):
            with open(filename, 'rb') as file:
                data = file.read()
                ss = StructStream(data)

                num_nodes = ss.read_next('<L', 4)
                num_vehicle_nodes = ss.read_next('<L', 4)
                num_ped_nodes = ss.read_next('<L', 4)
                num_navi_nodes = ss.read_next('<L', 4)
                num_links = ss.read_next('<L', 4)

                for i in range(num_vehicle_nodes):
                    node = VehicleNode(ss.read_next_bytes(VehicleNode.SIZEOF))
                    assert node.node_id == len(self.vehicle_nodes[node.area_id])
                    self.vehicle_nodes[node.area_id].append(node)

                ss.skip(28 * num_ped_nodes) # not needed

                for i in range(num_navi_nodes):
                    node = NaviNode(ss.read_next_bytes(NaviNode.SIZEOF))
                    self.navi_nodes.append(node)

                for i in range(num_links):
                    link = NodeLink(ss.read_next_bytes(NodeLink.SIZEOF))
                    self.node_links.append(link)

                ss.skip(768) # unknown section

                ss.skip(num_navi_nodes * 2) # navi links, skip for now

                for i in range(num_navi_nodes):
                    self.link_lengths.append(ss.read_next_bytes(1))

        self.ff_acceleration_nodes = []
        ff_acceleration_coords = []
        for nodes in self.vehicle_nodes:
            for node in nodes:
                if node.dead_end or node.is_disabled or node.boats:
                    continue
                ff_acceleration_coords.append([node.x, node.y, node.z * 3.0])
                self.ff_acceleration_nodes.append(node)
        self.ff_acceleration = KDTree(ff_acceleration_coords, metric='manhattan')

    def find_node_for_firefighter_spawn(self, x, y, z, max_dist):
        d, i = self.ff_acceleration.query([[x, y, z]], k=1)
        node = self.ff_acceleration_nodes[i[0][0]]
        dist = abs(node.x - x) + abs(node.y - y) + 3 * abs(node.z - z)
        if dist < max_dist:
            return node
        return None

    def find_vehicle_node_closest_to_coords(self, x, y, z, max_dist, boats=False, allow_dead_ends=False, allow_disabled=False):
        closest_node = None
        closest_dist = max_dist

        for nodes in self.vehicle_nodes:
            for node in nodes:
                if node.boats != boats:
                    continue

                if not allow_dead_ends and node.dead_end:
                    continue

                if not allow_disabled and node.is_disabled:
                    continue

                dist = abs(node.x - x) + abs(node.y - y) + 3 * abs(node.z - z)
                if dist < closest_dist:
                    closest_node = node
                    closest_dist = dist

        return closest_node

    def find_vehicle_node_pair_closest_to_coords(self, x, y, z, min_dist, max_dist, ignore_disabled, ignore_between_levels, water_path):
        # TODO: Match behaviour in SA. It's unclear how this function works exactly because
        #       paths are encoded differently than in VC.
        pass

class FirefighterMissionSpawnPoint:
    def __init__(self, x, y, z, num_attempts, num_passengers):
        self.x = x
        self.y = y
        self.z = z
        self.num_attempts = num_attempts
        self.num_passengers = num_passengers

WORLD = World()

def dist_3d(x0, y0, z0, x1, y1, z1):
    return ((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)**0.5

class FirefighterMission:
    def __init__(self, num_unlocked_cities=0):
        self.num_unlocked_cities = num_unlocked_cities
        pass

    def _try_generate_spawn_spot(self, spawns, level, num_attempts, x, y, z):
        rand_radius = max(170, level * 60 + num_attempts)
        min_x = x - rand_radius
        max_x = x + rand_radius
        min_y = y - rand_radius
        max_y = y + rand_radius
        xx = random.uniform(min_x, max_x)
        yy = random.uniform(min_y, max_y)
        nearest_node = WORLD.find_node_for_firefighter_spawn(xx, yy, z, 500)

        # No node nearby
        if nearest_node is None:
            return None

        # Player too close
        if dist_3d(nearest_node.x, nearest_node.y, nearest_node.z, x, y, z) < 140:
            return None

        if num_attempts < 35:
            # Before attempt 35 require the same zone as player or first spawn if already happened
            if len(spawns) > 0:
                if not same_zone(nearest_node.x, nearest_node.y, nearest_node.z, spawns[0].x, spawns[0].y, spawns[0].z):
                    return None
            else:
                if not same_zone(nearest_node.x, nearest_node.y, nearest_node.z, x, y, z):
                    return None

        # Generate only in unlocked cities
        if self.num_unlocked_cities == 0:
            if nearest_node.x < 78.4427 and nearest_node.y < -699.519:
                return None
            if nearest_node.x < -252.6557 and nearest_node.y < -285.766:
                return None
            if nearest_node.x < -948.3447:
                return None
            if nearest_node.x > 1473.448 and nearest_node.y > 403.7353:
                return None
            if nearest_node.y > 578.6325:
                return None
            if nearest_node.x < 837.5551 and nearest_node.y > 347.4097:
                return None
        elif self.num_unlocked_cities == 1:
            if nearest_node.x < 1473.448 and nearest_node.y < 403.7353:
                return None
            if nearest_node.x < -1528.498 and nearest_node.y < 578.6325:
                return None
            if nearest_node.x < 837.5551 and nearest_node.x > -1528.498 and nearest_node.y > 347.4097:
                return None
            if nearest_node.y > 1380.0:
                return None
            if nearest_node.x < 2150.0 and nearest_node.x > 1970.0 and nearest_node.y < -2274.0 and nearest_node.y > -2670.0:
                return None
            if nearest_node.x < 2150.0 and nearest_node.x > 1590.0 and nearest_node.y < -2397.0 and nearest_node.y > -2670.0:
                return None
            if nearest_node.x < -1070.0 and nearest_node.x > -1737.0 and nearest_node.y < -185.0 and nearest_node.y > -590.0:
                return None
            if nearest_node.x < -1081.0 and nearest_node.x > -1600.0 and nearest_node.y < 415.0 and nearest_node.y > -185.0:
                return None
            if nearest_node.x < 1733.0 and nearest_node.x > 1500.0 and nearest_node.y < 1702.0 and nearest_node.y > 1529.0:
                return None

        # Distance to previous spawns
        for spawn in spawns:
            if dist_3d(nearest_node.x, nearest_node.y, nearest_node.z, spawn.x, spawn.y, spawn.z) < 20:
                return None

        return nearest_node

    def generate_level(self, level, x, y, z):
        assert level >= 1 and level <= 12
        spawns = []

        num_cars = (level + 3) // 4
        num_passengers = (level + 3) % 4

        for i in range(num_cars):
            num_attempts = 0
            while True:
                num_attempts += 1
                spawn_spot = self._try_generate_spawn_spot(spawns, level, num_attempts, x, y, z)

                if spawn_spot is not None:
                    spawns.append(FirefighterMissionSpawnPoint(spawn_spot.x, spawn_spot.y, spawn_spot.z, num_attempts, num_passengers))
                    break

        return spawns

'''
ff = FirefighterMission(0)
d = []
for i in range(1000):
    print(i)
    spawns = ff.generate_level(12, 2865.0, -798.0, 20.0)
    d0 = dist_3d(spawns[0].x, spawns[0].y, spawns[0].z, spawns[1].x, spawns[1].y, spawns[1].z)
    d1 = dist_3d(spawns[1].x, spawns[1].y, spawns[1].z, spawns[2].x, spawns[2].y, spawns[2].z)
    d2 = dist_3d(spawns[2].x, spawns[2].y, spawns[2].z, spawns[0].x, spawns[0].y, spawns[0].z)
    d.append(d0+d1+d2 - max(d0, d1, d2))
print('Min: ', min(d))
print('Max: ', max(d))
print('Avg: ', sum(d) / len(d))
'''

RADAR_IMAGE = im = plt.imread('./assets/radar.png')

plt.rcParams["figure.figsize"] = [9, 9]
fig, ax = plt.subplots()
im = ax.imshow(im, extent=[-3000, 3000, -3000, 3000])
sp, = ax.plot([], [], label='', ms=10, color='r', marker='o', ls='')

def generate_next(event):
    ff = FirefighterMission(0)
    spawns = ff.generate_level(12, 2865.0, -798.0, 20.0)
    for spawn in spawns:
        print(spawn.x, spawn.y, spawn.z, spawn.num_attempts, get_zone(spawn.x, spawn.y, spawn.z).name)

    x = []
    y = []
    for spawn in spawns:
        x.append(spawn.x)
        y.append(spawn.y)

    sp.set_data(x, y)
    fig.canvas.draw()

ax_button = fig.add_axes([0.9, 0.0, 0.1, 0.075])
bnext = Button(ax_button, 'GENERATE')
bnext.on_clicked(generate_next)

plt.show()