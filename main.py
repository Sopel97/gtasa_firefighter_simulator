import struct
import glob
import random
import itertools
import argparse
import copy
import sys
import time
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.widgets import Button
from sklearn.neighbors import KDTree
from matplotlib import collections as mc
from matplotlib.patches import Circle

from zone import *

NODES_GLOB = './nodes/nodes*.dat'

RADAR_IMAGE_BW = plt.imread('./assets/radar_bw.png')
RADAR_IMAGE_EXTENTS = [-2998, 3002, -2998, 3002]

def dist_3d(x0, y0, z0, x1, y1, z1):
    return ((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)**0.5

def dist_3d_manhattan(x0, y0, z0, x1, y1, z1):
    return abs(x0-x1) + abs(y0-y1) + abs(z0-z1)

def dist_2d_manhattan(x0, y0, x1, y1):
    return abs(x0-x1) + abs(y0-y1)

def dist_3d_max(x0, y0, z0, x1, y1, z1):
    return max([abs(x0-x1), abs(y0-y1), abs(z0-z1)])

def dist_2d_max(x0, y0, x1, y1):
    return max([abs(x0-x1), abs(y0-y1)])

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
        self.path_width = ss.read_next('<B', 1) / 8
        self.flood_fill = ss.read_next('<B', 1)

        flags_stream = BitStream(ss.read_next('<L', 4))

        self.num_links = flags_stream.read_next_int(4)
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
        self.node_links = [[] for i in range(64)]
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

                area_id = 0
                for i in range(num_vehicle_nodes):
                    node = VehicleNode(ss.read_next_bytes(VehicleNode.SIZEOF))
                    assert node.node_id == len(self.vehicle_nodes[node.area_id])
                    self.vehicle_nodes[node.area_id].append(node)
                    area_id = node.area_id

                ss.skip(28 * num_ped_nodes) # not needed

                for i in range(num_navi_nodes):
                    node = NaviNode(ss.read_next_bytes(NaviNode.SIZEOF))
                    self.navi_nodes.append(node)

                for i in range(num_links):
                    link = NodeLink(ss.read_next_bytes(NodeLink.SIZEOF))
                    self.node_links[area_id].append(link)

                ss.skip(768) # unknown section

                ss.skip(num_navi_nodes * 2) # navi links, skip for now

                for i in range(num_navi_nodes):
                    self.link_lengths.append(ss.read_next_bytes(1))

        self.prepare_ff_acceleration()

        self.prepare_graph_for_pathfinding()

    def prepare_graph_for_pathfinding(self):
        self.node_graph = nx.Graph()

        self.node_acceleration_nodes = []
        node_acceleration_coords = []

        # Add nodes first
        for nodes_in_area in self.vehicle_nodes:
            for node in nodes_in_area:
                if node.boats:
                    continue

                self.node_acceleration_nodes.append(node)
                node_acceleration_coords.append([node.x, node.y, node.z])

                self.node_graph.add_node((node.area_id, node.node_id))

        # Only add edges after all nodes are added, just so the semantics are clear.
        for nodes_in_area in self.vehicle_nodes:
            for node in nodes_in_area:
                if node.boats:
                    continue
                for j in range(node.num_links):
                    link = self.node_links[node.area_id][node.link_id + j]
                    neighbour_node = self.vehicle_nodes[link.area_id][link.node_id]
                    dist = dist_3d(neighbour_node.x, neighbour_node.y, neighbour_node.z, node.x, node.y, node.z)
                    # TODO: add some heuristics? for example increase weight for inclines and reduce for declines?
                    self.node_graph.add_edge((node.area_id, node.node_id), (link.area_id, link.node_id), weight=dist)

        # add hardcoded edges for connectivity
        def add_edge(a0, n0, a1, n1):
            node0 = self.vehicle_nodes[a0][n0]
            node1 = self.vehicle_nodes[a1][n1]
            dist = dist_3d(node0.x, node0.y, node0.z, node1.x, node1.y, node1.z)
            self.node_graph.add_edge((a0, n0), (a1, n1), weight=dist)

        add_edge(14, 196, 6, 357) # LS airport entrance
        add_edge(6, 357, 6, 306) # LS airport interconnectivity

        add_edge(38, 696, 38, 526) # on some dumb patch of land in northern LS
        add_edge(38, 627, 38, 574)
        add_edge(38, 640, 38, 524)

        # TODO: what do we do with open areas? how do we connect the nodes on there?
        #       for highways we can maybe define stretches of parallel paths
        #       and connects all nodes within some distance (or % closest) between these paths
        #       but for larger empty grass areas it's a bit of an issue
        #       also, we might need to move to a directed graph if we go for cuts...

        self.node_graph.add_edge((node.area_id, node.node_id), (link.area_id, link.node_id), weight=dist)

        self.node_acceleration = KDTree(node_acceleration_coords)

    def prepare_ff_acceleration(self):
        self.ff_acceleration_nodes = []
        ff_acceleration_coords = []
        ff_acceleration_coords_2d = []
        for area_id in range(len(self.vehicle_nodes)):
            nodes_in_area = self.vehicle_nodes[area_id]
            for i in range(len(nodes_in_area)):
                node = nodes_in_area[i]

                if node.is_disabled or node.boats: # must not be disabled nor for boats
                    continue
                if node.z > 1000.0: # filter out interior nodes (only bloodbowl place? is it actually valid?)
                    continue
                for j in range(node.num_links):
                    link = self.node_links[node.area_id][node.link_id + j]
                    neighbour_node = self.vehicle_nodes[link.area_id][link.node_id]

                    # must have a neighbour that's also not disabled nor for boats
                    if neighbour_node.is_disabled or neighbour_node.boats:
                        continue

                    # and at least 10.0 units of straight road
                    if dist_3d(neighbour_node.x, neighbour_node.y, neighbour_node.z, node.x, node.y, node.z) > 10.0:
                        ff_acceleration_coords.append([node.x, node.y, node.z * 3.0])
                        ff_acceleration_coords_2d.append([node.x, node.y])
                        self.ff_acceleration_nodes.append(node)
                        break

        self.ff_acceleration = KDTree(ff_acceleration_coords, metric='manhattan')
        self.ff_acceleration_2d = KDTree(ff_acceleration_coords_2d, metric='manhattan')

    def get_length_of_path_going_through_points(self, points):
        nodes = []
        length = 0
        for point in points:
            nodes.append(self.find_nearest_node(point[0], point[1], point[2]))

        for source, destination in zip(nodes[:-1], nodes[1:]):
            try:
                length += nx.dijkstra_path_length(self.node_graph, (source.area_id, source.node_id), (destination.area_id, destination.node_id))
            except:
                plot_pathfinding_graph({(source.area_id, source.node_id), (destination.area_id, destination.node_id)})

        return length

    def find_node_for_firefighter_spawn(self, x, y, z, max_dist):
        d, i = self.ff_acceleration.query([[x, y, 3*z]], k=1)
        node = self.ff_acceleration_nodes[i[0][0]]
        dist = abs(node.x - x) + abs(node.y - y) + 3 * abs(node.z - z)
        if dist < max_dist:
            return node
        return None

    def find_node_for_firefighter_spawn_2d(self, x, y, max_dist):
        d, i = self.ff_acceleration_2d.query([[x, y]], k=1)
        node = self.ff_acceleration_nodes[i[0][0]]
        dist = abs(node.x - x) + abs(node.y - y)
        if dist < max_dist:
            return node
        return None

    def find_nearest_node(self, x, y, z):
        d, i = self.node_acceleration.query([[x, y, z]], k=1)
        node = self.node_acceleration_nodes[i[0][0]]
        return node

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

class FirefighterMissionSpawnPoint:
    def __init__(self, x, y, z, num_attempts, num_passengers):
        self.x = x
        self.y = y
        self.z = z
        self.num_attempts = num_attempts
        self.num_passengers = num_passengers

WORLD = World()

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
        if num_attempts < 1000:
            nearest_node = WORLD.find_node_for_firefighter_spawn(xx, yy, z, 500)
        else:
            print(f'WARNING: more than 1000 spawn attempts at ({x}, {y}, {z})')
            nearest_node = WORLD.find_node_for_firefighter_spawn(xx, yy, z, 10000)

        # No node nearby
        if nearest_node is None:
            return None

        # Player too close
        if dist_3d(nearest_node.x, nearest_node.y, nearest_node.z, x, y, z) < 140:
            return None

        if num_attempts < 35 and len(spawns) > 0:
            # Before attempt 35 require the same zone as player or first spawn if already happened
            if not same_zone(nearest_node.x, nearest_node.y, nearest_node.z, spawns[0].x, spawns[0].y, spawns[0].z):
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

class Percentile:
    def __init__(self, pct):
        assert pct >= 0 and pct <= 100
        self.pct = pct

    def __str__(self):
        return f'{self.pct}th%'

    def get_from_array(self, arr):
        idx = int((len(arr)-1) * self.pct / 100)
        return arr[idx]

class HeatMap:
    def __init__(self, min_x, min_y, max_x, max_y, ideal_num_buckets):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.ideal_num_buckets = ideal_num_buckets

        self.width = max_x - min_x
        self.height = max_y - min_y
        self.area = self.width * self.height
        self.area_per_bucket = self.area / self.ideal_num_buckets
        self.bucket_size = self.area_per_bucket**0.5
        self.x_buckets = max(1, round(self.width / self.bucket_size))
        self.y_buckets = max(1, round(self.height / self.bucket_size))
        self.actual_num_buckets = self.x_buckets * self.y_buckets
        self.X, self.Y = np.meshgrid(np.linspace(min_x, max_x, self.x_buckets), np.linspace(min_y, max_y, self.y_buckets))
        self.Zs = None

    def process_in_buckets(self, func_name, func, samples_per_bucket, only_near_nodes=False, combine=['avg']):
        bucket_i = -1 # for some reason gets called twice on the first coord
        start_time = time.time()
        def impl(bx, by):
            nonlocal func
            nonlocal bucket_i
            bucket_i += 1

            bz = 0.0
            if only_near_nodes:
                nearest_node = WORLD.find_node_for_firefighter_spawn_2d(bx, by, 3000.0)
                if nearest_node is None or dist_2d_max(nearest_node.x, nearest_node.y, bx, by) > self.bucket_size * 2.0 + nearest_node.path_width:
                    print(f'Processing bucket: {bucket_i} / {self.actual_num_buckets} (skipped)')
                    return np.asarray([float('nan')] * len(combine))
                bz = nearest_node.z

            curr_time = time.time()
            diff_time = curr_time - start_time
            if bucket_i == 0:
                eta_time = 0
            else:
                eta_time = diff_time * (self.actual_num_buckets / bucket_i)
            print(f'{func_name}: Processing bucket: {bucket_i} / {self.actual_num_buckets}. Elapsed: {int(diff_time)}s, ETA: {int(eta_time-diff_time)}s')

            vs = []
            for i in range(samples_per_bucket):
                rx = random.uniform(-self.bucket_size / 2, self.bucket_size / 2)
                ry = random.uniform(-self.bucket_size / 2, self.bucket_size / 2)
                vs.append(func(bx+rx, by+ry, bz))
            vs.sort()
            results = []
            for comb in combine:
                if isinstance(comb, Percentile):
                    results.append(comb.get_from_array(vs))
                elif comb == 'avg' or comb == 'prob':
                    results.append(sum(vs) / len(vs))
                else:
                    assert False
            return np.asarray(results)

        Zs = np.frompyfunc(impl, 2, 1)(self.X, self.Y)
        Zs = np.array([e.tolist() for e in Zs.flatten()]).reshape(Zs.shape[0], Zs.shape[1], -1)
        Zs = np.dsplit(Zs, len(combine))
        self.Zs = [np.squeeze(Z) for Z in Zs]
        self.combine = combine

    def smoothen(self, factor):
        # TODO: this. scipy interp2d doesn't work very well, and completely breaks with nans
        pass

    def draw(self, i, fig, ax, *args, **kwargs):
        assert self.Zs[i] is not None
        c = ax.pcolormesh(self.X, self.Y, self.Zs[i], *args, **kwargs)
        fig.colorbar(c, ax=ax)

    def drawables(self):
        return enumerate(self.combine)

class SimulationArea:
    def __init__(self, min_x, min_y, max_x, max_y, ideal_num_buckets, samples_per_bucket, only_near_nodes=True):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.ideal_num_buckets = ideal_num_buckets
        self.samples_per_bucket = samples_per_bucket
        self.only_near_nodes = only_near_nodes

class ShowParams:
    def __init__(self, width_inches=9, height_inches=9, filename=None, dpi=300, title=None):
        self.width_inches = width_inches
        self.height_inches = height_inches
        self.filename = filename
        self.dpi = dpi
        self.title = title

    def with_title(self, title):
        cpy = copy.deepcopy(self)
        cpy.title = title
        return cpy

def add_combine_type_to_filename(filename, comb):
    parts = filename.split('.')
    suffix = ''
    if isinstance(comb, Percentile):
        suffix = f'{comb.pct}pct'
    elif comb == 'avg':
        suffix = 'avg'
    elif comb == 'prob':
        suffix = 'prob'
    else:
        assert False
    return ''.join(parts[:-1] + ['_', suffix]) + '.' + parts[-1]

def add_combine_type_to_title(title, comb):
    prefix = ''
    if isinstance(comb, Percentile):
        prefix = str(comb)
    elif comb == 'avg':
        prefix = 'Average'
    elif comb == 'prob':
        prefix = 'Probability'
    else:
        assert False
    return prefix + ' ' + title

def show_heatmap(H, show_params, *args, **kwargs):
    for i, comb in H.drawables():
        plt.rcParams["figure.figsize"] = [show_params.width_inches, show_params.height_inches]
        fig, ax = plt.subplots()
        im = ax.imshow(RADAR_IMAGE_BW, extent=RADAR_IMAGE_EXTENTS)
        draw_zones(ax)

        H.draw(i, fig, ax, *args, **kwargs)

        if show_params.title:
            fig.suptitle(add_combine_type_to_title(show_params.title, comb))

        if show_params.filename:
            plt.savefig(add_combine_type_to_filename(show_params.filename, comb), dpi=show_params.dpi)
        else:
            plt.show()

def make_and_show_heatmap(func_name, func, area, show_params, combine, *args, **kwargs):
    H = HeatMap(area.min_x, area.min_y, area.max_x, area.max_y, area.ideal_num_buckets)
    H.process_in_buckets(func_name, func, area.samples_per_bucket, area.only_near_nodes, combine=combine)
    show_heatmap(H, show_params, *args, **kwargs)

def visualize_buckets(area, show_params):
    @np.vectorize
    def sample_func(x, y, z):
        return random.random()

    make_and_show_heatmap('visualize_buckets', sample_func, area, show_params.with_title(f'Random noise'), ['avg'], cmap='jet', alpha=0.75)

COMBS = ['avg', Percentile(0), Percentile(25), Percentile(50), Percentile(75), Percentile(100)]

def plot_average_distance_to_farthest_spawn(ff, level, area, show_params):
    @np.vectorize
    def sample_func(x, y, z):
        spawns = ff.generate_level(level, x, y, z)
        return max(dist_3d(s.x, s.y, s.z, x, y, z) for s in spawns)

    make_and_show_heatmap('plot_average_distance_to_farthest_spawn', sample_func, area, show_params.with_title(f'distance (straight line) to farthest spawn on level {level}'), COMBS, cmap='jet', alpha=0.75)

def plot_average_distance_between_spawns(ff, level, area, show_params):
    @np.vectorize
    def sample_func(x, y, z):
        ds = []
        spawns = ff.generate_level(level, x, y, z)
        for i in range(len(spawns)):
            for j in range(i + 1, len(spawns)):
                s0 = spawns[i]
                s1 = spawns[j]
                d = dist_3d(s0.x, s0.y, s0.z, s1.x, s1.y, s1.z)
                ds.append(d)
        return sum(ds) / len(ds)

    make_and_show_heatmap('plot_average_distance_between_spawns', sample_func, area, show_params.with_title(f'distance (straight line) between spawns on level {level}'), COMBS, cmap='jet', alpha=0.75)

def plot_distance_to_closest_road(area, show_params):
    def sample_func(x, y, z):
        nearest_node = WORLD.find_node_for_firefighter_spawn_2d(x, y, 3000.0)
        return dist_2d_max(nearest_node.x, nearest_node.y, x, y)

    make_and_show_heatmap('plot_distance_to_closest_road', sample_func, area, show_params, cmap='jet', alpha=0.75)

def plot_probability_of_multizone_split(ff, level, area, show_params):
    def sample_func(x, y, z):
        spawns = ff.generate_level(level, x, y, z)
        for spawn in spawns[1:]: # using all spawns would also include spawns outside of player's zone
            if not same_zone(spawn.x, spawn.y, spawn.z, spawns[0].x, spawns[0].y, spawns[0].z):
                return 1
        return 0

    make_and_show_heatmap('plot_probability_of_multizone_split', sample_func, area, show_params.with_title(f'of spawns splitting zones on level {level}'), ['prob'], cmap='jet', alpha=0.75)

def plot_average_total_firefighter_distance(ff, start_level, area, show_params):
    @np.vectorize
    def sample_func(x, y, z):
        ds = []
        d = 0.0
        for level in range(start_level, 13):
            spawns = ff.generate_level(level, x, y, z)
            # we order spawns from farthest to nearest to the start position
            # this is not the ideal heuristics, but fairly close to what actually happens
            # TODO: TSP and actual road distance? needs pathfinding
            ordered_spawns = sorted(spawns, key=lambda s: -dist_3d(s.x, s.y, s.z, x, y, z))
            for s in ordered_spawns:
                d += dist_3d_manhattan(x, y, z, s.x, s.y, s.z)
                x = s.x
                y = s.y
                z = s.z
        ds.append(d)
        return sum(ds) / len(ds)

    make_and_show_heatmap('plot_average_total_firefighter_distance', sample_func, area, show_params.with_title(f'total firefighter distance (straight line, order by heuristic) starting at level {start_level}'), COMBS, cmap='jet', alpha=0.75)

def plot_probability_that_firefighter_stays_on_coast(ff, start_level, area, show_params):
    @np.vectorize
    def sample_func(x, y, z):
        for level in range(start_level, 13):
            spawns = ff.generate_level(level, x, y, z)
            # we order spawns from farthest to nearest to the start position
            # this is not the ideal heuristics, but fairly close to what actually happens
            # TODO: TSP and actual road distance? needs pathfinding
            ordered_spawns = sorted(spawns, key=lambda s: -dist_3d(s.x, s.y, s.z, x, y, z))
            for s in ordered_spawns:
                x = s.x
                y = s.y
                z = s.z
                if s.y > -100 or s.x < 2650 or (s.x < 2800 and s.y < -600):
                    return 0

        return 1

    make_and_show_heatmap('plot_probability_that_firefighter_stays_on_coast', sample_func, area, show_params.with_title(f'that firefighter stays on coast (LS beach) starting at level {start_level}'), ['prob'], cmap='jet', alpha=0.75)

def plot_average_distance_to_complete_and_drive_to_cj_house(ff, level, area, show_params):
    def sample_func(x, y, z):
        cj_x = 2490
        cj_y = -1690
        spawns = ff.generate_level(level, x, y, z)
        locs = [(s.x, s.y, s.z) for s in spawns]
        dists = []
        # find best permutation, we can do it by brute force
        for permuted_locs in itertools.permutations(locs):
            xx = x
            yy = y
            zz = z
            d = 0.0
            for s in permuted_locs + ((cj_x, cj_y, 20.0),):
                d += dist_3d_manhattan(xx, yy, zz, s[0], s[1], s[2])
                xx = s[0]
                yy = s[1]
                zz = s[2]
            dists.append(d)
        return min(dists)

    make_and_show_heatmap('plot_average_distance_to_complete_and_drive_to_cj_house', sample_func, area, show_params.with_title(f'total [firefighter+drive to CJ\'s home] distance (manhattan, order by heuristic) on level {level}'), COMBS, cmap='jet', alpha=0.75)

def plot_average_distance_to_complete_and_drive_to_cj_house_2(ff, level, area, show_params):
    def sample_func(x, y, z):
        cj_x = 2490
        cj_y = -1690
        spawns = ff.generate_level(level, x, y, z)
        locs = [(s.x, s.y, s.z) for s in spawns]
        dists = []
        # find best permutation, we can do it by brute force
        for permuted_locs in itertools.permutations(locs):
            d = WORLD.get_length_of_path_going_through_points(((x, y, z),) + permuted_locs + ((cj_x, cj_y, 20.0),))
            dists.append(d)
        return min(dists)

    make_and_show_heatmap('plot_average_distance_to_complete_and_drive_to_cj_house_2', sample_func, area, show_params.with_title(f'total [firefighter+drive to CJ\'s home] distance (road, order by heuristic) on level {level}'), COMBS, cmap='jet', alpha=0.75)

def plot_valid_ff_spawns():
    X = []
    Y = []
    for node in WORLD.ff_acceleration_nodes:
        X.append(node.x)
        Y.append(node.y)

    plt.rcParams["figure.figsize"] = [9, 9]
    fig, ax = plt.subplots()
    im = ax.imshow(RADAR_IMAGE_BW, extent=RADAR_IMAGE_EXTENTS)
    draw_zones(ax)

    c = ax.scatter(X, Y)

    plt.show()

def plot_pathfinding_graph(highlight_nodes=set()):
    circles = []
    for area_id, node_id in WORLD.node_graph.nodes():
        node = WORLD.vehicle_nodes[area_id][node_id]
        do_highlight = (area_id, node_id) in highlight_nodes
        if do_highlight:
            print(node.x, node.y, node.z, area_id, node_id)
        color = 'r' if do_highlight else 'b'
        radius = 20 if do_highlight else 5
        label = f'({area_id},{node_id})'
        circles.append(Circle((node.x, node.y), radius, color=color, label=label))

    lines = []
    for (area_id0, node_id0), (area_id1, node_id1) in WORLD.node_graph.edges():
        node0 = WORLD.vehicle_nodes[area_id0][node_id0]
        node1 = WORLD.vehicle_nodes[area_id1][node_id1]
        lines.append([(node0.x, node0.y), (node1.x, node1.y)])

    plt.rcParams["figure.figsize"] = [9, 9]
    fig, ax = plt.subplots()

    im = ax.imshow(RADAR_IMAGE_BW, extent=RADAR_IMAGE_EXTENTS)
    draw_zones(ax)

    lc = mc.PatchCollection(circles)
    ax.add_collection(lc)

    lc = mc.LineCollection(lines, linewidths=2)
    ax.add_collection(lc)

    def on_plot_hover(event):
        x = event.xdata
        y = event.ydata
        node = WORLD.find_nearest_node(x, y, 20.0)
        print(node.x, node.y, node.z, node.area_id, node.node_id)

    #fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)

    plt.show()

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

'''
RADAR_IMAGE = im = plt.imread('./assets/radar.png')

plt.rcParams["figure.figsize"] = [9, 9]
fig, ax = plt.subplots()
im = ax.imshow(im, extent=RADAR_IMAGE_EXTENTS)
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
'''

#plot_valid_ff_spawns()

#plot_pathfinding_graph()

#ff = FirefighterMission(0)
#plot_average_distance_to_farthest_spawn(ff, 1, 2700.0, -1200.0, 3000.0, -600.0, 128, 1)
#plot_probability_of_multizone_split(ff, 12, 2700.0, -1200.0, 3000.0, -600.0, 256, 32, True)
#plot_probability_of_multizone_split(ff, 12, 2500.0, -1900.0, 2950.0, 200.0, 1024, 4)
#plot_average_distance_between_spawns(ff, 12, 2500.0, -1900.0, 2950.0, 200.0, 2048, 64)
#plot_average_total_firefighter_distance(ff, 2500.0, -1900.0, 2950.0, 200.0, 1024, 16, True)
#plot_probability_that_firefighter_stays_on_coast(ff, 2750.0, -1200.0, 2950.0, -500.0, 512, 200, True)
#plot_distance_to_closest_road()
#plot_average_distance_to_complete_and_drive_to_cj_house_2(ff, 12, 1700.0, -2300.0, 2950.0, 400.0, 2048, 8, True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def cli_add_ff_arguments(parser):
    parser.add_argument('--num_unlocked_cities', type=int, default=0)

def cli_add_area_arguments(parser):
    parser.add_argument('--min_x', type=float, default=2000)
    parser.add_argument('--min_y', type=float, default=-2200)
    parser.add_argument('--max_x', type=float, default=2950)
    parser.add_argument('--max_y', type=float, default=400)
    parser.add_argument('--ideal_num_buckets', type=int, default=32*1024)
    parser.add_argument('--samples_per_bucket', type=int, default=32)
    parser.add_argument('--only_near_nodes', type=str2bool, default=True)

def cli_add_show_params_arguments(parser):
    parser.add_argument('--width_inches', type=float, default=9)
    parser.add_argument('--height_inches', type=float, default=9)
    parser.add_argument('--filename', type=str, default=None)
    parser.add_argument('--dpi', type=int, default=300)

def add_plot_command(subparsers, plot_func, cmd, lvl_param_name='level'):
    def handler_func(args):
        area = SimulationArea(args.min_x, args.min_y, args.max_x, args.max_y, args.ideal_num_buckets, args.samples_per_bucket, args.only_near_nodes)
        show_params = ShowParams(args.width_inches, args.height_inches, args.filename, args.dpi)
        num_unlocked_cities = args.num_unlocked_cities
        level = vars(args).get(lvl_param_name)
        ff = FirefighterMission(num_unlocked_cities)
        plot_func(ff, level, area, show_params)

    subparser = subparsers.add_parser(cmd)
    subparser.set_defaults(func=handler_func)

    cli_add_ff_arguments(subparser)
    cli_add_area_arguments(subparser)
    cli_add_show_params_arguments(subparser)
    subparser.add_argument(f'--{lvl_param_name}', type=int, default=1)

def add_bucket_visualization_command(subparsers, name):
    def handler_func(args):
        area = SimulationArea(args.min_x, args.min_y, args.max_x, args.max_y, args.ideal_num_buckets, args.samples_per_bucket, args.only_near_nodes)
        show_params = ShowParams(args.width_inches, args.height_inches, args.filename, args.dpi)
        visualize_buckets(area, show_params)

    subparser = subparsers.add_parser(name)
    subparser.set_defaults(func=handler_func)

    cli_add_area_arguments(subparser)
    cli_add_show_params_arguments(subparser)

def add_commands(subparsers):
    add_plot_command(subparsers, plot_average_distance_to_farthest_spawn, 'plot_average_distance_to_farthest_spawn')
    add_plot_command(subparsers, plot_average_distance_between_spawns, 'plot_average_distance_between_spawns')
    add_plot_command(subparsers, plot_probability_of_multizone_split, 'plot_probability_of_multizone_split')
    add_plot_command(subparsers, plot_average_distance_to_complete_and_drive_to_cj_house, 'plot_average_distance_to_complete_and_drive_to_cj_house')
    add_plot_command(subparsers, plot_average_distance_to_complete_and_drive_to_cj_house_2, 'plot_average_distance_to_complete_and_drive_to_cj_house_2')

    add_plot_command(subparsers, plot_average_total_firefighter_distance, 'plot_average_total_firefighter_distance', 'start_level')
    add_plot_command(subparsers, plot_probability_that_firefighter_stays_on_coast, 'plot_probability_that_firefighter_stays_on_coast', 'start_level')

    add_bucket_visualization_command(subparsers, 'visualize_buckets')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    add_commands(subparsers)
    args = parser.parse_args()
    if args.func:
        args.func(args)
