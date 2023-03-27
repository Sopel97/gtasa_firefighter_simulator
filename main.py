import struct
import glob

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
        self.traffic_level = flags_stream.read_next_int(2)
        self.roadblocks = flags_stream.read_next_bool()
        self.boats = flags_stream.read_next_bool()
        self.emergency_vehicles_only = flags_stream.read_next_bool()
        zero = flags_stream.read_next_int(1)
        assert(zero == 0)

        unknown = flags_stream.read_next_bool()

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

WORLD = World()
