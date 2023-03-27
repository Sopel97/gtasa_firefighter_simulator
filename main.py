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

class World:
    def __init__(self):
        self.vehicle_nodes = []
        for filename in glob.glob(NODES_GLOB):
            with open(filename, 'rb') as file:
                data = file.read()
                self.vehicle_nodes += self.extract_vehicle_nodes(data)
        print(f'Loaded {len(self.vehicle_nodes)} vehicle nodes.')

    def extract_vehicle_nodes(self, nodes_data):
        ss = StructStream(nodes_data)

        num_nodes = ss.read_next('<L', 4)
        num_vehicle_nodes = ss.read_next('<L', 4)
        num_ped_nodes = ss.read_next('<L', 4)
        num_navi_nodes = ss.read_next('<L', 4)
        num_links = ss.read_next('<L', 4)

        vehicle_nodes = []
        for i in range(num_vehicle_nodes):
            vehicle_nodes.append(VehicleNode(ss.read_next_bytes(VehicleNode.SIZEOF)))
        return vehicle_nodes

WORLD = World()
