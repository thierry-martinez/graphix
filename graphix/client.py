import dataclasses
import numpy as np

from graphix.clifford import CLIFFORD_CONJ, CLIFFORD, CLIFFORD_MUL
import graphix.ops
import graphix.sim.base_backend
import graphix.sim.statevec
import graphix.simulator

"""
Usage:

client = Client(pattern:Pattern, blind=False) ## For pure MBQC
sv_backend = StatevecBackend(client.pattern, meas_op = client.meas_op)

simulator = PatternSimulator(client.pattern, backend=sv_backend)
simulator.run()

"""

@dataclasses.dataclass
class MeasureParameters:
    plane: graphix.pauli.Plane
    angle: float
    s_domain: list[int]
    t_domain: list[int]
    vop: int

class Client:
    def __init__(self, pattern, blind=False, secrets={}):
        self.pattern = pattern

        """
        Database containing the "measurement configuration"
        - Node
        - Measurement parameters : plane, angle, X and Z dependencies
        - Measurement outcome
        """
        self.results = pattern.results.copy()
        self.measure_method = ClientMeasureMethod(self)
        self.measurement_db = {}
        self.init_measurement_db()
        self.byproduct_db = {}
        self.init_byproduct_db()
        self.backend_results = {}
        self.blind = blind
        # By default, no secrets
        self.r_secret = False
        self.secrets = {}
        # Initialize the secrets
        self.init_secrets(secrets)

    def init_secrets(self, secrets):
        if self.blind:
            if 'r' in secrets:
                # User wants to add an r-secret, either customized or generated on the fly
                self.r_secret = True
                self.secrets['r'] = {}
                r_size = len(secrets['r'].keys())
                # If the client entered an empty secret (need to generate it)
                if r_size == 0:
                    # Need to generate the bit for each measured qubit
                    for node in self.measurement_db:
                        self.secrets['r'][node] = np.random.randint(0, 2)

                # If the client entered a customized secret : need to check its validity
                elif self.is_valid_secret('r', secrets['r']):
                    self.secrets['r'] = secrets['r']
                    # TODO : add more rigorous test of the r-secret format
                else:
                    raise ValueError("`r` has wrong format.")
            # TODO : handle secrets `a`, `theta`

    def init_measurement_db(self):
        """
        Initializes the "database of measurement configurations and results" held by the customer
        according to the pattern desired
        Initial measurement outcomes are set to 0
        """
        for cmd in self.pattern:
            if cmd[0] == 'M':
                node = cmd[1]
                plane = graphix.pauli.Plane[cmd[2]]
                angle = cmd[3] * np.pi
                s_domain = cmd[4]
                t_domain = cmd[5]
                if len(cmd) == 7:
                    vop = cmd[6]
                else:
                    vop = 0
                self.measurement_db[node] = MeasureParameters(plane, angle, s_domain, t_domain, vop)
                # Erase the unnecessary items from the command to make sure they don't appear on the server's side
                del cmd[2:]

    def simulate_pattern(self):
        backend = graphix.sim.statevec.StatevectorBackend(pattern=self.pattern, measure_method=self.measure_method)
        sim = graphix.simulator.PatternSimulator(backend=backend, pattern=self.pattern)
        state = sim.run()
        self.backend_results = backend.results
        self.decode_output_state(backend, state)
        return state


    def decode_output_state(self, backend, state):
        if self.blind:
            for node in self.pattern.output_nodes:
                if node in self.byproduct_db:
                    z_decoding, x_decoding = self.decode_output(node)
                    if z_decoding:
                        state.evolve_single(op=graphix.ops.Ops.z, i=backend.node_index.index(node))
                    if x_decoding:
                        state.evolve_single(op=graphix.ops.Ops.x, i=backend.node_index.index(node))

    def get_secrets_size(self):
        secrets_size = {}
        for secret in self.secrets:
            secrets_size[secret] = len(self.secrets[secret])
        return secrets_size

    def init_byproduct_db(self):
        for node in self.pattern.output_nodes:
            self.byproduct_db[node] = {
                'z-domain': [],
                'x-domain': []
            }

        for cmd in self.pattern:
            if cmd[0] == 'Z' or cmd[0] == 'X':
                node = cmd[1]

                if cmd[0] == 'Z':
                    self.byproduct_db[node]['z-domain'] = cmd[2]
                if cmd[0] == 'X':
                    self.byproduct_db[node]['x-domain'] = cmd[2]

    def decode_output(self, node):
        z_decoding = 0
        x_decoding = 0
        if self.r_secret:
            for z_dep_node in self.byproduct_db[node]['z-domain']:
                z_decoding += self.secrets['r'][z_dep_node]
            z_decoding = z_decoding % 2
            for x_dep_node in self.byproduct_db[node]['x-domain']:
                x_decoding += self.secrets['r'][x_dep_node]
            x_decoding = x_decoding % 2

        return z_decoding, x_decoding

    def get_secrets_locations(self):
        locations = {}
        for secret in self.secrets:
            secret_dict = self.secrets[secret]
            secrets_location = secret_dict.keys()
            locations[secret] = secrets_location
        return locations

    def is_valid_secret(self, secret_type, custom_secret):
        if any((i != 0 and i != 1) for i in custom_secret.values()) :
            print(custom_secret)
            return False
        if secret_type == 'r':
            return set(custom_secret.keys()) == set(self.measurement_db.keys())


class ClientMeasureMethod(graphix.sim.base_backend.MeasureMethod):
    def __init__(self, client: Client):
        self.__client = client

    def get_measurement_description(self, cmd, results) -> graphix.sim.base_backend.MeasurementDescription:
        node = cmd[1]
        parameters = self.__client.measurement_db[node]
        r_value = 0 if not self.__client.r_secret else self.__client.secrets['r'][node]
        angle = parameters.angle + np.pi * r_value
        # extract signals for adaptive angle
        s_signal = np.sum(self.__client.results[j] for j in parameters.s_domain)
        t_signal = np.sum(self.__client.results[j] for j in parameters.t_domain)
        measure_update = graphix.pauli.MeasureUpdate.compute(
            parameters.plane, s_signal % 2 == 1, t_signal % 2 == 1, graphix.clifford.TABLE[parameters.vop]
        )
        angle = angle * measure_update.coeff + measure_update.add_term
        return graphix.sim.base_backend.MeasurementDescription(measure_update.new_plane, angle)

    def set_measure_result(self, cmd, result: bool) -> None:
        node = cmd[1]
        if self.__client.r_secret:
            r_value = self.__client.secrets['r'][node]
            result = (result + r_value) % 2
        self.__client.results[node] = result

