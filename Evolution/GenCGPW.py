
import numpy as np
import random


class GenCgpW:
    """
    Class representing a Weighted Cartesian Genetic Programming (CGP) genotype.

    This class initializes a CGP individual with weights and supports
    operations for converting between integer and real-valued genes.

    Attributes:
    ----------
    conf_net : object
        Configuration object for network specifications such as node count and input/output connections.
    shapegene : tuple or None
        The shape of the gene array, initially set to None.
    weightcolums : int
        Number of weight columns; default is 2.
    n_var_size : int
        Total number of variables needed in the genotype array.
    realgene : numpy.ndarray or None
        Real-valued representation of the gene; if None, generated from integer gene.
    gene : numpy.ndarray
        Integer representation of the CGP gene, initialized with zeros.
    gene_test : numpy.ndarray
        Additional gene array for testing or temporary operations.
    weight : numpy.ndarray
        Array storing weight values for each node connection.
    is_active : numpy.ndarray
        Boolean array indicating which nodes are active.
    is_pool : numpy.ndarray
        Boolean array indicating which nodes are pooling nodes.
    eval : Any
        Evaluation result or metric for the CGP individual, initially set to None.
    """

    def __init__(self, conf_net, realgene=None, weights=2):
        """
        Initialize a GenCgpW instance with specified configuration and optional real gene.

        Parameters:
        ----------
        conf_net : object
            Configuration object that contains parameters like node_num, out_num, max_in_num, etc.
        realgene : numpy.ndarray or None, optional
            Real-valued gene representation; if provided, initializes from this representation.
        weights : int, optional
            Number of columns for weights in the weight matrix; default is 2.
        """
        self.conf_net = conf_net
        self.shapegene = None  # To store the shape of the gene array if needed for future processing
        self.weightcolums = weights  # Number of weight columns in each gene

        # Calculate total variable size based on configuration
        self.n_var_size = (self.conf_net.node_num + self.conf_net.out_num) * (
                self.conf_net.max_in_num + 1 + self.weightcolums
        )

        self.realgene = realgene  # Real-valued gene, if provided
        # Initialize integer gene representation for nodes and outputs
        self.gene = np.zeros((self.conf_net.node_num + self.conf_net.out_num, self.conf_net.max_in_num + 1), dtype=int)
        self.gene_test = np.zeros_like(self.gene)  # Secondary gene array for testing purposes

        # Initialize weight matrix for nodes and outputs
        self.weight = np.zeros((self.conf_net.node_num + self.conf_net.out_num, self.weightcolums), dtype=int)

        # Initialize active and pooling status arrays
        self.is_active = np.empty(self.conf_net.node_num + self.conf_net.out_num, dtype=bool)
        self.is_pool = np.empty(self.conf_net.node_num + self.conf_net.out_num, dtype=bool)

        # Evaluation metric or result for the CGP individual
        self.eval = None

        # Create initial gene structure and set up the real or integer gene representations
        self.creategene()
        if realgene is None:
            self.toReal()
        else:
            self.to_int_cgp()

    def creategene(self):
        """
        Generate the genetic structure for the individual in the Cartesian Genetic Programming (CGP) representation.

        This method initializes the gene and weight matrices based on the configuration parameters
        of the network (such as the number of nodes, maximum input number, and number of rows and columns).
        The method also ensures that the gene connections follow the constraints defined by the network configuration.
        """
        # Loop over each node and output
        for n in range(self.conf_net.node_num + self.conf_net.out_num):
            # Determine the type of gene: function or output node
            if n < self.conf_net.node_num:
                type_num = self.conf_net.func_type_num  # Number of available function types for intermediate nodes
                self.gene[n][0] = np.random.randint(type_num)  # Assign a random function type to the node
            else:
                type_num = self.conf_net.out_type_num  # Number of available output types
                self.gene[n][0] = self.conf_net.out_type_id  # Assign the fixed output type ID for output nodes

            # Initialize the connection genes for this node
            col = np.min((int(n / self.conf_net.rows), self.conf_net.cols))  # Determine the column of the node
            max_connect_id = col * self.conf_net.rows + self.conf_net.input_num  # Max connection ID based on the column
            min_connect_id = (col - self.conf_net.level_back) * self.conf_net.rows + self.conf_net.input_num \
                if col - self.conf_net.level_back >= 0 else 0  # Min connection ID based on network depth

            # Assign random connections for the node's inputs
            for i in range(self.conf_net.max_in_num):
                self.gene[n][i + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)

            # Initialize weights for the node
            self.weight[n][0] = np.random.randint(len(self.conf_net.kernels))  # Randomly choose a kernel
            self.weight[n][1] = np.random.randint(len(self.conf_net.channels))  # Randomly choose a channel

        # After gene creation, check which nodes are active
        self.check_active()

    def check_active(self):
        """
        Clears the current activity status of all nodes and then starts from the output nodes
        to mark all active nodes in the network by traversing through their dependencies.

        This method ensures that the network correctly identifies all active nodes by checking the
        connections starting from output nodes and marking each node that is part of the execution path.
        """
        # Clear all active status flags
        self.is_active[:] = False

        # Start checking from the output nodes
        for n in range(self.conf_net.out_num):
            self._check_path_to_output(self.conf_net.node_num + n)

    def _check_path_to_output(self, node_index):
        """
        Recursively checks the path from a given node to output nodes and marks all reachable nodes as active.

        Args:
            node_index (int): The index of the current node to check.

        This method traverses the gene connections starting from the given node and marks all reachable
        nodes (input or intermediate) as active. This ensures that the activation flow is traced correctly
        for each node in the network.
        """
        if not self.is_active[node_index]:
            # Mark this node as active
            self.is_active[node_index] = True

            # Determine the type of node: output node or intermediate node
            gene_type = self.gene[node_index][0]

            # Define the number of inputs based on node type
            if node_index >= self.conf_net.node_num:  # output node
                input_count = self.conf_net.out_in_num  # Arity for output nodes
            else:  # intermediate node
                input_count = self.conf_net.func_in_num[gene_type]  # Arity for function nodes

            # Check all input connections for this node
            for i in range(input_count):
                connected_node_index = self.gene[node_index][i + 1]

                # If the connected node is not an input (i.e., a node in the network), check its path
                if connected_node_index >= self.conf_net.input_num:
                    # Adjust the index for network nodes
                    self._check_path_to_output(connected_node_index - self.conf_net.input_num)

    def active_net_list(self, extra=0):
        """
        Generates a list of active nodes in the network along with their connections, types, and weights.

        Args:
            extra (int): An optional parameter to add extra values to the connection list.

        Returns:
            list: A list containing the details of each active node including its type, connections,
                  kernel, and channel weights.
        """
        # Ensure the active status of nodes is updated
        self.check_active()

        net_list = []  # Initialize the list of active nodes
        active_count = np.arange(self.conf_net.input_num + self.conf_net.node_num + self.conf_net.out_num)

        # Update the active count with cumulative sum of active nodes
        active_count[self.conf_net.input_num:] = np.cumsum(self.is_active)

        for n, is_active in enumerate(self.is_active):
            if is_active:
                # Get node type, weights, and kernel/channel information
                gene_type = self.gene[n][0]
                weight_kernel = self.weight[n][0].tolist()
                weight_channel = self.weight[n][1].tolist()

                kernel = self.conf_net.kernels[weight_kernel]
                channel = self.conf_net.channels[weight_channel]

                if n < self.conf_net.node_num:  # Intermediate node
                    node_type = self.conf_net.func_type[gene_type]
                else:  # Output node
                    node_type = self.conf_net.out_type[0]

                # Get connections for the current node
                connections = [active_count[self.gene[n][i + 1]] for i in range(self.conf_net.max_in_num)]

                # Add extra connections if specified
                if extra != 0:
                    net_list.append([node_type] + np.add(connections, [extra, extra]).tolist() + [kernel] + [channel])
                else:
                    net_list.append([node_type] + connections + [kernel] + [channel])

        return net_list

    def active_net_list_one(self):
        """
        Attempts to generate a list of active nodes and handles failure by resetting the gene configuration.

        Returns:
            list: A list containing the details of each active node.
        """
        self.shapegene = self.gene.shape

        try:
            net = self.active_net_list_int()
        except:
            print('Repairing individual, previous failure: ')
            # Handle failure by resetting the last gene to output type and adjusting the others
            self.gene[self.shapegene[0] - 1][0] = self.conf_net.out_type_id
            for i in range(self.shapegene[0] - 2):
                if self.gene[i][0] == self.conf_net.out_type_id:
                    self.gene[i][0] = np.random.randint(len(self.conf_net.func_type))

        net = self.active_net_list_int()
        return net

    def active_net_list_complete(self):
        """
        Generates a complete list of nodes in the network along with their connections.

        Returns:
            list: A list of all nodes including input nodes, their types, and connections.
        """
        net_list = [["input", 0, 0]]  # Start with the input node

        for n, is_active in enumerate(self.is_active):
            if is_active:
                # Get node type and connections
                gene_type = self.gene[n][0]
                if n < self.conf_net.node_num:  # Intermediate node
                    node_type = self.conf_net.func_type[gene_type]
                else:  # Output node
                    node_type = self.conf_net.out_type[gene_type]

                connections = [self.gene[n][i + 1] for i in range(self.conf_net.max_in_num)]
                net_list.append([node_type] + connections)

        return net_list

    def toReal(self):
        """
        Converts the gene representation to real-valued parameters for the evolutionary algorithm.

        This method translates the integer gene values into continuous real values between specified ranges.
        The conversion considers the total number of possible functions, weights, and channels.
        The real-valued gene is stored in `self.realgene`.

        The method performs the following transformations:
        - The function type is converted to a floating-point value between 0 and 1.
        - The weights are similarly scaled to floating-point values between 0 and 1.
        - The input connections are also scaled based on their respective ranges.
        """
        # Initialize the real gene with zeros
        self.realgene = np.zeros(
            (self.conf_net.node_num + self.conf_net.out_num, self.conf_net.max_in_num + 1 + self.weightcolums)
        ).astype(float)

        # Get total number of function types, weight kernels, and channels
        func_total = len(self.conf_net.func_type)
        weight_total = len(self.conf_net.kernels)
        channel_total = len(self.conf_net.channels)

        # Loop through each gene to convert its values to real numbers
        for i in range(len(self.gene)):
            # Scale function type to a float between 0 and 1
            self.realgene[i, 0] = random.uniform(self.gene[i, 0] / func_total, (self.gene[i, 0] + 1) / func_total)

            # Ensure the last gene for output node is close to 1
            if i == len(self.gene) - 1:
                self.realgene[i, 0] = 0.9999  # Assign near 1 for the output node

            # Scale weight and channel to floats between 0 and 1
            self.realgene[i, self.conf_net.max_in_num + 1] = random.uniform(
                self.weight[i][0] / weight_total, (self.weight[i][0] + 1) / weight_total
            )
            self.realgene[i, self.conf_net.max_in_num + 2] = random.uniform(
                self.weight[i][1] / channel_total, (self.weight[i][1] + 1) / channel_total
            )

            # Scale input connections to floats between 0 and 1, relative to the node index
            for j in range(self.conf_net.max_in_num):
                self.realgene[i, j + 1] = random.uniform(self.gene[i, j + 1] / (i + 1),
                                                         (self.gene[i, j + 1] + 1) / (i + 1))

    def to_int_cgp(self):
        """
        Converts the real-valued gene parameters back to their integer representation for use in the genetic algorithm.

        This method performs the reverse operation of `toReal`, converting the real-valued genes back to integer
        gene values. The conversion scales the real numbers back to integer values corresponding to the
        number of possible functions, weights, and channels.
        """
        # Get total number of function types, weight kernels, and channels
        func_total = len(self.conf_net.func_type)
        weight_total = len(self.conf_net.kernels)
        channel_total = len(self.conf_net.channels)

        # Loop through each real gene to convert back to integers
        for i in range(len(self.realgene)):
            # Convert function type from real to integer
            self.gene[i, 0] = np.floor(self.realgene[i, 0] * func_total)

            # Ensure the last gene for output node has the correct output type
            if i == len(self.realgene) - 1:
                self.gene[i, 0] = self.conf_net.out_type_id  # Output node type

            # Convert weights and channels from real to integer
            self.weight[i][0] = np.floor(self.realgene[i, self.conf_net.max_in_num + 1] * weight_total)
            self.weight[i][1] = np.floor(self.realgene[i, self.conf_net.max_in_num + 2] * channel_total)

            # Convert input connections from real to integer, relative to node index
            for j in range(self.conf_net.max_in_num):
                self.gene[i, j + 1] = np.floor(self.realgene[i, j + 1] * (i + 1))


#class GenCgpW:
#     def __init__(self, conf_net, realgene=None, weights=2):
#         self.shapegene = None
#         self.weightcolums = weights
#         self.conf_net = conf_net
#         self.n_var_size = (self.conf_net.node_num + self.conf_net.out_num) * (
#                 self.conf_net.max_in_num + 1 + self.weightcolums)
#         self.realgene = realgene
#         self.gene = np.zeros((self.conf_net.node_num + self.conf_net.out_num, self.conf_net.max_in_num + 1)).astype(int)
#         self.gene_test = np.zeros(
#             (self.conf_net.node_num + self.conf_net.out_num, self.conf_net.max_in_num + 1)).astype(int)
#
#         self.weight = np.zeros((self.conf_net.node_num + self.conf_net.out_num, self.weightcolums)).astype(int)
#         self.is_active = np.empty(self.conf_net.node_num + self.conf_net.out_num).astype(bool)
#         self.is_pool = np.empty(self.conf_net.node_num + self.conf_net.out_num).astype(bool)
#         self.eval = None
#         self.creategene()
#         if realgene is None:
#             self.toReal()
#         else:
#             self.to_int_cgp()
#
#     def creategene(self):
#         for n in range(self.conf_net.node_num + self.conf_net.out_num):
#             # type gene
#             # type_num = self.conf_net.func_type_num if n < self.conf_net.node_num else self.conf_net.out_type_num
#
#             if n < self.conf_net.node_num:
#                 type_num = self.conf_net.func_type_num
#                 self.gene[n][0] = np.random.randint(type_num)
#             else:
#                 type_num = self.conf_net.out_type_num
#                 self.gene[n][0] = self.conf_net.out_type_id
#
#             # connection gene
#             col = np.min((int(n / self.conf_net.rows), self.conf_net.cols))
#             max_connect_id = col * self.conf_net.rows + self.conf_net.input_num
#             min_connect_id = (col - self.conf_net.level_back) * self.conf_net.rows + self.conf_net.input_num \
#                 if col - self.conf_net.level_back >= 0 else 0
#             for i in range(self.conf_net.max_in_num):
#                 self.gene[n][i + 1] = min_connect_id + np.random.randint(max_connect_id - min_connect_id)
#             self.weight[n][0] = np.random.randint(len(self.conf_net.kernels))
#             self.weight[n][1] = np.random.randint(len(self.conf_net.channels))
#
#         self.check_active()
#
#     def check_active(self):
#         # clear
#         self.is_active[:] = False
#         # start from output nodes
#         for n in range(self.conf_net.out_num):
#             self.__check_course_to_out(self.conf_net.node_num + n)
#
#     def __check_course_to_out(self, n):
#         if not self.is_active[n]:
#             self.is_active[n] = True
#             t = self.gene[n][0]
#             if n >= self.conf_net.node_num:  # output node
#                 in_num = self.conf_net.out_in_num  # arity arrays
#             else:  # intermediate node
#                 in_num = self.conf_net.func_in_num[t]  # arity arrays
#             for i in range(in_num):
#                 if self.gene[n][i + 1] >= self.conf_net.input_num:
#                     self.__check_course_to_out(self.gene[n][i + 1] - self.conf_net.input_num)
#
#     def active_net_list(self, extra=0):
#
#         self.check_active()
#         net_list = []
#         # if extra > 0:
#         #     net_list = []
#         # else:
#         #    net_list = [["input", 0, 0,0,0]]
#         # net_list = [["input", 0, 0, 0, 0]]
#         active_cnt = np.arange(self.conf_net.input_num + self.conf_net.node_num + self.conf_net.out_num)
#         active_cnt[self.conf_net.input_num:] = np.cumsum(self.is_active)
#
#         for n, is_a in enumerate(self.is_active):
#             if is_a:
#                 t = self.gene[n][0]
#                 w = self.weight[n][0].tolist()
#                 C = self.weight[n][1].tolist()
#
#                 kernel = self.conf_net.kernels[w]
#
#                 chann = self.conf_net.channels[C]
#                 if n < self.conf_net.node_num:  # intermediate node
#                     type_str = self.conf_net.func_type[t]
#                 else:  # output node
#                     type_str = self.conf_net.out_type[0]  # old =t
#                 connections = [active_cnt[self.gene[n][i + 1]] for i in range(self.conf_net.max_in_num)]
#                 if extra != 0:
#                     net_list.append([type_str] + np.add(connections, [extra, extra]).tolist() + [kernel] + [chann])
#                 else:
#                     net_list.append([type_str] + connections + [kernel] + [chann])
#
#         return net_list
#
#     def active_net_list_one(self):
#         self.shapegene = self.gene.shape
#         try:
#             f = self.active_net_list_int()
#         except:
#             print('reparing individual, previous fail: ')
#             self.gene[self.shapegene[0] - 1][0] = self.conf_net.out_type_id
#             for i in range(self.shapegene[0] - 2):
#                 if self.gene[i][0] == self.conf_net.out_type_id:
#                     self.gene[i][0] = np.random.randint(len(self.conf_net.func_type))
#
#         net = self.active_net_list_int()
#
#         return net
#
#     def active_net_list_complete(self):
#         net_list = [["input", 0, 0]]
#         for n, is_a in enumerate(self.is_active):
#             if is_a:
#                 t = self.gene[n][0]
#                 if n < self.conf_net.node_num:  # intermediate node
#                     type_str = self.conf_net.func_type[t]
#                 else:  # output node
#                     type_str = self.conf_net.out_type[t]
#
#                 connections = [self.gene[n][i + 1] for i in range(self.conf_net.max_in_num)]
#                 net_list.append([type_str] + connections)
#         return net_list
#
#     def toReal(self):
#         self.realgene = np.zeros(
#             (self.conf_net.node_num + self.conf_net.out_num, self.conf_net.max_in_num + 1 + self.weightcolums)).astype(
#             float)
#         functTotal = len(self.conf_net.func_type)  # + 1 cause output is a function too
#         weightTotal = len(self.conf_net.kernels)
#         chaTotal = len(self.conf_net.channels)
#         gen = self.gene
#         wgh = self.weight
#         for i in range(len(gen)):
#             self.realgene[i, 0] = random.uniform(gen[i, 0] / functTotal, (gen[i, 0] + 1) / functTotal)
#             if i == gen.shape[0] - 1:
#                 self.realgene[i, 0] = 0.9999
#             self.realgene[i, self.conf_net.max_in_num + 1] = random.uniform(wgh[i][0] / weightTotal,
#                                                                             (wgh[i][0] + 1) / weightTotal)
#             self.realgene[i, self.conf_net.max_in_num + 2] = random.uniform(wgh[i][1] / chaTotal,
#                                                                             (wgh[i][1] + 1) / chaTotal)
#
#             for j in range(self.conf_net.max_in_num):
#                 self.realgene[i, j + 1] = random.uniform(gen[i, j + 1] / (i + 1), (gen[i, j + 1] + 1) / (i + 1))
#
#     def to_int_cgp(self):
#         functTotal = len(self.conf_net.func_type)  # + 1 cause output is a function too
#         weightTotal = len(self.conf_net.kernels)
#         chaTotal = len(self.conf_net.channels)
#         gen = self.realgene
#         for i in range(len(gen)):
#
#             self.gene[i, 0] = np.floor(gen[i, 0] * functTotal)
#             if i == gen.shape[0] - 1:
#                 self.realgene[i, 0] = self.conf_net.out_type_id
#             self.weight[i][0] = np.floor(gen[i, self.conf_net.max_in_num + 1] * weightTotal)
#             self.weight[i][1] = np.floor(gen[i, self.conf_net.max_in_num + 2] * chaTotal)
#
#             for j in range(self.conf_net.max_in_num):
#                 self.gene[i, j + 1] = np.floor(gen[i, j + 1] * (i + 1))
