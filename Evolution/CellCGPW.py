import numpy as np
from Evolution.GenCGPW import GenCgpW

class CellCgpW:

    """
    A class to represent a Cell in Cartesian Genetic Programming (CGP).

    This class manages the configuration, creation, and activation of CGP-based
    neural network cells. Each cell consists of multiple blocks, which are normal
    and reduction blocks in the network's architecture.

    Attributes:
        conf_net (list): List of configurations for each layer in the network.
        reduction (int): Number of reduction blocks.
        normal (int): Number of normal blocks.
        blocks (int): Total number of blocks (normal + reduction).
        individual (list): List of GenCgpW instances representing each block.
        pool (list): Pool configuration for the model.
        n_var_size (int): Number of variables in the genotype of the CGP structure.
        shapegene (tuple): Shape of the gene for the CGP structure.
    """


    def __init__(self, conf_net, pool, N=3, R=2):
        """
        Initializes a CellCgpW instance with the specified network configuration.

        Args:
            conf_net (list): Configuration list for each layer in the network.
            pool (list): Pool configuration for the model.
            N (int): Number of normal blocks.
            R (int): Number of reduction blocks.
        """
        self.conf_net = conf_net
        self.reduction = R
        self.normal = N
        self.blocks = N + R
        self.pool = pool
        self.individual = [
            GenCgpW(conf_net[i], realgene=None, weights=2) for i in range(N)
        ]
        self.n_var_size = self.individual[0].n_var_size
        self.shapegene = self.individual[0].gene.shape

    def active_net_list(self, AuxHead=False):
        """
        Generates an active network list for the CGP cell with optional auxiliary head.

        Args:
            AuxHead (bool): Flag to add auxiliary head in the network.

        Returns:
            list: List of active nodes in the CGP network.
        """
        AuxHeaMaxIndex = []
        finalCGP = []

        for j in range(self.normal):
            try:
                self.individual[j].active_net_list()
            except:
                print(f'Repairing individual, previous fail at index: {j}')
                self.individual[j].gene[self.shapegene[0] - 1][0] = self.conf_net[j].out_type_id
                for i in range(self.shapegene[0] - 2):
                    if self.individual[j].gene[i][0] == self.conf_net[j].out_type_id:
                        self.individual[j].gene[i][0] = np.random.randint(len(self.conf_net[j].func_type))

        n0 = self.individual[0].active_net_list()
        n0.insert(0, ['input', 0, 0, 0, 0])

        if self.pool[0] == 1:
            n0[-1][0] = 'Max_Pool'
        else:
            n0.pop(-1)
        finalCGP.extend(n0)

        for i in range(self.normal - 2):
            tempNi = self.individual[i + 1].active_net_list(len(finalCGP) - 1)
            if self.pool[i + 1] == 1:
                tempNi[-1][0] = 'Max_Pool'
                AuxHeaMaxIndex.append(len(finalCGP) + len(tempNi) - 1)
            else:
                tempNi.pop(-1)
            finalCGP.extend(tempNi)

        tempfinal = self.individual[-1].active_net_list(len(finalCGP) - 1)
        tempfinal[-1][0] = 'G_Avg_pool'
        finalCGP.extend(tempfinal)

        finalCGP.append(['full', len(finalCGP) - 1, 0, 0, 0])
        if AuxHead:
            finalCGP.insert(len(finalCGP) - 1, ['AHead', AuxHeaMaxIndex[-1], 0, 0, 0])

        return finalCGP

    def active_net_list_sum(self):
        """
        Generates an active network list for the CGP cell using summation pooling.

        Returns:
            list: List of active nodes in the CGP network.
        """
        finalCGP = []

        for j in range(self.normal):
            try:
                self.individual[j].active_net_list()
            except:
                print(f'Repairing individual, previous fail at index: {j}')
                self.individual[j].gene[self.shapegene[0] - 1][0] = 0

        n0 = self.individual[0].active_net_list()
        n0[-1][0] = 'ConvBlock'
        finalCGP.extend(n0)

        for i in range(self.reduction - 1):
            off = 32 * (2 ** i)
            tempNi = self.individual[i + 1].active_net_list(len(finalCGP) - 1)
            tempNi[-1][0] = f'S_ConvBlock_{off}_1_0'
            tempNi[-1][1] = tempNi[0][1]
            finalCGP.extend(tempNi)
            finalCGP.append(['Sum', len(finalCGP) - 1, len(finalCGP) - 2])
            finalCGP.append(['Max_Pool', len(finalCGP) - 1, 0])

        tempfinal = self.individual[-1].active_net_list(len(finalCGP) - 1)
        tempfinal[-1][0] = f'S_ConvBlock_{off * 2}_1_0'
        finalCGP.extend(tempfinal)
        finalCGP.append(['Sum', len(finalCGP) - 1, len(finalCGP) - 2])
        finalCGP.append(['G_Avg_pool', len(finalCGP) - 1, 0])
        finalCGP.append(['full', len(finalCGP) - 1, 0])

        return finalCGP

    def active_net_list_nomax(self):
        """
        Generates an active network list for the CGP cell without max pooling.

        Returns:
            list: List of active nodes in the CGP network.
        """
        finalCGP = []

        for j in range(self.normal):
            try:
                self.individual[j].active_net_list()
            except:
                print(f'Repairing individual, previous fail at index: {j}')
                self.individual[j].gene[self.shapegene[0] - 1][0] = 0

        n0 = self.individual[0].active_net_list()
        n0.pop(-1)
        finalCGP.extend(n0)

        for i in range(self.normal - 2):
            tempNi = self.individual[i + 1].active_net_list(len(finalCGP) - 1)
            tempNi.pop(-1)
            finalCGP.extend(tempNi)

        tempfinal = self.individual[-1].active_net_list(len(finalCGP) - 1)
        tempfinal[-1][0] = 'G_Avg_pool'
        finalCGP.extend(tempfinal)
        finalCGP.append(['full', len(finalCGP) - 1, 0])

        return finalCGP

    def active_net_list_maxnormal(self):
        """
        Generates an active network list for the CGP cell with max pooling for all normal blocks.

        Returns:
            list: List of active nodes in the CGP network.
        """
        finalCGP = []

        for j in range(self.normal):
            try:
                self.individual[j].active_net_list()
            except:
                print(f'Repairing individual, previous fail at index: {j}')
                self.individual[j].gene[self.shapegene[0] - 1][0] = 0

        n0 = self.individual[0].active_net_list()
        n0[-1][0] = 'Max_Pool'
        finalCGP.extend(n0)

        for i in range(self.normal - 2):
            tempNi = self.individual[i + 1].active_net_list(len(finalCGP) - 1)
            tempNi[-1][0] = 'Max_Pool'
            finalCGP.extend(tempNi)

        tempfinal = self.individual[-1].active_net_list(len(finalCGP) - 1)
        tempfinal[-1][0] = 'G_Avg_pool'
        finalCGP.extend(tempfinal)
        finalCGP.append(['full', len(finalCGP) - 1, 0])

        return finalCGP



# import numpy as np
#
# from Evolution.GenCGPW import GenCgpW
#
#
# class CellCgpW:
#     def __init__(self, conf_net,P, N=3, R=2):
#         self.conf_net = conf_net
#         self.reduction = R
#         self.normal = N
#         self.blocks = N + R
#         self.individual = []
#         self.pool = P
#         for i in range(N):
#             self.individual.append(GenCgpW(conf_net[i], realgene=None,weights=2))
#
#         self.n_var_size = self.individual[0].n_var_size
#         self.shapegene = self.individual[0].gene.shape
#
#     def active_net_list(self,AuxHead=False):
#         AuxHeaMaxIndex=[]
#         for j in range(self.normal):
#             index = j
#             try:
#                 f = self.individual[index].active_net_list()
#             except:
#                 print('reparing individual, previous fail: ' + str(index))
#                 self.individual[index].gene[self.shapegene[0] - 1][0] = self.conf_net[index].out_type_id
#                 for i in range(self.shapegene[0] - 2):
#                     if self.individual[index].gene[i][0] == self.conf_net[index].out_type_id:
#                         self.individual[index].gene[i][0] = np.random.randint(len(self.conf_net[index].func_type))
#                         #self.conf_net[index].out_type_id - 1
#         finalCGP = []
#         n0 = self.individual[0].active_net_list()
#         n0.insert(0, ['input', 0, 0, 0 ,0])
#         if self.pool[0] == 1:
#             n0[len(n0) - 1][0] = 'Max_Pool'
#         else:
#             lenN0 = len(n0)
#             n0.pop(lenN0 - 1)
#         finalCGP.extend(n0)
#         for i in range(self.normal - 2):
#             tempNi = self.individual[i + 1].active_net_list(len(finalCGP) - 1)
#
#             if self.pool[i + 1] == 1:
#                 tempNi[len(tempNi) - 1][0] = 'Max_Pool'
#
#                 AuxHeaMaxIndex.append(len(finalCGP) + len(tempNi)-1)
#             else:
#                 lenNi = len(tempNi)
#                 tempNi.pop(lenNi - 1)
#
#             finalCGP.extend(tempNi)
#
#
#         tempfinal = self.individual[len(self.individual) - 1].active_net_list(len(finalCGP) - 1)
#
#         tempfinal[len(tempfinal) - 1][0] = 'G_Avg_pool'
#         finalCGP.extend(tempfinal)
#         # si = len(finalCGP)
#         # finalCGP.extend([['G_Avg_pool', si - 1, 0]])
#         si = len(finalCGP)
#         finalCGP.extend([['full', si - 1, 0, 0, 0]])
#         si = len(finalCGP)
#         if AuxHead:
#             finalCGP.insert(si-1,['AHead', AuxHeaMaxIndex[len(AuxHeaMaxIndex)-2], 0,0,0]) #-2
#
#         return finalCGP
#
#
#
#
#     def active_net_list_sum(self):
#
#         for j in range(self.normal):
#             index = j
#             try:
#                 f = self.individual[index].active_net_list()
#             except:
#                 print('reparing individual, previous fail: ' + str(index))
#                 self.individual[index].gene[self.shapegene[0] - 1][0] = 0
#
#         finalCGP = []
#
#         n0 = self.individual[0].active_net_list()
#
#         n0[len(n0) - 1][0] = 'S_ConvBlock_32_1_0'
#         n0[len(n0) - 1][1] = 0
#
#         finalCGP.extend(n0)
#
#         si = len(finalCGP)
#
#         finalCGP.extend([['Sum', si - 1, si - 2]])
#         si = len(finalCGP)
#         finalCGP.extend([['Max_Pool', si - 1, 0]])
#         off = 32
#         for i in range(self.reduction - 1):
#             off = off * 2
#             lenfinalcgp = len(finalCGP)
#             tempNi = self.individual[i + 1].active_net_list(lenfinalcgp - 1)
#
#             temsize = len(tempNi)
#
#             tempNi[temsize - 1][0] = 'S_ConvBlock_' + str(off) + '_1_0'
#             tempNi[temsize - 1][1] = tempNi[0][1]
#             finalCGP.extend(tempNi)
#             si = len(finalCGP)
#             finalCGP.extend([['Sum', si - 1, si - 2]])
#             si = len(finalCGP)
#             finalCGP.extend([['Max_Pool', si - 1, 0]])
#         tempfinal = self.individual[len(self.individual) - 1].active_net_list(len(finalCGP) - 1)
#         # finalCGP.extend(tempfinal)
#         off = off * 2
#         tempfinal[len(tempfinal) - 1][0] = 'S_ConvBlock_' + str(off) + '_1_0'
#         tempfinal[len(tempfinal) - 1][1] = tempfinal[0][1]
#         finalCGP.extend(tempfinal)
#         si = len(finalCGP)
#         finalCGP.extend([['Sum', si - 1, si - 2]])
#         si = len(finalCGP)
#         finalCGP.extend([['G_Avg_pool', si - 1, 0]])
#         si = len(finalCGP)
#         finalCGP.extend([['full', si - 1, 0]])
#         # tempfinal[len(tempfinal) - 1][0] = 'G_Avg_pool'
#         # tempfinal[len(tempfinal) - 1][2] = tempfinal[0][2]
#
#         return finalCGP
#
#     def active_net_list_nomax(self):
#
#         for j in range(self.normal):
#             index = j
#             try:
#                 f = self.individual[index].active_net_list()
#             except:
#                 print('reparing individual, previous fail: ' + str(index))
#
#                 self.individual[index].gene[self.shapegene[0] - 1][0] = 0
#
#         finalCGP = []
#         n0 = self.individual[0].active_net_list()
#         # n0[len(n0) - 1][0] = 'Max_Pool'
#         lenN0 = len(n0)
#         n0.pop(lenN0 - 1)
#         finalCGP.extend(n0)
#         for i in range(self.normal - 2):
#             tempNi = self.individual[i + 1].active_net_list(len(finalCGP) - 1)
#             lenNi = len(tempNi)
#             tempNi.pop(lenNi - 1)
#             # tempNi[len(tempNi) - 1][0] = 'Max_Pool'
#
#             finalCGP.extend(tempNi)
#         tempfinal = self.individual[len(self.individual) - 1].active_net_list(len(finalCGP) - 1)
#
#         tempfinal[len(tempfinal) - 1][0] = 'G_Avg_pool'
#         finalCGP.extend(tempfinal)
#         # si = len(finalCGP)
#         # finalCGP.extend([['G_Avg_pool', si - 1, 0]])
#         si = len(finalCGP)
#         finalCGP.extend([['full', si - 1, 0]])
#         return finalCGP
#
#
#
#     def active_net_list_maxnormal(self):
#
#         for j in range(self.normal):
#             index = j
#             try:
#                 f = self.individual[index].active_net_list()
#             except:
#                 print('reparing individual, previous fail: ' + str(index))
#                 self.individual[index].gene[self.shapegene[0] - 1][0] = 0
#
#         finalCGP = []
#         n0 = self.individual[0].active_net_list()
#
#         n0[len(n0) - 1][0] = 'Max_Pool'
#
#         finalCGP.extend(n0)
#         for i in range(self.normal - 2):
#             tempNi = self.individual[i + 1].active_net_list(len(finalCGP) - 1)
#
#             tempNi[len(tempNi) - 1][0] = 'Max_Pool'
#
#             finalCGP.extend(tempNi)
#         tempfinal = self.individual[len(self.individual) - 1].active_net_list(len(finalCGP) - 1)
#
#         tempfinal[len(tempfinal) - 1][0] = 'G_Avg_pool'
#         finalCGP.extend(tempfinal)
#         # si = len(finalCGP)
#         # finalCGP.extend([['G_Avg_pool', si - 1, 0]])
#         si = len(finalCGP)
#         finalCGP.extend([['full', si - 1, 0]])
#         return finalCGP
