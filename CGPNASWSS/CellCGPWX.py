import numpy as np

from Evolution.CellCGPW import CellCgpW


class CellCgpWX (CellCgpW):
    """
    A class to represent a cell in a Cartesian Genetic Programming (CGP) network.

    Attributes
    ----------
    conf_net : list
        Configuration of the network.
    reduction : int
        Number of reduction blocks.
    normal : int
        Number of normal blocks.
    blocks : int
        Total number of blocks.
    individual : list
        List of GenCgpW instances.
    pool : list
        Pool configuration.

    Methods
    -------
    active_net_list_S(stage=1):
        Generates the active network list for a given stage.
    S1(stage=1, CNN=None):
        Helper method to generate the active network list for stage 1.
    active_net_list(AuxHead=False):
        Generates the active network list.
    """

    def active_net_list_S(self, stage=1):
        """
        Generates the active network list for a given stage.

        Parameters
        ----------
        stage : int, optional
            The stage of the network (default is 1).

        Returns
        -------
        list
            The active network list for the specified stage.
        """
        if stage == 1:
            finalCGP = self.S1(stage)
        elif stage == 2:
            n1 = self.individual[1]
            finalCGP = self.S1(stage, CNN=n1)
        else:
            raise ValueError("Invalid stage value. Only stage 1 and 2 are supported.")
        return finalCGP


    def S1(self, stage=1, CNN=None):
        """
        Generates the active network list for stage 1.

        Parameters
        ----------
        stage : int, optional
            The stage of the network (default is 1).
        CNN : GenCgpW, optional
            An instance of GenCgpW for the second stage (default is None).

        Returns
        -------
        list
            The active network list for the specified stage.
        """
        CB = 'ConvBlock'
        PO = 'Max_Pool'
        finalCGP = []

        # Generate the active network list for the first individual
        n0 = self.individual[0].active_net_list()
        n0.insert(0, ['input', 0, 0, 0, 0])
        n0.pop()
        finalCGP.extend(n0)

        # If CNN is provided, generate the active network list for the second stage
        if CNN is not None:
            n1 = CNN.active_net_list(len(finalCGP))
            n1.pop()
            n_size = len(finalCGP) - 1
            a, b = finalCGP[n_size][1], finalCGP[n_size][2]

            finalCGP.extend([[PO, a + 1, b + 1, np.random.choice(self.conf_net[0].kernels),
                              np.random.choice(self.conf_net[stage].channels)]])
            finalCGP.extend(n1)

        # Generate the active network list for the remaining normal blocks
        for i in range(self.normal - stage):
            n_size = len(finalCGP) - 1
            a, b = finalCGP[n_size][1], finalCGP[n_size][2]

            finalCGP.extend([[PO, a + 1, b + 1, np.random.choice(self.conf_net[0].kernels),
                              np.random.choice(self.conf_net[i + 1].channels)]])
            finalCGP.extend([[CB, a + 2, b + 2, np.random.choice(self.conf_net[0].kernels),
                              np.random.choice(self.conf_net[i + 1].channels)]])

        # Add the final pooling and fully connected layers
        n_size = len(finalCGP) - 1
        a, b = finalCGP[n_size][1], finalCGP[n_size][2]
        gap = ['G_Avg_pool', a + 1, b + 1, np.random.choice(self.conf_net[0].kernels),
               np.random.choice(self.conf_net[self.normal - 1].channels)]
        finalCGP.extend([gap])

        full = ['full', a + 2, b + 2, np.random.choice(self.conf_net[0].kernels),
                np.random.choice(self.conf_net[self.normal - 1].channels)]
        finalCGP.extend([full])

        return finalCGP
