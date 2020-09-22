"""
Finite difference interpolator using folds
"""
import logging

import numpy as np

from LoopStructural.interpolators.finite_difference_interpolator import \
    FiniteDifferenceInterpolator

logger = logging.getLogger(__name__)


class FiniteDifferenceFoldInterpolator(FiniteDifferenceInterpolator):
    """

    """
    def __init__(self, support, fold):
        """
        A piecewise linear interpolator that can also use fold constraints defined in Laurent et al., 2016

        Parameters
        ----------
        support
            discrete support with nodes and elements etc
        fold FoldEvent
            a fold event with a valid geometry
        """

        FiniteDifferenceInterpolator.__init__(self, support)
        self.type = ['foldinterpolator']
        self.fold = fold

    def update_fold(self, fold):
        """

        Parameters
        ----------
        fold : FoldEvent
            a fold that contrains the geometry we are trying to add

        Returns
        -------

        """
        logger.error('updating fold, this should be done by accessing the fold attribute')
        self.fold = fold

    def add_fold_constraints(self, fold_orientation=10., fold_axis_w=10., fold_regularisation=.1,
                             fold_normalisation=1.,
                             fold_norm=1.):
        """

        Parameters
        ----------
        fold_orientation : double
            weight for the fold direction/orientation in the least squares system
        fold_axis_w : double
            weight for the fold axis in the least squares system
        fold_regularisation : double
            weight for the fold regularisation in the least squares system
        fold_normalisation : double
            weight for the fold norm constraint in the least squares system
        fold_norm
            length of the interpolation norm in the least squares system

        Returns
        -------

        Notes
        -----
        For more information about the fold weights see EPSL paper by Gautier Laurent 2016

        """
        # get the gradient of all of the elements of the mesh
        # eg = self.support.get_element_gradients(np.arange(self.support.n_elements))
        eg = self.support.calcul_T(self.support.barycentre())
        # get array of all nodes for all elements N,4,3
        nodes = self.support.nodes[self.support.get_elements()[np.arange(self.support.n_elements)]]
        # calculate the fold geometry for the elements barycentre
        deformed_orientation, fold_axis, dgz = \
            self.fold.get_deformed_orientation(self.support.barycentre())

        # # calculate element volume for weighting
        # vecs = nodes[:, 1:, :] - nodes[:, 0, None, :]
        # vol = np.abs(np.linalg.det(vecs)) / 6
        if fold_orientation is not None:
            """
            dot product between vector in deformed ori plane = 0
            """
            logger.info("Adding fold orientation constraint to %s w = %f"%(self.propertyname, fold_orientation))
            A = np.einsum('ij,ijk->ik', deformed_orientation, eg)
            A *= self.vol
            A *= fold_orientation
            B = np.zeros(self.support.n_elements)
            idc = self.support.get_elements()
            self.add_constraints_to_least_squares(A, B, idc)

        if fold_axis_w is not None:
            """
            dot product between axis and gradient should be 0
            """
            logger.info("Adding fold axis constraint to %s w = %f"%(self.propertyname,fold_axis_w))
            A = np.einsum('ij,ijk->ik', fold_axis, eg)
            A *= self.vol
            A *= fold_axis_w
            B = np.zeros(self.support.n_elements).tolist()
            self.add_constraints_to_least_squares(A, B, self.support.get_elements())

        if fold_normalisation is not None:
            """
            specify scalar norm in X direction
            """
            logger.info("Adding fold normalisation constraint to %s w = %f"%(self.propertyname,fold_normalisation))
            A = np.einsum('ij,ijk->ik', dgz, eg)
            A *= self.vol
            A *= fold_normalisation
            B = np.ones(self.support.n_elements)

            if fold_norm is not None:
                B[:] = fold_norm
            B *= fold_normalisation
            B *= self.vol
            self.add_constraints_to_least_squares(A, B, self.support.get_elements())

        # if fold_regularisation is not None:
        #     """
        #     fold constant gradient  
        #     """
        #     logger.info("Adding fold regularisation constraint to %s w = %f"%(self.propertyname,fold_regularisation))
        #     idc, c, ncons = fold_cg(eg, dgz, self.support.get_neighbours(), self.support.get_elements(), self.support.nodes)
        #     A = np.array(c[:ncons, :])
        #     A *= fold_regularisation
        #     B = np.zeros(A.shape[0])
        #     idc = np.array(idc[:ncons, :])
        #     self.add_constraints_to_least_squares(A, B, idc)
