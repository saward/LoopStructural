import logging

import numpy as np

from LoopStructural.interpolators.discete_interpolator import \
    DiscreteInterpolator
from LoopStructural.utils.helper import get_vectors

logger = logging.getLogger(__name__)


class P2Interpolator(DiscreteInterpolator):

    def __init__(self, mesh):
        """
        Piecewise Linear Interpolator
        Approximates scalar field by finding coefficients to a piecewise linear
        equation on a tetrahedral mesh. Uses constant gradient regularisation.

        Parameters
        ----------
        mesh - TetMesh
            interpolation support
        """

        DiscreteInterpolator.__init__(self, mesh)
        # whether to assemble a rectangular matrix or a square matrix
        self.interpolator_type = 'P2I'
        self.nx = len(self.support.nodes[self.region])
        self.support = mesh

        self.interpolation_weights = {'cgw': 0.1, 'cpw': 1., 'npw': 1.,
                                      'gpw': 1., 'tpw': 1.}
        self.__str = 'Piecewise Linear Interpolator with %i unknowns. \n' % \
                     self.nx

    def __str__(self):
        return self.__str

    def copy(self):
        pass

    def _setup_interpolator(self, **kwargs):
        """
        Searches through kwargs for any interpolation weights and updates
        the dictionary.
        Then adds the constraints to the linear system using the
        interpolation weights values
        Parameters
        ----------
        kwargs -
            interpolation weights

        Returns
        -------

        """
        # can't reset here, clears fold constraints
        # self.reset()
        logger.info("Setting up PLI interpolator for %s"%self.propertyname)
        for key in kwargs:
            if 'regularisation' in kwargs:
                self.interpolation_weights['cgw'] = 0.1 * kwargs[
                    'regularisation']
            self.up_to_date = False
            self.interpolation_weights[key] = kwargs[key]
        if self.interpolation_weights['cgw'] > 0.:
            self.up_to_date = False
            self.add_constant_gradient(self.interpolation_weights['cgw'])
            logger.info("Using constant gradient regularisation w = %f"
                        %self.interpolation_weights['cgw'])
        logger.info("Added %i gradient constraints, %i normal constraints,"
                    "%i tangent constraints and %i value constraints"
                    "to %s" % (self.n_g, self.n_n,
                               self.n_t, self.n_i, self.propertyname))
        self.add_gradient_ctr_pts(self.interpolation_weights['gpw'])
        self.add_norm_ctr_pts(self.interpolation_weights['npw'])
        self.add_ctr_pts(self.interpolation_weights['cpw'])
        self.add_tangent_ctr_pts(self.interpolation_weights['tpw'])

    def add_constant_gradient(self, w=0.1):
        """
        Add the constant gradient regularisation to the system

        Parameters
        ----------
        w (double) - weighting of the cg parameter

        Returns
        -------

        """
        pass

    def add_gradient_ctr_pts(self, w=1.0):
        """
        Adds gradient constraints to the least squares system with a weight
        defined by w
        Parameters
        ----------
        w - either numpy array of length number of

        Returns
        -------
        Notes
        -----
        Gradient constraints add a constraint that the gradient of the
        implicit function should
        be orthogonal to the strike vector and the dip vector defined by the
        normal.
        This does not control the direction of the gradient and therefore
        requires at least two other
        value constraints OR a norm constraint for the interpolant to solve.
        """
        pass

    def add_norm_ctr_pts(self, w=1.0):
        """
        Extracts the norm vectors from the interpolators p_n list and adds
        these to the implicit
        system

        Parameters
        ----------
        w : double
            weighting of the norm constraints in a least squares system

        Returns
        -------
        Notes
        -----
        Controls the direction and magnitude of the norm of the scalar field
        gradient.
        This constraint can conflict with value constraints if the magnitude
        of the vector doesn't
        match with the value constraints added to the implicit system.
        """

        pass

    def add_tangent_ctr_pts(self, w=1.0):
        """

        Parameters
        ----------
        w

        Returns
        -------

        """
        return

    def add_ctr_pts(self, w=1.0):  # for now weight all value points the same
        """
        Adds value constraints to the least squares system

        Parameters
        ----------
        w

        Returns
        -------

        """
        # get elements for points
        points = self.get_value_constraints()
        if points.shape[0] > 1:
            e, inside = self.support.elements_for_array(points[:, :3])
            # get barycentric coordinates for points
            nodes = self.support.nodes[self.support.elements[e]]
            vecs = nodes[:, 1:, :] - nodes[:, 0, None, :]
            vol = np.abs(np.linalg.det(vecs)) / 6

        pass

    def add_gradient_orthogonal_constraint(self, elements, normals, w=1.0,
                                           B=0):
        """
        constraints scalar field to be orthogonal to a given vector

        Parameters
        ----------
        elements
        normals
        w
        B

        Returns
        -------

        """
        pass
