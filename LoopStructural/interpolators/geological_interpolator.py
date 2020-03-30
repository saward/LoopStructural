import logging

import numpy as np

from LoopStructural.core.core.geological_points import IPoint, GPoint, \
    TPoint

logger = logging.getLogger(__name__)


class GeologicalInterpolator:

    def __init__(self):
        """
        This class is the base class for a geological interpolator and contains all of the
        main interface functions. Any class that is inheriting from this should be callable
        by using any of these functions. This will enable interpolators to be interchanged.
        """
        self.p_i = []  # interface points #   TODO create data container
        self.p_g = []  # gradient points
        self.p_t = []  # tangent points
        self.p_n = []  # norm constraints
        self.n_i = 0
        self.n_g = 0
        self.n_t = 0
        self.n_n = 0
        self.type = 'undefined'
        self.up_to_date = False
        self.constraints = []
        self.propertyname = 'defaultproperty'
        self.__str = 'Base Geological Interpolator'

    def __str__(self):
        
        return self.__str
    
    def set_property_name(self, name):
        """
        Set the name of the interpolated property
        Parameters
        ----------
        name : string
            name of the property to be saved on a mesh

        Returns
        -------

        """
        self.propertyname = name

    def add_strike_dip_and_value(self, pos, strike, dip, val, w1 = 1., w2 = 1.):
        """
        Add a gradient and value constraint at a location gradient is in the form of strike and dip with the rh thumb

        Parameters
        ----------
        pos : numpy array
            position of the orientation and value thats being added
        strike : double
            strike of the plane in right hand thumb rule
        dip : double
            dip of the plane
        val : double
            value of the interpolant

        Returns
        -------

        """
        self.n_g += 1
        self.p_g.append(GPoint(pos, strike, dip, w1))
        self.n_i = self.n_i + 1
        self.p_i.append(IPoint(pos, val, w2))
        self.up_to_date = False

    def add_point(self, pos, val, w=1.):
        """
        Add a value constraint to the interpolator
        Parameters
        ----------
        pos : numpy array
            location of the value constraint
        val : double
            value of the constraint

        Returns
        -------

        """

        self.n_i = self.n_i + 1
        self.p_i.append(IPoint(pos, val,w))

    def add_planar_constraint(self, position, vector, norm=True, w=1.):
        """
        Add a gradient constraint to the interpolator where the gradient is defined by a normal vector

        Parameters
        ----------
        pos
        val
        norm bool
            whether to constrain the magnitude and directon or only direction

        Returns
        -------

        """
        if norm:
            self.n_n = self.n_n+1
            self.p_n.append(GPoint(position,vector,w))
        elif norm is False:
            self.n_g = self.n_g+1
            self.p_g.append(GPoint(position,vector,w))

        self.up_to_date = False

    def add_strike_and_dip(self, pos, s, d, norm=True, w=1.):
        """
        Add gradient constraint to the interpolator where the gradient is defined by strike and dip

        Parameters
        ----------
        pos : 
        s
        d
        norm

        Returns
        -------


        """
        self.n_g += 1
        self.p_g.append(GPoint(pos, s, d,w))
        self.up_to_date = False

    def add_tangent_constraint(self, pos, val,w):
        """
        Add tangent constraint to the interpolator where the tangent is
        described by a vector
        :param pos:
        :param val:
        :return:
        """
        self.n_t = self.n_t + 1
        self.p_t.append(TPoint(pos, val,w))
        self.up_to_date = False

    # def add_tangent_constraint_angle(self, pos, strike):
    #     """
    #     Add tangent constraint to the interpolator where the trangent is
    #     described by the strike and dip
    #     :param pos:
    #     :param s:
    #     :param d:
    #     :return:
    #     """
    #     self.n_t = self.n_t + 1
    #     self.p_t.append(TPoint(pos, strike, d))
    #     self.up_to_date = False

    def add_data(self, data):
        """
        Adds a GeologicalData object to the interpolator
        :param data:
        :return:
        """
        if data.type == 'GPoint':
            if not data.norm:
                self.p_g.append(data)
                self.n_g += 1
            if data.norm:
                self.p_n.append(data)
                self.n_n += 1
            self.up_to_date = False
            return
        if data.type == 'IPoint':
            self.p_i.append(data)
            self.n_i += 1
            self.up_to_date = False
            return
        if data.type == 'TPoint':
            self.p_t.append(data)
            self.n_t += 1
            self.up_to_date = False
            return
        else:
            print("Did not add data", data.type)

    def get_value_constraints(self):
        """

        Returns
        -------
        numpy array
        """
        points = np.zeros((self.n_i,5))#array

        for i in range(self.n_i):
            points[i, :3] = self.p_i[i].pos
            points[i, 3] = self.p_i[i].val
            points[i, 4] = self.p_i[i].weight
        return points

    def get_gradient_constraints(self):
        """

        Returns
        -------
        numpy array
        """
        points = np.zeros((self.n_g, 7))  # array
        for i in range(self.n_g):
            points[i, :3] = self.p_g[i].pos
            points[i, 3:6] = self.p_g[i].vec
            points[i, 6] = self.p_g[i].weight
        return points

    def get_tangent_constraints(self):
        """

        Returns
        -------
        numpy array
        """
        points = np.zeros((self.n_t,7))  # array

        for i in range(self.n_t):
            points[i, :3] = self.p_t[i].pos
            points[i, 3:6] = self.p_t[i].dir
            points[i, 6] = self.p_t[i].weight
        return points

    def get_norm_constraints(self):
        """

        Returns
        -------
        numpy array
        """
        points = np.zeros((self.n_n,7))

        for i in range(self.n_n):
            points[i, :3] = self.p_n[i].pos
            points[i, 3:6] = self.p_n[i].vec
            points[i, 6] = self.p_n[i].weight
        return points

    def setup_interpolator(self, **kwargs):
        """
        Runs all of the required setting up stuff
        """
        self._setup_interpolator(**kwargs)

    def solve_system(self, **kwargs):
        """
        Solves the interpolation equations
        """
        self._solve(**kwargs)
        self.up_to_date = True
