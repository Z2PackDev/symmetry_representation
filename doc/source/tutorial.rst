.. (c) 2017-2018, ETH Zurich, Institut fuer Theoretische Physik
.. Author: Dominik Gresch <greschd@gmx.ch>


.. _tutorial:

Tutorial
========

In this short tutorial, we will describe the mathematical notation used in the ``symmetry-representation`` library, and show how the symmetry operations for InAs can be constructed.

Notation
--------

In general, the symmetries of a crystal can be described as a space group :math:`G`, with symmetry operations :math:`g \in G`. The real-space effect of a symmetry :math:`\{S_g | \boldsymbol{\alpha}_g \}` (using the so-called Seitz notation) is given by

.. math::

    g~\mathbf{r} = S_g \mathbf{r} + \boldsymbol{\alpha}_g,

where :math:`S_g` is a rotation matrix, and :math:`\boldsymbol{\alpha}_g` is a translation vector.

For a Hamiltonian :math:`\mathcal{H}(\mathbf{k})`, the effect of these symmetries can be captured in the symmetry constraint

.. math::

    \mathcal{H}(\mathbf{k}) = D^\mathbf{k}(g) \mathcal{H}(g^{-1}\mathbf{k}) D^\mathbf{k}(g^{-1}),

where :math:`D^\mathbf{k}(g)` are the symmetry representations. The :math:`\mathbf{k}` - dependence of these representations comes from the translational part of the symmetry operation, and can be expressed as

.. math::

    D^\mathbf{k}(g) = e^{i \boldsymbol{\alpha}_{g}.\mathbf{k}} D(g),

where :math:`D(g)` is a :math:`\mathbf{k}` - `independent` representation matrix. In the symmetry constraint, these :math:`\mathbf{k}` - dependent phase factors cancel out:

.. math::

    \mathcal{H}(\mathbf{k}) = D^\mathbf{k}(g) \mathcal{H}(g^{-1}\mathbf{k}) D^\mathbf{k}(g^{-1}).

For this reason, we consider only the :math:`\mathbf{k}` - independent representation matrices :math:`D(g)` in the symmetry-representation code.

Depending on the kind of symmetry operation, these representation matrices are either unitary or anti-unitary. That is, we can write them as either :math:`D(g) = U_g` (unitary) or :math:`D(g) = U_g \hat{K}` (anti-unitary), where :math:`U_g` is a unitary matrix, and :math:`\hat{K}` is the complex conjugation operator.


The :class:`.SymmetryOperation` and :class:`.SymmetryGroup` classes
-------------------------------------------------------------------

The main purpose of the symmetry-representation code is to provide classes which describe these symmetry operations and their representations. These can then be used by other codes, such as `TBmodels <http://z2pack.ethz.ch/tbmodels>`_ or `kdotp-symmetry <http://z2pack.ethz.ch/kdotp-symmetry>`_. It also provides helper functions which allow for more easily constructing such symmetry operations.

Within symmetry-representation, symmetry operations are described using the :class:`.SymmetryOperation` class. This class has the following parameters:

* ``rotation_matrix``: The rotation matrix :math:`S_g`, in reduced coordinates.
* ``translation_vector``: The translation vector :math:`\boldsymbol{\alpha}_{g}`, in reduced coordinates.
* ``repr_matrix``: The matrix :math:`U_g` of the representation matrix.
* ``has_cc``: A Boolean, which describes if the representation :math:`D(g)` contains the complex conjugation operator :math:`\hat{K}`.

These symmetry operations can be combined into a group, using the :class:`.SymmetryGroup` class. In addition to the list of symmetry operations, it has a Boolean parameter ``full_group``. This parameter describes if the symmetries given represent the full group (``True``), or only a generating subset (``False``).

Automatic construction from orbitals
------------------------------------

A common obstacle to creating such symmetry operations is that their representation matrices are not known. Commonly, only the shape of the orbitals which span the basis of a given model is known. For this case, symmetry-representation provides helper functions to create the symmetry operations directly from the orbitals, and real-space symmetry operations only.

Here, we show the use of these helper functions for the case of a model for InAs, with :math:`s` and :math:`p` orbitals located on the In atom, and :math:`p` orbitals located on As. The first task is to create a list of orbitals in the model. Each orbital requires three inputs:

* The position, in reduced coordinates.
* The shape of the orbital, as a string which expresses the shape in terms of cartesian coordinates ``x``, ``y`` and ``z``.
* The spin of the orbital, as a :class:`.Spin` instance.

For the case of InAs, the following code constructs the correct orbitals:

.. ipython::

    In [0]: import symmetry_representation as sr

    In [0]: pos_In = (0, 0, 0)

    In [0]: pos_As = (0.25, 0.25, 0.25)

    In [0]: orbitals = []

    In [0]: for spin in (sr.SPIN_UP, sr.SPIN_DOWN):
       ...:     orbitals.extend([
       ...:         sr.Orbital(position=pos_In, function_string=fct, spin=spin)
       ...:         for fct in sr.WANNIER_ORBITALS['s'] + sr.WANNIER_ORBITALS['p']
       ...:     ])
       ...:     orbitals.extend([
       ...:         sr.Orbital(position=pos_As, function_string=fct, spin=spin)
       ...:         for fct in sr.WANNIER_ORBITALS['p']
       ...:     ])

Here we used some pre-defined constants of the symmetry-representation code, namely the spins

.. ipython::

    In [0]: sr.SPIN_UP

    In [0]: sr.SPIN_DOWN

and the orbitals as created by the Wanier90 code

.. ipython::

    In [0]: sr.WANNIER_ORBITALS['s']

    In [0]: sr.WANNIER_ORBITALS['p']

Having defined the orbitals, we also need to obtain the real-space symmetry operations of InAs. Since it has a symmorphic symmetry group, we only need rotation matrices, in both cartesian and reduced coordinates. We can use the `pymatgen <http://pymatgen.org>`_ code to simplify this:

.. ipython::

    In [0]: import pymatgen as mg

    In [0]: structure = mg.Structure(
       ...:     lattice=[[0., 3.029, 3.029], [3.029, 0., 3.029], [3.029, 3.029, 0.]],
       ...:     species=['In', 'As'],
       ...:     coords=np.array([pos_In, pos_As])
       ...: )

    In [0]: analyzer = mg.symmetry.analyzer.SpacegroupAnalyzer(structure)

    In [0]: sym_ops = analyzer.get_symmetry_operations(cartesian=False)

    In [0]: sym_ops_cart = analyzer.get_symmetry_operations(cartesian=True)

And finally, we can use the symmetry group using the :meth:`.from_orbitals` method of the :class:`.SymmetryOperation` class:

.. ipython::

    In [0]: symmetry_group = sr.SymmetryGroup(
       ...:     symmetries=[
       ...:         sr.SymmetryOperation.from_orbitals(
       ...:             orbitals=orbitals,
       ...:             real_space_operator=sr.RealSpaceOperator.
       ...:             from_pymatgen(sym_reduced),
       ...:             rotation_matrix_cartesian=sym_cart.rotation_matrix,
       ...:             numeric=True
       ...:         ) for sym_reduced, sym_cart in zip(sym_ops, sym_ops_cart)
       ...:     ],
       ...:     full_group=True
       ...: )

The ``numeric`` flag determines whether ``numpy`` arrays (``True``) or ``sympy`` matrices (``False``) are created. The former are suited for use with the `TBmodels <http://z2pack.ethz.ch/tbmodels>`_ code, while the latter should be used for `kdotp-symmetry <http://z2pack.ethz.ch/kdotp-symmetry>`_. Note that in order to create correct ``sympy`` matrices, the real space matrices should `also` be sympy matrices.

Additionally, there is a special helper function to create a representation of the time-reversal symmetry operation:

.. ipython::

    In [0]: time_reversal = sr.get_time_reversal(orbitals=orbitals, numeric=True)
