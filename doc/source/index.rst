.. _home:

symmetry-representation
=======================

This is a tool to describe symmetry operations and their representations.

.. contents ::
    :local:


Installation
~~~~~~~~~~~~

You can install this tool with with pip:

.. code ::

    pip install symmetry-representation

Creating the :class:`.SymmetryOperation`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. todo ::
    update

In order to run the code, we must first specify the symmetries as described above. To do this, we can use the :class:`.SymmetryOperation` class, which is a :py:func:`namedtuple <collections.namedtuple>` with two attributes ``kmatrix`` and ``repr``.

The first attribute, ``kmatrix``, is just the :math:`\mathbf{k}`-space matrix for the symmetry.

The second, ``repr``, describes the symmetry representation which can either be a unitary matrix :math:`U`, or a unitary matrix and complex conjugation :math:`U \hat{K}`. Because of this, the ``repr`` is another :py:func:`namedtuple <collections.namedtuple>` called :class:`.Representation` with two attributes ``matrix`` and ``complex_conjugate``. The ``matrix`` is the unitary matrix :math:`U`, and ``complex_conjugate`` is a :py:class:`bool` describing whether the representation contains complex conjugation (``True``) or not (``False``).

The following code creates the symmetries described above:

.. toctree::
    :hidden:

    Usage <self>
    reference.rst
