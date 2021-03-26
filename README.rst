Pooling Network Library for Pyomo
=================================

This library provides classes and functions to describe pooling-like network
optimization problems.

The ``pooling_network`` library can be used to describe the network structure of
pooling problems and then generate Pyomo optimization problems from it.
Currently the library supports generating optimization problems using the PQ-formulation.

The library also includes a MIP-restriction of the pooling problem [DEY2015]_ that
performs well on dense instances, and a convex relaxation of the pooling problem
used to generate valid cuts [LUEDTKE2020]_.

Contributors
------------
.. list-table::
   :header-rows: 1
   :widths: 10 40 50

   * - GitHub
     - Name
     - Acknowledgements

   * - |fracek|_
     - Francesco Ceccon
     - This work was funded by an Engineering & Physical Sciences Research Council Research Fellowship [GrantNumber EP/P016871/1]

.. _fracek: https://github.com/fracek
.. |fracek| image:: https://avatars1.githubusercontent.com/u/282580?s=120&v=4
   :width: 80px

References
----------

.. [DEY2015] Dey, S.S. and Gupte, A., 2015.
   Analysis of MILP techniques for the pooling problem.
   Operations Research, 63(2), pp.412-427.

.. [LUEDTKE2020] Luedtke, J., D'ambrosio, C., Linderoth, J. and Schweiger, J., 2020.
   Strong convex nonlinear relaxations of the pooling problem.
   SIAM Journal on Optimization, 30(2), pp.1582-1609.


License
-------

Copyright 2020 Francesco Ceccon

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.