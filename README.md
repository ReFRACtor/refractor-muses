ReFRACtor
=========

Jet Propulsion Laboratory, California Institute of Technology. \
Copyright 2022 California Institute of Technology. \
U.S. Government sponsorship acknowledged.

(NPO 52201-CP)

The Reusable Framework for Retrieval of Atmospheric Composition (ReFRACtor) software transforms radiance data from multiple co-located instruments into physical quantities such as ozone volume mixing ratio using an optimal estimation-based retrieval process. It provides an extensible multiple instrument Earth science atmospheric composition radiative transfer and retrieval software framework that enables software reuse while enabling data fusion.

ReFRACtor is designed to connect components through abstract interfaces that make the fewest possible implementation assumptions. The isolation of details behind abstract interfaces allows the interchange of different component implementations without needing to modify existing tested code. The core framework software repository contains generic algorithms that are reusable by multiple instrument adaptation implementations. Adaptations are located in separate software repositories that contain instrument specific configuration, data and algorithms.

See [refractor-framework](https://github.com/ReFRACtor/framework) for more details

ReFRACtor/MUSES
---------------

This repository contains code for integrating ReFRACtor with the MUSES pipeline.
