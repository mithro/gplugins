# SPDX-License-Identifier: MIT
"""Simulator backends of gplugins.devsim, one module per simulator.

Each backend turns a :mod:`gplugins.devsim.device_description` into a solved device; that module
states the contract (the :class:`~gplugins.devsim.device_description.Backend` protocol).
:mod:`gplugins.devsim.backends.devsim` is the DEVSIM backend; the Charon backend is the package
:mod:`gplugins.devsim.backends.charon` (a stub so far; gf180-tcad spec §12.1). This package
imports no backend: import the one you use.
"""
