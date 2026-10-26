# SPDX-License-Identifier: MIT
"""Simulator backends of gplugins.devsim, one module per simulator.

Each backend turns a :mod:`gplugins.devsim.device_description` into a solved device; that module
states the contract. :mod:`gplugins.devsim.backends.devsim` is the DEVSIM backend. A Charon
backend (``charon.py``) is to follow (gf180-tcad spec §12.1). This package imports no backend:
import the one you use.
"""
