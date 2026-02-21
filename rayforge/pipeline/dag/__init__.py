"""
Dependency Graph (DAG) based pipeline execution system.

This module provides the core components for building and managing a
dependency graph of artifacts in the pipeline.
"""
# We must not import the scheduler module, as that drags in core in situations
# where we only want to import the graph or node modules.
