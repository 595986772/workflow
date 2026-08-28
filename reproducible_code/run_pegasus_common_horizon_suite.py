#!/usr/bin/env python3
"""Register the Pegasus 26k profile and delegate to the suite runner."""

import run_reproduction_suite
from pegasus_common_horizon_protocol import register_profile


if __name__ == "__main__":
    register_profile()
    run_reproduction_suite.main()
