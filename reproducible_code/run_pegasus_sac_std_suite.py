#!/usr/bin/env python3
"""Register the P7 method and delegate to the frozen reproduction runner."""

import run_reproduction_suite
from pegasus_sac_std_extension_protocol import register_suite_extension


if __name__ == "__main__":
    register_suite_extension()
    run_reproduction_suite.main()
