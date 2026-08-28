# DAOC Reference

## Bibliographic Record

Mohammad Reza Golzari Oskoui and Brunilde Sansò, "Distributed Dependency-Aware Task Offloading and Service Caching in Cloudlet-Based Edge Computing Networks," *IEEE Transactions on Services Computing*, vol. 19, no. 2, pp. 1120--1133, 2026.

- DOI: https://doi.org/10.1109/TSC.2026.3664339
- IEEE publication page: https://ieeexplore.ieee.org/document/11395623
- Author/institutional technical-report record: https://www.gerad.ca/en/papers/G-2026-32
- GERAD report identifier: `G-2026-32`

## Full-Text Boundary

The locally consulted file is the IEEE publisher PDF. Its first page states that personal use is permitted but republication or redistribution requires IEEE permission. Because this repository is public, that licensed copy is not committed here.

The GERAD page above is the author-affiliated public access point and provides its own PDF download link. Use that institutional record or the DOI to obtain the paper legally. This repository stores only citation metadata and links, not a redistributed publisher PDF.

## Role in This Project

This work is the primary baseline paper referred to as `DAOC` in the manuscript, code, experiment protocols, and Pro-review files. The pristine authors' implementation is pinned under `upstream/daoc`; the audited reproduction path and this project's extensions are under `reproducible_code/`. Claims about the paper itself should still be checked against the official article or the author-affiliated report.

## Official Source Code

- Authors' repository: https://github.com/MR-Golzari/distributed-dag-offloading-caching
- Pinned project submodule: `upstream/daoc`
- Pinned commit: `15596b3137e2e0a61d8b36c073c8a250deb5f2f5`

Initialize it with `git submodule update --init --recursive`. See `upstream/README.md` for the separation between the pristine DAOC source and this project's extended reproducibility code.
