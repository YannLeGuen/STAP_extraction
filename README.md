Description of the steps to extract the STAP section in the superior temporal sulcus

First, follow the steps describe in
https://github.com/YannLeGuen/sulcal_pits_analysis/blob/master/build_pits_database/README.md
to build your sulcal pits database

Second, back-project your sulcal pits group areals from the fsaverage onto the native white mesh

Notably each subject needs to have:
- a white mesh for both hemisphere
- a sulcal pits texture on this native mesh
- a geodesic depth texture on this native mesh
- a DPF (depth potential function) on this native mesh
- a texture containing the group areals on this native mesh

Third, locate the "deep" sulcal pits of each subject in each group areal

Fourth, extract the shortest geodesic paths between sulcal pits STS b-STS c, and STS c-STS d.
These two set of vertices are then concatenated as one geodesic path.
According to Leroy et al (2015) Talaraich coordinates of the STAP, the posterior border of the STAP roughly
corresponds to the border between areal STS c and STS d.
Thus, the previous STAP geodesic path is truncated at the border between these two areals.

Sulcal interruptions, so-called plis de passage, are then identified using thresholds on the DPF and geodesic depth.
These thresholds were benchmarked on data used in Leroy et al (2015) for which plis de passage were manually annotated.

The last two steps can be repeated between any pair of sulcal pits to identify a pli de passage (sulcal interruption),
in every segment between two pits.