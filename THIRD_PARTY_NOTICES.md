# Third-party notices

`template/run_alphafold.py` is vendored from
`sokrypton/alphafold3` tag `v3.1.18` and retains its DeepMind copyright and
Apache License 2.0 notice.

The runtime image contains ColabFold, AlphaFold 3 source, and third-party
scientific software under their respective licences. The converted BioEmu2
weights remain subject to the licence and notices published with the original
BioEmu2 model and the `sokrypton/af3-any-model` distribution. The app does not
redistribute those weights; administrators download them into runtime storage.

`template/3Dmol.min.js` is vendored from the `3dmol` npm distribution
(v2.4.0, David Koes and contributors, BSD-style licence; the full licence
text ships alongside it upstream) and is copied next to `report.html` at
report-build time so the interactive structure view loads from the same
origin with no external service.

Preserve model-produced `TERMS_OF_USE.md` files with every result.
