# Archived Python Runtime Modules

This directory holds Python modules that used to live in the shipped
`voyager_index._internal.inference` tree but are no longer part of the active
runtime surface.

They remain in-repo for historical reference and archaeology, but they are not
packaged, documented, or exercised by the main shard-first CI lanes.

Current archived areas:

- `inference/index_gpu/`: backward-compatibility GPU re-export package
- `inference/gym/`: vector-environment experimentation code
- `inference/control/`: experimental control-law primitives
- `inference/distributed/`: unfinished distributed router prototype
- `inference/index_core/gem_segment_manager.py`: unused sealed GEM wrapper
