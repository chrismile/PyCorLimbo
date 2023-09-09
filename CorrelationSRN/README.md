# CorrelationSRN

In this directory, an example is given how PyCorLimbo can be used for training a scene representation network (SRN)
to learn the reconstruction of correlation fields. 

For more details on using SRNs for encoding correlation fields refer to:
https://www.cs.cit.tum.de/cg/research/publications/2023/neural-fields-for-interactive-visualization-of-statistical-dependencies-in-3d-simulation-ensembles/

Prerequisites:
- PyCorLimbo (see parent directory)
- [PyCoriander](https://github.com/chrismile/PyCoriander)
- [PyTorch](https://pytorch.org/), [Numpy](https://numpy.org/), [netCDF4](https://github.com/Unidata/netcdf4-python),
  [commentjson](https://github.com/vaidik/commentjson), [Numba](https://numba.pydata.org/)
- A NetCDF scalar volume data set with time or ensemble dimension.
- A .zip file containing three network description files: `config.json`, `config_encoder.json`, `config_decoder.json`
  (for more details see: https://zenodo.org/record/8186686)
