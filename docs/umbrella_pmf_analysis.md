# Umbrella PMF analysis

Production umbrella calculations should use real biased windows with overlap. The generic analysis API provides a pure-Python fallback and is structured so WHAM/MBAR implementations can be selected when available. Histogram proxy analysis is diagnostic only and must not be used as a production PMF unless explicitly allowed.
