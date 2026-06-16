# Data directory

This directory holds datasets and precomputed graph structures. None of it is
tracked in git (see `.gitignore`).

## Solov'ev datasets (`Solovev/`)

Per-geometry `.mat` files with arrays `coeff` (N, s) and `sol` (N, s, s).
Generate them with:

```bash
python scripts/generate_data.py --out_dir data/Solovev --workers 8
```

or download the released archive and unpack here.

## Graph cache (`graph_cache/`)

The Sp²GNO operator uses precomputed per-geometry graph structures to avoid
recomputing the KNN graph, edge weights, Laplacian eigenmodes, and Lipschitz
embeddings on every run:

```
<geom>_U.pt           truncated graph-Laplacian eigenvectors
<geom>_edge_ind.pt    KNN edge index (k = 50 on the 32x32 grid)
<geom>_edge_wei.pt    edge weights (clipped inverse Euclidean distance)
<geom>_lip_embeds.pt  Lipschitz node embeddings
```

These are produced by the Sp²GNO setup utilities (`stone/utilities.py`,
`stone/lipschitz.py`). They are released with the dataset archive; regenerate
them only if you change the grid resolution or KNN parameters.
