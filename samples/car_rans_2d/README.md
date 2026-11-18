# car_rans_2d

This is the original `car_rans` case transferred to the supplied `2d.stl`.
The original velocity, turbulence model, physical properties, numerical
schemes, run length, write interval, and force settings are retained.

OpenFOAM's `snappyHexMesh` cannot operate on a mesh containing `empty`
patches.  The case therefore uses a thin 3D slab with `symmetryPlane` on both
sides.  This is a quasi-2D model: the solution is constrained in the thin
direction while the mesh remains valid for snappyHexMesh.

Run from any directory:

```sh
/Users/allebedev/Desktop/openfoam-pinns/car_rans_2d/Allrun
```

The automatically sampled centre plane is written below
`postProcessing/cutPlane/`.

Clean the generated mesh, time directories, VTK files, post-processing
results, and logs while preserving the case setup:

```sh
/Users/allebedev/Desktop/openfoam-pinns/car_rans_2d/AllClean
```
