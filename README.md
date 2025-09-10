# Open Cube Encoding
Geometric Encoding Scheme Using Rotationally Unique Open Cubes

<img width="700" alt="d7e4aaa1-a52c-401e-99c3-b35a18f64064" src="https://github.com/user-attachments/assets/30b7e7e5-eec3-4c6c-9bc6-4d6bbb79206b" />

# Abstract

A data encoding method based on rotationally unique open cubes.

Each cube serves as a geometric token parameterized by type, orientation, and placement.

By assembling sequences of such tokens, digital data can be transformed into 3D spatial structures.

<img width="233" height="229" alt="image" src="https://github.com/user-attachments/assets/ab329918-ac83-47d1-a2bb-f9760cf62ab6" />

A three-dimensional primitive: the open cube, derived from a skeletal cube with selective edge removal.

By enforcing rules of connectivity, dimensionality, and rotational uniqueness,

we obtain a finite library of distinct cube types.

When combined with rotations and placements, these cubes form a rich symbol space.

# Definitions & Encoding Capacity

- Library `C`       : a finite set of M rotationally unique open cubes.
- Rotations `R`     : allowable orientations in 3D space (typically 24 for a cube).
- Placements `P`    : relative adjacency options (e.g., 6 face-to-face positions).
- Token             : `t = (c, r, p) where c ∈ C, r ∈ R, p ∈ P`
 
## Capacity
```
S = M × R × P
Bits per token = log (S)
```

# Information

This project was inspired and credits where credits are due by:

https://www.youtube.com/watch?v=_BrFKp-U8GI
