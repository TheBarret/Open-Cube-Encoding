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

# Proof of Concept

```bash
python demo.py --demo --headless
=== Open Cube Encoding System ===

Generating open cube library...
  Testing configurations with 3 edges...
Generated 16 unique open cubes
Initialized encoder: M=16, R=24, P=6
Bits per token: 11.17
Original message: 'Hello, World!'
Encoded tokens: [(6, 16, 3), (5, 0, 5), (12, 22, 0), (12, 22, 0), (10, 0, 2), (12, 11, 0), (5, 7, 4), (12, 11, 1), (10, 0, 2), (10, 0, 4), (12, 22, 0), (9, 3, 4), (1, 6, 3)]
Decoded message: 'Hello, World!'
Encoding successful: True

Generating 3D structure...

Library statistics:
  Unique cubes: 16
  Total symbol space: 2304
  Bits per symbol: 11.17
Generated mesh with 2574 vertices
```

Visual render:  

<img width="240" alt="image" src="https://github.com/user-attachments/assets/92676f50-965f-43fc-881f-5601cf6525ab" />
<img width="240" alt="image" src="https://github.com/user-attachments/assets/8d21ef46-6a21-4f0d-99aa-637d1fd6512f" />


# Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Encode a message (prints tokens as JSON)
python demo.py encode --message "Hello, World!" --nonce 12345 --key 000000

# Decode tokens (provide tokens JSON)
python demo.py decode --tokens "[[6,16,3],[5,0,5]]" --nonce 12345 --key 000000

# Render tokens to OBJ (no display in headless)
python demo.py render --tokens "[[6,16,3],[5,0,5]]" --output encoded_message.obj --headless

# Show library stats
python demo.py stats
```

Flags:
- `--key`: master key bytes (ASCII) for deterministic PRF
- `--nonce`: nonce bytes (ASCII) to derive per-stream keys
- `--max-cubes`: limit library size (default 16)
- `--headless`: skip visualization window


# Information

This project was inspired and credits where credits are due by:

https://www.youtube.com/watch?v=_BrFKp-U8GI
