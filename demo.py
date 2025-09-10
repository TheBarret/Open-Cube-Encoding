import numpy as np
import trimesh
from hashlib import sha256
from hmac import HMAC
from itertools import combinations, product
import json
from typing import List, Tuple, Set, Dict, Optional

class OpenCube:
    """Represents an open cube with selective edge removal"""
    
    def __init__(self, edges: Set[Tuple[int, int]]):
        """
        Initialize with a set of edges (vertex pairs)
        Vertices are numbered 0-7 following standard cube indexing:
        Bottom face: 0,1,2,3 (counterclockwise)
        Top face: 4,5,6,7 (counterclockwise, matching bottom)
        """
        self.edges = frozenset(edges)
        self.vertices = self._get_vertices()
        
    def _get_vertices(self) -> Set[int]:
        """Extract unique vertices from edges"""
        vertices = set()
        for v1, v2 in self.edges:
            vertices.add(v1)
            vertices.add(v2)
        return vertices
    
    def is_connected(self) -> bool:
        """Check if the cube forms a connected graph"""
        if not self.vertices:
            return False
            
        # BFS to check connectivity
        visited = set()
        queue = [next(iter(self.vertices))]
        
        while queue:
            vertex = queue.pop(0)
            if vertex in visited:
                continue
            visited.add(vertex)
            
            # Find neighbors
            for v1, v2 in self.edges:
                if v1 == vertex and v2 not in visited:
                    queue.append(v2)
                elif v2 == vertex and v1 not in visited:
                    queue.append(v1)
        
        return len(visited) == len(self.vertices)
    
    def is_3dimensional(self) -> bool:
        """Check if the cube spans all three dimensions"""
        if len(self.vertices) < 4:
            return False
            
        # Standard cube vertex coordinates
        coords = {
            0: (0, 0, 0), 1: (1, 0, 0), 2: (1, 1, 0), 3: (0, 1, 0),
            4: (0, 0, 1), 5: (1, 0, 1), 6: (1, 1, 1), 7: (0, 1, 1)
        }
        
        vertex_coords = [coords[v] for v in self.vertices]
        
        # Check if vertices span all three dimensions
        x_coords = set(coord[0] for coord in vertex_coords)
        y_coords = set(coord[1] for coord in vertex_coords)
        z_coords = set(coord[2] for coord in vertex_coords)
        
        return len(x_coords) > 1 and len(y_coords) > 1 and len(z_coords) > 1
    
    def get_rotations(self) -> List['OpenCube']:
        """Generate all 24 possible rotations of the cube"""
        # 24 rotation matrices for cube orientations
        rotations = []
        
        # Standard cube vertex coordinates
        coords = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top
        ])
        
        # Generate 24 rotation matrices (6 faces × 4 rotations each)
        for face in range(6):
            for rot in range(4):
                # Apply rotation
                if face == 0:  # +Z up
                    matrix = trimesh.transformations.rotation_matrix(rot * np.pi/2, [0,0,1])
                elif face == 1:  # -Z up
                    matrix = trimesh.transformations.rotation_matrix(np.pi, [1,0,0])
                    matrix = matrix @ trimesh.transformations.rotation_matrix(rot * np.pi/2, [0,0,1])
                elif face == 2:  # +Y up
                    matrix = trimesh.transformations.rotation_matrix(-np.pi/2, [1,0,0])
                    matrix = matrix @ trimesh.transformations.rotation_matrix(rot * np.pi/2, [0,0,1])
                elif face == 3:  # -Y up
                    matrix = trimesh.transformations.rotation_matrix(np.pi/2, [1,0,0])
                    matrix = matrix @ trimesh.transformations.rotation_matrix(rot * np.pi/2, [0,0,1])
                elif face == 4:  # +X up
                    matrix = trimesh.transformations.rotation_matrix(np.pi/2, [0,1,0])
                    matrix = matrix @ trimesh.transformations.rotation_matrix(rot * np.pi/2, [0,0,1])
                elif face == 5:  # -X up
                    matrix = trimesh.transformations.rotation_matrix(-np.pi/2, [0,1,0])
                    matrix = matrix @ trimesh.transformations.rotation_matrix(rot * np.pi/2, [0,0,1])
                
                # Apply transformation
                rotated_coords = trimesh.transformations.transform_points(coords, matrix)
                
                # Round to avoid floating point errors
                rotated_coords = np.round(rotated_coords).astype(int)
                
                # Create mapping from old vertices to new vertices
                vertex_mapping = {}
                for old_v, new_coord in enumerate(rotated_coords):
                    # Find which original vertex this coordinate corresponds to
                    for orig_v, orig_coord in enumerate(coords):
                        if np.allclose(new_coord, orig_coord, atol=0.1):
                            vertex_mapping[old_v] = orig_v
                            break
                
                # Map edges using the vertex mapping
                new_edges = set()
                for v1, v2 in self.edges:
                    if v1 in vertex_mapping and v2 in vertex_mapping:
                        new_v1, new_v2 = vertex_mapping[v1], vertex_mapping[v2]
                        new_edges.add((min(new_v1, new_v2), max(new_v1, new_v2)))
                
                rotations.append(OpenCube(new_edges))
        
        return rotations
    
    def __hash__(self):
        return hash(self.edges)
    
    def __eq__(self, other):
        return isinstance(other, OpenCube) and self.edges == other.edges
    
    def __repr__(self):
        return f"OpenCube({len(self.edges)} edges, {len(self.vertices)} vertices)"


class OpenCubeLibrary:
    """Generates and manages the library of rotationally unique open cubes"""
    
    def __init__(self, max_cubes: int = 32):
        self.max_cubes = max_cubes
        self.cubes: List[OpenCube] = []
        self._generate_library()
    
    def _generate_library(self):
        """Generate all valid rotationally unique open cubes"""
        print("Generating open cube library...")
        
        # All possible edges in a cube
        full_edges = {
            # Bottom face edges
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top face edges
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7)
        }
        
        unique_cubes = set()
        
        # Try different numbers of edges (from minimal to full cube)
        for num_edges in range(3, len(full_edges) + 1):
            if len(unique_cubes) >= self.max_cubes:
                break
                
            print(f"  Testing configurations with {num_edges} edges...")
            
            for edges in combinations(full_edges, num_edges):
                cube = OpenCube(set(edges))
                
                # Check if it meets our criteria
                if not cube.is_connected():
                    continue
                if not cube.is_3dimensional():
                    continue
                
                # Check rotational uniqueness
                is_unique = True
                for existing_cube in unique_cubes:
                    if self._are_rotationally_equivalent(cube, existing_cube):
                        is_unique = False
                        break
                
                if is_unique:
                    unique_cubes.add(cube)
                    if len(unique_cubes) >= self.max_cubes:
                        break
        
        self.cubes = list(unique_cubes)
        print(f"Generated {len(self.cubes)} unique open cubes")
    
    def _are_rotationally_equivalent(self, cube1: OpenCube, cube2: OpenCube) -> bool:
        """Check if two cubes are rotationally equivalent"""
        rotations = cube1.get_rotations()
        return cube2 in rotations
    
    def get_cube(self, index: int) -> OpenCube:
        """Get cube by index"""
        return self.cubes[index % len(self.cubes)]
    
    def size(self) -> int:
        """Get library size"""
        return len(self.cubes)


class OpenCubeEncoder:
    """Complete encoding system using open cubes"""
    
    def __init__(self, master_key: bytes = b'default_key', max_cubes: int = 16):
        self.master_key = master_key
        self.library = OpenCubeLibrary(max_cubes)
        self.M = self.library.size()
        self.R = 24  # rotations
        self.P = 6   # placements (face adjacencies)
        
        print(f"Initialized encoder: M={self.M}, R={self.R}, P={self.P}")
        print(f"Bits per token: {np.log2(self.M * self.R * self.P):.2f}")
    
    def _prf(self, key: bytes, data: bytes) -> int:
        """randomizer"""
        return int.from_bytes(HMAC(key, data, sha256).digest(), 'big')
    
    def _derive_keys(self, nonce: bytes) -> Tuple[bytes, bytes, bytes]:
        """Derive subkeys for encoding"""
        k_type = HMAC(self.master_key, nonce + b'type', sha256).digest()
        k_rot = HMAC(self.master_key, nonce + b'rot', sha256).digest()
        k_place = HMAC(self.master_key, nonce + b'place', sha256).digest()
        return k_type, k_rot, k_place
    
    def encode(self, message: str, nonce: bytes = b'default_nonce') -> List[Tuple[int, int, int]]:
        """Encode message into cube tokens"""
        k_type, k_rot, k_place = self._derive_keys(nonce)
        tokens = []
        
        for byte in message.encode('utf-8'):
            byte_data = bytes([byte])
            
            cube_type = self._prf(k_type, byte_data) % self.M
            rotation = self._prf(k_rot, byte_data) % self.R
            placement = self._prf(k_place, byte_data) % self.P
            
            tokens.append((cube_type, rotation, placement))
        
        return tokens
    
    def decode(self, tokens: List[Tuple[int, int, int]], nonce: bytes = b'default_nonce') -> str:
        """Decode cube tokens back to message"""
        k_type, k_rot, k_place = self._derive_keys(nonce)
        message_bytes = []
        
        # Brute force
        for cube_type, rotation, placement in tokens:
            for byte_val in range(256):
                byte_data = bytes([byte_val])
                
                test_type = self._prf(k_type, byte_data) % self.M
                test_rot = self._prf(k_rot, byte_data) % self.R
                test_place = self._prf(k_place, byte_data) % self.P
                
                if (test_type == cube_type and 
                    test_rot == rotation and 
                    test_place == placement):
                    message_bytes.append(byte_val)
                    break
        
        return bytes(message_bytes).decode('utf-8')
    
    def tokens_to_3d_structure(self, tokens: List[Tuple[int, int, int]], 
                             grid_size: Tuple[int, int, int] = (10, 10, 10)) -> np.ndarray:
        """Convert tokens to 3D voxel structure with proper placement logic"""
        grid = np.zeros(grid_size, dtype=int)  # 0 = empty, >0 = cube type
        
        current_pos = np.array([0, 0, 0])
        
        for i, (cube_type, rotation, placement) in enumerate(tokens):
            # Place current cube
            if all(0 <= current_pos[j] < grid_size[j] for j in range(3)):
                grid[tuple(current_pos)] = cube_type + 1  # +1 to avoid 0 (empty)
            
            # Determine next position based on placement
            # 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
            directions = [
                (1, 0, 0), (-1, 0, 0),  # X directions
                (0, 1, 0), (0, -1, 0),  # Y directions
                (0, 0, 1), (0, 0, -1)   # Z directions
            ]
            
            direction = directions[placement]
            current_pos = current_pos + np.array(direction)
            
            # Wrap around if out of bounds
            current_pos = current_pos % np.array(grid_size)
        
        return grid
    
    def create_mesh_from_tokens(self, tokens: List[Tuple[int, int, int]]) -> trimesh.Trimesh:
        """Create a 3D mesh visualization from tokens"""
        grid = self.tokens_to_3d_structure(tokens)
        return self._grid_to_mesh(grid)
    
    def _grid_to_mesh(self, grid: np.ndarray) -> trimesh.Trimesh:
        """Convert voxel grid to trimesh object"""
        meshes = []
        
        for x in range(grid.shape[0]):
            for y in range(grid.shape[1]):
                for z in range(grid.shape[2]):
                    if grid[x, y, z] > 0:
                        # Create cube with color based on cube type
                        cube_type = grid[x, y, z] - 1
                        
                        # Get the actual open cube geometry
                        open_cube = self.library.get_cube(cube_type)
                        mesh = self._create_open_cube_mesh(open_cube, (x, y, z))
                        meshes.append(mesh)
        
        if meshes:
            return trimesh.util.concatenate(meshes)
        else:
            # Return empty mesh if no voxels
            return trimesh.Trimesh()
    
    def _create_open_cube_mesh(self, open_cube: OpenCube, position: Tuple[int, int, int]) -> trimesh.Trimesh:
        """Create mesh representation of an open cube"""
        # Standard cube vertex coordinates
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top
        ]) + np.array(position)
        
        # Create edges as thin cylinders
        edge_meshes = []
        for v1, v2 in open_cube.edges:
            start = vertices[v1]
            end = vertices[v2]
            
            # Create cylinder along edge
            direction = end - start
            length = np.linalg.norm(direction)
            
            if length > 0:
                cylinder = trimesh.creation.cylinder(radius=0.05, height=length)
                
                # Orient cylinder along edge
                z_axis = np.array([0, 0, 1])
                edge_axis = direction / length
                
                if not np.allclose(edge_axis, z_axis):
                    rotation_axis = np.cross(z_axis, edge_axis)
                    if np.linalg.norm(rotation_axis) > 1e-6:
                        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                        angle = np.arccos(np.clip(np.dot(z_axis, edge_axis), -1, 1))
                        rotation_matrix = trimesh.transformations.rotation_matrix(angle, rotation_axis)
                        cylinder.apply_transform(rotation_matrix)
                
                # Position cylinder
                center = (start + end) / 2
                translation = trimesh.transformations.translation_matrix(center)
                cylinder.apply_transform(translation)
                
                edge_meshes.append(cylinder)
        
        if edge_meshes:
            return trimesh.util.concatenate(edge_meshes)
        else:
            return trimesh.Trimesh()


def main():
    print("=== Open Cube Encoding System ===\n")
    
    # Initialize encoder
    encoder = OpenCubeEncoder(master_key=b'000000', max_cubes=16)
    
    # Test message
    message = "Hello, World!"
    print(f"Original message: '{message}'")
    
    # Encode
    nonce = b'12345'
    tokens = encoder.encode(message, nonce)
    print(f"Encoded tokens: {tokens}")
    
    # Decode
    decoded = encoder.decode(tokens, nonce)
    print(f"Decoded message: '{decoded}'")
    print(f"Encoding successful: {message == decoded}")
    
    # Create 3D visualization
    print("\nGenerating 3D structure...")
    mesh = encoder.create_mesh_from_tokens(tokens)
    
    # Library statistics
    print(f"\nLibrary statistics:")
    print(f"  Unique cubes: {encoder.library.size()}")
    print(f"  Total symbol space: {encoder.M * encoder.R * encoder.P}")
    print(f"  Bits per symbol: {np.log2(encoder.M * encoder.R * encoder.P):.2f}")

    
    if len(mesh.vertices) > 0:
        print(f"Generated mesh with {len(mesh.vertices)} vertices")
        
        # Show visualization (requires display)
        try:
            mesh.show()
            mesh.export('encoded_message.obj')
            print("Saved as 'encoded_message.obj'")
        except:
            print("Display not available!")
            
    else:
        print("No mesh generated")
    

if __name__ == "__main__":
    main()
