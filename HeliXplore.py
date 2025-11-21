######################################################################################################
######################################################################################################
######################################################################################################
##### HeliXplore: a Python package for analyzing multi-strand helix deformations in biomolecules #####
################################################# v1.0.0 #############################################
######################################################################################################
######################################################################################################
######################################################################################################
import numpy as np # checks for dependency
import pandas as pd # checks for dependency
import matplotlib.pyplot as plt # checks for dependency
from scipy.spatial.transform import Rotation # checks for dependency
from scipy.optimize import minimize # checks for dependency
import sys # in-built in Python
import datetime # in-built in Python
import warnings # in-built in Python
import argparse # in-built in Python
import os # in-built in Python
import sys # in-built in Python
from pathlib import Path # in-built in Python
# from collections import defaultdict
warnings.filterwarnings('ignore')

"""
These are the main trajectory parsers for the code.
read_tinker_arc takes in the filename and the filters, and returns the number of frames, number of helix units, 
and a coordinates array of shape (frames, units, 3). read_traj_pdb does the same for a PDB.
They check for consistent atom counts across frames, validates frame structure, and filter for helix units only.
Replace these functions with your equivalent function for other trajectory file types.
"""

def read_tinker_arc(filename, filter_names=None, filter_types=None):
    """
    Reads a Tinker ARC trajectory file and extracts atom coordinates.
    
    Parameters:
    filename : str
        Path to the Tinker ARC file
    filter_names : list of str, optional
        Atom names to filter (e.g., ["CA", "CB", "N"]). Matches second column in ARC file.
    filter_types : list of int, optional
        Atom types to filter (e.g., [129, 130, 156]). Matches sixth column in ARC file.
        If both filter_names and filter_types are provided, filter_types takes priority.
        If neither is provided, defaults to filtering for atom name "CA".
    
    Returns:
    tuple : (num_frames, num_ca_atoms, coordinates_array) or (None, None, None) on error
        - num_frames: int, total number of frames read
        - num_ca_atoms: int, number of filtered atoms per frame
        - coordinates_array: numpy array of shape (num_frames, num_ca_atoms, 3)
    """
    
    # Convert single values to lists for uniform handling
    if filter_names is not None and not isinstance(filter_names, list):
        filter_names = [filter_names]
    if filter_types is not None and not isinstance(filter_types, list):
        filter_types = [filter_types]
    
    # Determine filtering mode and print what we're doing
    if filter_names is not None and len(filter_names) > 0 and filter_types is not None and len(filter_types) > 0:
        print(f"Both atom names {filter_names} and atom types {filter_types} provided.")
        print(f"Using atom types {filter_types} (more exact).")
        use_type = True
        use_name = False
    elif filter_types is not None and len(filter_types) > 0:
        print(f"Filtering by atom types: {filter_types}")
        use_type = True
        use_name = False
    elif filter_names is not None and len(filter_names) > 0:
        print(f"Filtering by atom names: {filter_names}")
        use_type = False
        use_name = True
    else:
        print("No filter provided. Defaulting to atom name 'CA'.")
        use_type = False
        use_name = True
        filter_names = ["CA"]
    
    # Initialize variables to track frame data
    all_frame_coordinates = []  # Will store CA coordinates for each frame
    expected_num_atoms = None   # Number of atoms in first frame (for consistency check)
    expected_num_ca = None      # Number of helix units in first frame (for consistency check)
    frame_number = 0            # Counter for current frame number
    
    try:
        # Open file and read line by line to avoid loading entire file into memory
        with open(filename, 'r') as f:
            current_line = f.readline()
            
            # Main loop: process the file until we reach the end
            while current_line:
                # Check if this line starts a new frame (contains only one word: atom count)
                words = current_line.rstrip().split()
                
                # If line has exactly one word, it's the start of a frame
                if len(words) == 1:
                    frame_number += 1
                    
                    # Parse the number of atoms in this frame
                    try:
                        num_atoms = int(words[0])
                    except ValueError:
                        print(f"Error: Frame {frame_number} has invalid atom count: {words[0]}")
                        return None, None, None
                    
                    # Check if atom count is consistent with first frame
                    if expected_num_atoms is None:
                        expected_num_atoms = num_atoms  # Set expected count from first frame
                    elif num_atoms != expected_num_atoms:
                        print(f"Error: Frame {frame_number} has {num_atoms} atoms, expected {expected_num_atoms}")
                        return None, None, None
                    
                    # Skip the box dimensions line (second line of frame)
                    box_line = f.readline()
                    if not box_line:
                        print(f"Error: Frame {frame_number} is incomplete (missing box dimensions line)")
                        return None, None, None
                    
                    # Read atom lines until we hit the next frame or end of file
                    frame_ca_coords = []  # Store CA coordinates for this frame
                    atom_lines_read = 0   # Count how many atom lines we've read
                    
                    while True:
                        # Peek at the next line
                        next_line = f.readline()
                        
                        # Check if we've reached end of file
                        if not next_line:
                            # Verify we read the correct number of atom lines
                            if atom_lines_read != num_atoms:
                                print(f"Error: Frame {frame_number} has {atom_lines_read} atom lines, expected {num_atoms}")
                                return None, None, None
                            break
                        
                        # Check if this line starts a new frame
                        next_words = next_line.rstrip().split()
                        if len(next_words) == 1:
                            # This is the start of the next frame
                            # Verify we read the correct number of atom lines
                            if atom_lines_read != num_atoms:
                                print(f"Error: Frame {frame_number} has {atom_lines_read} atom lines, expected {num_atoms}")
                                return None, None, None
                            # Set current_line to this new frame start line for next iteration
                            current_line = next_line
                            break
                        
                        # This is an atom line - parse it
                        atom_lines_read += 1
                        
                        # Extract atom name and coordinates
                        # Format: words[1] = atom name, words[2,3,4] = x,y,z coordinates, words[5] = atom type
                        if len(next_words) < 6:
                            print(f"Error: Frame {frame_number}, atom line {atom_lines_read} has insufficient data")
                            return None, None, None
                        
                        # Determine if this atom matches our filter
                        matches_filter = False
                        
                        if use_type:
                            # Filter by atom type (sixth column)
                            try:
                                atom_type_value = int(next_words[5])
                                if atom_type_value in filter_types:
                                    matches_filter = True
                            except (ValueError, IndexError) as e:
                                print(f"Error: Frame {frame_number}, atom line {atom_lines_read} has invalid atom type")
                                return None, None, None
                        elif use_name:
                            # Filter by atom name (second column)
                            atom_name = next_words[1]
                            if atom_name in filter_names:
                                matches_filter = True
                        
                        # If atom matches filter, extract coordinates
                        if matches_filter:
                            try:
                                x = float(next_words[2])
                                y = float(next_words[3])
                                z = float(next_words[4])
                                frame_ca_coords.append([x, y, z])
                            except (ValueError, IndexError) as e:
                                print(f"Error: Frame {frame_number}, atom line {atom_lines_read} has invalid coordinates")
                                return None, None, None
                    
                    # Check if number of filtered atoms is consistent across frames
                    num_ca_this_frame = len(frame_ca_coords)
                    if expected_num_ca is None:
                        expected_num_ca = num_ca_this_frame  # Set expected count from first frame
                    elif num_ca_this_frame != expected_num_ca:
                        print(f"Error: Frame {frame_number} has {num_ca_this_frame} filtered atoms, expected {expected_num_ca}")
                        return None, None, None
                    
                    # Append this frame's CA coordinates to our collection
                    all_frame_coordinates.append(frame_ca_coords)
                    
                    # If we reached end of file, break out of main loop
                    if not next_line:
                        break
                else:
                    # This shouldn't happen in a well-formed file
                    # Move to next line
                    current_line = f.readline()
        
        # Convert list of frame coordinates to numpy array with shape (frames, CAs, 3)
        if len(all_frame_coordinates) == 0:
            print("Error: No frames found in file")
            return None, None, None
        
        coordinates_array = np.array(all_frame_coordinates)
        
        # Return the results
        return frame_number, expected_num_ca, coordinates_array
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None, None, None
    except Exception as e:
        print(f"Error: Unexpected error while reading file: {str(e)}")
        return None, None, None

def read_traj_pdb(filename, filter_names=None, filter_types=None):
    """
    Reads a PDB trajectory file and extracts atom coordinates.
    
    Parameters:
    filename : str
        Path to the PDB trajectory file
    filter_names : list of str, optional
        Atom names to filter (e.g., [" CA ", " CB ", " N  "]). Matches columns 13-16 in PDB file.
        Note: PDB atom names are 4 characters with specific spacing (e.g., " CA " not "CA")
    filter_types : list of int, optional
        Not applicable for PDB files. If provided without filter_names, will switch to default.
    
    Returns:
    tuple : (num_frames, num_filtered_atoms, coordinates_array) or (None, None, None) on error
        - num_frames: int, total number of frames read
        - num_filtered_atoms: int, number of filtered atoms per frame
        - coordinates_array: numpy array of shape (num_frames, num_filtered_atoms, 3)
    """
    
    # Convert single values to lists for uniform handling
    if filter_names is not None and not isinstance(filter_names, list):
        filter_names = [filter_names]
    if filter_types is not None and not isinstance(filter_types, list):
        filter_types = [filter_types]
    
    # Determine filtering mode and print what we're doing
    if filter_types is not None and len(filter_types) > 0:
        if filter_names is None or len(filter_names) == 0:
            print(f"Warning: Atom types {filter_types} provided, but PDB files don't have atom types.")
            print("Switching to default: filtering by atom name ' CA '")
            filter_names = [" CA "]
        else:
            print(f"Warning: Atom types {filter_types} provided, but PDB files don't have atom types.")
            print(f"Using atom names {filter_names} instead.")
    
    if filter_names is not None and len(filter_names) > 0:
        print(f"Filtering by atom names: {filter_names}")
    else:
        print("No filter provided. Defaulting to atom name ' CA '")
        filter_names = [" CA "]
    
    # Initialize variables to track frame data
    all_frame_coordinates = []  # Will store filtered coordinates for each frame
    expected_num_filtered = None  # Number of filtered atoms in first frame
    frame_number = 0            # Counter for current frame number
    
    try:
        # Open file and read all lines
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if this line starts a new frame (ATOM or HETATM with serial number 1)
            if (line.startswith("ATOM") or line.startswith("HETATM")):
                # Extract serial number (columns 7-11)
                try:
                    serial = int(line[6:11].strip())
                except (ValueError, IndexError):
                    i += 1
                    continue
                
                if serial == 1:
                    # Verify this is truly a frame start by checking next few atoms
                    is_frame_start = True
                    for check_idx in range(1, min(3, len(lines) - i)):
                        next_line = lines[i + check_idx]
                        if next_line.startswith("ATOM") or next_line.startswith("HETATM"):
                            try:
                                next_serial = int(next_line[6:11].strip())
                                if next_serial != check_idx + 1:
                                    is_frame_start = False
                                    break
                            except (ValueError, IndexError):
                                is_frame_start = False
                                break
                        elif not next_line.startswith("CRYST1") and not next_line.startswith("MODEL"):
                            # Allow CRYST1 or MODEL before atoms, but not other records
                            if check_idx == 1:  # First line after serial 1 might be CRYST1
                                continue
                            is_frame_start = False
                            break
                    
                    if not is_frame_start:
                        i += 1
                        continue
                    
                    frame_number += 1
                    frame_filtered_coords = []  # Store filtered coordinates for this frame
                    
                    # Check for optional CRYST1 line (box dimensions) before atom 1
                    # Go back one line to see if there's a CRYST1
                    if i > 0 and lines[i-1].startswith("CRYST1"):
                        pass  # Just skip it, already moved past it
                    
                    # Read all atom lines in this frame
                    j = i
                    while j < len(lines):
                        atom_line = lines[j]
                        
                        # Check if this is an atom line
                        if not (atom_line.startswith("ATOM") or atom_line.startswith("HETATM")):
                            # Check if this might be the start of next frame
                            if atom_line.startswith("CRYST1") or atom_line.startswith("MODEL"):
                                # Peek ahead to see if ATOM/HETATM 1 follows
                                if j + 1 < len(lines):
                                    next_line = lines[j + 1]
                                    if (next_line.startswith("ATOM") or next_line.startswith("HETATM")):
                                        try:
                                            next_serial = int(next_line[6:11].strip())
                                            if next_serial == 1:
                                                # This is the start of next frame
                                                break
                                        except (ValueError, IndexError):
                                            pass
                            elif atom_line.startswith("ENDMDL") or atom_line.startswith("END"):
                                # End of this frame
                                j += 1
                                break
                            j += 1
                            continue
                        
                        # Parse atom line using column positions (1-indexed -> 0-indexed)
                        # Columns 13-16 = atom name (0-indexed: 12-16)
                        # Columns 31-38 = X (0-indexed: 30-38)
                        # Columns 39-46 = Y (0-indexed: 38-46)
                        # Columns 47-54 = Z (0-indexed: 46-54)
                        
                        if len(atom_line) < 54:
                            print(f"Error: Frame {frame_number}, line too short: {atom_line.strip()}")
                            return None, None, None
                        
                        atom_name = atom_line[12:16]  # 4 characters
                        
                        # Check if this atom matches our filter
                        if atom_name in filter_names:
                            try:
                                x = float(atom_line[30:38].strip())
                                y = float(atom_line[38:46].strip())
                                z = float(atom_line[46:54].strip())
                                frame_filtered_coords.append([x, y, z])
                            except (ValueError, IndexError) as e:
                                print(f"Error: Frame {frame_number}, invalid coordinates in line: {atom_line.strip()}")
                                return None, None, None
                        
                        j += 1
                    
                    # Check if number of filtered atoms is consistent across frames
                    num_filtered_this_frame = len(frame_filtered_coords)
                    if expected_num_filtered is None:
                        expected_num_filtered = num_filtered_this_frame
                        if expected_num_filtered == 0:
                            print(f"Error: No atoms matching filter {filter_names} found in first frame")
                            return None, None, None
                    elif num_filtered_this_frame != expected_num_filtered:
                        print(f"Error: Frame {frame_number} has {num_filtered_this_frame} filtered atoms, expected {expected_num_filtered}")
                        return None, None, None
                    
                    # Append this frame's coordinates to our collection
                    all_frame_coordinates.append(frame_filtered_coords)
                    
                    # Move to the position after this frame
                    i = j
                else:
                    i += 1
            else:
                i += 1
        
        # Convert list of frame coordinates to numpy array
        if len(all_frame_coordinates) == 0:
            print("Error: No frames found in file")
            return None, None, None
        
        coordinates_array = np.array(all_frame_coordinates)
        
        # Return the results
        return frame_number, expected_num_filtered, coordinates_array
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None, None, None
    except Exception as e:
        print(f"Error: Unexpected error while reading file: {str(e)}")
        return None, None, None

####################################################################
"""
SECTION I
Python code to calculate INTRA-strand deformations (rise,radius,twist)
for systems with any num_strand number of helices.
Basic plotting and metric validations added.
"""
####################################################################
class HelixDeformationAnalyzer:
    def __init__(self, coordinates, num_strands, weights=None):
        """
        Initialize the analyzer with helix coordinates.
        Parameters:
        coordinates: numpy array of shape [frames, strand_length*num_strands, 3]
        num_strands: number of strands in the helix
        weights: list of weights for [rise, radius, twist] in combined deformation score
        """
        self.coords = coordinates
        self.n_frames, self.n_total_atoms, _ = coordinates.shape
        self.num_strands = num_strands
        self.strand_length = self.n_total_atoms // num_strands
        self.n_tripeptides = self.strand_length // 3
        
        # Set default weights if none provided
        if weights is None:
            weights = [1.0, 1.0, 1.0]
        self.weights = np.array(weights)
        
        # Results storage
        self.rise_values = None
        self.radius_values = None
        self.twist_values = None
        self.deformation_metrics = None
        
    def get_triangle_centroids(self, frame, tripeptide_idx):
        """
        Get centroids of triangles for a tripeptide.
        Parameters:
        frame: frame index
        tripeptide_idx: tripeptide index (0 to n_tripeptides-1)
        Returns:
        centroids: array of shape [3, 3] - three centroids with xyz coordinates
        """
        centroids = []
        start_res = tripeptide_idx * 3
        
        for i in range(3):
            res_idx = start_res + i
            if res_idx >= self.strand_length:
                break
            # Get helix units from all strands for this residue
            strand_atoms = []
            for strand in range(self.num_strands):
                atom_idx = res_idx + strand * self.strand_length
                strand_atoms.append(self.coords[frame, atom_idx, :])
            # Calculate centroid 
            centroid = np.mean(strand_atoms, axis=0)
            centroids.append(centroid)
        
        return np.array(centroids)
    
    def fit_helical_axis(self, centroids):
        """
        Fit a line to get helical axis.
        Parameters:
        centroids: array of shape [3, 3]
        Returns:
        axis_direction: unit vector along helical axis
        axis_point: a point on the axis
        """
        # Use first centroid as reference point
        axis_point = centroids[0]
        # Fit line using least squares
        # Direction vector from first to last centroid as initial guess
        if len(centroids) >= 2:
            direction = centroids[-1] - centroids[0]
            if np.linalg.norm(direction) > 1e-10:
                axis_direction = direction / np.linalg.norm(direction)
            else:
                axis_direction = np.array([0, 0, 1])  # Default z-axis
        else:
            axis_direction = np.array([0, 0, 1])
        return axis_direction, axis_point
    
    def calculate_helical_parameters(self, centroids):
        """
        Calculate rise, radius, and twist for a tripeptide.
        Parameters:
        centroids: array of shape [3, 3]
        Returns:
        rise: average rise per residue
        radius: average radius
        twist: average twist per residue (in radians)
        """
        if len(centroids) < 3:
            return 0.0, 0.0, 0.0
        # Get helical axis
        axis_direction, axis_point = self.fit_helical_axis(centroids)
        # Calculate rise per residue
        rises = []
        for i in range(len(centroids) - 1):
            vec = centroids[i + 1] - centroids[i]
            rise = np.dot(vec, axis_direction)
            rises.append(rise)
        avg_rise = np.mean(rises) if rises else 0.0
        
        # Calculate radius
        radii = []
        for centroid in centroids:
            # Vector from axis point to centroid
            vec_to_centroid = centroid - axis_point
            # Project onto axis
            proj_on_axis = np.dot(vec_to_centroid, axis_direction) * axis_direction
            # Perpendicular component gives radius
            perpendicular = vec_to_centroid - proj_on_axis
            radius = np.linalg.norm(perpendicular)
            radii.append(radius)
        avg_radius = np.mean(radii)
        
        # Calculate twist per residue
        twists = []
        for i in range(len(centroids) - 1):
            # Project centroids onto plane perpendicular to axis
            vec1 = centroids[i] - axis_point
            vec2 = centroids[i + 1] - axis_point
            
            # Remove component along axis
            proj1 = vec1 - np.dot(vec1, axis_direction) * axis_direction
            proj2 = vec2 - np.dot(vec2, axis_direction) * axis_direction
            
            # Calculate angle between projections
            if np.linalg.norm(proj1) > 1e-10 and np.linalg.norm(proj2) > 1e-10:
                cos_angle = np.dot(proj1, proj2) / (np.linalg.norm(proj1) * np.linalg.norm(proj2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                twist = np.arccos(cos_angle)
                twists.append(twist)
        
        avg_twist = np.mean(twists) if twists else 0.0
        return avg_rise, avg_radius, avg_twist

    def calculate_helical_parameters_per_strand(self, centroids, frame, tripeptide_idx):
        """
        Calculate rise, radius, and twist for each strand separately.
        Parameters:
        centroids: array of shape [3, 3]
        Returns:
        rise_per_strand: array of shape [num_strands] - rise for each strand
        radius_per_strand: array of shape [num_strands] - radius for each strand  
        twist_per_strand: array of shape [num_strands] - twist for each strand (in radians)
        """
        if len(centroids) < 3:
            return np.zeros(self.num_strands), np.zeros(self.num_strands), np.zeros(self.num_strands)
        
        # Get helical axis
        axis_direction, axis_point = self.fit_helical_axis(centroids)
        
        # Calculate rise for each strand
        rise_per_strand = np.zeros(self.num_strands)
        for i in range(min(2, self.num_strands)):  # Two consecutive pairs (or less if fewer strands)
            vec = centroids[i + 1] - centroids[i]
            rise_per_strand[i] = np.dot(vec, axis_direction)
        # For remaining strands, use the average or calculate from first to last
        for i in range(2, self.num_strands):
            rise_per_strand[i] = np.dot(centroids[2] - centroids[0], axis_direction) / 2.0

        # Calculate radius for each strand using actual CA positions
        start_res = tripeptide_idx * 3
        radius_per_strand = np.zeros(self.num_strands)
        for strand in range(self.num_strands):
            radii = []
            for res in range(3):  # Three consecutive residues
                res_idx = start_res + res
                if res_idx >= self.strand_length:
                    break
                # Get CA position for this strand at this residue
                atom_idx = res_idx + strand * self.strand_length
                ca_position = self.coords[frame, atom_idx, :]
                # Vector from axis point to CA position
                vec_to_ca = ca_position - axis_point
                # Project onto axis
                proj_on_axis = np.dot(vec_to_ca, axis_direction) * axis_direction
                # Perpendicular component gives radius
                perpendicular = vec_to_ca - proj_on_axis
                radius = np.linalg.norm(perpendicular)
                radii.append(radius)
            radius_per_strand[strand] = np.mean(radii) if radii else 0.0

        # Calculate twist for each strand
        twist_per_strand = np.zeros(self.num_strands)
        for i in range(min(2, self.num_strands)):  # Two consecutive pairs (or less if fewer strands)
            # Project centroids onto plane perpendicular to axis
            vec1 = centroids[i] - axis_point
            vec2 = centroids[i + 1] - axis_point
            # Remove component along axis
            proj1 = vec1 - np.dot(vec1, axis_direction) * axis_direction
            proj2 = vec2 - np.dot(vec2, axis_direction) * axis_direction
            # Calculate angle between projections
            if np.linalg.norm(proj1) > 1e-10 and np.linalg.norm(proj2) > 1e-10:
                cos_angle = np.dot(proj1, proj2) / (np.linalg.norm(proj1) * np.linalg.norm(proj2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                twist_per_strand[i] = np.arccos(cos_angle)
        # For remaining strands, calculate twist from first to last centroid
        for i in range(2, self.num_strands):
            vec1 = centroids[0] - axis_point
            vec2 = centroids[2] - axis_point
            proj1 = vec1 - np.dot(vec1, axis_direction) * axis_direction
            proj2 = vec2 - np.dot(vec2, axis_direction) * axis_direction
            if np.linalg.norm(proj1) > 1e-10 and np.linalg.norm(proj2) > 1e-10:
                cos_angle = np.dot(proj1, proj2) / (np.linalg.norm(proj1) * np.linalg.norm(proj2))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                twist_per_strand[i] = np.arccos(cos_angle) / 2.0  # Divide by 2 since it's over 2 steps
        
        return rise_per_strand, radius_per_strand, twist_per_strand

    def analyze_deformation(self):
        """
        Analyze deformation across all frames and tripeptides.
        """
        # Initialize storage arrays
        rise_all = np.zeros((self.n_frames, self.n_tripeptides))
        radius_all = np.zeros((self.n_frames, self.n_tripeptides))
        twist_all = np.zeros((self.n_frames, self.n_tripeptides))
        
        # Calculate helical parameters for all frames and tripeptides
        for frame in range(self.n_frames):
            for tripeptide_idx in range(self.n_tripeptides):
                centroids = self.get_triangle_centroids(frame, tripeptide_idx)
                if len(centroids) == 3:
                    rise, radius, twist = self.calculate_helical_parameters(centroids)
                    rise_all[frame, tripeptide_idx] = rise
                    radius_all[frame, tripeptide_idx] = radius
                    twist_all[frame, tripeptide_idx] = twist
        
        # Store reference values (frame 0)
        rise_ref = rise_all[0, :]
        radius_ref = radius_all[0, :]
        twist_ref = twist_all[0, :]

        # Calculate deviations from reference
        rise_dev = rise_all - rise_ref[np.newaxis, :]
        radius_dev = radius_all - radius_ref[np.newaxis, :]
        twist_dev = twist_all - twist_ref[np.newaxis, :]
        
        # Calculate statistics
        self.deformation_metrics = {
            'rise': {
                'mean': np.mean(rise_dev, axis=0),
                'std': np.std(rise_dev, axis=0),
                'deviations': rise_dev
            },
            'radius': {
                'mean': np.mean(radius_dev, axis=0),
                'std': np.std(radius_dev, axis=0),
                'deviations': radius_dev
            },
            'twist': {
                'mean': np.mean(twist_dev, axis=0),
                'std': np.std(twist_dev, axis=0),
                'deviations': twist_dev
            }
        }
        
        # Store raw values for reference
        self.rise_values = rise_all
        self.radius_values = radius_all
        self.twist_values = twist_all
        
        return self.deformation_metrics
    
    def analyze_deformation_time(self):
        """
        Analyze deformation across all frames and tripeptides, over time.
        """
        # Initialize storage arrays
        rise_all = np.zeros((self.n_frames, self.n_tripeptides))
        radius_all = np.zeros((self.n_frames, self.n_tripeptides))
        twist_all = np.zeros((self.n_frames, self.n_tripeptides))
        
        # Calculate helical parameters for all frames and tripeptides
        for frame in range(self.n_frames):
            for tripeptide_idx in range(self.n_tripeptides):
                centroids = self.get_triangle_centroids(frame, tripeptide_idx)
                if len(centroids) == 3:
                    rise, radius, twist = self.calculate_helical_parameters(centroids)
                    rise_all[frame, tripeptide_idx] = rise
                    radius_all[frame, tripeptide_idx] = radius
                    twist_all[frame, tripeptide_idx] = twist
        
        # Store reference values (frame 0)
        rise_ref = rise_all[0, :]
        radius_ref = radius_all[0, :]
        twist_ref = twist_all[0, :]
        
        # Calculate deviations from reference
        rise_dev = rise_all - rise_ref[np.newaxis, :]
        radius_dev = radius_all - radius_ref[np.newaxis, :]
        twist_dev = twist_all - twist_ref[np.newaxis, :]
        
        # Calculate statistics
        self.deformation_metrics = {
            'rise': {
                'mean': np.mean(rise_dev, axis=1),
                'std': np.std(rise_dev, axis=1),
                'deviations': rise_dev
            },
            'radius': {
                'mean': np.mean(radius_dev, axis=1),
                'std': np.std(radius_dev, axis=1),
                'deviations': radius_dev
            },
            'twist': {
                'mean': np.mean(twist_dev, axis=1),
                'std': np.std(twist_dev, axis=1),
                'deviations': twist_dev
            }
        }
        
        # Store raw values for reference
        self.rise_values = rise_all
        self.radius_values = radius_all
        self.twist_values = twist_all
        
        return self.deformation_metrics
    
    def analyze_deformation_time_per_strand(self):
        """
        Analyze deformation across all frames and tripeptides for each strand separately.
        Returns actual values, not deviations.
        """
        # Initialize storage arrays for each strand
        rise_all = np.zeros((self.n_frames, self.n_tripeptides, self.num_strands))
        radius_all = np.zeros((self.n_frames, self.n_tripeptides, self.num_strands))
        twist_all = np.zeros((self.n_frames, self.n_tripeptides, self.num_strands))
        
        # Calculate helical parameters for all frames and tripeptides
        for frame in range(self.n_frames):
            for tripeptide_idx in range(self.n_tripeptides):
                centroids = self.get_triangle_centroids(frame, tripeptide_idx)
                if len(centroids) == 3:
                    rise_per_strand, radius_per_strand, twist_per_strand = self.calculate_helical_parameters_per_strand(centroids, frame, tripeptide_idx)
                    rise_all[frame, tripeptide_idx, :] = rise_per_strand
                    radius_all[frame, tripeptide_idx, :] = radius_per_strand
                    twist_all[frame, tripeptide_idx, :] = twist_per_strand
        
        # Calculate statistics for each strand (actual values, not deviations)
        self.deformation_metrics_per_strand = {
            'rise': {},
            'radius': {},
            'twist': {}
        }
        
        for strand in range(self.num_strands):
            strand_name = f'strand{strand + 1}'
            
            self.deformation_metrics_per_strand['rise'][strand_name] = {
                'mean': np.mean(rise_all[:, :, strand], axis=1),  # Average over tripeptides for each frame
                'std': np.std(rise_all[:, :, strand], axis=1),
                'values': rise_all[:, :, strand]  # Raw values
            }
            
            self.deformation_metrics_per_strand['radius'][strand_name] = {
                'mean': np.mean(radius_all[:, :, strand], axis=1),
                'std': np.std(radius_all[:, :, strand], axis=1),
                'values': radius_all[:, :, strand]
            }
            
            self.deformation_metrics_per_strand['twist'][strand_name] = {
                'mean': np.mean(twist_all[:, :, strand], axis=1),
                'std': np.std(twist_all[:, :, strand], axis=1),
                'values': twist_all[:, :, strand]
            }
        
        return self.deformation_metrics_per_strand
    
    def analyze_deformation_by_residue(self):
        """
        Analyze deformation across all frames and residues (instead of tripeptides).
        """
        # Initialize storage arrays
        rise_all = np.zeros((self.n_frames, self.strand_length - 2))  # -2 because we need 3 consecutive residues
        radius_all = np.zeros((self.n_frames, self.strand_length - 2))
        twist_all = np.zeros((self.n_frames, self.strand_length - 2))
        
        # Calculate helical parameters for all frames and residues
        for frame in range(self.n_frames):
            for res_idx in range(self.strand_length - 2):  # Need at least 3 consecutive residues
                # Get centroids for this residue and the next two
                centroids = []
                for i in range(3):  # Three consecutive residues starting from res_idx
                    current_res = res_idx + i
                    
                    # Get helix units from all strands for this residue
                    strand_atoms = []
                    for strand in range(self.num_strands):
                        atom_idx = current_res + strand * self.strand_length
                        strand_atoms.append(self.coords[frame, atom_idx, :])
                    
                    # Calculate centroid of the polygon formed by all strands
                    centroid = np.mean(strand_atoms, axis=0)
                    centroids.append(centroid)
                
                centroids = np.array(centroids)
                
                if len(centroids) == 3:
                    rise, radius, twist = self.calculate_helical_parameters(centroids)
                    rise_all[frame, res_idx] = rise
                    radius_all[frame, res_idx] = radius
                    twist_all[frame, res_idx] = twist
        
        # Store reference values (frame 0)
        rise_ref = rise_all[0, :]
        radius_ref = radius_all[0, :]
        twist_ref = twist_all[0, :]
        
        # Calculate deviations from reference
        rise_dev = rise_all - rise_ref[np.newaxis, :]
        radius_dev = radius_all - radius_ref[np.newaxis, :]
        twist_dev = twist_all - twist_ref[np.newaxis, :]
        
        # Calculate statistics
        self.deformation_metrics_residue = {
            'rise': {
                'mean': np.mean(rise_dev, axis=0),
                'std': np.std(rise_dev, axis=0),
                'deviations': rise_dev
            },
            'radius': {
                'mean': np.mean(radius_dev, axis=0),
                'std': np.std(radius_dev, axis=0),
                'deviations': radius_dev
            },
            'twist': {
                'mean': np.mean(twist_dev, axis=0),
                'std': np.std(twist_dev, axis=0),
                'deviations': twist_dev
            }
        }
        
        # Store raw values for reference
        self.rise_values_residue = rise_all
        self.radius_values_residue = radius_all
        self.twist_values_residue = twist_all
        
        return self.deformation_metrics_residue

    def calculate_combined_deformation(self, weights=None):
        """
        Calculate combined deformation score. Not directly used in tests.
        Parameters:
        weights: list of weights for [rise, radius, twist]. If None, uses self.weights
        Returns:
        combined_score: array of combined deformation scores per tripeptide
        """
        if weights is None:
            weights = self.weights
        else:
            weights = np.array(weights)
        
        if self.deformation_metrics is None:
            raise ValueError("Must run analyze_deformation() first")
        
        # Get RMS deviations for each parameter
        rise_rms = np.sqrt(np.mean(self.deformation_metrics['rise']['deviations']**2, axis=0))
        radius_rms = np.sqrt(np.mean(self.deformation_metrics['radius']['deviations']**2, axis=0))
        twist_rms = np.sqrt(np.mean(self.deformation_metrics['twist']['deviations']**2, axis=0))
        
        # Normalize by typical scales (optional - you can adjust these)
        # These are one set of probable estimates for collagen - not actually used in the original paper
        # Adjust based on your system, change to 1 for no scaling
        rise_scale = 1.0  # Angstroms
        radius_scale = 1.0  # Angstroms
        twist_scale = 1.0  # radians
        
        # Calculate weighted combined score
        combined_score = np.sqrt(
            weights[0] * (rise_rms / rise_scale)**2 +
            weights[1] * (radius_rms / radius_scale)**2 +
            weights[2] * (twist_rms / twist_scale)**2
        )
        
        return combined_score
    
    def plot_deformation_analysis(self, weights=None, figsize=(12, 10)):
        """
        Plot the deformation analysis results.
        Parameters:
        weights: weights for combined score. If None, uses self.weights
        figsize: figure size
        """
        if self.deformation_metrics is None:
            raise ValueError("Must run analyze_deformation() first")
        
        tripeptide_numbers = np.arange(1, self.n_tripeptides + 1)
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        
        # Plot 1: Rise deformation
        ax1 = axes[0]
        rise_mean = self.deformation_metrics['rise']['mean']
        rise_std = self.deformation_metrics['rise']['std']
        ax1.errorbar(tripeptide_numbers, rise_mean, yerr=rise_std, 
                    marker='o', capsize=3, capthick=1, label='Rise deformation')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_ylabel('Rise Deviation (Å)')
        ax1.set_title('Axial Deformation (Rise per Residue)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Radius deformation
        ax2 = axes[1]
        radius_mean = self.deformation_metrics['radius']['mean']
        radius_std = self.deformation_metrics['radius']['std']
        ax2.errorbar(tripeptide_numbers, radius_mean, yerr=radius_std, 
                    marker='s', capsize=3, capthick=1, label='Radius deformation', color='orange')
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Radius Deviation (Å)')
        ax2.set_title('Radial Deformation')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Plot 3: Twist deformation
        ax3 = axes[2]
        twist_mean = self.deformation_metrics['twist']['mean']
        twist_std = self.deformation_metrics['twist']['std']
        ax3.errorbar(tripeptide_numbers, twist_mean, yerr=twist_std, 
                    marker='^', capsize=3, capthick=1, label='Twist deformation', color='green')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_ylabel('Twist Deviation (rad)')
        ax3.set_title('Torsional Deformation')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Plot 4: Combined deformation score - if needed
        ax4 = axes[3]
        combined_score = self.calculate_combined_deformation(weights)
        ax4.plot(tripeptide_numbers, combined_score, 'ro-', linewidth=2, 
                label=f'Combined Score (weights: {weights if weights else self.weights})')
        ax4.set_ylabel('Combined Deformation Score')
        ax4.set_xlabel('Tripeptide Number')
        ax4.set_title('Combined Deformation Score')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig, axes
        
    def save_deformation_data_time(self, filename, weights=[1.0,1.0,1.0]):
        """
        Save deformation data to a text file.
        
        Parameters:
        filename: str, name of the output file
        """
        if self.deformation_metrics is None:
            raise ValueError("Must run analyze_deformation() first")
        
        # Prepare data
        frame_numbers = np.arange(1, self.n_frames + 1)
        rise_mean = self.deformation_metrics['rise']['mean']
        rise_std = self.deformation_metrics['rise']['std']
        radius_mean = self.deformation_metrics['radius']['mean']
        radius_std = self.deformation_metrics['radius']['std']
        twist_mean = self.deformation_metrics['twist']['mean']
        twist_std = self.deformation_metrics['twist']['std']

        # # DOES NOT PRINT COMBINED DEFORMATION - DEFINE WEIGHTS IN POST-PROCESSING
        # combined_mean = np.sqrt(weights[0] * rise_mean + weights[1] * radius_mean + weights[2] * twist_mean)
        # combined_std = np.sqrt(weights[0] * rise_std**2 + weights[1] * radius_std**2 + weights[2] * twist_std**2)

        # Note that for collagen you might get all twists = 0 (for ideal packing)
        # Optional: remove Twist entirely
        # combined_mean = (weights[0] * rise_mean + weights[1] * radius_mean)/2
        # combined_std = np.sqrt(weights[0] * rise_std**2 + weights[1] * radius_std**2)
        
        # Write to file
        with open(filename, 'w') as f:
            # Write header
            # f.write("Frame\tMean_Rise\tStd_Rise\tMean_Radius\tStd_Radius\tMean_Twist\tStd_Twist\tCombined_Mean\tCombined_Std\n")
            f.write("Frame\tMean_Rise_over_residues\tStd_Rise_over_residues\tMean_Radius_over_residues\tStd_Radius_over_residues\tMean_Twist_over_residues\tStd_Twist_over_residues\n")
            
            # Write data rows
            for i in range(self.n_frames):
                f.write(f"{frame_numbers[i]}\t{rise_mean[i]:.3f}\t{rise_std[i]:.3f}\t")
                f.write(f"{radius_mean[i]:.3f}\t{radius_std[i]:.3f}\t")
                # f.write(f"{twist_mean[i]:.3f}\t{twist_std[i]:.3f}\t")
                # f.write(f"{combined_mean[i]:.3f}\t{combined_std[i]:.3f}\n")
                f.write(f"{twist_mean[i]:.3f}\t{twist_std[i]:.3f}\n")
        
        print(f"Deformation data saved to {filename}")

    def save_deformation_data_time_per_strand(self, filename):
        """
        Save per-strand deformation data to a text file.
        
        Parameters:
        filename: str, name of the output file
        """
        if not hasattr(self, 'deformation_metrics_per_strand') or self.deformation_metrics_per_strand is None:
            raise ValueError("Must run analyze_deformation_time_per_strand() first")
        
        # Prepare data
        frame_numbers = np.arange(1, self.n_frames + 1)
        
        # Write to file
        with open(filename, 'w') as f:
            # Write header - dynamically generate based on number of strands
            header_parts = ["Frame"]
            
            # Add rise columns for all strands
            for i in range(self.num_strands):
                header_parts.append(f"Mean_Rise_Strand{i+1}")
            
            # Add radius columns for all strands
            for i in range(self.num_strands):
                header_parts.append(f"Mean_Radius_Strand{i+1}")
            
            # Add twist columns for all strands
            for i in range(self.num_strands):
                header_parts.append(f"Mean_Twist_Strand{i+1}")
            
            f.write("\t".join(header_parts) + "\n")
            
            # Write data rows
            for frame in range(self.n_frames):
                row_parts = [str(frame_numbers[frame])]
                
                # Add rise values for all strands
                for strand in range(self.num_strands):
                    strand_name = f'strand{strand + 1}'
                    rise_val = self.deformation_metrics_per_strand['rise'][strand_name]['mean'][frame]
                    row_parts.append(f"{rise_val:.3f}")
                
                # Add radius values for all strands
                for strand in range(self.num_strands):
                    strand_name = f'strand{strand + 1}'
                    radius_val = self.deformation_metrics_per_strand['radius'][strand_name]['mean'][frame]
                    row_parts.append(f"{radius_val:.3f}")
                
                # Add twist values for all strands
                for strand in range(self.num_strands):
                    strand_name = f'strand{strand + 1}'
                    twist_val = self.deformation_metrics_per_strand['twist'][strand_name]['mean'][frame]
                    row_parts.append(f"{twist_val:.3f}")
                
                f.write("\t".join(row_parts) + "\n")
        
        print(f"Per-strand deformation data saved to {filename}")

    def save_deformation_data_residue(self, filename, weights=[1.0, 1.0, 1.0]):
        """
        Save residue-based deformation data to a text file.
        
        Parameters:
        filename: str, name of the output file
        weights: list of weights for [rise, radius, twist] in combined score
        """
        if not hasattr(self, 'deformation_metrics_residue') or self.deformation_metrics_residue is None:
            raise ValueError("Must run analyze_deformation_by_residue() first")
        
        # Prepare data
        n_residues = self.strand_length - 2  # Available residues for analysis
        residue_numbers = np.arange(1, n_residues + 1)
        rise_mean = self.deformation_metrics_residue['rise']['mean']
        rise_std = self.deformation_metrics_residue['rise']['std']
        radius_mean = self.deformation_metrics_residue['radius']['mean']
        radius_std = self.deformation_metrics_residue['radius']['std']
        twist_mean = self.deformation_metrics_residue['twist']['mean']
        twist_std = self.deformation_metrics_residue['twist']['std']

        # # DOES NOT PRINT COMBINED DEFORMATION - DEFINE WEIGHTS IN POST-PROCESSING
        # combined_mean = (weights[0] * rise_mean + weights[1] * radius_mean + weights[2] * twist_mean) / 3
        # combined_std = np.sqrt(weights[0] * rise_std**2 + weights[1] * radius_std**2 + weights[2] * twist_std**2)
        
        # Note that for collagen you might get all twists = 0 (for ideal packing)
        # Optional: remove Twist entirely
        # combined_mean = (weights[0] * rise_mean + weights[1] * radius_mean)/2
        # combined_std = np.sqrt(weights[0] * rise_std**2 + weights[1] * radius_std**2)
        
        # Write to file
        with open(filename, 'w') as f:
            # Write header
            # f.write("Residue\tMean_Rise\tStd_Rise\tMean_Radius\tStd_Radius\tMean_Twist\tStd_Twist\tCombined_Mean\tCombined_Std\n")
            f.write("Residue_number\tMean_Rise_over_time\tStd_Rise_over_time\tMean_Radius_over_time\tStd_Radius_over_time\tMean_Twist_over_time\tStd_Twist_over_time\n")
            
            # Write data rows
            for i in range(n_residues):
                f.write(f"{residue_numbers[i]}\t{rise_mean[i]:.3f}\t{rise_std[i]:.3f}\t")
                f.write(f"{radius_mean[i]:.3f}\t{radius_std[i]:.3f}\t")
                # f.write(f"{twist_mean[i]:.3f}\t{twist_std[i]:.3f}\t")
                # f.write(f"{combined_mean[i]:.3f}\t{combined_std[i]:.3f}\n")
                f.write(f"{twist_mean[i]:.3f}\t{twist_std[i]:.3f}\n")
        
        print(f"Residue-based deformation data saved to {filename}")
        print(f"Note that the number of residues here is {residue_numbers[-1]} as the last two are not printed.")

def read_deformation_data(filename):
    """
    Read deformation data from a text file created by save_deformation_data().
    
    Parameters:
    filename: str, name of the input file
    
    Returns:
    dict: Dictionary containing the deformation data with keys:
        - 'tripeptide': array of tripeptide numbers
        - 'rise_mean': array of mean rise deviations
        - 'rise_std': array of rise standard deviations
        - 'radius_mean': array of mean radius deviations
        - 'radius_std': array of radius standard deviations
        - 'twist_mean': array of mean twist deviations
        - 'twist_std': array of twist standard deviations
    """
    try:
        data = np.loadtxt(filename, delimiter='\t', skiprows=1)
        
        # Extract columns
        result = {
            'tripeptide': data[:, 0].astype(int),
            'rise_mean': data[:, 1],
            'rise_std': data[:, 2],
            'radius_mean': data[:, 3],
            'radius_std': data[:, 4],
            'twist_mean': data[:, 5],
            'twist_std': data[:, 6]
        }
        
        print(f"Successfully read deformation data from {filename}")
        print(f"Data contains {len(result['tripeptide'])} tripeptides")
        
        return result
        
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        return None

def read_deformation_data_time(filename):
    """
    Read deformation data from a text file created by save_deformation_data_time().
    
    Parameters:
    filename: str, name of the input file
    
    Returns:
    dict: Dictionary containing the deformation data with keys:
        - 'frame': array of frame numbers
        - 'rise_mean': array of mean rise deviations
        - 'rise_std': array of rise standard deviations
        - 'radius_mean': array of mean radius deviations
        - 'radius_std': array of radius standard deviations
        - 'twist_mean': array of mean twist deviations
        - 'twist_std': array of twist standard deviations
    """
    try:
        data = np.loadtxt(filename, delimiter='\t', skiprows=1)
        
        # Extract columns
        result = {
            'frame': data[:, 0].astype(int),
            'rise_mean': data[:, 1],
            'rise_std': data[:, 2],
            'radius_mean': data[:, 3],
            'radius_std': data[:, 4],
            'twist_mean': data[:, 5],
            'twist_std': data[:, 6]
        }
        
        print(f"Successfully read deformation data from {filename}")
        print(f"Data contains {len(result['frame'])} Frames")
        
        return result
        
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None
    except Exception as e:
        print(f"Error reading file {filename}: {str(e)}")
        return None

def plot_deformation_from_file(filename, plot_file, weights=[1.0, 1.0, 1.0], figsize=(12, 10)):
    """
    Plot deformation analysis directly from a saved file.
    
    Parameters:
    filename: str, name of the input file
    weights: list of weights for [rise, radius, twist] in combined score
    figsize: tuple, figure size
    
    Returns:
    fig, axes: matplotlib figure and axes objects
    """
    data = read_deformation_data(filename)
    if data is None:
        return None, None
    
    weights = np.array(weights)
    """
    #### Calculate combined score, again some scaling values if needed
    # rise_scale = 1.0  # Angstroms
    # radius_scale = 1.0  # Angstroms
    # twist_scale = 1.0  # radians
    # combined_score = np.sqrt(
    #     weights[0] * (data['rise_std'] / rise_scale)**2 +
    #     weights[1] * (data['radius_std'] / radius_scale)**2 +
    #     weights[2] * (data['twist_std'] / twist_scale)**2
    # )
    """

    # if "combined_mean" not in data:
    #     # Combined mean score from the means
    #     data["combined_mean"] = (weights[0] * data['rise_mean'] + 
    #                         weights[1] * data['radius_mean'] + 
    #                         weights[2] * data['twist_mean'])/3
    # if "combined_std" not in data:
    #     # Combined std score from the standard deviations  
    #     data["combined_std"] = np.sqrt(weights[0] * data['rise_std']**2 + 
    #                         weights[1] * data['radius_std']**2 + 
    #                         weights[2] * data['twist_std']**2)
    """"
    Once again, if you need to remove Twist entirely:
    if "combined_mean" not in data:
        # Combined mean score from the means
        data["combined_mean"] = (weights[0] * data['rise_mean'] + 
                            weights[1] * data['radius_mean'])/2
    if "combined_std" not in data:
        # Combined std score from the standard deviations  
        data["combined_std"] = np.sqrt(weights[0] * data['rise_std']**2 + 
                            weights[1] * data['radius_std']**2)
    """

    # Create plots
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Rise deformation
    ax1 = axes[0]
    ax1.errorbar(data['tripeptide'], data['rise_mean'], yerr=data['rise_std'], marker='o', color='blue',capsize=3, capthick=1, label='Rise deformation')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Rise Deviation (Å)')
    # ax1.set_title('Axial Deformation (Rise per Residue)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1,data['tripeptide'][-1]+1)
    ax1.set_xticks(range(1, int(np.amax(data['tripeptide'])) + 1))
    ax1.legend(loc='upper right')
    
    # Plot 2: Radius deformation
    ax2 = axes[1]
    ax2.errorbar(data['tripeptide'], data['radius_mean'], yerr=data['radius_std'], marker='s', color='red', capsize=3, capthick=1, label='Radius deformation')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Radius Deviation (Å)')
    # ax2.set_title('Radial Deformation')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1,data['tripeptide'][-1]+1)
    ax2.set_xticks(range(1, int(np.amax(data['tripeptide'])) + 1))
    ax2.legend(loc='upper right')
    
    # # Plot 3: Twist deformation
    # ax3 = axes[2]
    # ax3.errorbar(data['tripeptide'], data['twist_mean'], yerr=data['twist_std'], marker='^', color='green', capsize=3, capthick=1, label='Twist deformation')
    # ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    # ax3.set_ylabel('Twist Deviation (rad)')
    # # ax3.set_title('Torsional Deformation')
    # ax3.grid(True, alpha=0.3)
    # ax3.set_xticks(range(1, int(np.amax(data['tripeptide'])) + 1))
    # ax3.legend(loc='upper right')
    
    # # Plot 4: Combined deformation score
    # ax4 = axes[3]
    # # ax4.errorbar(data['tripeptide'], combined_mean, yerr=combined_std, marker='o', capsize=3, capthick=1, linewidth=2, label=f'Combined Score (weights: {weights})')
    # ax4.errorbar(data['tripeptide'], data["combined_mean"], yerr=data["combined_std"], marker='o', color='black', capsize=3, capthick=1, linewidth=2, label=f'Combined Score')
    # ax4.set_ylabel('Combined Deformation Score')
    # ax4.set_xlabel('Tripeptide Number')
    # # ax4.set_title('Combined Deformation Score')
    # ax4.grid(True, alpha=0.3)
    # ax4.set_xticks(range(1, int(np.amax(data['tripeptide'])) + 1))
    # ax4.legend(loc='upper right')

    plt.tight_layout()
    # plt.show()
    plt.savefig(plot_file, dpi=600, transparent=False)
    plt.close(fig)
    print(f"Residue-based deformation data ploted to {plot_file}")

    return

def plot_time_deformation_from_file(filename,plot_file, ff=30, weights=[1.0, 1.0, 1.0], figsize=(12, 10)):
    """
    Plot deformation analysis directly from a saved file, over time.
    
    Parameters:
    filename: str, name of the input file
    weights: list of weights for [rise, radius, twist] in combined score
    figsize: tuple, figure size
    
    Returns:
    fig, axes: matplotlib figure and axes objects
    """
    data = read_deformation_data_time(filename)
    if data is None:
        return None, None
    
    weights = np.array(weights)
    """
    #### Calculate combined score, again some scaling values if needed
    # rise_scale = 1.5  # Angstroms
    # radius_scale = 5.0  # Angstroms
    # twist_scale = 0.2  # radians
    # combined_score = np.sqrt(
    #     weights[0] * (data['rise_std'] / rise_scale)**2 +
    #     weights[1] * (data['radius_std'] / radius_scale)**2 +
    #     weights[2] * (data['twist_std'] / twist_scale)**2
    # )
    """
    # if "combined_mean" not in data:
    #     # Combined mean score from the means
    #     data["combined_mean"] = (weights[0] * data['rise_mean'] + 
    #                         weights[1] * data['radius_mean'] + 
    #                         weights[2] * data['twist_mean'])/3
    # if "combined_std" not in data:
    #     # Combined std score from the standard deviations  
    #     data["combined_std"] = np.sqrt(weights[0] * data['rise_std']**2 + 
    #                         weights[1] * data['radius_std']**2 + 
    #                         weights[2] * data['twist_std']**2)
    """"
    Once again, if you need to remove Twist entirely:
    if "combined_mean" not in data:
        # Combined mean score from the means
        data["combined_mean"] = (weights[0] * data['rise_mean'] + 
                            weights[1] * data['radius_mean'])/2
    if "combined_std" not in data:
        # Combined std score from the standard deviations  
        data["combined_std"] = np.sqrt(weights[0] * data['rise_std']**2 + 
                            weights[1] * data['radius_std']**2)
    """

    # Create plots, the scaling on the x axis by 1000 is for ps->ns
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Plot 1: Rise deformation
    ax1 = axes[0]
    ax1.errorbar(data['frame'], data['rise_mean'], yerr=data['rise_std'], marker='o', color='blue',capsize=3, capthick=1, label='Rise deformation')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Rise Deviation (Å)')
    # ax1.set_title('Axial Deformation (Rise per Residue)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, int(np.amax(data['frame'])) + 1))
    ax1.legend(loc='upper right')
    
    # Plot 2: Radius deformation
    ax2 = axes[1]
    ax2.errorbar(data['frame'], data['radius_mean'], yerr=data['radius_std'], marker='s', color='red', capsize=3, capthick=1, label='Radius deformation')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Radius Deviation (Å)')
    # ax2.set_title('Radial Deformation')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(1, int(np.amax(data['frame'])) + 1))
    ax2.legend(loc='upper right')
    
    # Plot 3: Twist deformation
    ax3 = axes[2]
    ax3.errorbar(data['frame'], data['twist_mean'], yerr=data['twist_std'], marker='^', color='green', capsize=3, capthick=1, label='Twist deformation')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Twist Deviation (rad)')
    # ax3.set_title('Torsional Deformation')
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(range(1, int(np.amax(data['frame'])) + 1))
    ax3.legend(loc='upper right')
    
    # # Plot 4: Combined deformation score
    # ax4 = axes[3]
    # ax4.errorbar(data['frame'], data["combined_mean"], yerr=data["combined_std"], marker='o', color='black', capsize=3, capthick=1, linewidth=2, label=f'Combined Score')
    # ax4.set_ylabel('Combined Deformation Score')
    # ax4.set_xlabel('Time, t / ns')
    # # ax4.set_title('Combined Deformation Score')
    # ax4.grid(True, alpha=0.3)
    # ax4.set_xticks(range(1, int(np.amax(data['frame'])) + 1))
    # ax4.legend(loc='upper right')

    # # Print some files
    # f = open(f"{descriptor}/combined_score.dat",'w')
    # for i in range (len(data['combined_mean'])):
    #     f.write(f"{data['frame'][i]/1000}\t{data['combined_mean'][i]}\t{data['combined_std'][i]}\n")
    # f.close()

    # f = open(f"{descriptor}/rise.dat",'w')
    # for i in range (0,len(data['frame'])):
    #     f.write(f"{data['frame'][i]/1000}\t{data['rise_mean'][i]}\t{data['rise_std'][i]}\n")
    # f.close()

    # f = open(f"{descriptor}/radius.dat",'w')
    # for i in range (0,len(data['frame'])):
    #     f.write(f"{data['frame'][i]/1000}\t{data['radius_mean'][i]}\t{data['radius_std'][i]}\n")
    # f.close()

    plt.tight_layout()
    # plt.show()
    plt.savefig(plot_file, dpi=600, transparent=False)
    print(f"Time-based deformation data plotted to {plot_file}")
    plt.close(fig)

    return

####################################################################
"""
SECTION II
Python code to calculate helical regularity,
axial shift, local deformations, interhelical metrics.
This is generalized for any number of helices.
"""
####################################################################

def analyze_multi_helix_deformation(coordinates, num_strands, strand_length=None):
    """
    Analyzes deformation of multi-strand helix from ideal structure.
    
    Parameters:
    -----------
    coordinates : numpy.ndarray
        Array of shape [frames, num_strands*strand_length, 3] containing C-alpha coordinates
    num_strands : int
        Number of strands (must be > 0)
    strand_length : int, optional
        Length of each individual helix. If None, will be inferred
        
    Returns:
    --------
    dict
        Dictionary containing various metrics of deformation with the following structure:
        - 'frame_metrics': list of length [frames]
            Each element is a dict containing:
            - 'frame': int, frame index
            - 'individual_helix_metrics': list of length [num_strands]
                Each element contains strand-specific metrics:
                - 'strand_index': int (0 to num_strands-1)
                - 'axial_shift': float, displacement along helical axis
                - 'helical_regularity': float, consistency score (lower = more regular)
                - 'local_deformation_mean': float, average local deformation
                - 'local_deformation_max': float, maximum local deformation
                - 'local_deformation_profile': list of length [strand_length], per-residue deformation scores
            - 'interhelix_metrics': dict with inter-strand relationship metrics
            - 'overall_frame_score': float, combined deformation score for this frame
        The following metrics are optional - can be calculated, but better if calculated in post-processing.
        - 'overall_deformation_score': float, mean deformation across all frames
        - 'max_deformation_frame': int, frame index with highest deformation (only if frames > 1)
        - 'deformation_trend': dict, trend analysis across frames (only if frames > 1)
    """
    if num_strands <= 0:
        raise ValueError("Number of strands must be greater than 0")
    
    frames, total_length, dims = coordinates.shape
    
    if strand_length is None:
        strand_length = total_length // num_strands
    
    # First frame is the ideal structure
    ideal_structure = coordinates[0]
    
    # Separate the strands
    def get_strands(frame):
        strands = []
        for i in range(num_strands):
            start_idx = i * strand_length
            end_idx = (i + 1) * strand_length
            strands.append(frame[start_idx:end_idx])
        return strands
    
    # Get ideal strands
    ideal_strands = get_strands(ideal_structure)
    
    # Results container
    results = {
        'frame_metrics': [],
        'overall_deformation_score': None
    }
    
    # Process each frame
    for frame_idx in range(frames):
        frame = coordinates[frame_idx]
        strands = get_strands(frame)
        
        frame_result = {
            'frame': frame_idx,
            'individual_helix_metrics': [],
            'overall_frame_score': 0
        }
        
        # Process each strand
        for i, (strand, ideal_strand) in enumerate(zip(strands, ideal_strands)):            
            # 1. Calculate strand shift (renamed from axial_shift to shift)
            shift = calculate_shift(strand, ideal_strand)
            
            # 2. Calculate helical regularity (variance in rise and twist)
            regularity = calculate_helical_regularity(strand)
            
            # 3. Calculate local deformations
            local_deformations = calculate_local_deformations(strand, ideal_strand)

            # Add safety check
            if len(local_deformations) == 0:
                print(f"Error: local_deformations is empty for strand {i} in frame {frame_idx}")
                print(f"  Strand shape: {strand.shape}, Ideal strand shape: {ideal_strand.shape}")
                return None

            # Store metrics for this strand (renamed axial_shift to shift)
            strand_metrics = {
                'strand_index': i,
                'shift': shift,  # Changed from 'axial_shift'
                'helical_regularity': regularity,
                'local_deformation_mean': np.mean(local_deformations),
                'local_deformation_max': np.max(local_deformations),
                'local_deformation_profile': local_deformations.tolist()
            }
            
            frame_result['individual_helix_metrics'].append(strand_metrics)

        # Calculate interhelical metrics only if there are multiple strands
        if num_strands > 1:
            frame_result['interhelix_metrics'] = calculate_interhelical_metrics(strands, ideal_strands)
            # Add pairwise axial shifts
            pairwise_data = calculate_axial_shift_mn(strands, ideal_strands)
            frame_result['pairwise_axial_shifts'] = pairwise_data['pairwise_shifts']
            frame_result['pairwise_shift_labels'] = pairwise_data['pair_labels']
        
        # Calculate overall frame deformation score
        frame_score = calculate_frame_deformation_score(frame_result)
        frame_result['overall_frame_score'] = frame_score
        
        results['frame_metrics'].append(frame_result)
    
    # Calculate overall deformation score across all frames
    if frames > 1:
        frame_scores = [r['overall_frame_score'] for r in results['frame_metrics']]
        results['overall_deformation_score'] = np.mean(frame_scores)
        results['max_deformation_frame'] = np.argmax(frame_scores)
        results['deformation_trend'] = calculate_deformation_trend(frame_scores)
    else:
        results['overall_deformation_score'] = results['frame_metrics'][0]['overall_frame_score']
    
    return results

def calculate_helical_parameters_by_strand(strand):
    """
    Calculate helical parameters (pitch, radius, rise, twist) for a single helix.
    
    Parameters:
    -----------
    strand : numpy.ndarray
        Array of shape [strand_length, 3] containing C-alpha coordinates
        
    Returns:
    --------
    dict
        Dictionary containing helical parameters:
        - 'axis': numpy.ndarray of shape [3], unit vector representing helical axis direction
        - 'centroid': numpy.ndarray of shape [3], center point of the helix
        - 'radius': float, average distance from helix axis to strand points
        - 'rise_per_residue': float, average axial distance between consecutive residues
        - 'twist_per_residue': float, average angular rotation between consecutive residues (radians)
        - 'pitch': float, axial distance for one complete helical turn
    """
    # Calculate helical axis using PCA
    centroid = np.mean(strand, axis=0)
    centered_coords = strand - centroid
    u, s, vh = np.linalg.svd(centered_coords)
    principal_axis = vh[0]
    
    # Project points onto the axis
    projections = np.dot(centered_coords, principal_axis)
    
    # Calculate distances from axis
    projection_vectors = np.outer(projections, principal_axis)
    distances = np.linalg.norm(centered_coords - projection_vectors, axis=1)
    
    # Estimate radius
    radius = np.mean(distances)
    
    # Estimate pitch by looking at the axial component
    sorted_indices = np.argsort(projections)
    ordered_projections = projections[sorted_indices]
    
    # Estimate rise per residue
    rise_per_residue = np.mean(np.diff(ordered_projections))
    
    # To get twist, we need to look at the angular component
    # Project points onto plane perpendicular to axis
    plane_coords = centered_coords - np.outer(projections, principal_axis)
    
    # Calculate angles in this plane
    angles = np.arctan2(np.dot(plane_coords, vh[2]), np.dot(plane_coords, vh[1]))
    
    # Unwrap angles
    angles = np.unwrap(angles)
    
    # Calculate twist per residue (average angle difference)
    twist_per_residue = np.mean(np.abs(np.diff(angles[sorted_indices])))
    
    # Calculate pitch from rise and twist
    pitch = (2 * np.pi / twist_per_residue) * rise_per_residue if twist_per_residue > 0 else 0
    
    return {
        'axis': principal_axis,
        'centroid': centroid,
        'radius': radius,
        'rise_per_residue': rise_per_residue,
        'twist_per_residue': twist_per_residue,
        'pitch': pitch
    }

def calculate_shift(strand, ideal_strand):
    """
    Calculate the shift of a strand compared to its ideal position.
    
    Parameters:
    -----------
    strand : numpy.ndarray
        Array of shape [strand_length, 3] containing C-alpha coordinates
    ideal_strand : numpy.ndarray
        Array of shape [strand_length, 3] containing ideal C-alpha coordinates
        
    Returns:
    --------
    float
        Shift value in coordinate units (typically Angstroms).
        Represents the absolute displacement of the strand centroid along the ideal helical axis.
        Higher values indicate greater displacement from the ideal position.
    """
    # Calculate centroids
    strand_centroid = np.mean(strand, axis=0)
    ideal_centroid = np.mean(ideal_strand, axis=0)
    
    # Get helical parameters to find axes
    strand_params = calculate_helical_parameters_by_strand(strand)
    ideal_params = calculate_helical_parameters_by_strand(ideal_strand)
    
    # Calculate transformation to align helical axes
    def objective_function(params):
        angle = params[0]
        axis = params[1:4]
        axis = axis / np.linalg.norm(axis)
        
        # Create rotation
        r = Rotation.from_rotvec(angle * axis)
        rotated_strand_axis = r.apply(strand_params['axis'])
        
        # Return alignment error
        return 1 - np.abs(np.dot(rotated_strand_axis, ideal_params['axis']))
    
    # Initial guess: rotation around cross product of axes
    cross_product = np.cross(strand_params['axis'], ideal_params['axis'])
    if np.linalg.norm(cross_product) < 1e-6:
        # Axes are parallel or anti-parallel
        if np.dot(strand_params['axis'], ideal_params['axis']) > 0:
            # Parallel, no rotation needed
            rot_angle = 0
            rot_axis = np.array([1, 0, 0])  # Arbitrary
        else:
            # Anti-parallel, rotate 180 degrees around arbitrary perpendicular axis
            rot_angle = np.pi
            # Find perpendicular vector
            if np.abs(strand_params['axis'][0]) < np.abs(strand_params['axis'][1]):
                rot_axis = np.cross(strand_params['axis'], [1, 0, 0])
            else:
                rot_axis = np.cross(strand_params['axis'], [0, 1, 0])
            rot_axis = rot_axis / np.linalg.norm(rot_axis)
    else:
        # Normal case: axes at an angle
        rot_axis = cross_product / np.linalg.norm(cross_product)
        rot_angle = np.arccos(np.clip(np.dot(strand_params['axis'], ideal_params['axis']), -1, 1))
        
    initial_guess = np.array([rot_angle, rot_axis[0], rot_axis[1], rot_axis[2]])
    
    # Optimize
    result = minimize(objective_function, initial_guess, method='BFGS')
    
    # Apply rotation to align axes
    optimized_angle = result.x[0]
    optimized_axis = result.x[1:4]
    optimized_axis = optimized_axis / np.linalg.norm(optimized_axis)
    
    r = Rotation.from_rotvec(optimized_angle * optimized_axis)
    aligned_strand_centroid = r.apply(strand_centroid - ideal_centroid) + ideal_centroid
    
    # Project displacement vector onto aligned helical axis
    displacement = aligned_strand_centroid - ideal_centroid
    shift = np.abs(np.dot(displacement, ideal_params['axis']))
    
    return shift

def calculate_axial_shift_mn(strands, ideal_strands):
    """
    Calculate pairwise axial shifts between strand pairs.
    
    For each pair of strands (m, n), calculates the projection of the 
    centroid difference vector onto the average helical axis:
    s^{mn}(t) = |v_avg · (c^m(t) - c^n(t))|
    
    Parameters:
    -----------
    strands : list
        List of numpy arrays, each of shape [strand_length, 3] containing 
        C-alpha coordinates for each strand
    ideal_strands : list
        List of numpy arrays containing ideal C-alpha coordinates
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'pairwise_shifts': list of float, axial shifts for each strand pair
          Length is num_strands * (num_strands - 1) / 2
        - 'pair_labels': list of str, labels identifying each pair (e.g., "1-2", "1-3", etc.)
    """
    num_strands = len(strands)
    # Calculate helical parameters for each strand to get axes
    strand_params = [calculate_helical_parameters_by_strand(strand) for strand in strands]
    # Calculate average helical axis
    axes = np.array([params['axis'] for params in strand_params])
    avg_axis = np.mean(axes, axis=0)
    # Normalize the average axis
    avg_axis = avg_axis / np.linalg.norm(avg_axis)
    centroids = np.array([params['centroid'] for params in strand_params])
    # Calculate pairwise shifts
    pairwise_shifts = []
    pair_labels = []
    
    for m in range(num_strands):
        for n in range(m + 1, num_strands):
            # Difference vector between centroids
            diff_vector = centroids[m] - centroids[n]
            # Project onto average axis and take absolute value
            shift_mn = np.abs(np.dot(avg_axis, diff_vector))
            pairwise_shifts.append(shift_mn)
            pair_labels.append(f"{m+1}-{n+1}")
    
    return {
        'pairwise_shifts': pairwise_shifts,
        'pair_labels': pair_labels
    }

def calculate_helical_regularity(strand):
    """
    Calculate how regular the helix is in terms of consistency of rise, twist, and radius.
    
    Parameters:
    -----------
    strand : numpy.ndarray
        Array of shape [strand_length, 3] containing C-alpha coordinates
        
    Returns:
    --------
    float
        Regularity score (dimensionless). Lower values indicate more regular helices.
        Perfect helices approach 0, while highly irregular helices have higher scores.
    """
    # Get helical parameters (axis, centroid, etc.)
    params = calculate_helical_parameters_by_strand(strand)
    centroid = params['centroid']
    axis = params['axis']
    centered_coords = strand - centroid
    projections = np.dot(centered_coords, axis)
    sorted_indices = np.argsort(projections)
    ordered_projections = projections[sorted_indices]
    local_rises = np.diff(ordered_projections)
    projection_vectors = np.outer(projections, axis)
    plane_coords = centered_coords - projection_vectors
    local_radii = np.linalg.norm(plane_coords, axis=1)
    
    u, s, vh = np.linalg.svd(plane_coords)
    ref_vec1 = vh[0]
    ref_vec2 = vh[1]
    
    angles = np.arctan2(np.dot(plane_coords, ref_vec2), np.dot(plane_coords, ref_vec1))
    angles = np.unwrap(angles)
    ordered_angles = angles[sorted_indices]
    ordered_radii = local_radii[sorted_indices]
    local_twists = np.diff(ordered_angles)

    rise_std = np.std(local_rises)
    twist_std = np.std(local_twists)
    radius_std = np.std(ordered_radii)
    
    rise_variation = rise_std / np.mean(np.abs(local_rises)) if np.mean(np.abs(local_rises)) > 0 else 0
    twist_variation = twist_std / np.mean(np.abs(local_twists)) if np.mean(np.abs(local_twists)) > 0 else 0
    radius_variation = radius_std / np.mean(ordered_radii) if np.mean(ordered_radii) > 0 else 0
    
    regularity_score = rise_variation + twist_variation + radius_variation
    
    return regularity_score

def calculate_local_deformations(strand, ideal_strand):
    """
    Calculate local deformations along the helix compared to ideal structure (Kabsch algorithm).
    
    Parameters:
    -----------
    strand : numpy.ndarray
        Array of shape [strand_length, 3] containing C-alpha coordinates
    ideal_strand : numpy.ndarray
        Array of shape [strand_length, 3] containing ideal C-alpha coordinates
        
    Returns:
    --------
    numpy.ndarray
        Array of shape [strand_length] containing local deformation scores.
        Each element represents the deviation (in coordinate units, typically Angstroms) 
        of that residue from its ideal position after optimal local alignment.
        Uses a sliding window approach to account for local flexibility while
        measuring deviations from ideal geometry at each residue position.
    """
    strand_length = len(strand)
    local_deformations = np.zeros(strand_length)
    
    # Optimally align the entire strand to the ideal
    strand_centroid = np.mean(strand, axis=0)
    ideal_centroid = np.mean(ideal_strand, axis=0)
    # Center both
    centered_strand = strand - strand_centroid
    centered_ideal = ideal_strand - ideal_centroid
    # Find optimal rotation using Kabsch algorithm
    H = np.dot(centered_strand.T, centered_ideal)
    U, S, Vt = np.linalg.svd(H)
    # Ensure proper rotation (not reflection)
    R = np.dot(U, Vt)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(U, Vt)
    
    # Apply rotation
    aligned_strand = np.dot(centered_strand, R) + ideal_centroid
    # Now calculate local deformations using sliding window approach
    window_size = min(5, strand_length // 2)  # Use 5 residues or half the strand
    
    for i in range(strand_length):
        # Define window around residue i
        start = max(0, i - window_size)
        end = min(strand_length, i + window_size + 1)
        window_ideal = ideal_strand[start:end]
        window_aligned = aligned_strand[start:end]
        # Center window
        window_ideal_center = np.mean(window_ideal, axis=0)
        window_aligned_center = np.mean(window_aligned, axis=0)
        centered_window_ideal = window_ideal - window_ideal_center
        centered_window_aligned = window_aligned - window_aligned_center
        # Find local optimal rotation
        H_local = np.dot(centered_window_aligned.T, centered_window_ideal)
        U_local, S_local, Vt_local = np.linalg.svd(H_local)
        # Ensure proper rotation
        R_local = np.dot(U_local, Vt_local)
        if np.linalg.det(R_local) < 0:
            Vt_local[-1, :] *= -1
            R_local = np.dot(U_local, Vt_local)
        
        # Apply local rotation
        locally_aligned = np.dot(centered_window_aligned, R_local) + window_ideal_center
        # Calculate deviation at residue i
        i_local = i - start  # Index in the window
        residue_deviation = np.linalg.norm(locally_aligned[i_local] - window_ideal[i_local])
        # Store local deformation
        local_deformations[i] = residue_deviation
    
    return local_deformations

def calculate_interhelical_metrics(strands, ideal_strands):
    """
    Calculate metrics related to the relationship between all helices.
    
    Parameters:
    -----------
    strands : list
        List of numpy arrays representing the helices
    ideal_strands : list
        List of numpy arrays representing the ideal helices
        
    Returns:
    --------
    dict
        Dictionary containing interhelical metrics with arrays of size [num_pairs] where
        num_pairs = num_strands * (num_strands - 1) / 2 (all unique strand pairs):
        
        - 'axis_distances': list of float, minimum distances between helical axes for each pair
        - 'ideal_axis_distances': list of float, ideal minimum distances between helical axes
        - 'axis_distance_deviations': list of float, relative deviations from ideal axis distances
        
        - 'axis_angles': list of float, angles between helical axes in radians for each pair  
        - 'ideal_axis_angles': list of float, ideal angles between helical axes in radians
        - 'axis_angle_deviations': list of float, relative deviations from ideal axis angles
        
        - 'centroid_distances': list of float, distances between strand centroids for each pair
        - 'ideal_centroid_distances': list of float, ideal distances between strand centroids  
        - 'centroid_distance_deviations': list of float, relative deviations from ideal centroid distances
        
        And optional metrics, recommended for post-processing instead.
        - 'interhelical_geometry_deviation': float, overall geometric deviation score (0-1 scale)
          representing the mean of all relative deviations across distance, angle, and centroid metrics
    """
    num_strands = len(strands)
    strand_params = [calculate_helical_parameters_by_strand(strand) for strand in strands]
    ideal_params = [calculate_helical_parameters_by_strand(strand) for strand in ideal_strands]
    
    axis_distances = []
    ideal_axis_distances = []
    axis_angles = []
    ideal_axis_angles = []
    centroid_distances = []
    ideal_centroid_distances = []
    
    # Process all pairs of strands
    for i in range(num_strands):
        for j in range(i+1, num_strands):
            # Calculate minimum distance between skew lines (helical axes)
            axis_i = strand_params[i]['axis']
            axis_j = strand_params[j]['axis']
            centroid_i = strand_params[i]['centroid']
            centroid_j = strand_params[j]['centroid']
            # Vector connecting centroids
            connecting_vector = centroid_j - centroid_i
            # Calculate minimum distance between skew lines
            cross_product = np.cross(axis_i, axis_j)
            cross_norm = np.linalg.norm(cross_product)
            if cross_norm < 1e-10: # For numerical stability
                # Parallel axes
                projection = np.dot(connecting_vector, axis_i)
                perp_vector = connecting_vector - projection * axis_i
                distance = np.linalg.norm(perp_vector)
            else:
                # Skew axes
                distance = np.abs(np.dot(connecting_vector, cross_product)) / cross_norm
            axis_distances.append(distance)
            
            # Calculate angle between axes
            cos_angle = np.dot(axis_i, axis_j)
            # Clamp to prevent numerical issues
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            axis_angles.append(angle)
            
            # Calculate centroid distance
            centroid_distance = np.linalg.norm(centroid_i - centroid_j)
            centroid_distances.append(centroid_distance)
            
            # Do the same for ideal strands
            ideal_axis_i = ideal_params[i]['axis']
            ideal_axis_j = ideal_params[j]['axis']
            ideal_centroid_i = ideal_params[i]['centroid']
            ideal_centroid_j = ideal_params[j]['centroid']
            ideal_connecting_vector = ideal_centroid_j - ideal_centroid_i
            ideal_cross_product = np.cross(ideal_axis_i, ideal_axis_j)
            ideal_cross_norm = np.linalg.norm(ideal_cross_product)
            if ideal_cross_norm < 1e-10:
                ideal_projection = np.dot(ideal_connecting_vector, ideal_axis_i)
                ideal_perp_vector = ideal_connecting_vector - ideal_projection * ideal_axis_i
                ideal_distance = np.linalg.norm(ideal_perp_vector)
            else:
                ideal_distance = np.abs(np.dot(ideal_connecting_vector, ideal_cross_product)) / ideal_cross_norm
            ideal_axis_distances.append(ideal_distance)
            # Ideal angle
            ideal_cos_angle = np.dot(ideal_axis_i, ideal_axis_j)
            ideal_cos_angle = np.clip(ideal_cos_angle, -1.0, 1.0)
            ideal_angle = np.arccos(ideal_cos_angle)
            ideal_axis_angles.append(ideal_angle)
            # Ideal centroid distance
            ideal_centroid_distance = np.linalg.norm(ideal_centroid_i - ideal_centroid_j)
            ideal_centroid_distances.append(ideal_centroid_distance)
    
    # Calculate deviation from ideal geometry
    axis_distance_deviations = [
        np.abs(dist - ideal) / ideal if ideal > 0 else dist
        for dist, ideal in zip(axis_distances, ideal_axis_distances)]
    axis_angle_deviations = [
        np.abs(angle - ideal) / ideal if ideal > 0 else angle
        for angle, ideal in zip(axis_angles, ideal_axis_angles)]
    centroid_distance_deviations = [
        np.abs(dist - ideal) / ideal if ideal > 0 else dist
        for dist, ideal in zip(centroid_distances, ideal_centroid_distances)]
    
    # Calculate overall interhelical geometry deviation - optional
    interhelical_geometry_deviation = (
        np.mean(axis_distance_deviations) + 
        np.mean(axis_angle_deviations) + 
        np.mean(centroid_distance_deviations)) / 3.0
    
    return {
        'axis_distances': axis_distances,
        'ideal_axis_distances': ideal_axis_distances,
        'axis_distance_deviations': axis_distance_deviations,
        'axis_angles': axis_angles,
        'ideal_axis_angles': ideal_axis_angles,
        'axis_angle_deviations': axis_angle_deviations,
        'centroid_distances': centroid_distances,
        'ideal_centroid_distances': ideal_centroid_distances,
        'centroid_distance_deviations': centroid_distance_deviations,
        'interhelical_geometry_deviation': interhelical_geometry_deviation}

def calculate_frame_deformation_score(frame_result):
    """
    Calculate an overall deformation score for a frame. Better if done in post-processing.
    
    Parameters:
    -----------
    frame_result : dict
        Dictionary containing metrics for a frame
        
    Returns:
    --------
    float
        Overall deformation score
    """
    # Extract relevant metrics
    individual_metrics = frame_result['individual_helix_metrics']
    
    # Check if we have multiple strands (interhelix_metrics only exists for multi-strand)
    has_multiple_strands = 'interhelix_metrics' in frame_result
    
    if has_multiple_strands:
        # Multi-strand case: include interhelical geometry
        interhelix_metrics = frame_result['interhelix_metrics']
        
        # Weights for different components
        weights = {
            'shift': 0.2,  
            'helical_regularity': 0.2,
            'local_deformation': 0.2,
            'interhelical_geometry': 0.4
        }
        
        # Calculate individual helix component
        individual_scores = []
        for strand_metrics in individual_metrics:
            strand_score = (
                weights['shift'] * strand_metrics['shift'] + 
                weights['helical_regularity'] * strand_metrics['helical_regularity'] +
                weights['local_deformation'] * strand_metrics['local_deformation_mean']
            )
            individual_scores.append(strand_score)
        
        # Average score across all strands
        individual_score = np.mean(individual_scores)
        
        # Interhelical geometry score
        interhelical_score = interhelix_metrics['interhelical_geometry_deviation']
        
        # Combine scores
        overall_score = (
            (1 - weights['interhelical_geometry']) * individual_score +
            weights['interhelical_geometry'] * interhelical_score
        )
        
    else:
        # Single strand case: only individual helix metrics
        # Weights for single strand (no interhelical component)
        weights = {
            'shift': 0.33,  
            'helical_regularity': 0.33,
            'local_deformation': 0.34
        }
        
        # Calculate score (should only be one strand)
        strand_metrics = individual_metrics[0]
        overall_score = (
            weights['shift'] * strand_metrics['shift'] + 
            weights['helical_regularity'] * strand_metrics['helical_regularity'] +
            weights['local_deformation'] * strand_metrics['local_deformation_mean']
        )
    
    return overall_score

def calculate_deformation_trend(frame_scores):
    """
    Calculate trend in deformation across frames.
    Parameters:
    -----------
    frame_scores : list
        List of deformation scores for each frame
        
    Returns:
    --------
    dict
        Dictionary containing trend metrics
    """
    scores = np.array(frame_scores)
    frames = np.arange(len(scores))
    # Simple linear regression
    mean_frame = np.mean(frames)
    mean_score = np.mean(scores)
    numerator = np.sum((frames - mean_frame) * (scores - mean_score))
    denominator = np.sum((frames - mean_frame) ** 2)
    if denominator == 0:
        slope = 0
    else:
        slope = numerator / denominator
    intercept = mean_score - slope * mean_frame
    # Calculate R^2
    predicted = slope * frames + intercept
    ss_total = np.sum((scores - mean_score) ** 2)
    ss_residual = np.sum((scores - predicted) ** 2)
    if ss_total == 0:
        r_squared = 0
    else:
        r_squared = 1 - (ss_residual / ss_total)
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'increasing': slope > 0,
        'trend_significance': abs(r_squared)}

def print_results_as_table(results, output_file, num_strands):
    """
    Print results as a formatted table.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from analyze_multi_helix_deformation
    num_strands : int
        Number of strands in the analysis
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the results in tabular format
    """
    # Create a list of column names
    columns = []
    # Add columns for each strand's metrics (changed axial_shift to shift)
    for strand_idx in range(num_strands):
        for metric in ['shift', 'helical_regularity', 'local_deformation_mean', 'local_deformation_max']:
            columns.append(f"strand{strand_idx+1}_{metric}")
    
    # Add pairwise axial shift columns
    if num_strands > 1:
        num_pairs = num_strands * (num_strands - 1) // 2
        # Get pair labels from first frame
        pair_labels = results['frame_metrics'][0]['pairwise_shift_labels']
        for label in pair_labels:
            columns.append(f"axial_shift_{label}")
    
    # Add interhelix metrics columns with pair labels
    if num_strands > 1:
        num_pairs = num_strands * (num_strands - 1) // 2
        # Get pair labels from first frame
        pair_labels = results['frame_metrics'][0]['pairwise_shift_labels']
        interhelix_metrics = [
            'axis_distance_deviations',
            'axis_angle_deviations', 
            'centroid_distance_deviations']
        for metric in interhelix_metrics:
            for label in pair_labels:
                columns.append(f"{metric}_{label}")
        columns.append('interhelical_geometry_deviation')
    
    # Create data rows
    data = []
    frame_metrics = results['frame_metrics']
    for frame_result in frame_metrics:
        row = []
        # Add strand metrics (changed axial_shift to shift)
        for strand_idx in range(num_strands):
            strand_metrics = frame_result['individual_helix_metrics'][strand_idx]
            for metric in ['shift', 'helical_regularity', 'local_deformation_mean', 'local_deformation_max']:
                row.append(strand_metrics[metric])
        
        # Add pairwise axial shifts
        if num_strands > 1:
            for shift_val in frame_result['pairwise_axial_shifts']:
                row.append(shift_val)
        
        # Add interhelix metrics
        if num_strands > 1:
            interhelix = frame_result['interhelix_metrics']
            for metric in interhelix_metrics:
                for val in interhelix[metric]:
                    row.append(val)
            row.append(interhelix['interhelical_geometry_deviation'])
        
        data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    # Add frame index as the first column
    frame_indices = [frame_result['frame'] for frame_result in frame_metrics]
    df.insert(0, 'frame', frame_indices)
    df.to_csv(output_file, index=False)
    return df

####################################################################
"""
SECTION III
Python code to calculate INTER-helix deformations using a 
cross-sectional triangle for collagen or any other triple helices.
Note that this only works for triple helices.
For other values of num_strands, scroll to Section II. 
Basic plotting and metric validations also added.
"""
####################################################################

def calculate_triangle_properties(p1, p2, p3):
    """
    Calculate area, perimeter, and isoperimetric ratio for a triangle.
    
    Parameters:
    p1, p2, p3: numpy arrays of shape (3,) representing triangle vertices
    
    Returns:
    area, perimeter, isoperimetric_ratio
    """
    # Calculate side lengths
    a = np.linalg.norm(p2 - p1)
    b = np.linalg.norm(p3 - p2)
    c = np.linalg.norm(p1 - p3)
    
    # Calculate area using cross product
    area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
    
    # Calculate perimeter
    perimeter = a + b + c
    
    # Calculate isoperimetric ratio (4πA/P²)
    if perimeter > 0:
        isoperimetric_ratio = (4 * np.pi * area) / (perimeter ** 2)
    else:
        isoperimetric_ratio = 0
    
    return area, perimeter, isoperimetric_ratio

def analyze_collagen_deformation(coordinates, output_file):
    """ 
    Parameters:
    coordinates: numpy array of shape [frames, strand_length*3, 3]
    output_file: string, output filename
    
    The function calculates shape and size deformation metrics for each residue
    at each frame and writes to a file.
    """
    frames, total_atoms, coords = coordinates.shape
    strand_length = total_atoms // 3
    print(f"Processing {frames} frames with {strand_length} residues per strand")
    
    results = []
    
    # Loop through each residue position
    for residue_idx in range(strand_length):
        if (residue_idx % 3 == 0):
            print(f"Processing Tripeptide {residue_idx // 3 + 1}/{strand_length // 3}")
        
        # Get indices for helix units from each strand for this residue
        ca1_idx = residue_idx                    # Strand 1
        ca2_idx = strand_length + residue_idx    # Strand 2
        ca3_idx = 2 * strand_length + residue_idx # Strand 3
        # Get reference triangle (frame 0)
        ref_p1 = coordinates[0, ca1_idx, :]
        ref_p2 = coordinates[0, ca2_idx, :]
        ref_p3 = coordinates[0, ca3_idx, :]
        ref_area, ref_perimeter, ref_ir = calculate_triangle_properties(ref_p1, ref_p2, ref_p3)
        # Process each frame
        for frame in range(frames):
            # Get current triangle
            p1 = coordinates[frame, ca1_idx, :]
            p2 = coordinates[frame, ca2_idx, :]
            p3 = coordinates[frame, ca3_idx, :]
            # Calculate current properties
            area, perimeter, ir = calculate_triangle_properties(p1, p2, p3)
            # Calculate deformation metrics
            if ref_area > 0:
                size_deformation = (area - ref_area) / ref_area
            else:
                size_deformation = 0
            if ref_ir > 0:
                shape_deformation = (ir - ref_ir) / ref_ir
            else:
                shape_deformation = 0
            # Store results
            results.append({
                'frame': frame,
                'residue': residue_idx + 1,
                'shape_deformation': shape_deformation,
                'size_deformation': size_deformation
            })
    df = pd.DataFrame(results)
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Results of inter-helix metrics over time written to {output_file}")
    return df

def calculate_statistics(input_file, output_file):
    """
    Calculate mean and standard deviation for deformation metrics over residues/tripeptides based on definition.
    Parameters:
    input_file: string, input filename from analyze_collagen_deformation
    output_file: string, output filename for statistics
    """
    df = pd.read_csv(input_file, sep='\t')
    print(f"Calculating statistics from {input_file}")
    stats = df.groupby('residue').agg({'shape_deformation': ['mean', 'std'],'size_deformation': ['mean', 'std']}).round(6)
    # Flatten column names
    stats.columns = ['mean_shape', 'std_shape', 'mean_size', 'std_size']
    # Reset index to get residue as a column
    stats = stats.reset_index()
    stats.columns = ['residue_number', 'mean_shape', 'std_shape', 'mean_size', 'std_size']
    stats.to_csv(output_file, sep='\t', index=False)
    print(f"Statistics of inter-helix metrics per residue averaged over time written to {output_file}")
    return stats

def calculate_statistics_over_time(input_file, output_file):
    """
    Calculate mean and standard deviation for deformation metrics, this time over time.
    Parameters:
    input_file: string, input filename from analyze_collagen_deformation
    output_file: string, output filename for statistics
    """
    df = pd.read_csv(input_file, sep='\t')
    print(f"Calculating statistics from {input_file}")
    stats = df.groupby('frame').agg({'shape_deformation': ['mean', 'std'],'size_deformation': ['mean', 'std']}).round(6)
    # Flatten column names
    stats.columns = ['mean_shape', 'std_shape', 'mean_size', 'std_size']
    # Reset index to get residue as a column
    stats = stats.reset_index()
    stats.columns = ['frame', 'mean_shape', 'std_shape', 'mean_size', 'std_size']
    # Create a display version with custom labels
    display_stats = stats.rename(columns={
        'frame': 'frame_number',
        'mean_shape': 'Mean_Shape_averaged_over_residues',
        'std_shape': 'STD_Shape_averaged_over_residues',
        'mean_size': 'Mean_Size_averaged_over_residues',
        'std_size': 'STD_Size_averaged_over_residues' })
    display_stats.to_csv(output_file, sep='\t', index=False)
    print(f"Statistics of inter-helix metrics averaged over residues written to {output_file}")
    return stats

def plot_deformation_statistics(stats_file, output_plot, plot=True, save=False, figsize=(12, 8)):
    """
    Plot mean shape and size deformation over residues with error bars.
    
    Parameters:
    stats_file: string, input statistics file
    output_plot: string, output plot filename
    figsize: tuple, figure size (width, height)
    """
    
    df = pd.read_csv(stats_file, sep='\t')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot shape deformation
    ax1.errorbar(df['residue_number'], df['mean_shape'], yerr=df['std_shape'],
                 marker='o', markersize=4, capsize=3, capthick=1, 
                 color='blue', ecolor='blue', linewidth=2, elinewidth=1)
    ax1.set_ylabel('Shape Deformation\n(Isoperimetric Ratio Change)', fontsize=12)
    ax1.set_title('Collagen Triple Helix Deformation Analysis', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot size deformation
    ax2.errorbar(df['residue_number'], df['mean_size'], yerr=df['std_size'],
                 marker='s', markersize=4, capsize=3, capthick=1,
                 color='red', ecolor='red', linewidth=2, elinewidth=1)
    ax2.set_ylabel('Size Deformation\n(Area Change)', fontsize=12)
    ax2.set_xlabel('Residue Number', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save == True :
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"Residue-based deformation plotted to {output_plot}")
    if plot == True :
        plt.show()
    plt.close(fig)
    
    print("\nSummary Statistics:")
    print(f"Shape deformation - Mean: {df['mean_shape'].mean():.4f}, Max: {df['mean_shape'].max():.4f}")
    print(f"Size deformation - Mean: {df['mean_size'].mean():.4f}, Max: {df['mean_size'].max():.4f}")
    print(f"Most deformed residue (shape): {df.loc[df['mean_shape'].idxmax(), 'residue_number']}")
    print(f"Most deformed residue (size): {df.loc[df['mean_size'].idxmax(), 'residue_number']}")

def plot_deformation_statistics_over_time(stats_file, output_plot, plot=True, save=False, figsize=(12, 8)):
    """
    Plot mean shape and size deformation over time with error bars. The "/1000" for x axis is for ps->ns.
    
    Parameters:
    stats_file: string, input statistics file
    output_plot: string, output plot filename
    figsize: tuple, figure size (width, height)
    """
    df = pd.read_csv(stats_file, sep='\t')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot shape deformation
    ax1.errorbar(df['frame_number']/1000, df['Mean_Shape_averaged_over_residues'], yerr=df['STD_Shape_averaged_over_residues'],
                 marker='o', markersize=4, capsize=3, capthick=1, 
                 color='blue', ecolor='blue', linewidth=2, elinewidth=1)
    ax1.set_ylabel('Shape Deformation\n(Isoperimetric Ratio Change)', fontsize=12)
    ax1.set_title('Collagen Triple Helix Deformation Analysis', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot size deformation
    ax2.errorbar(df['frame_number']/1000, df['Mean_Size_averaged_over_residues'], yerr=df['STD_Size_averaged_over_residues'],
                 marker='s', markersize=4, capsize=3, capthick=1,
                 color='red', ecolor='red', linewidth=2, elinewidth=1)
    ax2.set_ylabel('Size Deformation\n(Area Change)', fontsize=12)
    ax2.set_xlabel('Time, t / ns', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save == True :
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"Time-based deformation plotted to {output_plot}")
    if plot == True :
        plt.show()
    plt.close(fig)

def plot_combined_deformation(stats_file, shape_weight, size_weight, output_plot, plot=True, save=False, ff= 30, figsize=(10, 6)):
    """
    Plot combined shape and size deformation over residues with user-defined weights.
    
    Parameters:
    stats_file: string, input statistics file
    shape_weight: float, weight for shape deformation (default 0.5)
    size_weight: float, weight for size deformation (default 0.5)
    output_plot: string, output plot filename
    figsize: tuple, figure size (width, height)
    """
    df = pd.read_csv(stats_file, sep='\t')
    combined_mean = (shape_weight * df['mean_shape'] + size_weight * df['mean_size'])
    # For weighted sum: σ_combined = √(w1²σ1² + w2²σ2²)
    combined_std = np.sqrt((shape_weight * df['std_shape'])**2 + (size_weight * df['std_size'])**2)
    
    plt.figure(figsize=figsize)
    plt.errorbar(df['residue_number'], combined_mean, yerr=combined_std, marker='o', markersize=5, capsize=4, capthick=1.5, color='purple', ecolor='purple', linewidth=2, elinewidth=1)
    plt.xlabel('Residue Number', fontsize=ff)
    plt.ylabel('Total Deformation', fontsize=ff)
    # plt.title(f'Combined Collagen Deformation\n(Shape weight: {shape_weight}, Size weight: {size_weight})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    max_val = len(df['mean_shape'])
    plt.xlim(0, max_val+1)
    plt.xticks(range(0, int(max_val) + 1, 3),fontsize=ff)
    plt.yticks(fontsize=ff)
    # plt.text(0.02, 0.98, f'Shape: {shape_weight:.1f}\nSize: {size_weight:.1f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    if save == True :
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"Combined deformation plot saved to {output_plot}")
    if plot == True :
        plt.show()
    
    # Print summary statistics
    print(f"\nCombined Deformation Statistics:")
    print(f"Mean: {combined_mean.mean():.4f}")
    print(f"Max: {combined_mean.max():.4f}")
    print(f"Most deformed residue: {df.loc[combined_mean.idxmax(), 'residue_number']}")
    print(f"Standard deviation range: {combined_std.min():.4f} - {combined_std.max():.4f}")

def plot_combined_deformation_over_time(descriptor,stats_file, shape_weight, size_weight, output_plot, plot=True, save=False, ff= 30, figsize=(10, 6)):
    """
    Plot combined shape and size deformation over time with user-defined weights. The "1000" in plotting is for ease and ps->ns.
    
    Parameters:
    stats_file: string, input statistics file
    shape_weight: float, weight for shape deformation (default 0.5)
    size_weight: float, weight for size deformation (default 0.5)
    output_plot: string, output plot filename
    figsize: tuple, figure size (width, height)
    """
    df = pd.read_csv(stats_file, sep='\t')
    df['mean_shape'] = pd.to_numeric(df['mean_shape'], errors='coerce')
    df['std_shape'] = pd.to_numeric(df['std_shape'], errors='coerce')
    df['mean_size'] = pd.to_numeric(df['mean_size'], errors='coerce')
    df['std_size'] = pd.to_numeric(df['std_size'], errors='coerce')
    combined_mean = (shape_weight * df['mean_shape'] + size_weight * df['mean_size'])
    # For weighted sum: σ_combined = √(w1²σ1² + w2²σ2²)
    combined_std = np.sqrt((shape_weight * df['std_shape'])**2 + (size_weight * df['std_size'])**2)
    
    f = open(f"{descriptor}_combtime.dat",'w')
    for i in range (len(combined_mean)):
        f.write(f"{combined_mean[i]}\t{combined_std[i]}\n")
    f.close()

    plt.figure(figsize=figsize)
    plt.errorbar((df['frame'][::1000])/1000, combined_mean[::1000], yerr=combined_std[::1000], marker='o', markersize=5, capsize=4, capthick=1.5, color='purple', ecolor='purple', linewidth=2, elinewidth=1)
    plt.xlabel('Time, t / ns', fontsize=ff)
    plt.ylabel('Total Deformation', fontsize=ff)
    # plt.title(f'Combined Collagen Deformation\n(Shape weight: {shape_weight}, Size weight: {size_weight})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(bottom=0)
    plt.yticks(fontsize=ff)
    plt.xticks(fontsize=ff)
    # plt.text(0.02, 0.98, f'Shape: {shape_weight:.1f}\nSize: {size_weight:.1f}', transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    if save == True :
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"Combined deformation plot saved to {output_plot}")
    if plot == True :
        plt.show()

def analyze_tripeptides_plots_correlations(shapes_array, sizes_array, special_tripeptides, glucose_labels, glucose_colors, tripeptide_labels, grid_colors, plot1=False, plot2=False, plot3=False):
    """
    Analyze tripeptide deformation data and create various plots for correlation or sanity checks.
    
    Parameters:
    -----------
    shapes_array : numpy array, shape (4, frames, tripeps*3)
        Shape deformation data for 4 glucose concentrations
    sizes_array : numpy array, shape (4, frames, tripeps*3) 
        Size deformation data for 4 glucose concentrations
    special_tripeptides : list of 4 ints
        Tripeptide indices to analyze (1-indexed)
    glucose_labels : list of 4 strings
        Labels for glucose concentrations
    tripeptide_labels : list of 4 strings
        Labels for tripeptides
    grid_colors : list of 4 strings
        Colors for correlation plot grids
    """
    n_glucose, frames, _ = shapes_array.shape
    time = np.arange(frames) / 1000  # Convert frames to time
    space = 100

    # Step 1: Extract and average tripeptide data
    tripeptide_shapes = np.zeros((n_glucose, frames, len(special_tripeptides)))
    tripeptide_sizes = np.zeros((n_glucose, frames, len(special_tripeptides)))
    
    for i, tripeptide_idx in enumerate(special_tripeptides):
        # Convert 1-indexed to 0-indexed and get residue indices
        start_residue = (tripeptide_idx - 1) * 3
        end_residue = start_residue + 3
        
        # Average over the 3 residues for each tripeptide
        tripeptide_shapes[:, :, i] = np.mean(shapes_array[:, :, start_residue:end_residue], axis=2)
        tripeptide_sizes[:, :, i] = np.mean(sizes_array[:, :, start_residue:end_residue], axis=2)

    if plot1 == True:
        # Plot 1: Lines for tripeptides, separate plots for each glucose concentration
        fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
        axes1 = axes1.flatten()
        
        for glucose_idx in range(n_glucose):
            ax = axes1[glucose_idx]
            
            # Create subplots within each glucose concentration plot
            fig_sub, (ax_size, ax_shape) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Size plot (left)
            for trip_idx, trip_label in enumerate(tripeptide_labels):
                ax_size.plot(time[::space], tripeptide_sizes[glucose_idx, :, trip_idx][::space], label=f"Tripeptide {trip_label}", linewidth=3)
            ax_size.set_xlabel('Time (ns)', fontsize=20)
            ax_size.set_ylabel(r'$\langle \text{Area}(k,t) \rangle _m$ / $\AA^2$', fontsize=20)
            ax_size.tick_params(axis='both', labelsize=15)
            # ax_size.set_title(f'{glucose_labels[glucose_idx]} - Size', fontsize=20)
            ax_size.legend(fontsize=20, loc="upper left")
            ax_size.set_xlim(0,100)
            ax_size.grid(True, alpha=0.3)
            
            # Shape plot (right)
            for trip_idx, trip_label in enumerate(tripeptide_labels):
                ax_shape.plot(time[::space], tripeptide_shapes[glucose_idx, :, trip_idx][::space], label=f"Tripeptide {trip_label}", linewidth=3)
            ax_shape.set_xlabel('Time (ns)', fontsize=20)
            ax_shape.set_ylabel(r'$\langle \text{IP}(k,t) \rangle _m$', fontsize=20)
            ax_shape.tick_params(axis='both', labelsize=15)
            # ax_shape.set_title(f'{glucose_labels[glucose_idx]} - Shape', fontsize=20)
            ax_shape.legend(fontsize=20, loc="upper left")
            ax_shape.set_xlim(0,100)
            ax_shape.grid(True, alpha=0.3)
            
            plt.suptitle(f'Glucose Concentration: {glucose_labels[glucose_idx]}', fontsize=25)
            plt.tight_layout()
            plt.show()
    
    if plot2 == True:
        # Plot 2: Lines for glucose concentrations, separate plots for each tripeptide
        for trip_idx, trip_label in enumerate(tripeptide_labels):
            fig2, (ax_size, ax_shape) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Size plot (left)
            for glucose_idx, glucose_label in enumerate(glucose_labels):
                ax_size.plot(time[::space], tripeptide_sizes[glucose_idx, :, trip_idx][::space], label=glucose_label, color=glucose_colors[glucose_idx], linewidth=3)
            ax_size.set_xlabel('Time (ns)', fontsize=20)
            ax_size.set_ylabel(r'$\langle \text{Area}(k,t) \rangle _m$ / $\AA^2$', fontsize=20)
            ax_size.tick_params(axis='both', labelsize=15)
            # ax_size.set_title(f'{trip_label} - Size', fontsize=20)
            ax_size.legend(fontsize=20, loc="upper left")
            ax_size.set_xlim(0,100)
            ax_size.grid(True, alpha=0.3)
            
            # Shape plot (right)
            for glucose_idx, glucose_label in enumerate(glucose_labels):
                ax_shape.plot(time[::space], tripeptide_shapes[glucose_idx, :, trip_idx][::space], label=glucose_label, color=glucose_colors[glucose_idx], linewidth=3)
            ax_shape.set_xlabel('Time (ns)', fontsize=20)
            ax_shape.set_ylabel(r'$\langle \text{IP}(k,t) \rangle _m$', fontsize=20)
            ax_shape.tick_params(axis='both', labelsize=15)
            # ax_shape.set_title(f'{trip_label} - Shape', fontsize=20)
            ax_shape.legend(fontsize=20, loc="upper left")
            ax_shape.set_xlim(0,100)
            ax_shape.grid(True, alpha=0.3)
            
            plt.suptitle(f'Tripeptide: {trip_label}', fontsize=25)
            plt.tight_layout()
            plt.show()
    
    if plot3 == True:
        # Plot 3: Correlation plots - 2x2 grid for each glucose concentration
        for glucose_idx, (glucose_label, color) in enumerate(zip(glucose_labels, grid_colors)):
            fig3, axes3 = plt.subplots(2, 2, figsize=(12, 10))
            
            for trip_idx, trip_label in enumerate(tripeptide_labels):
                row = trip_idx // 2
                col = trip_idx % 2
                ax = axes3[row, col]
                
                # Create hexbin correlation plot
                hb = ax.hexbin(tripeptide_sizes[glucose_idx, :, trip_idx], tripeptide_shapes[glucose_idx, :, trip_idx], gridsize=30, cmap=color, mincnt=1)
                
                ax.set_xlabel(r'$\langle \text{Area}(k,t) \rangle _m$ / $\AA^2$', fontsize=20)
                ax.set_ylabel(r'$\langle \text{IP}(k,t) \rangle _m$', fontsize=20)
                ax.tick_params(axis='both', labelsize=15)
                # ax.set_title(f'{trip_label}', fontsize=20)
                ax.grid(True, alpha=0.3)
                
                label_text = f"Tripeptide {trip_label}"
                ax.text(0.95, 0.95, label_text,transform=ax.transAxes,  # Use axes coordinates (0 to 1)
                    fontsize=16,verticalalignment='top',horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

                # # Add colorbar
                # plt.colorbar(hb, ax=ax, label='Count')
                # Add colorbar with values divided by 1000
                cbar = plt.colorbar(hb, ax=ax)
                cbar.set_label('Time (ns)', fontsize=18)
                # Modify colorbar tick labels to show values divided by 1000
                cbar_ticks = cbar.get_ticks()
                cbar.set_ticklabels([f'{tick/1000:.2f}' for tick in cbar_ticks], fontsize=16)
            
            plt.suptitle(f'Size vs Shape Correlation - {glucose_label}', fontsize=16)
            plt.tight_layout()
            plt.show()


#####################################################################
#####################################################################
############################ PYTHON MAIN ############################
#####################################################################
#####################################################################

def run_various_deformations(strand_length,coordinates,output_pwd,num_strands=3,plot_flag=True):
# def run_various_deformations(pwd,output_pwd,trajectory_file,len_strand=12,num_strands=3,plot_flag=True):    
    # Run Section I and II always
    if (num_strands > 0):
        ####################################################################
        # Commands to run Section I for any number of helices
        ####################################################################
        """
        You only need to run the calculations once. Usually the MD trajectories are large files, so calculation will take a while.
        So run the calculation once (Steps 1 and 2), then comment out those lines, and plot however you feel like.
        There are only some basic plots, so you'll need to write your own plotting code based on use-case.
        """
        print("-" * 100)
        print(f"Printouts for Section I for {num_strands:.0f} helices")
        print("This calculates INTRA-strand deformations for any multi-helix system")
        print("All output files start with a prefix \" Section1_\"")
        print("-" * 100)
        weightages = [1.0,1.0,1.0] # weights as rise,radius,twist
        # frames, strand_length, coordinates = get_coordinates(f"{pwd}/{trajectory_file}",len_strand*num_strands)
        # coordinates = np.array(coordinates)
        # print(f"Number of Frames: {frames}")
        # print(f"Length of Strand: {strand_length}")
        # print(f"Shape of Coordinates Array: {coordinates.shape}")
        # Step 1: Make the object of the class, get deformation data
        analyzer = HelixDeformationAnalyzer(coordinates, num_strands=num_strands, weights=weightages)
        results = analyzer.analyze_deformation()
        analyzer.analyze_deformation_by_residue()
        results_time = analyzer.analyze_deformation_time()
        results = analyzer.analyze_deformation_time_per_strand()
        # Step 2: Save if needed
        analyzer.save_deformation_data_residue(f"{output_pwd}/Section1_multi_helix_deform_results_unit.dat", weights=[1.0, 1.0, 1.0])
        analyzer.save_deformation_data_time(f"{output_pwd}/Section1_multi_helix_deform_results_time.dat",weights=weightages)
        analyzer.save_deformation_data_time_per_strand(f"{output_pwd}/Section1_multi_helix_deform_results_time_per_strand.dat")
        if (plot_flag == True):
            # Step 3: Plot the way you feel like
            plot_deformation_from_file(f"{output_pwd}/Section1_multi_helix_deform_results_unit.dat",f"{output_pwd}/Section1_multi_helix_deform_results_plot.png", weights=weightages) 
            plot_time_deformation_from_file(filename=f"{output_pwd}/Section1_multi_helix_deform_results_time.dat",plot_file=f"{output_pwd}/Section1_multi_helix_deform_results_plot_over_time.png",ff=30,weights=weightages)
        print("-" * 100)
        print("-" * 100)

        ####################################################################
        # Commands to run Section II for any number of helices
        ####################################################################
        # FOR ANY OTHER TYPE OF MULTI-HELIX SYSTEM
        """
        You only need to run the calculations once. Usually the MD trajectories are large files, so calculation will take a while.
        So run the calculation once (Steps 1 and 2), then comment out those lines, and plot however you feel like.
        Here, you'll need to write your own plotting code based on use-case.
        """
        print("-" * 100)
        print(f"Printouts for Section II for {num_strands:.0f} helices")
        print("This calculates helical regularity, parameter by strand, axial shift, local deformations, interhelical metrics over strand pairs")
        print("All output files start with a prefix \" Section2_\"")
        print("-" * 100)
        # Step 1: Analyze deformation
        # frames, strand_length, coordinates = get_coordinates(f"{pwd}/{trajectory_file}",len_strand*num_strands)
        # coordinates = np.array(coordinates)
        # print(f"Number of Frames: {frames}")
        # print(f"Length of Strand: {strand_length}")
        # print(f"Shape of Coordinates Array: {coordinates.shape}")
        # print("Interhelical metrics are printed in the order:\n 1 - between strands 1 and 2 \n 2 - between strands 1 and 3 \n ... \n n-1 - between strands 1 and n \n n - between strands 2 and 3 \n ... \n n(n-1) - between strands n-1 and n")
        results = analyze_multi_helix_deformation(coordinates, num_strands=num_strands, strand_length=strand_length//3)
        # Step 2: Print to a file
        df = print_results_as_table(results, output_file=f"{output_pwd}/Section2_multi_helix_deformation_table.dat", num_strands=num_strands)
        # Step 3 (Not Added Here): Plot using columns of df the way you like
        print("-" * 100)
        print("-" * 100)

    #  If num_strands = 3, run Section III
    if (num_strands == 3):
        ## FOR A COLLAGEN TRIPLE HELIX OR ANY TRIPLE HELIX
        ####################################################################
        # Commands to run Section III for triple helix
        ####################################################################
        """
        You only need to run the calculations once. Usually the MD trajectories are large files, so calculation will take a while.
        So run the calculation once (Steps 1 and 2), then comment out those lines, and plot however you feel like.
        There are only some basic plots, so you'll need to write your own plotting code based on use-case.
        """
        print("-" * 100)
        print("Printouts for Section III for triple helix")
        print("This calculates INTER-strand deformations for collagen or any other triple helices")
        print("All output files start with a prefix \" Section3_\"")
        print("-" * 100)
        shape_weights = np.array([1])          # default value
        size_weights = np.array([1])           # default value
        # frames, strand_length, coordinates = get_coordinates(f"{pwd}/{trajectory_file}",len_strand*num_strands)
        # coordinates = np.array(coordinates)
        # print(f"Number of Frames: {frames}")
        # print(f"Length of Strand: {strand_length}")
        # print(f"Shape of Coordinates Array: {coordinates.shape}")    
        # Step 1: Analyze deformation
        df_raw = analyze_collagen_deformation(coordinates,output_file=f"{output_pwd}/Section3_triad_deformation.dat")
        # Step 2: Calculate statistics
        df_stats = calculate_statistics(input_file=f"{output_pwd}/Section3_triad_deformation.dat", output_file=f"{output_pwd}/Section3_triad_deformation_by_unit.dat")
        df_stats_time = calculate_statistics_over_time(input_file=f"{output_pwd}/Section3_triad_deformation.dat", output_file=f"{output_pwd}/Section3_triad_deformation_by_time.dat")
        print("Terminated successfully!")
        if (plot_flag == True):
            # Step 3: Plot based on need, in any order you see fit
            plot_deformation_statistics(stats_file=f"{output_pwd}/Section3_triad_deformation_by_unit.dat",output_plot=f"{output_pwd}/Section3_triad_deformation_by_unit.png", plot=False, save=plot_flag)
            plot_deformation_statistics_over_time(stats_file=f"{output_pwd}/Section3_triad_deformation_by_time.dat",output_plot=f"{output_pwd}/Section3_triad_deformation_by_time.png", plot=False, save=plot_flag)
        print("-" * 100)
        print("-" * 100)
        
"""
These build up the commands to be used in Linux, 
but you can work with this in Python if need be (using the commented def line).
# path_to_working_directory = "path/to/working/directory"
# path_to_output_directory = "path/to/output/directory"
# trajectory_filename = "trajectory.ext"
# number_of_strands = 3
# strand_length = 12 
# run_various_deformations(pwd=path_to_working_directory,output_pwd=path_to_output_directory,trajectory_file=trajectory_filename,len_strand=strand_length,num_strands=number_of_strands)
Command-line interface follows next:
Supports flexible file path input and automatic path resolution.
Also supports input calculation if strand_length is not provided.
"""

def parse_file_path(filepath):
    """
    Parse a file path and return the directory and filename separately.
    
    Args:
        filepath (str): Full path to file, relative path, or just filename
        
    Returns:
        tuple: (directory_path, filename)
    """
    path_obj = Path(filepath)
    
    if path_obj.is_absolute():
        # Absolute path provided
        directory = str(path_obj.parent)
        filename = path_obj.name
    elif "/" in filepath or "\\" in filepath:
        # Relative path with directory
        directory = str(path_obj.parent)
        filename = path_obj.name
        # Convert to absolute path
        directory = str(Path(directory).resolve())
    else:
        # Just filename provided, use current directory
        directory = str(Path.cwd())
        filename = filepath
    
    return directory, filename

def validate_file_exists(filepath):
    """
    Validate that the file exists and is readable.
    
    Args:
        filepath (str): Path to the file
        
    Returns:
        bool: True if file exists and is readable
    """
    path_obj = Path(filepath)
    return path_obj.exists() and path_obj.is_file()

def format_atom_name_for_pdb(atom_name):
    """
    Format atom name to standard PDB 4-character format.
    
    PDB standard formatting:
    - Element in columns 13-14 (0-indexed: 12-13) if 2 characters
    - Element in column 14 (0-indexed: 13) if 1 character
    - Total of 4 characters with appropriate spacing
    
    Common examples:
    - CA  -> " CA "
    - C   -> " C  "
    - N   -> " N  "
    - O   -> " O  "
    - CB  -> " CB "
    - OXT -> " OXT"
    - H   -> " H  "
    """
    # Remove any existing spaces
    clean_name = atom_name.strip()
    
    # # Standard 2-letter atom names (Greek letters, beta carbons, etc.)
    # two_letter_atoms = ['CA', 'CB', 'CG', 'CD', 'CE', 'CZ', 'CH', 
    #                     'OG', 'OD', 'OE', 'OH',
    #                     'SG', 'SD',
    #                     'ND', 'NE', 'NH', 'NZ']
    
    # # Standard 3-letter atom names
    # three_letter_atoms = ['OXT', 'OG1', 'OD1', 'OD2', 'OE1', 'OE2',
    #                       'CG1', 'CG2', 'CD1', 'CD2', 'CE1', 'CE2', 'CE3',
    #                       'CZ2', 'CZ3', 'CH2',
    #                       'ND1', 'ND2', 'NE1', 'NE2', 'NH1', 'NH2']
    
    if len(clean_name) == 1:
        # Single character: " X  " (space, char, space, space)
        return f"{clean_name}   "
    elif len(clean_name) == 2:
        # Two characters: " XX " (space, char, char, space)
        return f" {clean_name} "
    elif len(clean_name) == 3:
        # Three characters: " XXX" (space, char, char, char)
        return f" {clean_name}"
    elif len(clean_name) == 4:
        # Already 4 characters, assume it's formatted
        return clean_name
    else:
        # Longer than 4 characters, truncate to 4
        return clean_name[:4]

def detect_file_type(filename):
    """
    Detect file type based on file extension.
    
    Returns:
    str : 'arc' for Tinker ARC files, 'pdb' for PDB files, None for unknown
    """
    ext = Path(filename).suffix.lower()
    if ext == '.arc':
        return 'arc'
    elif ext == '.pdb':
        return 'pdb'
    else:
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Analyze molecular trajectories with various deformations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
    python %(prog)s /path/to/trajectory.arc --strands 5
    python %(prog)s data/trajectory.arc --strands 2 --length 15 --atom-type 50 51 52
    python %(prog)s /path/to/trajectory.pdb --strands 3 --length 20 --atom-name " CA " " N " "C" --atom-type 401 402
    
    *** FOR PDB FILES: Atom names must include proper spacing (4 characters).
    If not provided, %(prog)s will autocorrect. Note that autocorrect may not always be accurate.
    python %(prog)s trajectory.pdb --strands 2 --length 15 --atom-name " CA " " N  " " C  "

    *** NOTE: If the length of strand of the multi-helix size is not provided, %(prog)s will autocalculate. 
    Make sure your inputs are properly taken into account. Make sure to input the number of strands.
    False validations can occur if the inputs are wrong but the product matches the helix units.
    %(prog)s assumes all strands have the same number of units.

    *** RUNNING EXAMPLES: To replicate the examples provided on GitHub, run:
    python %(prog)s singlehelix.arc --strands 1 --atom-type 401 402
    python %(prog)s doublehelix.arc --strands 2 --atom-name "CA" "N"
    python %(prog)s triplehelix.arc --strands 3 --length 15
        """
    )
    # Generate timestamped filename for logging (moved up for early logging)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"tempfile_{timestamp}.out"
    # Check dependencies first and log the results
    def check_dependencies():
        """Check if all required packages are installed."""
        required_packages = ['numpy', 'pandas', 'matplotlib', 'scipy']
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        return missing_packages
    # Open log file early for dependency checking
    try:
        with open(log_filename, "w") as log_file:
            original_stdout = sys.stdout
            sys.stdout = log_file
            
            missing_deps = check_dependencies()
            
            if missing_deps:
                print("=" * 60)
                print("DEPENDENCY CHECK FAILED")
                print("=" * 60)
                print("The following required packages are missing:")
                for package in missing_deps:
                    print(f"  - {package}")
                print("\nPlease install the missing packages using:")
                print("pip install " + " ".join(missing_deps))
                print("=" * 60)
                
                # Restore stdout before exiting
                sys.stdout = original_stdout
                print(f"Dependency check failed. Details logged to: {log_filename}")
                sys.exit(1)
            else:
                print("=" * 60)
                print("DEPENDENCY CHECK PASSED")
                print("=" * 60)
                print("All required packages are available:")
                required_packages = ['numpy', 'pandas', 'matplotlib', 'scipy']
                for package in required_packages:
                    try:
                        module = __import__(package)
                        version = getattr(module, '__version__', 'unknown')
                        print(f"  ✓ {package} (version: {version})")
                    except:
                        print(f"  ✓ {package} (version: unknown)")
                print("=" * 60)
            
            sys.stdout = original_stdout
            
    except Exception as e:
        print(f"Error creating log file {log_filename}: {e}", file=sys.stderr)
        sys.exit(1)
    
    # File input options - mutually exclusive group
    file_group = parser.add_mutually_exclusive_group(required=True)
    file_group.add_argument(
        "filepath",
        nargs="?",
        help="Path to trajectory file (can be absolute, relative, or just filename)")
    file_group.add_argument(
        "-f", "--file",
        help="Path to trajectory file (alternative to positional argument)")
    # Optional parameters
    parser.add_argument(
        "-s", "--strands",
        type=int,
        help="Number of strands")
    parser.add_argument(
        "-l", "--length",
        type=int,
        help="Strand length")
    parser.add_argument(
        "--atom-name",
        type=str,
        nargs='*',
        help='Atom name(s) to filter. For PDB files (columns 13-16), include spaces (e.g., " CA " " N  " " C  "). For ARC files, spaces are optional.')
    parser.add_argument(
        "--atom-type",
        type=int,
        nargs='*',
        help="Atom type(s) to filter (e.g., 129 130 156). Only applies to ARC files. Sixth column in ARC file. Can specify multiple types. Takes priority over atom name.")
    parser.add_argument(
        "--no-plot",
        action="store_false",
        dest="plot",
        help="Disable default plotting")
    parser.add_argument(
        "-w", "--workdir",
        help="Working directory (default: directory containing the trajectory file)")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output")
    parser.add_argument(
        "--check-file",
        action="store_true",
        help="Check if trajectory file exists before running analysis")
    parser.set_defaults(plot=True)
    
    # Check if --help was requested
    if '--help' in sys.argv or '-h' in sys.argv:
        # Print help to terminal
        parser.print_help()
        print("-" * 50)
        print(f"\nOutput will be directed to: {log_filename}")
        print("-" * 50)
        
        # Create log file for help message
        with open(log_filename, "a") as log_file:
            original_stdout = sys.stdout
            sys.stdout = log_file
            print("\n" + "=" * 60)
            print("HELP MESSAGE REQUESTED")
            print("=" * 60)
            print(parser.format_help())
            sys.stdout = original_stdout
        sys.exit(0)

    # Now parse arguments normally
    args = parser.parse_args()
    
    filepath = args.filepath if args.filepath else args.file
    if args.workdir:
        # Use provided working directory
        path_to_working_directory = str(Path(args.workdir).resolve())
        trajectory_filename = Path(filepath).name
    else:
        # Extract directory and filename from filepath
        path_to_working_directory, trajectory_filename = parse_file_path(filepath)

    if not isinstance(args.strands, int) or args.strands <= 0:
        print("Error: Number of strands must be a positive integer", file=sys.stderr)
        sys.exit(1)

    # Check if file exists (optional)
    if args.check_file:
        full_filepath = os.path.join(path_to_working_directory, trajectory_filename)
        if not validate_file_exists(full_filepath):
            print(f"Error: Trajectory file not found: {full_filepath}", file=sys.stderr)
            sys.exit(1)

    # Verbose output
    if args.verbose:
        print("Configuration:")
        print(f"  Original input: {filepath}")
        print(f"  Resolved working directory: {path_to_working_directory}")
        print(f"  Trajectory filename: {trajectory_filename}")
        print(f"  Number of strands: {args.strands}")
        print(f"  Strand Length: {args.length}")
        if args.atom_name:
            print(f"  Atom name filter: {args.atom_name}")
        if args.atom_type:
            print(f"  Atom type filter: {args.atom_type}")
        print("-" * 50)
        print("-" * 50)
        print("-" * 50)

    # Continue with the main analysis (appending to the same log file)
    try:
        with open(log_filename, "a") as log_file:
            original_stdout = sys.stdout
            sys.stdout = log_file
            
            print("\n" + "=" * 60)
            print("STARTING MAIN ANALYSIS")
            print("=" * 60)
            
            # Detect file type
            print("\n" + "=" * 60)
            print("FILE TYPE DETECTION")
            print("=" * 60)
            file_type = detect_file_type(trajectory_filename)
            print(f"Trajectory filename: {trajectory_filename}")
            print(f"Detected file type: {file_type.upper() if file_type else 'UNKNOWN'}")
            
            if file_type is None:
                print(f"\nERROR: Unknown file type for '{trajectory_filename}'")
                print(f"Supported file types: .arc (Tinker ARC), .pdb (PDB)")
                print("\nAnalysis cannot proceed. Exiting.")
                sys.stdout = original_stdout
                print(f"Error: Unknown file type. Check {log_filename} for details.", file=sys.stderr)
                sys.exit(1)
            
            if file_type == 'arc':
                print(f"Will use function: read_tinker_arc()")
            else:  # pdb
                print(f"Will use function: read_traj_pdb()")
            
            # Process atom name and type filters based on file type
            print("\n" + "=" * 60)
            print("PROCESSING ATOM FILTERS")
            print("=" * 60)
            
            filter_names_arg = None
            filter_types_arg = None
            
            if file_type == 'arc':
                print("File type: ARC (Tinker)")
                
                # Handle atom names for ARC files
                if args.atom_name and len(args.atom_name) > 0:
                    print(f"\nOriginal atom names provided: {args.atom_name}")
                    # Strip leading/trailing spaces for ARC files
                    filter_names_arg = [name.strip() for name in args.atom_name]
                    print(f"Processed atom names (spaces stripped): {filter_names_arg}")
                else:
                    print(f"\nNo atom names provided.")
                
                # Handle atom types for ARC files
                if args.atom_type and len(args.atom_type) > 0:
                    filter_types_arg = args.atom_type
                    print(f"Atom types provided: {filter_types_arg}")
                    if filter_names_arg:
                        print(f"NOTE: Both atom names and types provided. Atom types will take priority.")
                else:
                    print(f"No atom types provided.")
                
                # Determine what will be used
                if filter_types_arg:
                    print(f"\nFinal filter decision: Using atom types {filter_types_arg}")
                elif filter_names_arg:
                    print(f"\nFinal filter decision: Using atom names {filter_names_arg}")
                else:
                    print(f"\nFinal filter decision: Using default atom name ['CA']")
                    
            else:  # pdb
                print("File type: PDB")
                
                # Check if atom types were provided for PDB
                if args.atom_type and len(args.atom_type) > 0:
                    print(f"\nWARNING: Atom types {args.atom_type} provided, but PDB files do not use atom types.")
                    print(f"Atom types will be ignored for PDB file.")
                    filter_types_arg = None  # Explicitly set to None for PDB
                
                # Handle atom names for PDB files
                if args.atom_name and len(args.atom_name) > 0:
                    print(f"\nOriginal atom names provided: {args.atom_name}")
                    
                    # Check if autocorrection is needed
                    needs_correction = False
                    corrected_names = []
                    
                    for name in args.atom_name:
                        if len(name) != 4:
                            needs_correction = True
                            corrected = format_atom_name_for_pdb(name)
                            corrected_names.append(corrected)
                            print(f"  Autocorrecting: '{name}' -> '{corrected}' (PDB requires 4 characters)")
                        else:
                            corrected_names.append(name)
                    
                    filter_names_arg = corrected_names
                    
                    if needs_correction:
                        print(f"\nProcessed atom names (after autocorrection): {filter_names_arg}")
                        print(f"NOTE: Atom names were autocorrected to match PDB 4-character format.")
                    else:
                        print(f"Processed atom names (already in correct format): {filter_names_arg}")
                else:
                    print(f"\nNo atom names provided.")
                    if args.atom_type and len(args.atom_type) > 0:
                        print(f"Since atom types don't apply to PDB, switching to default atom name [' CA ']")
                        filter_names_arg = [" CA "]
                
                # Determine what will be used
                if filter_names_arg:
                    print(f"\nFinal filter decision: Using atom names {filter_names_arg}")
                else:
                    print(f"\nFinal filter decision: Using default atom name [' CA ']")
            
            # Read trajectory using appropriate function
            print("\n" + "=" * 60)
            print("READING TRAJECTORY FILE")
            print("=" * 60)
            
            full_trajectory_path = f"{path_to_working_directory}/{trajectory_filename}"
            print(f"Reading file: {full_trajectory_path}")
            
            if file_type == 'arc':
                print(f"Calling read_tinker_arc() with:")
                print(f"  filter_names: {filter_names_arg}")
                print(f"  filter_types: {filter_types_arg}")
                print()
                
                traj_frames, traj_strand_length, traj_coordinates = read_tinker_arc(
                    full_trajectory_path,
                    filter_names=filter_names_arg,
                    filter_types=filter_types_arg
                )
            else:  # pdb
                print(f"Calling read_traj_pdb() with:")
                print(f"  filter_names: {filter_names_arg}")
                print(f"  filter_types: {filter_types_arg}")
                print()
                
                traj_frames, traj_strand_length, traj_coordinates = read_traj_pdb(
                    full_trajectory_path,
                    filter_names=filter_names_arg,
                    filter_types=filter_types_arg
                )
            
            # Check if reading was successful
            if traj_frames is None or traj_strand_length is None or traj_coordinates is None:
                print(f"\nERROR: Failed to read trajectory file.")
                print(f"Check error messages above for details.")
                print("\nAnalysis cannot proceed. Exiting.")
                sys.stdout = original_stdout
                print(f"Error: Failed to read trajectory. Check {log_filename} for details.", file=sys.stderr)
                sys.exit(1)
            
            traj_coordinates = np.array(traj_coordinates)
            print(f"\nTrajectory reading successful!")
            print(f"Number of Frames: {traj_frames}")
            print(f"Shape of Coordinates Array: {traj_coordinates.shape}")

            # Check if coordinates array is empty
            if traj_coordinates.size == 0 or traj_coordinates.shape[1] == 0:
                error_msg = "\n" + "=" * 60 + "\n"
                error_msg += "ERROR: EMPTY COORDINATES ARRAY\n"
                error_msg += "=" * 60 + "\n"
                error_msg += "The coordinates array is empty. This typically means:\n"
                error_msg += "  - The specified atom name(s) were not found in the input file\n"
                error_msg += "  - The specified atom type(s) were not found in the input file\n"
                error_msg += "\nFilters used:\n"
                if filter_types_arg:
                    error_msg += f"  Atom types: {filter_types_arg}\n"
                if filter_names_arg:
                    error_msg += f"  Atom names: {filter_names_arg}\n"
                error_msg += "\nPlease verify that:\n"
                error_msg += "  1. The atom names/types exist in your trajectory file\n"
                error_msg += "  2. The atom names are correctly formatted (especially for PDB files)\n"
                error_msg += "  3. The atom types match those in your ARC file (if applicable)\n"
                error_msg += "\nAnalysis cannot proceed. Exiting.\n"
                error_msg += f"Full log available at: {log_filename} \n"
                error_msg += "=" * 60
                
                print(error_msg)
                sys.stdout = original_stdout
                print(error_msg, file=sys.stderr)
                sys.exit(1)

            # Auto-calculate missing parameter if only one is provided
            actual_total_cas = traj_coordinates.shape[1]

            # Function to print to both log and terminal
            def print_dual(message):
                """Print to both log file and terminal"""
                print(message)  # To log file
                sys.stdout.flush()
                temp_stdout = sys.stdout
                sys.stdout = original_stdout
                print(message)  # To terminal
                sys.stdout = temp_stdout

            # Validate user inputs against trajectory shape
            print_dual("\n" + "=" * 60)
            print_dual("VALIDATING USER INPUTS")
            print_dual("=" * 60)
            
            # Add filtering information summary
            if file_type == 'arc':
                if filter_names_arg and filter_types_arg:
                    print_dual(f"Filtering: Using atom types {filter_types_arg} (atom names {filter_names_arg} provided but types take priority)")
                elif filter_types_arg:
                    print_dual(f"Filtering: Using atom types {filter_types_arg}")
                elif filter_names_arg:
                    print_dual(f"Filtering: Using atom names {filter_names_arg}")
                else:
                    print_dual(f"Filtering: Using default atom name ['CA']")
            else:  # pdb
                if filter_names_arg:
                    print_dual(f"Filtering: Using atom names {filter_names_arg}")
                else:
                    print_dual(f"Filtering: Using default atom name [' CA ']")
            print_dual("")

            # Check if one or both parameters need to be calculated
            if args.strands is None and args.length is None:
                print_dual("ERROR: Both --strands and --length are missing!")
                print_dual("  Please provide at least one of these parameters.")
                print_dual("\nAnalysis cannot proceed. Exiting.")
                print_dual(f"Full log available at: {log_filename} \n")
                sys.stdout = original_stdout
                sys.exit(1)

            elif args.strands is None:
                # Calculate number of strands from length
                print_dual(f"User specified:")
                print_dual(f"  Length per strand: {args.length}")
                print_dual(f"  Number of strands: NOT PROVIDED (will auto-calculate)")
                
                if actual_total_cas % args.length != 0:
                    print_dual(f"\n✗ INPUT VALIDATION FAILED")
                    print_dual(f"  Cannot evenly divide {actual_total_cas} helix units by length {args.length}")
                    print_dual(f"  Result would be: {actual_total_cas / args.length} strands (not an integer)")
                    print_dual(f"\nERROR: Please provide a length that evenly divides {actual_total_cas}")
                    print_dual("\nAnalysis cannot proceed. Exiting.")
                    print_dual(f"Full log available at: {log_filename} \n")
                    sys.stdout = original_stdout
                    sys.exit(1)
                
                args.strands = actual_total_cas // args.length
                print_dual(f"\n  Auto-calculated number of strands: {actual_total_cas} ÷ {args.length} = {args.strands}")
                # print_dual(f"  Expected total helix units: {args.strands} × {args.length} = {args.strands * args.length}")

            elif args.length is None:
                # Calculate length per strand from number of strands
                print_dual(f"User specified:")
                print_dual(f"  Number of strands: {args.strands}")
                print_dual(f"  Length per strand: NOT PROVIDED (will auto-calculate)")
                
                if actual_total_cas % args.strands != 0:
                    print_dual(f"\n✗ INPUT VALIDATION FAILED")
                    print_dual(f"  Cannot evenly divide {actual_total_cas} helix units by {args.strands} strands")
                    print_dual(f"  Result would be: {actual_total_cas / args.strands} atoms/strand (not an integer)")
                    print_dual(f"\nERROR: Please provide a number of strands that evenly divides {actual_total_cas}")
                    print_dual("\nAnalysis cannot proceed. Exiting.")
                    print_dual(f"Full log available at: {log_filename} \n")
                    sys.stdout = original_stdout
                    sys.exit(1)
                
                args.length = actual_total_cas // args.strands
                print_dual(f"\n  Auto-calculated length per strand: {actual_total_cas} ÷ {args.strands} = {args.length}")
                # print_dual(f"  Expected total helix units: {args.strands} × {args.length} = {args.strands * args.length}")

            else:
                # Both parameters provided by user
                print_dual(f"User specified:")
                print_dual(f"  Number of strands: {args.strands}")
                print_dual(f"  Length per strand: {args.length}")
                print_dual(f"  Expected total helix units: {args.strands} × {args.length} = {args.strands * args.length}")

            print_dual(f"\nTrajectory contains:")
            print_dual(f"  Total helix units: {actual_total_cas}")

            # Check if inputs match trajectory
            expected_total_cas = args.strands * args.length

            if expected_total_cas == actual_total_cas:
                print_dual(f"\n✓ INPUT VALIDATION PASSED")
                print_dual(f"  User inputs match trajectory shape: {expected_total_cas} = {actual_total_cas}")
                print_dual(f"  Final configuration: {args.strands} strands × {args.length} atoms/strand")
                print_dual(f"\n  ======= NOTE: False validations can occur if the inputs are wrong but the product matches the helix units.")
                print_dual(f"  ======= ALWAYS double check your inputs for accurate results.")
            else:
                print_dual(f"\n✗ INPUT VALIDATION FAILED")
                print_dual(f"  User inputs DO NOT match trajectory shape!")
                print_dual(f"  Expected: {args.strands} strands × {args.length} atoms/strand = {expected_total_cas} total atoms")
                print_dual(f"  Found in trajectory: {actual_total_cas} total helix units")
                print_dual(f"\nERROR: Mismatch detected. Please check your inputs:")
                print_dual(f"  - Verify the number of strands (--strands {args.strands})")
                print_dual(f"  - Verify the length per strand (--length {args.length})")
                print_dual(f"  - Ensure their product equals {actual_total_cas}")
                print_dual("\nAnalysis cannot proceed. Exiting.")
                print_dual(f"Full log available at: {log_filename} \n")

                # Restore stdout before exiting
                sys.stdout = original_stdout
                sys.exit(1)

            # If validation passed, continue with analysis
            print("\n" + "=" * 60)
            print("CREATING OUTPUT DIRECTORY")
            print("=" * 60)
            
            # Create new folder name
            new_folder_name = f"HeliXplore_{timestamp}"
            full_path = os.path.join(path_to_working_directory, new_folder_name)
            print(f"Output directory: {full_path}")
            
            # Make the directory
            os.makedirs(full_path, exist_ok=True)
            print(f"Directory created successfully.")
            
            # Run the analysis
            print("\n" + "=" * 60)
            print("RUNNING DEFORMATION ANALYSIS")
            print("=" * 60)
            run_various_deformations(strand_length=traj_strand_length, coordinates=traj_coordinates, output_pwd=full_path, num_strands=args.strands, plot_flag=args.plot)
            
            # Restore stdout
            sys.stdout = original_stdout
            
            print(f"\nAnalysis completed successfully!")
            print(f"Results saved to: {full_path}")
            print(f"Full log available at: {log_filename} \n")

    except Exception as e:
        print(f"Error during analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()