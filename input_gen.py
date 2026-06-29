#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import numpy as np
import time
from rdkit import Chem
from rdkit.Geometry import Point3D

# ---------------------------
# Info
# ---------------------------

def info():
    print(' ---------- \n'
          '    INFO \n'
          ' ---------- \n'
        
          'The script is intended to simplify generation of input xyz files \n'
          'by reasonable alignment of (un)optimized molecules on graphene \n'
          'and subsequent geometry optimization on a trimmed graphene with fixed coords. \n'
          'The optimized molecule is finally re-aligned on a larger graphene \n'
          'for simulation with GRegOptimizer. \n'

          'GReg-InputGen is distributed in the hope that it will be useful, \n'
          'but WITHOUT ANY WARRANTY; without even the implied warranty of \n'
          'MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n'

          ' ---------- \n'
          '    INFO \n'
          ' ----------')

# ---------------------------
# Basic XYZ / RDKit reader
# ---------------------------

def read_xyz(filename):
    mol = Chem.MolFromXYZFile(filename)
    if mol is None:
        raise ValueError(f"Could not read XYZ file: {filename}")

    conf = mol.GetConformer()
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = np.array([
        [conf.GetAtomPosition(i).x,
         conf.GetAtomPosition(i).y,
         conf.GetAtomPosition(i).z]
        for i in range(mol.GetNumAtoms())
    ], dtype=float)

    return mol, atoms, coords


def write_xyz(filename, atoms, coords, comment=""):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{len(atoms)}\n")
        f.write(comment + "\n")
        for atom, xyz in zip(atoms, coords):
            f.write(f"{atom:2s} {xyz[0]:15.8f} {xyz[1]:15.8f} {xyz[2]:15.8f}\n")


def read_xyz_xtb(filename):
    ### ignore any xtb parameters
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty XYZ file: {filename}")

    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"First line is not a valid atom count in {filename!r}") from exc

    atom_lines = lines[2:2 + n_atoms]
    if len(atom_lines) < n_atoms:
        raise ValueError(f"XYZ file {filename!r} contains fewer atom lines than declared.")

    atoms = []
    coords = []
    for idx, line in enumerate(atom_lines, start=3):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed atom line {idx} in {filename!r}: {line.strip()}")
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return atoms, np.array(coords, dtype=float)


def append_xtb_fix_block(filename, first_fixed_atom, last_fixed_atom):
    with open(filename, "a", encoding="utf-8") as f:
        f.write(
            "\n$opt\n"
            "  engine=rf\n"
            "$fix\n"
            f"  atoms: {first_fixed_atom} - {last_fixed_atom}\n"
            "$end\n"
        )

# ---------------------------
# Alignment
# ---------------------------

def best_fit_plane(coords):
    center = coords.mean(axis=0)
    shifted = coords - center
    _, _, vh = np.linalg.svd(shifted)
    normal = vh[-1]
    return center, normal / np.linalg.norm(normal)


def rotation_matrix_from_vectors(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.isclose(c, 1.0):
        return np.eye(3)

    if np.isclose(c, -1.0):
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v /= np.linalg.norm(v)
        return -np.eye(3) + 2 * np.outer(v, v)

    s = np.linalg.norm(v)
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

    return np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)


def pca_thickness_axis(coords):
    center = coords.mean(axis=0)
    shifted = coords - center
    cov = np.cov(shifted.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    thickness_axis = eigvecs[:, np.argmin(eigvals)]
    thickness_axis /= np.linalg.norm(thickness_axis)
    return center, thickness_axis


def align_molecule_to_graphene(graphene_coords, molecule_coords, distance=3.6):
    graphene_center, graphene_normal = best_fit_plane(graphene_coords)

    if graphene_normal[2] < 0:
        graphene_normal *= -1.0

    molecule_center, thickness_axis = pca_thickness_axis(molecule_coords)

    # Rotate molecule so its smallest PCA dimension points along graphene normal.
    R = rotation_matrix_from_vectors(thickness_axis, graphene_normal)
    shifted = molecule_coords - molecule_center
    rotated = shifted @ R.T

    # Put molecular center above xy center of graphene.
    aligned = rotated + graphene_center

    # Shift along graphene normal so closest molecular atom is distance Angstrom away.
    signed_distances = (aligned - graphene_center) @ graphene_normal
    min_distance = signed_distances.min()
    aligned += (distance - min_distance) * graphene_normal

    return aligned


def set_rdkit_coords(mol, coords):
    conf = mol.GetConformer()
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))

# ---------------------------
# Graphene trimming / edge repair
# ---------------------------

def graphene_plane_basis(graphene_coords):
    center, normal = best_fit_plane(graphene_coords)
    if normal[2] < 0:
        normal *= -1.0

    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, normal)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])

    u = trial - np.dot(trial, normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    return center, normal, u, v


def project_to_graphene_plane(coords, origin, u, v):
    rel = coords - origin
    return np.column_stack((rel @ u, rel @ v))


def pairwise_degrees(coords, cutoff):
    n = len(coords)
    degrees = np.zeros(n, dtype=int)
    for i in range(n):
        d = np.linalg.norm(coords - coords[i], axis=1)
        degrees[i] = int(np.count_nonzero((d < cutoff) & (d > 1e-8)))
    return degrees


def prune_uncoordinated_carbons(carbon_atoms, carbon_coords, cutoff=1.6, min_degree=2):
    atoms = list(carbon_atoms)
    coords = np.array(carbon_coords, dtype=float)

    while len(coords) > 0:
        degrees = pairwise_degrees(coords, cutoff)
        keep = degrees >= min_degree
        if np.all(keep):
            return atoms, coords
        atoms = [atom for atom, k in zip(atoms, keep) if k]
        coords = coords[keep]

    return [], np.empty((0, 3), dtype=float)


def add_hydrogens_to_edge_carbons(carbon_atoms, carbon_coords, bond_cutoff=1.6, bond_length=1.09):
    atoms = list(carbon_atoms)
    coords = [np.array(c, dtype=float) for c in carbon_coords]
    carbon_coords = np.array(carbon_coords, dtype=float)

    for i, pc in enumerate(carbon_coords):
        d = np.linalg.norm(carbon_coords - pc, axis=1)
        neighbors = [j for j, dist in enumerate(d) if 1e-8 < dist < bond_cutoff]

        # Degree-2 carbons are graphene edge carbons and need one H cap.
        if len(neighbors) == 2:
            v1 = carbon_coords[neighbors[0]] - pc
            v2 = carbon_coords[neighbors[1]] - pc
            direction = -(v1 + v2)
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                continue
            direction /= norm
            atoms.append("H")
            coords.append(pc + bond_length * direction)

    return atoms, np.array(coords, dtype=float)


def verify_no_degree1_carbons(carbon_coords, cutoff=1.6):
    if len(carbon_coords) == 0:
        raise ValueError("No graphene carbon atoms remain after trimming.")
    degrees = pairwise_degrees(np.array(carbon_coords, dtype=float), cutoff)
    bad = np.where(degrees == 1)[0]
    if len(bad) > 0:
        raise AssertionError(f"Found {len(bad)} degree-1 graphene carbon(s) after trimming/passivation.")


def prepare_trimmed_graphene_for_xtb(
    graphene_atoms,
    graphene_coords,
    molecule_coords,
    buffer=4.0,
    bond_cutoff=1.6,
    edge_ch_bond_length=1.09,
):

    graphene_atoms = np.array(graphene_atoms, dtype=object)
    graphene_coords = np.array(graphene_coords, dtype=float)

    carbon_mask = graphene_atoms == "C"
    carbon_atoms = graphene_atoms[carbon_mask].tolist()
    carbon_coords = graphene_coords[carbon_mask]

    origin, normal, u, v = graphene_plane_basis(carbon_coords)
    mol_uv = project_to_graphene_plane(np.array(molecule_coords, dtype=float), origin, u, v)
    graph_uv = project_to_graphene_plane(carbon_coords, origin, u, v)

    min_u, min_v = mol_uv.min(axis=0) - buffer
    max_u, max_v = mol_uv.max(axis=0) + buffer

    keep = (
        (graph_uv[:, 0] >= min_u) & (graph_uv[:, 0] <= max_u) &
        (graph_uv[:, 1] >= min_v) & (graph_uv[:, 1] <= max_v)
    )

    trimmed_atoms = [a for a, k in zip(carbon_atoms, keep) if k]
    trimmed_coords = carbon_coords[keep]

    trimmed_atoms, trimmed_coords = prune_uncoordinated_carbons(
        trimmed_atoms, trimmed_coords, cutoff=bond_cutoff, min_degree=2
    )
    verify_no_degree1_carbons(trimmed_coords, cutoff=bond_cutoff)

    capped_atoms, capped_coords = add_hydrogens_to_edge_carbons(
        trimmed_atoms,
        trimmed_coords,
        bond_cutoff=bond_cutoff,
        bond_length=edge_ch_bond_length,
    )

    # Re-check carbon network after passivation
    carbon_coords_final = np.array([c for a, c in zip(capped_atoms, capped_coords) if a == "C"])
    verify_no_degree1_carbons(carbon_coords_final, cutoff=bond_cutoff)

    return capped_atoms, capped_coords


# ---------------------------
# xtb workflow
# ---------------------------

def write_xtb_ready_system(
    filename,
    molecule_atoms,
    molecule_coords,
    graphene_atoms,
    graphene_coords,
):

    atoms = list(molecule_atoms) + list(graphene_atoms)
    coords = np.vstack([molecule_coords, graphene_coords])
    write_xyz(filename, atoms, coords, comment="Aligned molecule on trimmed, H-passivated graphene; graphene fixed for xTB")

    first_fixed = len(molecule_atoms) + 1
    last_fixed = len(atoms)
    append_xtb_fix_block(filename, first_fixed, last_fixed)
    return first_fixed, last_fixed


def run_xtb_optimization(xyz_path, workdir, xtb_executable="xtb", method="gfnff", solvent=None):
    os.makedirs(workdir, exist_ok=True)
    xyz_name = os.path.basename(xyz_path)
    local_xyz = os.path.join(workdir, xyz_name)
    shutil.copyfile(xyz_path, local_xyz)

    cmd = [xtb_executable, f"--{method.lower()}", xyz_name, "--opt"]
    if solvent:
        cmd += ["--alpb", solvent]

    print("...running", " ".join(cmd), f"in {workdir}")
    result = subprocess.run(
        cmd,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=False,
        check=False,
    )

    stdout_text = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""

    log_path = os.path.join(workdir, "xtb_optimization.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(stdout_text)

    if result.returncode != 0:
        raise RuntimeError(
            f"xTB optimization failed with return code {result.returncode}. See {log_path}"
        )

    optimized_xyz = os.path.join(workdir, "xtbopt.xyz")
    if not os.path.exists(optimized_xyz):
        raise RuntimeError(f"xTB completed but did not produce {optimized_xyz}. See {log_path}")

    print(f"...saved xTB log: {log_path}")
    print(f"...saved optimized geometry: {optimized_xyz}")
    return optimized_xyz, log_path


def place_optimized_molecule_on_full_graphene(
    optimized_xyz,
    original_graphene_atoms,
    original_graphene_coords,
    molecule_atom_count,
    output_xyz="optimized_on_full_graphene.xyz",
):

    opt_atoms, opt_coords = read_xyz_xtb(optimized_xyz)

    if len(opt_atoms) < molecule_atom_count:
        raise ValueError(
            f"Optimized file {optimized_xyz!r} has only {len(opt_atoms)} atoms, "
            f"but {molecule_atom_count} molecule atoms were expected."
        )

    optimized_molecule_atoms = opt_atoms[:molecule_atom_count]
    optimized_molecule_coords = opt_coords[:molecule_atom_count]

    full_atoms = list(original_graphene_atoms) + optimized_molecule_atoms
    full_coords = np.vstack([
        np.array(original_graphene_coords, dtype=float),
        optimized_molecule_coords,
    ])

    write_xyz(
        output_xyz,
        full_atoms,
        full_coords,
        comment="xTB-optimized molecule placed back onto original full graphene",
    )
    print(f"...saved optimized molecule on full graphene: {output_xyz}")
    return output_xyz

# ---------------------------
# Viewer
# ---------------------------

def open_in_ovito(xyz_file):
    ovito = r"C:\Program Files\OVITO_Basic\ovito.exe"   # adjust if necessary

    if not os.path.isfile(ovito):
        raise FileNotFoundError(f"OVITO not found: {ovito}")

    subprocess.Popen([ovito, os.path.abspath(xyz_file)])

# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Place a molecule on graphene, trim/passivate graphene, and run constrained xTB optimization."
    )
    parser.add_argument("graphene_xyz")
    parser.add_argument("molecule_xyz")
    parser.add_argument("-o", "--output", default="graphene_with_molecule.xyz")
    parser.add_argument("-v", "--view", action='store_true')
    parser.add_argument("-d", "--distance", type=float, default=3.6,
                        help="Closest molecule-graphene distance in Angstrom")
    parser.add_argument("--buffer", type=float, default=4.0,
                        help="Graphene trimming buffer around the projected molecule in Angstrom")
    parser.add_argument("--bond-cutoff", type=float, default=1.6,
                        help="C-C cutoff used for graphene coordination analysis")
    parser.add_argument("--edge-ch", type=float, default=1.09,
                        help="C-H bond length used for edge passivation")
    parser.add_argument("--xtb-input", default="xtb_ready.xyz",
                        help="xTB-ready XYZ file with fixed graphene block")
    parser.add_argument("--xtb-workdir", default="xtb_optimization",
                        help="Working directory for xTB optimization")
    parser.add_argument("--xtb", default="xtb",
                        help="xTB executable name or path")
    parser.add_argument("--method", default="gfnff",
                        help="xTB method, e.g. gfnff, gfn2, gfn1")
    parser.add_argument("--solvent", default=None,
                        help="Optional ALPB solvent name, e.g. toluene")
    parser.add_argument("--skip-xtb", action="store_true",
                        help="Only prepare the xTB-ready geometry; do not run xTB")
    parser.add_argument("--full-output", default="optimized_on_full_graphene.xyz",
                        help="Output XYZ for the optimized molecule placed back on the original full graphene")
    parser.add_argument("--optimized-xyz", default=None,
                        help="Existing xtbopt.xyz to place back on full graphene, useful with --skip-xtb")

    args = parser.parse_args()

    info()

    time.sleep(3.14)

    graphene_mol, graphene_atoms, graphene_coords = read_xyz(args.graphene_xyz)
    molecule_mol, molecule_atoms, molecule_coords = read_xyz(args.molecule_xyz)

    aligned_molecule_coords = align_molecule_to_graphene(
        graphene_coords,
        molecule_coords,
        distance=args.distance,
    )

    # Write the untrimmed aligned structure
    aligned_atoms = list(graphene_atoms) + list(molecule_atoms)
    aligned_coords = np.vstack([graphene_coords, aligned_molecule_coords])
    write_xyz(args.output, aligned_atoms, aligned_coords, comment="Aligned molecule on full graphene")

    if args.view:
        open_in_ovito(args.output)

    # Prepare the trimmed, passivated graphene for xtb.
    trimmed_graphene_atoms, trimmed_graphene_coords = prepare_trimmed_graphene_for_xtb(
        graphene_atoms=graphene_atoms,
        graphene_coords=graphene_coords,
        molecule_coords=aligned_molecule_coords,
        buffer=args.buffer,
        bond_cutoff=args.bond_cutoff,
        edge_ch_bond_length=args.edge_ch,
    )

    first_fixed, last_fixed = write_xtb_ready_system(
        filename=args.xtb_input,
        molecule_atoms=molecule_atoms,
        molecule_coords=aligned_molecule_coords,
        graphene_atoms=trimmed_graphene_atoms,
        graphene_coords=trimmed_graphene_coords,
    )

    print(f"...saved aligned full system: {args.output}")
    print(f"...saved xTB-ready system: {args.xtb_input}")
    print(f"...fixed graphene atom range for xTB: {first_fixed}-{last_fixed}")
    print(f"...trimmed graphene atoms including edge H caps: {len(trimmed_graphene_atoms)}")

    optimized_xyz = args.optimized_xyz

    if not args.skip_xtb:
        optimized_xyz, _ = run_xtb_optimization(
            xyz_path=args.xtb_input,
            workdir=args.xtb_workdir,
            xtb_executable=args.xtb,
            method=args.method,
            solvent=args.solvent,
        )
    else:
        print("...skipped xTB optimization.")

    if optimized_xyz is not None:
        place_optimized_molecule_on_full_graphene(
            optimized_xyz=optimized_xyz,
            original_graphene_atoms=graphene_atoms,
            original_graphene_coords=graphene_coords,
            molecule_atom_count=len(molecule_atoms),
            output_xyz=args.full_output,
        )
    else:
        print("...no optimized XYZ was provided or produced; full-graphene reconstruction was skipped.")

    if args.view:
        open_in_ovito(args.full_output)


if __name__ == "__main__":
    main()
