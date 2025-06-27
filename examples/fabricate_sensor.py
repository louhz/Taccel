import json
import os
import os.path as osp
import pickle as pkl

import numpy as np
import pyvista as pv
import trimesh as tm
from meshpy.tet import MeshInfo, build
from scipy.spatial import Delaunay

markers = True


MAX_V = 1e-05
mesh_file = "assets/robots/franka_description/meshes/visual/pad.stl"


def get_stick_mask_fn(verts):
    return verts[:, 2] > 0.003 - 1e-4


def get_coat_mask_fn(verts):
    return verts[:, 2] < 1e-4


marker_ls_x = np.linspace(-0.0133, 0.0133, 8)
marker_ls_y = np.linspace(-0.0133, 0.0133, 8)
marker_ls_z = 0.0 * np.ones(1)

dump_dir = osp.dirname(mesh_file)
os.makedirs(dump_dir, exist_ok=True)
mesh_vtk_file = dump_dir + f"/pad_maxv={MAX_V}.vtk"
mesh_metadata_file = dump_dir + f"/pad_maxv={MAX_V}.pkl"
if markers:
    marker_grid = np.array(np.meshgrid(marker_ls_x, marker_ls_y, marker_ls_z)).T.reshape(-1, 3)


def find_bc_coords(mesh, points):
    cells = mesh.faces.reshape(-1, 4)[:, 1:]
    closest_point, distance, face_index = tm.Trimesh(mesh.points, cells).nearest.on_surface(points)
    bounding_faces = cells[face_index]
    print(bounding_faces.shape)

    # To store barycentric coordinates
    simplex_indices = []
    barycentric_coords = []

    # Iterate over each point and find the triangle that contains the closest point
    for point_idx, point in enumerate(points):
        # Find which triangle includes this closest point
        for cell in cells:
            if sorted(bounding_faces[point_idx]) == sorted(cell):
                # Get the vertices of the triangle
                v0, v1, v2 = mesh.points[cell]

                # Compute vectors
                v0v1 = v1 - v0
                v0v2 = v2 - v0
                v0p = point - v0

                # Compute dot products
                dot00 = np.dot(v0v2, v0v2)
                dot01 = np.dot(v0v2, v0v1)
                dot02 = np.dot(v0v2, v0p)
                dot11 = np.dot(v0v1, v0v1)
                dot12 = np.dot(v0v1, v0p)

                # Compute barycentric coordinates
                inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
                u = (dot11 * dot02 - dot01 * dot12) * inv_denom
                v = (dot00 * dot12 - dot01 * dot02) * inv_denom
                w = 1 - u - v

                simplex_indices.append(cell)
                barycentric_coords.append((w, v, u))
                break

    return np.array(simplex_indices), np.array(barycentric_coords)


def proceed():
    # Load triangular meh file
    mesh = tm.load(mesh_file)

    # Subdivide into tetrahedral mesh
    mesh_info = MeshInfo()
    mesh_info.set_points(mesh.vertices)
    mesh_info.set_facets(mesh.faces)
    tet_mesh = build(mesh_info, max_volume=MAX_V)

    # Export tet mesh
    points = np.array(tet_mesh.points)
    cells = np.array([[4] + list(cell) for cell in tet_mesh.elements])
    cell_types = np.array([pv.CellType.TETRA] * len(cells))
    tetmesh_usgrid = pv.UnstructuredGrid(cells, cell_types, points)
    surf_mesh = tetmesh_usgrid.extract_surface()
    actual_max_v = tetmesh_usgrid.compute_cell_sizes()["Volume"].max().item()
    print(f"Got {len(cells)} cells with {len(points)} points, {len(surf_mesh.points)} on the surface, and max volume {actual_max_v} (target {MAX_V})")

    tetmesh_usgrid.save(mesh_vtk_file)

    surf_pv_mesh = tetmesh_usgrid.extract_surface()
    surf_verts = surf_pv_mesh.points
    surf_tris = surf_pv_mesh.faces.reshape(-1, 4)[:, 1:]
    surf_tris = np.concatenate([surf_tris, surf_tris[:, ::-1]], axis=0)
    surf_trimesh = tm.Trimesh(vertices=surf_verts, faces=surf_tris)
    surf_trimesh.export(mesh_vtk_file.replace(".vtk", ".stl"))

    # Compute marker positions as baryentrc cordites
    if markers:
        print(f"Computing bc coordinates for {len(marker_grid)} markers")
        simplex_indices_surf, barycentric_coords = find_bc_coords(surf_mesh, marker_grid)
        simplex_indices_body = surf_mesh["vtkOriginalPointIds"][simplex_indices_surf]

    # Compute sticking masks
    stick_mask = get_stick_mask_fn(points)
    coat_mask = get_coat_mask_fn(points)
    surface_coat_mask = get_coat_mask_fn(surf_mesh.points)
    print(f"Found {stick_mask.sum()} sticking vertices and {coat_mask.sum()} coated vertices .")

    pkl.dump(
        {
            "max_v": actual_max_v,
            "target_v": MAX_V,
            "stick_mask": stick_mask,
            "coat_mask": coat_mask,
            "coat_mask_surf": surface_coat_mask,
            "surface_index": simplex_indices_surf if markers else [],
            "marker_vert_idx": simplex_indices_body.tolist() if markers else [],
            "marker_vert_idx_surf": simplex_indices_surf.tolist() if markers else [],
            "marker_bc_coords": barycentric_coords.tolist() if markers else [],
        },
        open(mesh_metadata_file, "wb"),
    )

    print(f"num of points: {len(points)}")
    print(f"num of coat mask: {len(coat_mask)}")

    # Visualization
    stick_positions = points[np.where(stick_mask)[0]]
    coat_positions = points[np.where(coat_mask)[0]]
    stick_pv = pv.PolyData(stick_positions)
    coat_pv = pv.PolyData(coat_positions)
    plotter = pv.Plotter()
    if markers:
        simplex_positions = surf_mesh.points[simplex_indices_surf]
        marker_positions = (simplex_positions * barycentric_coords[..., None]).sum(axis=-2)
        point_cloud_pv = pv.PolyData(marker_grid)
        marker_recover_pv = pv.PolyData(marker_positions)
        plotter.add_points(point_cloud_pv, color="green", point_size=15, render_points_as_spheres=True, opacity=0.5)
        plotter.add_points(marker_recover_pv, color="red", point_size=10, render_points_as_spheres=True, opacity=1.0)
    plotter.add_mesh(tetmesh_usgrid, color="white", show_edges=True, opacity=0.5)
    plotter.add_points(coat_pv, color="white", point_size=10, render_points_as_spheres=True, opacity=1.0)
    plotter.add_points(stick_pv, color="purple", point_size=10, render_points_as_spheres=True, opacity=1.0)
    plotter.show()


if __name__ == "__main__":
    proceed()
