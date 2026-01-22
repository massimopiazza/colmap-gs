import bpy
import bmesh
from mathutils import Vector, Matrix
from mathutils.bvhtree import BVHTree
import math

# -----------------------------
# CONFIG (tune these)
# -----------------------------
CFG = {
    "new_object_name": "BarrierUmbrella",

    # "Umbrella" margin (meters): expands boundary outward so it surely covers selection
    "margin_m": 0.5,

    # Sheet resolution: higher => more vertices to shrinkwrap (but more overfit risk)
    "subdivide_cuts": 30,

    # How much to smooth / low-pass after shrinkwrap
    "smooth_iterations": 40,
    "smooth_factor": 0.45,

    # Shrinkwrap: makes the sheet wrap onto your terrain
    "use_shrinkwrap": True,
    "shrinkwrap_offset_m": 2.0,   # stand-off distance

    # Shrinkwrap method:
    # - "NEAREST_SURFACEPOINT" (your original)
    # - "PROJECT" (often better to avoid snapping to wrong side on complex terrain)
    "shrinkwrap_method": "NEAREST_SURFACEPOINT",

    # If using PROJECT, project along clearance axis (AUTO) or force "X"/"Y"/"Z"
    "shrinkwrap_project_axis": "AUTO",  # "AUTO" | "X" | "Y" | "Z"
    "shrinkwrap_project_positive": True,
    "shrinkwrap_project_negative": True,

    # Apply modifiers to get a final clean mesh for area measurement/export
    "apply_modifiers": True,

    # If True, compute hull in WORLD_XY (good if your terrain is "Z-up").
    # If False, hull plane is built from average selected surface normal.
    "force_world_xy": False,

    # Axis that defines "above". Default Z+ (Blender world up).
    "clearance_axis": "Z+",

    # Minimum positive delta along clearance_axis between umbrella and terrain (meters)
    "axis_clearance_m": 0.05,

    # Raycast max distance along the axis (meters)
    "axis_raycast_max_m": 2000.0,

    # Small start offset so the ray doesn't immediately hit the umbrella itself
    "axis_raycast_start_eps_m": 0.01,

    # If a ray doesn't hit (edge cases), use nearest-point fallback
    "axis_use_nearest_fallback": True,

    # NEW: place the initial umbrella polygon above the selection along clearance_axis before shrinkwrap
    # (helps avoid shrinkwrap snapping to an unintended "backside")
    "initial_lift_along_axis_m": 1.0,

    # Apply scale to terrain to make units consistent
    "apply_scale_to_terrain": True,
}

# -----------------------------
# Utilities
# -----------------------------
def ensure_object_mode():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')

def ensure_edit_mode():
    # Works for mesh objects; Blender mode string is 'EDIT', context becomes 'EDIT_MESH'
    if bpy.context.mode != 'EDIT_MESH':
        bpy.ops.object.mode_set(mode='EDIT')

def set_active(obj, deselect_others=False):
    if deselect_others:
        bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

def apply_scale(obj):
    ensure_object_mode()
    set_active(obj, deselect_others=True)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

def MatrixIdentity():
    return Matrix.Identity(4)

def _axis_tag_to_vec(tag: str) -> Vector:
    tag = tag.strip().upper()
    mapping = {
        "X+": Vector(( 1, 0, 0)),
        "X-": Vector((-1, 0, 0)),
        "Y+": Vector(( 0, 1, 0)),
        "Y-": Vector(( 0,-1, 0)),
        "Z+": Vector(( 0, 0, 1)),
        "Z-": Vector(( 0, 0,-1)),
    }
    if tag not in mapping:
        raise ValueError(f"Invalid clearance_axis='{tag}'. Use one of {list(mapping.keys())}.")
    return mapping[tag].normalized()

# -----------------------------
# Selection / Terrain detection (KEY FIX)
# -----------------------------
def _mesh_has_any_selection_in_edit(obj) -> bool:
    """obj must be a mesh in Edit Mode."""
    bm = bmesh.from_edit_mesh(obj.data)
    # Fast checks
    for v in bm.verts:
        if v.select:
            return True
    for f in bm.faces:
        if f.select:
            return True
    return False

def find_terrain_object_from_selection():
    """
    Robustly find the mesh object that actually has a selection.
    Works even if the active object is not the terrain (e.g., umbrella from last run),
    and also in multi-object edit mode.
    """
    # 1) Prefer objects currently in edit mode (multi-object edit)
    objs_in_edit = getattr(bpy.context, "objects_in_mode_unique_data", None)
    if objs_in_edit:
        for obj in objs_in_edit:
            if obj and obj.type == 'MESH' and obj.mode == 'EDIT':
                if _mesh_has_any_selection_in_edit(obj):
                    return obj

    # 2) Otherwise try active object: temporarily enter edit mode and check selection
    obj = bpy.context.active_object
    if not obj or obj.type != 'MESH':
        raise RuntimeError("No active mesh object. Select your terrain mesh and try again.")

    # Ensure it's the active object and in edit mode for selection read
    set_active(obj, deselect_others=False)
    ensure_edit_mode()
    if _mesh_has_any_selection_in_edit(obj):
        return obj

    # 3) No selection found anywhere
    raise RuntimeError(
        "No selected vertices/faces found on any mesh in Edit Mode. "
        "Make sure you are selecting AOI vertices/faces on the terrain mesh (Edit Mode)."
    )

# -----------------------------
# Geometry helpers
# -----------------------------
def get_selected_world_points_and_mean_normal(obj):
    """
    Collect selected verts (and verts of selected faces if needed) as world-space points.
    Also estimate a mean normal from linked faces for a stable projection plane.
    """
    set_active(obj, deselect_others=False)
    ensure_edit_mode()

    me = obj.data
    bm = bmesh.from_edit_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    sel_verts = [v for v in bm.verts if v.select]
    if not sel_verts:
        sel_faces = [f for f in bm.faces if f.select]
        if sel_faces:
            sel_verts = list({v for f in sel_faces for v in f.verts})
        else:
            raise RuntimeError("Selection is empty on the detected terrain object.")

    M = obj.matrix_world
    pts = [M @ v.co for v in sel_verts]

    n_sum = Vector((0,0,0))
    face_count = 0
    for v in sel_verts:
        for f in v.link_faces:
            n_sum += (obj.matrix_world.to_3x3() @ f.normal)
            face_count += 1

    if face_count == 0 or n_sum.length < 1e-9:
        mean_n = Vector((0,0,1))
    else:
        mean_n = n_sum.normalized()

    return pts, mean_n

def make_basis_from_normal(n: Vector):
    n = n.normalized()
    up = Vector((0,0,1))
    u = n.cross(up)
    if u.length < 1e-6:
        u = n.cross(Vector((1,0,0)))
    u.normalize()
    v = n.cross(u)
    v.normalize()
    return u, v, n

def project_to_2d(pts3d, origin, u, v):
    return [((p-origin).dot(u), (p-origin).dot(v)) for p in pts3d]

def cross(o, a, b):
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def convex_hull_2d(points):
    pts = sorted(set(points))
    if len(pts) < 3:
        raise RuntimeError("Need at least 3 distinct points to build a hull.")

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

def polygon_area_signed(poly):
    a = 0.0
    for i in range(len(poly)):
        x1,y1 = poly[i]
        x2,y2 = poly[(i+1)%len(poly)]
        a += x1*y2 - x2*y1
    return 0.5*a

def line_intersection(p1, d1, p2, d2):
    det = d1[0]*(-d2[1]) - d1[1]*(-d2[0])
    if abs(det) < 1e-12:
        return ((p1[0]+p2[0])*0.5, (p1[1]+p2[1])*0.5)
    rhs = (p2[0]-p1[0], p2[1]-p1[1])
    t = (rhs[0]*(-d2[1]) - rhs[1]*(-d2[0])) / det
    return (p1[0] + t*d1[0], p1[1] + t*d1[1])

def offset_convex_polygon_ccw(poly, offset):
    if offset <= 0:
        return poly

    if polygon_area_signed(poly) < 0:
        poly = list(reversed(poly))

    def norm2(d):
        l = math.hypot(d[0], d[1])
        if l < 1e-12:
            return (1.0, 0.0)
        return (d[0]/l, d[1]/l)

    out = []
    n = len(poly)

    for i in range(n):
        p_prev = poly[(i-1) % n]
        p_curr = poly[i]
        p_next = poly[(i+1) % n]

        e1 = (p_curr[0]-p_prev[0], p_curr[1]-p_prev[1])
        e2 = (p_next[0]-p_curr[0], p_next[1]-p_curr[1])

        d1 = norm2(e1)
        d2 = norm2(e2)

        n1 = (d1[1], -d1[0])
        n2 = (d2[1], -d2[0])

        p1 = (p_curr[0] + n1[0]*offset, p_curr[1] + n1[1]*offset)
        p2 = (p_curr[0] + n2[0]*offset, p_curr[1] + n2[1]*offset)

        ip = line_intersection(p1, d1, p2, d2)
        out.append(ip)

    return out

def build_mesh_ngon(obj, verts_world):
    mesh = obj.data
    mesh.clear_geometry()

    inv = obj.matrix_world.inverted()
    verts_local = [inv @ p for p in verts_world]

    mesh.from_pydata(verts_local, [], [list(range(len(verts_local)))])
    mesh.update(calc_edges=True)

    # Triangulate for robustness downstream
    ensure_object_mode()
    set_active(obj, deselect_others=True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
    bpy.ops.object.mode_set(mode='OBJECT')

def add_modifiers(obj, target_terrain):
    ensure_object_mode()
    set_active(obj, deselect_others=True)

    # --- Optional shading (not required for area) ---
    try:
        bpy.ops.object.shade_smooth()
        # Blender 4.1+ may have this operator; safe to try
        try:
            bpy.ops.object.shade_smooth_by_angle(angle=math.radians(30.0), keep_sharp_edges=True)
        except Exception:
            pass
    except Exception:
        pass

    # --- Shrinkwrap ---
    if CFG.get("use_shrinkwrap", True):
        sw = obj.modifiers.new("Umbrella_Shrinkwrap", type='SHRINKWRAP')
        sw.target = target_terrain
        sw.offset = CFG.get("shrinkwrap_offset_m", 0.05)

        method = CFG.get("shrinkwrap_method", "NEAREST_SURFACEPOINT").upper()
        if method == "PROJECT":
            sw.wrap_method = 'PROJECT'

            axis = CFG.get("shrinkwrap_project_axis", "AUTO").upper()
            if axis == "AUTO":
                axis_tag = CFG["clearance_axis"].upper()
                axis = axis_tag[0]  # X/Y/Z from "Z+"

            # reset and enable desired project axis flags if present
            for a in ("use_project_x","use_project_y","use_project_z"):
                if hasattr(sw, a):
                    setattr(sw, a, False)

            if axis == "X" and hasattr(sw, "use_project_x"): sw.use_project_x = True
            if axis == "Y" and hasattr(sw, "use_project_y"): sw.use_project_y = True
            if axis == "Z" and hasattr(sw, "use_project_z"): sw.use_project_z = True

            if hasattr(sw, "use_positive_direction"):
                sw.use_positive_direction = bool(CFG.get("shrinkwrap_project_positive", True))
            if hasattr(sw, "use_negative_direction"):
                sw.use_negative_direction = bool(CFG.get("shrinkwrap_project_negative", True))
        else:
            sw.wrap_method = 'NEAREST_SURFACEPOINT'

    # --- Smooth ---
    sm = obj.modifiers.new("Umbrella_Smooth", type='SMOOTH')
    sm.iterations = CFG.get("smooth_iterations", 40)
    sm.factor = CFG.get("smooth_factor", 0.45)

def apply_all_modifiers(obj):
    ensure_object_mode()
    set_active(obj, deselect_others=True)
    for m in list(obj.modifiers):
        try:
            bpy.ops.object.modifier_apply(modifier=m.name)
        except Exception as e:
            print(f"Warning: couldn't apply modifier {m.name}: {e}")

def edit_subdivide(obj, cuts):
    ensure_object_mode()
    set_active(obj, deselect_others=True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.subdivide(number_cuts=cuts)
    bpy.ops.object.mode_set(mode='OBJECT')

def compute_area_world(obj):
    ensure_object_mode()
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    M = obj.matrix_world
    area = 0.0
    for f in bm.faces:
        verts = [M @ v.co for v in f.verts]
        o = verts[0]
        for i in range(1, len(verts)-1):
            area += ((verts[i] - o).cross(verts[i+1] - o)).length * 0.5
    bm.free()
    return area

# -----------------------------
# Axis clearance (your version, cleaned + still version-robust)
# -----------------------------
def _build_bvh_for_object(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()

    if hasattr(BVHTree, "FromObject"):
        try:
            bvh = BVHTree.FromObject(obj, depsgraph)  # newer signature
            return bvh, (lambda: None)
        except TypeError:
            bvh = BVHTree.FromObject(
                obj, bpy.context.scene,
                deform=True, render=False, cage=False, epsilon=0.0
            )
            return bvh, (lambda: None)

    obj_eval = obj.evaluated_get(depsgraph)
    me = obj_eval.to_mesh()
    bm = bmesh.new()
    bm.from_mesh(me)

    bvh = BVHTree.FromBMesh(bm, epsilon=0.0)

    def cleanup():
        bm.free()
        obj_eval.to_mesh_clear()

    return bvh, cleanup

def enforce_axis_clearance(
    umbrella_obj,
    terrain_obj,
    axis_tag="Z+",
    clearance_m=0.05,
    ray_max_m=2000.0,
    ray_start_eps_m=0.01,
    use_nearest_fallback=True,
):
    bvh, cleanup_bvh = _build_bvh_for_object(terrain_obj)
    try:
        a_world = _axis_tag_to_vec(axis_tag)
        ray_dir_world = (-a_world).normalized()

        terr_M = terrain_obj.matrix_world
        terr_M_inv = terr_M.inverted()
        terr_R_inv = terr_M_inv.to_3x3()

        umb_M = umbrella_obj.matrix_world
        umb_M_inv = umb_M.inverted()

        me = umbrella_obj.data
        moved = 0
        min_delta_before = None

        for v in me.vertices:
            p_world = umb_M @ v.co
            ray_origin_world = p_world + a_world * ray_start_eps_m

            ray_origin_local = terr_M_inv @ ray_origin_world
            ray_dir_local = (terr_R_inv @ ray_dir_world).normalized()

            hit = bvh.ray_cast(ray_origin_local, ray_dir_local, ray_max_m)
            hit_loc_local = hit[0] if hit else None

            if hit_loc_local is None and use_nearest_fallback:
                nearest = bvh.find_nearest(ray_origin_local)
                if nearest:
                    hit_loc_local = nearest[0]

            if hit_loc_local is None:
                continue

            hit_world = terr_M @ hit_loc_local
            delta = (p_world - hit_world).dot(a_world)

            if (min_delta_before is None) or (delta < min_delta_before):
                min_delta_before = delta

            if delta < clearance_m:
                p_world = p_world + (clearance_m - delta) * a_world
                v.co = umb_M_inv @ p_world
                moved += 1

        me.update()
        print(f"[AxisClearance] axis={axis_tag} clearance={clearance_m}m | adjusted_verts={moved} | min_delta_before={min_delta_before}")

    finally:
        cleanup_bvh()

# -----------------------------
# MAIN
# -----------------------------
def run():
    # KEY FIX: determine terrain from where the selection actually is
    terrain = find_terrain_object_from_selection()
    if not terrain or terrain.type != 'MESH':
        raise RuntimeError("Could not detect a terrain mesh from selection.")

    # Apply scale for consistent units if requested
    if CFG.get("apply_scale_to_terrain", True):
        apply_scale(terrain)

    # Collect selected points + mean normal
    pts3d, mean_n = get_selected_world_points_and_mean_normal(terrain)

    # Basis for hull plane
    if CFG["force_world_xy"]:
        u = Vector((1,0,0))
        v = Vector((0,1,0))
        n = Vector((0,0,1))
    else:
        u, v, n = make_basis_from_normal(mean_n)

    # Centroid as origin
    origin = Vector((0,0,0))
    for p in pts3d:
        origin += p
    origin /= len(pts3d)

    # NEW: lift the construction plane along clearance axis before shrinkwrap
    a = _axis_tag_to_vec(CFG["clearance_axis"])
    t_max = max(p.dot(a) for p in pts3d)
    t_origin = origin.dot(a)
    lift = (t_max - t_origin) + max(0.0, CFG.get("initial_lift_along_axis_m", 0.0))
    origin = origin + a * lift

    pts2d = project_to_2d(pts3d, origin, u, v)

    hull = convex_hull_2d(pts2d)
    hull_off = offset_convex_polygon_ccw(hull, CFG["margin_m"])

    verts_world = [origin + u*p[0] + v*p[1] for p in hull_off]

    # Create umbrella object
    obj_name = CFG["new_object_name"]
    mesh = bpy.data.meshes.new(obj_name + "_Mesh")
    umbrella = bpy.data.objects.new(obj_name, mesh)
    bpy.context.collection.objects.link(umbrella)
    umbrella.matrix_world = MatrixIdentity()

    build_mesh_ngon(umbrella, verts_world)

    # Subdivide to provide drape resolution
    edit_subdivide(umbrella, CFG["subdivide_cuts"])

    # Modifiers: shrinkwrap + smooth
    add_modifiers(umbrella, terrain)

    if CFG["apply_modifiers"]:
        apply_all_modifiers(umbrella)

    # Axis clearance clamp
    enforce_axis_clearance(
        umbrella_obj=umbrella,
        terrain_obj=terrain,
        axis_tag=CFG["clearance_axis"],
        clearance_m=CFG["axis_clearance_m"],
        ray_max_m=CFG["axis_raycast_max_m"],
        ray_start_eps_m=CFG["axis_raycast_start_eps_m"],
        use_nearest_fallback=CFG["axis_use_nearest_fallback"],
    )

    area_m2 = compute_area_world(umbrella)
    print(f"[{obj_name}] One-sided area estimate: {area_m2:.3f} m^2")

    # Make umbrella active for convenience (optional)
    set_active(umbrella, deselect_others=False)

    return umbrella, area_m2

umbrella_obj, area = run()
