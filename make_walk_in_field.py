r"""Build a 15-second walk through a field, with a bird flyby and coffee sip.

The walker wears a bucket hat, red-and-white flannel, cargo pants and hiking boots.

Usage (from PowerShell):
  & "C:\Program Files\Blender Foundation\Blender 5.3\blender.exe" -b -P make_walk_in_field.py -- [--still F[,F..]] [--render] [--preview]

  --still F[,F..] render PNG frame(s) to render/still_FFF.png
  --render       render the full animation to render/walk_in_field.mp4
  --preview      lower resolution / samples for quick tests
Without --still/--render the scene is only built and saved to walk_in_field.blend.
"""

import math
import os
import random
import sys
import warnings

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

warnings.filterwarnings("ignore", category=DeprecationWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "render")

FPS = 30
FRAMES = 15 * FPS
CYCLE = 45                   # keep the original step length at the slower pace
SPEED = 0.8 / FPS            # metres per frame (two thirds of the original)
STRIDE = SPEED * CYCLE       # distance a foot travels per cycle
STANCE = 0.6                 # fraction of the cycle a foot is planted
START_X = 0.0
PATH_Y = 0.6                 # lateral position of the walker on the road
LOOK_HEIGHT = 0.92
LENS = 35
BIRD_ENTER, BIRD_EXIT = 5, 9

THIGH = 0.46
SHIN = 0.46
ANKLE_H = 0.08               # ankle height above ground when the foot is flat
HIP_WALK = 0.972             # mean pelvis height while walking (legs almost straight)
BOB = 0.021                  # pelvis rise/fall: high at mid-stance, low at double support
FOOT_Y = 0.078               # lateral foot placement (half the step width)
HEEL_X, TOE_X = -0.070, 0.165  # ground pivots under the boot, relative to the ankle
HEEL_STRIKE = -0.24          # foot pitch at heel contact (toe up)
TOE_OFF = 0.42               # foot pitch when the toe leaves the ground
LOAD, HEEL_OFF = 0.10, 0.34  # stance phases: heel rocker ends / heel starts to rise
# swing keys: (swing phase, fraction of the way forward, ankle height, foot pitch)
SWING_KEYS = ((0.22, 0.12, 0.168, 0.26), (0.50, 0.50, 0.128, 0.02), (0.78, 0.88, 0.110, -0.15))

rng = random.Random(7)


# ---------------------------------------------------------------- utilities
def args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opts = {"still": [], "render": False, "preview": False}
    i = 0
    while i < len(argv):
        if argv[i] == "--still":
            opts["still"] = [int(v) for v in argv[i + 1].split(",")]
            i += 1
        elif argv[i] == "--render":
            opts["render"] = True
        elif argv[i] == "--preview":
            opts["preview"] = True
        i += 1
    return opts


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def material(name, color, roughness=0.6):
    mat = bpy.data.materials.new(name)
    try:
        mat.use_nodes = True
    except Exception:
        pass
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def gradient_material(name, stops, roughness=0.55):
    """Colour ramp driven by the object's generated Z (bottom -> top)."""
    mat = material(name, (1, 1, 1), roughness)
    nt = mat.node_tree
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = stops[0][0], (*stops[0][1], 1)
    els[1].position, els[1].color = stops[-1][0], (*stops[-1][1], 1)
    for pos, col in stops[1:-1]:
        e = els.new(pos)
        e.color = (*col, 1)
    nt.links.new(coord.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return mat


def noise_material(name, col_a, col_b, scale=2.0, roughness=0.9):
    mat = material(name, col_a, roughness)
    nt = mat.node_tree
    coord = nt.nodes.new("ShaderNodeTexCoord")
    noise = nt.nodes.new("ShaderNodeTexNoise")
    nt.links.new(coord.outputs["Object"], noise.inputs["Vector"])
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = 8.0
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[0].color = (*col_a, 1)
    ramp.color_ramp.elements[1].position = 0.65
    ramp.color_ramp.elements[1].color = (*col_b, 1)
    nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return mat


def smooth(obj):
    for p in obj.data.polygons:
        p.use_smooth = True


def empty(name, parent=None, loc=(0, 0, 0), size=0.1):
    obj = link(bpy.data.objects.new(name, None))
    obj.empty_display_size = size
    obj.parent = parent
    obj.location = loc
    return obj


# ---------------------------------------------------------------- environment
def build_world(scene):
    world = bpy.data.worlds.new("Sky")
    scene.world = world
    try:
        world.use_nodes = True
    except Exception:
        pass
    nt = world.node_tree
    bg = nt.nodes.get("Background") or nt.nodes.new("ShaderNodeBackground")
    out = nt.nodes.get("World Output") or nt.nodes.new("ShaderNodeOutputWorld")
    coord = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (1.0, 0.55, 0.28, 1)      # warm horizon
    els[1].position, els[1].color = 0.6, (0.12, 0.28, 0.70, 1)     # deep blue zenith
    for pos, col in ((0.05, (1.0, 0.74, 0.48)), (0.22, (0.36, 0.56, 0.88))):
        e = els.new(pos)
        e.color = (*col, 1)
    nt.links.new(coord.outputs["Generated"], sep.inputs[0])
    nt.links.new(sep.outputs["Z"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = 1.0
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])


def build_sun():
    light = bpy.data.lights.new("Sun", "SUN")
    light.energy = 4.0
    light.color = (1.0, 0.80, 0.58)
    light.angle = math.radians(1.5)
    sun = link(bpy.data.objects.new("Sun", light))
    travel = Vector((0.25, 1.0, -0.5)).normalized()   # warm side light, long soft shadows
    sun.rotation_euler = travel.to_track_quat("-Z", "Y").to_euler()


def box(name, loc, dims, mat):
    """Cube with real (applied) dimensions so procedural textures are not stretched."""
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc)
    obj = bpy.context.active_object
    obj.name = name
    for v in obj.data.vertices:
        v.co = Vector((v.co.x * dims[0], v.co.y * dims[1], v.co.z * dims[2]))
    obj.data.materials.append(mat)
    return obj


def build_ground():
    field_mat = noise_material("FieldMud", (0.10, 0.08, 0.04), (0.28, 0.25, 0.09), scale=0.05)
    bpy.ops.mesh.primitive_plane_add(size=600, location=(0, 0, -0.12))
    ground = bpy.context.active_object
    ground.name = "Ground"
    ground.data.materials.append(field_mat)

    road_mat = noise_material("Dirt", (0.30, 0.19, 0.10), (0.50, 0.36, 0.22), scale=1.5)
    box("DirtRoad", (-90, 0, -0.1), (240, 2.8, 0.2), road_mat)      # top surface at z = 0

    # grassy edges along the road
    grass_mat = noise_material("RoadGrass", (0.10, 0.16, 0.04), (0.22, 0.28, 0.07), scale=2.0)
    for y in (-1.45, 1.45):
        box("RoadEdge", (-90, y, -0.07), (240, 0.25, 0.16), grass_mat)


def make_plant_mesh(name, blades, height):
    """A clump of curved, tapering blades (rice / wheat style)."""
    bm = bmesh.new()
    for _ in range(blades):
        a = rng.uniform(0, 2 * math.pi)
        lean = rng.uniform(0.05, 0.45)
        h = height * rng.uniform(0.75, 1.1)
        r = rng.uniform(0, 0.04)
        base = Vector((math.cos(a) * r, math.sin(a) * r, 0))
        fwd = Vector((math.cos(a), math.sin(a), 0))
        side = Vector((-math.sin(a), math.cos(a), 0))

        def pt(s):
            return base + fwd * (h * lean * s ** 1.6) + Vector((0, 0, h * s * (1 - 0.25 * lean * s)))

        w0, w1 = 0.012, 0.007
        v0l = bm.verts.new(pt(0) - side * w0)
        v0r = bm.verts.new(pt(0) + side * w0)
        v1l = bm.verts.new(pt(0.55) - side * w1)
        v1r = bm.verts.new(pt(0.55) + side * w1)
        tip = bm.verts.new(pt(1.0))
        bm.faces.new((v0l, v0r, v1r, v1l))
        bm.faces.new((v1l, v1r, tip))
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    return mesh


def scatter(name, mesh, points, origin):
    """Instance `mesh` on every point using vertex instancing (fast & lightweight)."""
    pts = bpy.data.meshes.new(name + "_pts")
    pts.from_pydata([(x - origin[0], y - origin[1], 0) for x, y in points], [], [])
    holder = link(bpy.data.objects.new(name, pts))
    holder.location = origin
    holder.instance_type = "VERTS"
    holder.show_instancer_for_render = False
    holder.show_instancer_for_viewport = False
    plant = link(bpy.data.objects.new(name + "_plant", mesh))
    plant.parent = holder
    return holder


def build_crops():
    mat = gradient_material("Crop", [
        (0.0, (0.06, 0.10, 0.02)),
        (0.45, (0.30, 0.34, 0.07)),
        (1.0, (0.85, 0.60, 0.16)),
    ])
    variants = []
    for i, (blades, h) in enumerate(((16, 0.78), (14, 0.70), (18, 0.84))):
        m = make_plant_mesh(f"CropMesh{i}", blades, h)
        m.materials.append(mat)
        variants.append(m)

    def rows(x0, x1, y0, y1, dx, dy, jitter):
        out = [[] for _ in variants]
        y = y0
        while y < y1:
            x = x0
            while x < x1:
                px = x + rng.uniform(-jitter, jitter)
                py = y + rng.uniform(-jitter, jitter)
                out[rng.randrange(len(variants))].append((px, py))
                x += dx
            y += dy
        return out

    regions = [
        # near field, dense rows parallel to the road
        ("Near_R", (-35, 14, 1.6, 22), 0.25, 0.30),
        ("Near_L", (-35, 14, -14, -1.6), 0.25, 0.30),
        # far field, sparser (detail is lost with distance anyway)
        ("Far_X", (-110, -35, -40, -1.6), 0.40, 0.36),
        ("Far_X2", (-110, -35, 1.6, 45), 0.40, 0.36),
        ("Far_R", (-35, 14, 22, 45), 0.40, 0.36),
        ("Far_L", (-35, 14, -40, -14), 0.40, 0.36),
    ]
    total = 0
    for name, (x0, x1, y0, y1), dx, dy in regions:
        groups = rows(x0, x1, y0, y1, dx, dy, 0.05)
        origin = ((x0 + x1) / 2, (y0 + y1) / 2, -0.12)
        for i, pts in enumerate(groups):
            scatter(f"{name}_{i}", variants[i], pts, origin)
            total += len(pts)
    print(f"[field] {total} crop clumps")


def build_scenery():
    hill_mat = material("Hills", (0.16, 0.22, 0.28), 0.9)
    for i in range(9):
        x = rng.uniform(-260, -170)
        y = rng.uniform(-200, 200)
        bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1, location=(x, y, -8))
        h = bpy.context.active_object
        h.name = f"Hill{i}"
        h.scale = (rng.uniform(40, 80), rng.uniform(40, 90), rng.uniform(18, 35))
        h.data.materials.append(hill_mat)
        smooth(h)

    leaf_mat = noise_material("Leaves", (0.05, 0.12, 0.03), (0.13, 0.22, 0.05), scale=3.0)
    trunk_mat = material("Trunk", (0.12, 0.08, 0.05), 0.9)
    spots = [(rng.uniform(-140, -60), rng.uniform(-70, 70)) for _ in range(30)]
    spots += [(x, 4.5 + rng.uniform(0, 1.5)) for x in range(-95, -40, 11)]   # roadside row
    for i, (x, y) in enumerate(spots):
        s = rng.uniform(0.8, 1.5)
        bpy.ops.mesh.primitive_cylinder_add(radius=0.18 * s, depth=3 * s, location=(x, y, 1.5 * s))
        bpy.context.active_object.data.materials.append(trunk_mat)
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=2.2 * s, location=(x, y, 3.8 * s))
        crown = bpy.context.active_object
        crown.scale = (1, 1, 0.85)
        crown.data.materials.append(leaf_mat)


# ---------------------------------------------------------------- mesh building
def smoothstep(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def sgnpow(v, e):
    return math.copysign(abs(v) ** e, v)


def catmull_rom(pts, samples=8):
    """Smooth curve through the tuples in `pts` (any dimension)."""
    out = []
    n = len(pts)
    for i in range(n - 1):
        p0, p1, p2, p3 = pts[max(i - 1, 0)], pts[i], pts[i + 1], pts[min(i + 2, n - 1)]
        for k in range(samples):
            t = k / samples
            out.append(tuple(
                0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t * t
                       + (3 * b - a - 3 * c + d) * t * t * t)
                for a, b, c, d in zip(p0, p1, p2, p3)))
    out.append(tuple(pts[-1]))
    return out


def profile_at(keys, s):
    """Linear interpolation in a [(s, value, ...), ...] table."""
    if s <= keys[0][0]:
        return keys[0][1:]
    for (s0, *v0), (s1, *v1) in zip(keys, keys[1:]):
        if s <= s1:
            t = (s - s0) / (s1 - s0)
            return tuple(a + (b - a) * t for a, b in zip(v0, v1))
    return keys[-1][1:]


def stations(s0, s1, step, extra=()):
    """Evenly spaced samples in [s0, s1] that always include the `extra` key positions."""
    keys = sorted({round(v, 5) for v in extra if s0 <= v <= s1} | {s0, s1})
    n = max(1, round((s1 - s0) / step))
    vals = list(keys)
    for i in range(1, n):
        v = s0 + (s1 - s0) * i / n
        if all(abs(v - k) > step * 0.3 for k in keys):
            vals.append(v)
    return sorted(vals)


class MeshBuilder:
    """Accumulates lofted tubes and blobs into one mesh, with optional per-vertex bone weights."""

    def __init__(self):
        self.verts, self.faces, self.mats, self.weights = [], [], [], []

    def _vert(self, co, w=None):
        self.verts.append(tuple(co))
        self.weights.append(w or {})
        return len(self.verts) - 1

    def _bridge(self, a, b, mat):
        if isinstance(a, int):
            quads = [(a, b[j], b[(j + 1) % len(b)]) for j in range(len(b))]
        elif isinstance(b, int):
            quads = [(a[j], a[(j + 1) % len(a)], b) for j in range(len(a))]
        else:
            n = len(a)
            quads = [(a[j], a[(j + 1) % n], b[(j + 1) % n], b[j]) for j in range(n)]
        self.faces += quads
        self.mats += [mat] * len(quads)

    def loft(self, centers, radii, ring=16, mats=0, ref=(1, 0, 0), refs=None, shape=2.0,
             caps=(0.8, 0.8), cap_rings=4, weight=None):
        """Tube through `centers` with (super)elliptic cross-sections.

        radii[i] is r or (r_ref, r_side); `ref`/`refs` orient the r_ref axis.
        mats is one material index or one per segment; weight(i, co) -> {group: w}.
        caps are the end-cap lengths relative to the end radius (0 = open)."""
        centers = [Vector(c) for c in centers]
        n = len(centers)
        radii = [(r, r) if isinstance(r, (int, float)) else tuple(r) for r in radii]
        e = 2.0 / shape
        frames, nrm = [], Vector(ref)
        for i in range(n):
            t = (centers[min(i + 1, n - 1)] - centers[max(i - 1, 0)]).normalized()
            r = Vector(refs[i]) if refs else nrm
            if abs(r.normalized().dot(t)) > 0.99:
                r = Vector((0, 1, 0)) if abs(t.y) < 0.9 else Vector((1, 0, 0))
            nrm = (r - t * r.dot(t)).normalized()
            frames.append((t, nrm, t.cross(nrm)))

        def wts(i, co):
            return weight(i, co) if weight else None

        def ring_at(c, i, scale=1.0):
            _, nv, bv = frames[i]
            ra, rb = radii[i]
            idx = []
            for j in range(ring):
                ph = 2 * math.pi * j / ring
                co = (c + nv * (ra * scale * sgnpow(math.cos(ph), e))
                      + bv * (rb * scale * sgnpow(math.sin(ph), e)))
                idx.append(self._vert(co, wts(i, co)))
            return idx

        def cap(i, sign, length):
            t = frames[i][0] * sign
            ln = length * min(radii[i])
            pole = centers[i] + t * ln
            out = [(self._vert(pole, wts(i, pole)), i)]
            for k in range(cap_rings - 1, 0, -1):
                th = math.pi / 2 * k / cap_rings
                out.append((ring_at(centers[i] + t * (ln * math.sin(th)), i, math.cos(th)), i))
            return out

        rows = cap(0, -1, caps[0]) if caps[0] else []
        rows += [(ring_at(centers[i], i), i) for i in range(n)]
        if caps[1]:
            rows += reversed(cap(n - 1, 1, caps[1]))
        for (a, ia), (b, ib) in zip(rows, rows[1:]):
            m = mats if isinstance(mats, int) else mats[min(ia, ib, len(mats) - 1)]
            self._bridge(a, b, m)

    def blob(self, center, half, exp=2.0, seg=24, rings=12, mat=0, w=None,
             warp=None, keep=None, rot=None):
        """Superellipsoid (exp=2 -> ellipsoid, larger -> rounded box)."""
        c = Vector(center)
        e = 2.0 / exp
        rmat = Euler(rot).to_matrix() if rot else None

        def vert(nv):
            p = Vector([sgnpow(v, e) * h for v, h in zip(nv, half)])
            if warp:
                p = warp(nv, p)
            if rmat:
                p = rmat @ p
            return self._vert(c + p, w), (keep(nv) if keep else True)

        top = vert(Vector((0, 0, 1)))
        grid = []
        for k in range(1, rings):
            th = math.pi * k / rings
            grid.append([vert(Vector((math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph),
                                      math.cos(th))))
                         for ph in (2 * math.pi * j / seg for j in range(seg))])
        bottom = vert(Vector((0, 0, -1)))

        def add(face):
            if all(ok for _, ok in face):
                self.faces.append(tuple(i for i, _ in face))
                self.mats.append(mat)

        for j in range(seg):
            j1 = (j + 1) % seg
            add((top, grid[0][j1], grid[0][j]))
            add((grid[-1][j], grid[-1][j1], bottom))
            for k in range(rings - 2):
                add((grid[k][j], grid[k][j1], grid[k + 1][j1], grid[k + 1][j]))

    def build(self, name, materials, subsurf=(1, 2)):
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(self.verts, [], self.faces)
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(mesh)
        bm.free()
        for m in materials:
            mesh.materials.append(m)
        for poly, m in zip(mesh.polygons, self.mats):
            poly.material_index = m
            poly.use_smooth = True
        obj = link(bpy.data.objects.new(name, mesh))
        groups = {}
        for vi, w in enumerate(self.weights):
            for g, val in w.items():
                if val > 1e-4:
                    if g not in groups:
                        groups[g] = obj.vertex_groups.new(name=g)
                    groups[g].add([vi], val, "REPLACE")
        if subsurf:
            mod = obj.modifiers.new("Subdivision", "SUBSURF")
            mod.levels, mod.render_levels = subsurf
        return obj


def attach(obj, parent):
    """Parent an object whose mesh is modelled in rest-pose world space."""
    obj.parent = parent
    obj.matrix_parent_inverse = parent.matrix_world.inverted()
    return obj


def finger(mb, pts, r, mat=0, taper=0.8):
    pts = catmull_rom([tuple(p) for p in pts], 3)
    n = len(pts)
    mb.loft(pts, [r * (1 - (1 - taper) * i / (n - 1)) for i in range(n)], ring=10, mats=mat,
            caps=(0.9, 0.9), cap_rings=3)


def chain(base, direction, axis, lengths, bends):
    pts = [Vector(base)]
    d = Vector(direction).normalized()
    for ln, b in zip(lengths, bends):
        d = Matrix.Rotation(b, 3, axis) @ d
        pts.append(pts[-1] + d * ln)
    return pts


# ---------------------------------------------------------------- character
# Rest pose (world space, walker root at the origin, all joints un-rotated).
HIP_Z = 0.93
SPINE_Z = HIP_Z + 0.02
SHOULDER_Y, SHOULDER_Z = 0.235, SPINE_Z + 0.50
UPPER_ARM, FOREARM = 0.30, 0.255
HIP_Y = 0.095
CARRY_YAW = 0.35            # inward rotation of the coffee forearm
HAND_YAW = 0.33             # resulting horizontal heading of that forearm
HEAD_PIVOT = Vector((0.005, 0, 0.61))  # relative to the spine, behind the jaw
MOUTH = Vector((0.109, 0, 0.643))     # relative to the spine at rest

TORSO_SHAPE = 2.4
TORSO_KEYS = [  # z, half-depth (x), half-width (y), x-offset
    (0.80, 0.060, 0.080, -0.010),
    (0.83, 0.098, 0.140, -0.012),
    (0.87, 0.116, 0.168, -0.012),
    (0.93, 0.121, 0.180, -0.008),
    (1.00, 0.112, 0.170, 0.000),
    (1.06, 0.104, 0.158, 0.005),
    (1.14, 0.108, 0.160, 0.008),
    (1.24, 0.120, 0.170, 0.010),
    (1.33, 0.125, 0.178, 0.008),
    (1.40, 0.118, 0.196, 0.000),
    (1.45, 0.098, 0.190, -0.010),
    (1.49, 0.075, 0.140, -0.012),
    (1.52, 0.058, 0.085, -0.010),
    (1.55, 0.050, 0.054, -0.005),
    (1.60, 0.046, 0.047, 0.000),
    (1.67, 0.045, 0.045, 0.005),
]
TORSO_DENSE = catmull_rom(TORSO_KEYS, 16)
BELT = (0.988, 1.012)
COLLAR = (1.528, 1.546)

LEG_KEYS = [(0.0, 0.082), (0.10, 0.083), (0.22, 0.075), (0.36, 0.063), (0.46, 0.058),
            (0.55, 0.060), (0.66, 0.057), (0.78, 0.052), (0.835, 0.054), (0.86, 0.058),
            (0.87, 0.058)]
SLEEVE_END = 0.529
ARM_KEYS = [(0.0, 0.060), (0.06, 0.058), (0.14, 0.054), (0.22, 0.050), (0.30, 0.046),
            (0.36, 0.047), (0.43, 0.041), (0.498, 0.034), (0.504, 0.037), (0.524, 0.036),
            (SLEEVE_END, 0.034), (0.534, 0.028), (0.555, 0.027)]


def torso_dims(z, bumps=True):
    a, b, cx = profile_at(TORSO_DENSE, z)
    if bumps and (BELT[0] <= z <= BELT[1] or COLLAR[0] <= z <= COLLAR[1]):
        a, b = a + 0.007, b + 0.007
    return a, b, cx


def torso_point(z, phi_deg):
    a, b, cx = torso_dims(z, bumps=False)
    ph = math.radians(phi_deg)
    e = 2 / TORSO_SHAPE
    return Vector((cx + a * sgnpow(math.cos(ph), e), b * sgnpow(math.sin(ph), e), z))


def torso_normal(z, phi_deg):
    dp = torso_point(z, phi_deg + 0.5) - torso_point(z, phi_deg - 0.5)
    dz = torso_point(z + 0.003, phi_deg) - torso_point(z - 0.003, phi_deg)
    return dp.cross(dz).normalized()


def build_armature():
    bones = {"Pelvis": ((0, 0, HIP_Z), (0, 0, HIP_Z + 0.07)),
             "Spine": ((0, 0, SPINE_Z), (0, 0, SHOULDER_Z))}
    for side, sy in (("L", 1), ("R", -1)):
        sy_ = SHOULDER_Y * sy
        bones[f"Shoulder.{side}"] = ((0, sy_, SHOULDER_Z), (0, sy_, SHOULDER_Z - UPPER_ARM))
        bones[f"Elbow.{side}"] = ((0, sy_, SHOULDER_Z - UPPER_ARM),
                                  (0, sy_, SHOULDER_Z - UPPER_ARM - FOREARM))
        bones[f"Hip.{side}"] = ((0, HIP_Y * sy, HIP_Z), (0, HIP_Y * sy, HIP_Z - THIGH))
        bones[f"Knee.{side}"] = ((0, HIP_Y * sy, HIP_Z - THIGH),
                                 (0, HIP_Y * sy, HIP_Z - THIGH - SHIN))
    data = bpy.data.armatures.new("BodyRig")
    arm = link(bpy.data.objects.new("BodyRig", data))
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    for name, (head, tail) in bones.items():
        eb = data.edit_bones.new(name)
        eb.head, eb.tail = head, tail
    bpy.ops.object.mode_set(mode="OBJECT")
    # every bone follows the animated empty of the same name, relative to its rest pose
    for name in bones:
        target = bpy.data.objects[name]
        con = arm.pose.bones[name].constraints.new("CHILD_OF")
        con.target = target
        con.inverse_matrix = target.matrix_world.inverted()
    return arm


def flannel_material():
    mat = material("Red and white flannel", (0.62, 0.018, 0.026), 0.83)
    nt = mat.node_tree
    uv = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(uv.outputs["UV"], sep.inputs[0])

    def math_node(op, a, b):
        node = nt.nodes.new("ShaderNodeMath")
        node.operation = op
        nt.links.new(a, node.inputs[0])
        if isinstance(b, (float, int)):
            node.inputs[1].default_value = b
        else:
            nt.links.new(b, node.inputs[1])
        return node.outputs[0]

    def stripe(axis):
        phase = math_node("FRACT", math_node("MULTIPLY", axis, 11), 0)
        return math_node("LESS_THAN", phase, 0.52)

    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    ramp.color_ramp.elements[0].color = (0.93, 0.89, 0.82, 1)
    ramp.color_ramp.elements[1].color = (0.40, 0.008, 0.017, 1)
    ramp.color_ramp.elements.new(0.45).color = (0.65, 0.088, 0.106, 1)
    ramp.color_ramp.elements.new(0.95).color = (0.40, 0.008, 0.017, 1)
    strength = math_node("MULTIPLY", math_node("ADD", stripe(sep.outputs["X"]), stripe(sep.outputs["Y"])), 0.5)
    nt.links.new(strength, ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return mat


def build_body(arm, skin, shirt, pants, belt, work_trim, topstitch, shirt_trim, buttons):
    """One continuous, skinned body: torso + neck, legs and arms."""
    SKIN, SHIRT, PANTS, BELT_M, WORK, STITCH, SHIRT_TRIM, BUTTON = range(8)
    mb = MeshBuilder()

    zs = stations(0.80, 1.67, 0.015, (0.984, BELT[0], BELT[1], 1.016,
                                      1.524, COLLAR[0], COLLAR[1], 1.550))
    dims = [torso_dims(z) for z in zs]

    def torso_mat(z):
        if z < 0.987:
            return PANTS
        if z < 1.013:
            return BELT_M
        return SHIRT if z < 1.547 else SKIN

    def torso_w(_, co):
        low = 0.55 * smoothstep(0.95, 0.83, co.z)
        tl, tr = low * smoothstep(0.02, 0.09, co.y), low * smoothstep(0.02, 0.09, -co.y)
        s = smoothstep(0.97, 1.06, co.z)
        rest = 1 - tl - tr
        return {"Pelvis": rest * (1 - s), "Spine": rest * s, "Hip.L": tl, "Hip.R": tr}

    mb.loft([(cx, 0, z) for z, (_, _, cx) in zip(zs, dims)], [(a, b) for a, b, _ in dims],
            ring=36, mats=[torso_mat((z0 + z1) / 2) for z0, z1 in zip(zs, zs[1:])],
            shape=TORSO_SHAPE, caps=(0.6, 0.5), weight=torso_w)

    def limb(top, keys, step, ring, mat_fn, weight_fn, caps):
        ss = stations(0.0, keys[-1][0], step, [k[0] for k in keys])
        mb.loft([Vector(top) - Vector((0, 0, s)) for s in ss],
                [profile_at(keys, s)[0] for s in ss], ring=ring,
                mats=[mat_fn((s0 + s1) / 2) for s0, s1 in zip(ss, ss[1:])],
                caps=caps, weight=lambda i, co: weight_fn(ss[i]))

    sleeve_ranges = []
    for side, sy in (("L", 1), ("R", -1)):
        def leg_w(s, side=side):
            k = smoothstep(0.40, 0.50, s)
            return {f"Hip.{side}": 1 - k, f"Knee.{side}": k}

        def arm_w(s, side=side):
            k = smoothstep(0.26, 0.34, s)
            return {f"Shoulder.{side}": 1 - k, f"Elbow.{side}": k}

        work_leg = [(s, r + 0.006 * smoothstep(0.04, 0.20, s)) for s, r in LEG_KEYS]
        limb((0, HIP_Y * sy, HIP_Z), work_leg, 0.02, 24, lambda s: PANTS, leg_w, (0.8, 0.25))
        hip_w = {f"Hip.{side}": 1.0}
        pocket_y = sy * (HIP_Y + 0.073)
        mb.blob((0.002, pocket_y, HIP_Z - 0.205), (0.055, 0.014, 0.083),
                exp=5, mat=WORK, w=hip_w)
        mb.blob((0.003, pocket_y + sy * 0.008, HIP_Z - 0.146), (0.058, 0.011, 0.022),
                exp=5, mat=PANTS, w=hip_w)
        mb.blob((0.004, pocket_y + sy * 0.019, HIP_Z - 0.149), (0.006, 0.003, 0.006),
                seg=12, rings=6, mat=STITCH, w=hip_w)
        ss = stations(0.30, 0.80, 0.03)
        centers = [(0.007, sy * (HIP_Y + profile_at(work_leg, s)[0]), HIP_Z - s) for s in ss]
        mb.loft(centers, [0.0016] * len(ss), ring=6, mats=STITCH, weight=lambda i, _: leg_w(ss[i]))

        first = len(mb.verts)
        limb((0, SHOULDER_Y * sy, SHOULDER_Z), ARM_KEYS, 0.015, 20,
             lambda s: SHIRT if s < 0.504 else SHIRT_TRIM if s < SLEEVE_END else SKIN, arm_w, (0.9, 0.5))
        sleeve_ranges.append((range(first, len(mb.verts)), SHOULDER_Y * sy))

    spine_w = {"Spine": 1.0}
    placket = [(torso_point(z, 0).x + 0.002, 0, z) for z in stations(1.06, 1.51, 0.02)]
    mb.loft(placket, [(0.004, 0.010)] * len(placket), ring=12, mats=SHIRT_TRIM, shape=3,
            weight=lambda _, __: spine_w)
    for z in (1.095, 1.165, 1.235, 1.305, 1.375, 1.445):
        mb.blob(torso_point(z, 0) + Vector((0.009, 0, 0)), (0.003, 0.005, 0.005),
                seg=12, rings=6, mat=BUTTON, w=spine_w)
    for sign in (-1, 1):
        collar = [torso_point(z, sign * angle) + Vector((0.010, 0, 0.005))
                  for z, angle in ((1.447, 21), (1.475, 22), (1.504, 24), (1.532, 54))]
        mb.loft(collar, [(0.003, r) for r in (0.004, 0.014, 0.019, 0.010)], ring=12,
                mats=SHIRT_TRIM, ref=(1, 0, 0), shape=3, weight=lambda _, __: spine_w)

    body = mb.build("Body", [skin, shirt, pants, belt, work_trim, topstitch, shirt_trim, buttons], subsurf=None)
    # Rest-pose UVs keep the woven pattern on the cloth as the armature bends.
    sleeves = {i: y for indices, y in sleeve_ranges for i in indices}
    uv = body.data.uv_layers.new(name="Flannel weave")
    for face in body.data.polygons:
        if face.material_index != SHIRT:
            continue
        coords = []
        is_sleeve = face.vertices[0] in sleeves
        tiles = 4 if is_sleeve else 11
        for i in face.vertices:
            x, y, z = body.data.vertices[i].co
            if is_sleeve:
                angle = math.atan2(y - sleeves[i], x)
            else:
                depth, width, cx = torso_dims(z)
                angle = math.atan2(y / width, (x - cx) / depth)
            coords.append((angle % math.tau / math.tau, z))
        seam = max(u for u, _ in coords) - min(u for u, _ in coords) > 0.5
        for loop, (u, z) in zip(face.loop_indices, coords):
            uv.data[loop].uv = ((u + (seam and u < 0.5)) * tiles / 11, z)
    mod = body.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    mod.use_deform_preserve_volume = True
    sub = body.modifiers.new("Subdivision", "SUBSURF")
    sub.levels, sub.render_levels = 1, 2
    return body


def build_head(head_pivot, skin, hair, eye, lips):
    H = Vector((0.012, 0, SPINE_Z + 0.695))
    half = (0.103, 0.092, 0.118)

    def warp(nv, p):
        x, y, z = p
        if nv.z < 0:                                   # narrower jaw, small chin
            k = (-nv.z) ** 1.5
            y *= 1 - 0.30 * k
            x *= 1 - 0.08 * k
            if nv.x > 0:
                x += 0.010 * smoothstep(0.3, 0.8, -nv.z) * nv.x
        if nv.x < 0:                                   # rounder back of the skull
            x *= 1.07
        return Vector((x, y, z))

    SKIN, EYE, HAIR, LIPS = range(4)
    mb = MeshBuilder()
    mb.blob(H, half, seg=32, rings=18, mat=SKIN, warp=warp)
    mb.blob(H + Vector((0.098, 0, -0.012)), (0.022, 0.014, 0.024), mat=SKIN, rot=(0, 0.35, 0))
    for sy in (1, -1):
        mb.blob(H + Vector((-0.006, 0.091 * sy, -0.004)), (0.017, 0.008, 0.027), mat=SKIN,
                seg=16, rings=8, rot=(0, 0, -0.25 * sy))
        mb.blob(H + Vector((0.084, 0.037 * sy, 0.014)), (0.012, 0.013, 0.011), mat=EYE,
                seg=12, rings=6)
        mb.blob(H + Vector((0.092, 0.038 * sy, 0.036)), (0.005, 0.019, 0.0045), mat=HAIR,
                seg=12, rings=6, rot=(-0.12 * sy, 0, 0))
    mb.blob(H + Vector((0.092, 0, -0.052)), (0.005, 0.018, 0.004), mat=LIPS, seg=12, rings=6)
    attach(mb.build("Head", [skin, eye, hair, lips]), head_pivot)

    def keep(nv):
        thr = 0.05 - 0.45 * max(0.0, -nv.x) + 0.40 * smoothstep(0.3, 0.7, nv.x)
        return nv.z > thr

    hb = MeshBuilder()
    hb.blob(H, (half[0] * 1.06, half[1] * 1.08, half[2] * 1.05), seg=32, rings=18,
            warp=warp, keep=keep)
    attach(hb.build("Hair", [hair]), head_pivot)


def build_bucket_hat(head_pivot):
    canvas = material("Bucket hat canvas", (0.32, 0.37, 0.19), 0.88)
    lining = material("Bucket hat lining", (0.19, 0.22, 0.103), 0.88)
    thread = material("Bucket hat stitching", (0.43, 0.47, 0.27), 0.83)
    mb = MeshBuilder()
    profile = [(0.012, 1.793), (0.081, 1.794), (0.099, 1.789),
               (0.105, 1.777), (0.118, 1.704), (0.122, 1.696),
               (0.142, 1.684), (0.169, 1.666), (0.175, 1.662),
               (0.174, 1.656), (0.168, 1.654), (0.140, 1.674),
               (0.117, 1.692), (0.106, 1.713), (0.092, 1.775), (0.012, 1.782)]
    n = 64

    def point(r, z, a):
        front_lift = 0.015 * smoothstep(0.115, 0.17, r) * max(0, math.cos(a))
        return (0.008 + 1.08 * r * math.cos(a), r * math.sin(a), z + 0.045 * r * math.cos(2 * a) + front_lift)

    rings = [[mb._vert(point(r, z, j * math.tau / n)) for j in range(n)] for r, z in profile]
    mb._bridge(mb._vert((0.008, 0, 1.793)), rings[0], 0)
    for i, (a, b) in enumerate(zip(rings, rings[1:])):
        mb._bridge(a, b, 0 if i < 9 else 1)
    mb._bridge(rings[-1], mb._vert((0.008, 0, 1.782)), 1)
    for radius, z in ((0.119, 1.708), (0.153, 1.678), (0.166, 1.669)):
        for j in range(n):
            angle = j * math.tau / n
            mb.loft([point(radius, z, angle), point(radius, z, angle + 0.067)],
                    [0.0009, 0.0009], ring=6, mats=2, cap_rings=2)
    for sign in (-1, 1):
        for x in (-0.027, 0.030):
            mb.blob((x, sign * 0.111, 1.744), (0.0045, 0.002, 0.0045), seg=12, rings=8, mat=1)
    attach(mb.build("BucketHat", [canvas, lining, thread], subsurf=(1, 1)), head_pivot)


def build_backpack(spine, fabric, trim, buckle):
    FAB, TRIM, BUCK = range(3)
    mb = MeshBuilder()
    mb.blob((-0.205, 0, 1.215), (0.075, 0.150, 0.205), exp=3.0, seg=32, rings=16, mat=FAB)
    mb.blob((-0.215, 0, 1.415), (0.090, 0.158, 0.040), exp=3.0, seg=32, rings=10, mat=FAB)
    mb.blob((-0.283, 0, 1.120), (0.036, 0.115, 0.095), exp=3.5, seg=28, rings=12, mat=TRIM)
    mb.blob((-0.318, 0.06, 1.195), (0.004, 0.006, 0.013), exp=3.0, seg=8, rings=6, mat=BUCK)

    # lid compression straps with buckles
    for y in (0.075, -0.075):
        pts = [(-0.16, y, 1.452), (-0.24, y, 1.458), (-0.297, y, 1.43), (-0.287, y, 1.36),
               (-0.284, y, 1.30)]
        pts = catmull_rom(pts, 5)
        mb.loft(pts, [(0.003, 0.013)] * len(pts), ring=10, mats=TRIM, shape=3.0,
                refs=[Vector(p) - Vector((-0.2, y, 1.25)) for p in pts], caps=(0.3, 0.3))
        mb.blob((-0.292, y, 1.30), (0.006, 0.018, 0.014), exp=5, seg=12, rings=6, mat=BUCK)
    # grab handle
    handle = catmull_rom([(-0.15, -0.035, 1.44), (-0.15, -0.02, 1.47), (-0.15, 0.02, 1.47),
                          (-0.15, 0.035, 1.44)], 5)
    mb.loft(handle, [(0.004, 0.009)] * len(handle), ring=8, mats=TRIM, refs=[(1, 0, 0)] * len(handle),
            caps=(0.3, 0.3))
    mb.verts = [(x + 0.017, y, z) for x, y, z in mb.verts]     # snug against the back

    # shoulder straps hugging the torso: (z, angle around the body in degrees, gap)
    padded = [(1.36, 160, 0.05), (1.40, 152, 0.02), (1.45, 135, 0.008), (1.49, 110, 0.006),
              (1.50, 90, 0.006), (1.48, 68, 0.006), (1.43, 48, 0.006), (1.35, 36, 0.006),
              (1.26, 33, 0.006), (1.19, 34, 0.006)]
    webbing = [(1.20, 34, 0.004), (1.14, 48, 0.004), (1.09, 75, 0.004), (1.06, 105, 0.004),
               (1.05, 135, 0.004), (1.05, 155, 0.03)]

    def strap(spec, half_t, half_w, sign):
        centers, refs = [], []
        for z, ph, off in catmull_rom(spec, 6):
            n = torso_normal(z, ph * sign)
            centers.append(torso_point(z, ph * sign) + n * (off + half_t))
            refs.append(n)
        mb.loft(centers, [(half_t, half_w)] * len(centers), ring=12, mats=TRIM, refs=refs,
                shape=3.0, caps=(0.3, 0.3))

    for sign in (1, -1):
        strap(padded, 0.0065, 0.024, sign)
        strap(webbing, 0.0025, 0.012, sign)
        p = torso_point(1.19, 34 * sign) + torso_normal(1.19, 34 * sign) * 0.014
        mb.blob(p, (0.006, 0.016, 0.012), exp=5, seg=12, rings=6, mat=BUCK,
                rot=(0, 0, math.radians(34 * sign)))
    # sternum strap with a centre buckle
    strap([(1.30, a, 0.012) for a in (34, 17, 0, -17, -34)], 0.0025, 0.009, 1)
    p = torso_point(1.30, 0) + torso_normal(1.30, 0) * 0.02
    mb.blob(p, (0.006, 0.02, 0.012), exp=5, seg=12, rings=6, mat=BUCK)

    attach(mb.build("Backpack", [fabric, trim, buckle]), spine)


def build_left_hand(elbow, skin):
    """Relaxed hand hanging from the swinging left arm (palm facing the body)."""
    W = Vector((0, SHOULDER_Y, SHOULDER_Z - UPPER_ARM - FOREARM))
    mb = MeshBuilder()
    mb.blob(W + Vector((0, -0.002, -0.048)), (0.040, 0.017, 0.048), seg=20, rings=12)
    for x, scale, r in ((0.026, 0.92, 0.0102), (0.009, 1.0, 0.0105), (-0.008, 0.94, 0.010),
                        (-0.024, 0.76, 0.0088)):
        finger(mb, chain(W + Vector((x, -0.004, -0.088)), (x * 0.6, 0, -1), Vector((1, 0, 0)),
                         [0.040 * scale, 0.026 * scale, 0.020 * scale], [-0.12, -0.35, -0.30]), r)
    finger(mb, [W + Vector(p) for p in ((0.022, -0.010, -0.018), (0.040, -0.020, -0.040),
                                        (0.048, -0.030, -0.060), (0.050, -0.036, -0.075))], 0.0125)
    attach(mb.build("Hand.L", [skin]), elbow)


CUP_AXIS = (0.056, 0.063)


def build_coffee_hand(hand_yaw, skin, paper, sleeve, lid):
    """Right hand gripping a takeaway coffee cup.

    Modelled in the HandYaw frame: +X along the forearm, +Y towards the body, Z up."""
    ax, ay = CUP_AXIS
    mb = MeshBuilder()
    mb.blob((0.046, 0.0, -0.008), (0.048, 0.016, 0.035), seg=20, rings=12)
    R = 0.0525
    for z, length, r in ((0.016, 0.070, 0.0105), (-0.002, 0.078, 0.0105),
                         (-0.019, 0.072, 0.010), (-0.034, 0.058, 0.009)):
        a0 = math.radians(-50)
        arc = [(ax + R * math.cos(a0 + length / R * k / 4), ay + R * math.sin(a0 + length / R * k / 4), z)
               for k in range(5)]
        finger(mb, [(0.090, 0.006, z)] + arc, r)
    thumb = [(0.020, 0.010, 0.020)] + [
        (ax + 0.053 * math.cos(math.radians(a)), ay + 0.053 * math.sin(math.radians(a)), z)
        for a, z in ((-125, 0.024), (-145, 0.026), (-165, 0.028), (-185, 0.030))]
    finger(mb, thumb, 0.0125, taper=0.76)
    hand = mb.build("Hand.R", [skin])
    hand.parent = hand_yaw

    PAPER, SLEEVE, LID = range(3)
    profile = [(-0.075, 0.0310, PAPER), (-0.040, 0.0339, PAPER), (-0.0395, 0.0369, SLEEVE),
               (0.0215, 0.0420, SLEEVE), (0.022, 0.0392, PAPER), (0.046, 0.0428, PAPER),
               (0.047, 0.0448, LID), (0.056, 0.0448, LID), (0.058, 0.0405, LID),
               (0.064, 0.0360, LID)]
    cb = MeshBuilder()
    cb.loft([(ax, ay, z) for z, _, _ in profile], [r for _, r, _ in profile], ring=32,
            mats=[m for *_, m in profile], caps=(0.05, 0.25))
    cup = cb.build("CoffeeCup", [paper, sleeve, lid], subsurf=(1, 1))
    cup.parent = hand_yaw


def build_boot(ankle, sy, upper, rubber, midsole, mesh, lace):
    A = Vector((0, HIP_Y * sy, HIP_Z - THIGH - SHIN))
    mb = MeshBuilder()
    mb.blob(A + Vector((0.055, 0, -0.064)), (0.135, 0.055, 0.012), exp=5, seg=32, rings=10, mat=2)
    mb.blob(A + Vector((0.055, 0, -0.073)), (0.133, 0.055, 0.007), exp=5, seg=32, rings=10, mat=1)
    for x in (-0.061, -0.025, 0.011, 0.048, 0.085, 0.122, 0.158):
        for side in (-1, 1):
            mb.blob(A + Vector((x, side * 0.041, -0.077)), (0.013, 0.014, 0.006),
                    exp=4, seg=12, rings=6, mat=1, rot=(0, 0, side * 0.24))
    foot = [(-0.055, -0.030, 0.035, 0.040), (-0.020, -0.032, 0.042, 0.045),
            (0.040, -0.040, 0.036, 0.046), (0.100, -0.048, 0.026, 0.045),
            (0.150, -0.052, 0.020, 0.041), (0.178, -0.054, 0.016, 0.035)]
    mb.loft([A + Vector((x, 0, z)) for x, z, _, _ in foot], [(ra, rb) for *_, ra, rb in foot],
            ring=24, ref=(0, 0, 1), mats=[0, 0, 0, 1, 1], caps=(0.7, 0.9))
    shaft = [(-0.04, 0.048), (0.015, 0.049), (0.060, 0.051), (0.071, 0.052)]
    mb.loft([A + Vector((-0.008, 0, z)) for z, _ in shaft], [r for _, r in shaft], ring=20,
            mats=[0, 0, 3], caps=(0.5, 0.25))
    for side in (-1, 1):
        mb.blob(A + Vector((0.042, side * 0.037, -0.023)), (0.068, 0.013, 0.023),
                exp=3, seg=20, rings=10, mat=3, rot=(0, 0.34, 0))
    stations_xz = [(0.036, 0.051), (0.048, 0.037), (0.064, 0.024),
                   (0.083, 0.014), (0.104, 0.005), (0.122, -0.006)]
    for i, (x, z) in enumerate(stations_xz):
        for side in (-1, 1):
            mb.blob(A + Vector((x, side * 0.024, z - 0.003)), (0.006, 0.007, 0.005),
                    seg=12, rings=6, mat=1)
            if i + 1 < len(stations_xz):
                nx, nz = stations_xz[i + 1]
                pts = [A + Vector(p) for p in ((x, side * 0.024, z), ((x + nx) / 2, 0, (z + nz) / 2 + 0.003),
                                               (nx, -side * 0.024, nz))]
                mb.loft(pts, [0.0026] * 3, ring=8, mats=4, cap_rings=2)
    attach(mb.build(f"HikingBoot.{'L' if sy > 0 else 'R'}", [upper, rubber, midsole, mesh, lace]), ankle)


def build_character():
    skin = material("Skin", (0.80, 0.55, 0.40), 0.5)
    shirt = flannel_material()
    shirt_trim = material("Flannel collar and cuffs", (0.30, 0.011, 0.020), 0.83)
    buttons = material("Shirt buttons", (0.82, 0.77, 0.67), 0.43)
    pants = material("Pants", (0.13, 0.12, 0.11), 0.8)
    work_trim = material("Workwear pocket fabric", (0.105, 0.096, 0.086), 0.86)
    topstitch = material("Workwear topstitch", (0.22, 0.20, 0.17), 0.85)
    belt = material("Belt", (0.10, 0.06, 0.03), 0.45)
    boot_upper = material("Hiking boot nubuck", (0.30, 0.20, 0.11), 0.8)
    rubber = material("Hiking boot tread and toe guard", (0.025, 0.028, 0.028), 0.85)
    midsole = material("Hiking boot midsole", (0.34, 0.32, 0.25), 0.78)
    boot_mesh = material("Hiking boot mesh", (0.050, 0.063, 0.055), 0.82)
    laces = material("Hiking boot laces", (0.73, 0.38, 0.095), 0.72)
    hair = material("Hair", (0.04, 0.03, 0.025), 0.5)
    eye = material("Eye", (0.02, 0.02, 0.02), 0.3)
    lips = material("Lips", (0.50, 0.24, 0.20), 0.5)
    fabric = material("PackFabric", (0.42, 0.10, 0.07), 0.75)
    trim = material("PackTrim", (0.04, 0.04, 0.045), 0.7)
    buckle = material("Buckle", (0.015, 0.015, 0.015), 0.35)
    paper = material("CupPaper", (0.88, 0.86, 0.80), 0.5)
    kraft = material("CupSleeve", (0.50, 0.33, 0.18), 0.85)
    lid = material("CupLid", (0.95, 0.95, 0.93), 0.3)

    rig = {}
    root = empty("Walker", size=0.4)
    rig["root"] = root
    rig["pelvis"] = pelvis = empty("Pelvis", root, (0, 0, HIP_Z))
    rig["spine"] = spine = empty("Spine", pelvis, (0, 0, SPINE_Z - HIP_Z))
    for side, sy in (("L", 1), ("R", -1)):
        sh = empty(f"Shoulder.{side}", spine, (0, SHOULDER_Y * sy, SHOULDER_Z - SPINE_Z))
        rig[f"shoulder.{side}"] = sh
        rig[f"elbow.{side}"] = empty(f"Elbow.{side}", sh, (0, 0, -UPPER_ARM))
        hip = empty(f"Hip.{side}", pelvis, (0, HIP_Y * sy, 0))
        knee = empty(f"Knee.{side}", hip, (0, 0, -THIGH))
        rig[f"hip.{side}"], rig[f"knee.{side}"] = hip, knee
        rig[f"ankle.{side}"] = empty(f"Ankle.{side}", knee, (0, 0, -SHIN))
    rig["look"] = empty("LookAt", root, (0, 0, LOOK_HEIGHT))
    rig["head"] = empty("HeadPivot", spine, HEAD_PIVOT)

    # the coffee hand stays upright: world-aligned grip, yawed to follow the forearm
    grip = empty("Grip.R", rig["elbow.R"], (0, 0, -FOREARM))
    con = grip.constraints.new("COPY_ROTATION")
    con.target = root
    rig["handyaw"] = empty("HandYaw.R", grip)

    bpy.context.view_layer.update()            # rest-pose matrices for attach / Child Of
    arm = build_armature()
    build_body(arm, skin, shirt, pants, belt, work_trim, topstitch, shirt_trim, buttons)
    build_head(rig["head"], skin, hair, eye, lips)
    build_bucket_hat(rig["head"])
    build_backpack(spine, fabric, trim, buckle)
    build_left_hand(rig["elbow.L"], skin)
    build_coffee_hand(rig["handyaw"], skin, paper, kraft, lid)
    for side, sy in (("L", 1), ("R", -1)):
        build_boot(rig[f"ankle.{side}"], sy, boot_upper, rubber, midsole, boot_mesh, laces)
    return rig


def pivot_ankle(pivot_x, pitch, lever_x):
    """Ankle X/Z when the foot is pitched about a ground contact `lever_x` ahead of the ankle."""
    c, s = math.cos(pitch), math.sin(pitch)
    return pivot_x - lever_x * c + ANKLE_H * s, lever_x * s + ANKLE_H * c


def stance_pose(p, plant):
    """Heel rocker -> flat foot -> heel rise onto the toe. Returns ankle x, z and foot pitch."""
    if p < LOAD:
        pitch = HEEL_STRIKE * (1 - p / LOAD) ** 2
        return (*pivot_ankle(plant + HEEL_X, pitch, HEEL_X), pitch)
    if p > HEEL_OFF:
        pitch = TOE_OFF * ((p - HEEL_OFF) / (STANCE - HEEL_OFF)) ** 2
        return (*pivot_ankle(plant + TOE_X, pitch, TOE_X), pitch)
    return plant, ANKLE_H, 0.0


def hermite(keys, s):
    """Piecewise cubic through (s, value, slope) keys; slope None = Catmull-Rom tangent."""
    i = 0
    while i < len(keys) - 2 and s > keys[i + 1][0]:
        i += 1

    def slope(j):
        if keys[j][2] is not None:
            return keys[j][2]
        return (keys[j + 1][1] - keys[j - 1][1]) / (keys[j + 1][0] - keys[j - 1][0])

    (s0, v0, _), (s1, v1, _) = keys[i], keys[i + 1]
    h = s1 - s0
    t = (s - s0) / h
    t2, t3 = t * t, t * t * t
    return ((2 * t3 - 3 * t2 + 1) * v0 + (t3 - 2 * t2 + t) * h * slope(i)
            + (3 * t2 - 2 * t3) * v1 + (t3 - t2) * h * slope(i + 1))


def swing_pose(s, plant):
    """Toe-off to heel strike, velocity-matched to the stance phases at both ends."""
    eps = 1e-4
    scale = (1 - STANCE) / eps
    start, before = stance_pose(STANCE, plant), stance_pose(STANCE - eps, plant)
    end, after = stance_pose(0.0, plant + STRIDE), stance_pose(eps, plant + STRIDE)
    out = []
    for c in range(3):
        keys = [(0.0, start[c], (start[c] - before[c]) * scale)]
        for ks, frac, z, pitch in SWING_KEYS:
            keys.append((ks, (start[c] + frac * (end[c] - start[c]), z, pitch)[c], None))
        keys.append((1.0, end[c], (after[c] - end[c]) * scale))
        out.append(hermite(keys, s))
    return tuple(out)


def gait_pose(frame, offset):
    """World ankle X, ankle height and foot pitch (positive = heel up) of one foot."""
    ph = frame / CYCLE + offset
    n = math.floor(ph)
    p = ph - n
    plant = START_X + STRIDE * (STANCE / 2 - offset + n)
    if p < STANCE:
        return stance_pose(p, plant)
    return swing_pose((p - STANCE) / (1 - STANCE), plant)


def leg_ik(hip, foot):
    """Two-bone IK in the pelvis frame: the leg plane is rolled about X to reach the foot,
    thigh/shin angles are measured forward from straight down inside that plane."""
    d = foot - hip
    roll = math.atan2(d.y, -d.z)
    dx, dz = d.x, -math.hypot(d.y, d.z)
    dist = min(math.hypot(dx, dz), THIGH + SHIN - 1e-4)
    alpha = math.atan2(dx, -dz)
    a = math.acos(max(-1.0, min(1.0, (THIGH ** 2 + dist ** 2 - SHIN ** 2) / (2 * THIGH * dist))))
    thigh = alpha + a                      # knee points forward
    kx, kz = THIGH * math.sin(thigh), -THIGH * math.cos(thigh)
    shin = math.atan2(dx - kx, -(dz - kz))
    return roll, thigh, shin


def animate_reaction(rig, frame, spine_origin, spine_rot, twist):
    t = (frame - 1) / FPS
    bird_weight = smoothstep(BIRD_ENTER + 0.10, BIRD_ENTER + 0.65, t) * (
        1 - smoothstep(BIRD_EXIT - 0.05, BIRD_EXIT + 0.40, t))
    sip = smoothstep(9.45, 10.36, t) * (1 - smoothstep(10.98, 11.84, t))
    tilt = smoothstep(10.15, 10.55, t) * (1 - smoothstep(10.98, 11.40, t))
    head = rig["head"]
    gaze = spine_rot.transposed() @ (bird_position(frame) - spine_origin) - HEAD_PIVOT
    # the neck only covers part of the gaze (the eyes do the rest), easing into its limits
    pitch = 0.58 * math.tanh(0.8 * math.atan2(gaze.z, math.hypot(gaze.x, gaze.y)) / 0.58)
    yaw = 0.80 * math.tanh(0.8 * math.atan2(gaze.y, gaze.x) / 0.80)
    # the head follows the target through a critically damped spring, lagging naturally
    state = rig.setdefault("gaze_state", [0.0, 0.0, 0.0, 0.0])   # pitch, yaw, velocities
    omega, dt = 11.0, 1 / FPS
    for i, target in enumerate((bird_weight * pitch, bird_weight * yaw)):
        state[i + 2] += (omega * omega * (target - state[i]) - 2 * omega * state[i + 2]) * dt
        state[i] += state[i + 2] * dt
    look_pitch, look_yaw = state[0], state[1]
    # walking: the head counters the shoulder twist to keep the gaze steady
    # sipping: chin dips to meet the lid, then the head tips back as the cup tilts
    head.rotation_euler = (0, -look_pitch + 0.05 * (sip - tilt) - 0.30 * tilt,
                           look_yaw - 0.6 * (1 - bird_weight) * twist - 0.13 * sip)
    head.keyframe_insert("rotation_euler", frame=frame)

    hand = rig["handyaw"]
    hand.rotation_euler = (0, -0.62 * tilt, HAND_YAW + twist + 0.17 * sip)
    if sip > 0:
        shoulder, elbow = rig["shoulder.R"], rig["elbow.R"]
        upper = shoulder.rotation_euler.to_matrix()
        lower = upper @ elbow.rotation_euler.to_matrix()
        origin = shoulder.location
        upper_vec = upper @ Vector((0, 0, -UPPER_ARM))
        carry_wrist = origin + upper_vec + lower @ Vector((0, 0, -FOREARM))

        mouth = HEAD_PIVOT + head.rotation_euler.to_matrix() @ (MOUTH - HEAD_PIVOT)
        rim = Vector((CUP_AXIS[0] - 0.034, CUP_AXIS[1], 0.065))
        wrist_offset = spine_rot.transposed() @ (hand.rotation_euler.to_matrix() @ rim)
        target = carry_wrist.lerp(mouth - wrist_offset + Vector((0.010, -0.006, -0.002)), sip)

        # Solve the two arm lengths with the elbow biased down and out from the ribs.
        reach = target - origin
        axis = reach.normalized()
        span = min(reach.length, UPPER_ARM + FOREARM - 0.001)
        along = (span**2 + UPPER_ARM**2 - FOREARM**2) / (2 * span)
        pole = upper_vec.normalized().lerp(Vector((0.40, -0.62, -0.70)), sip)
        side = (pole - axis * pole.dot(axis)).normalized()
        new_upper = axis * along + side * math.sqrt(UPPER_ARM**2 - along**2)
        new_lower = axis * span - new_upper
        upper_rotation = upper_vec.rotation_difference(new_upper) @ upper.to_quaternion()
        lower_vec = lower @ Vector((0, 0, -FOREARM))
        lower_rotation = lower_vec.rotation_difference(new_lower) @ lower.to_quaternion()
        shoulder.rotation_euler = upper_rotation.to_euler("XYZ", shoulder.rotation_euler)
        elbow.rotation_euler = (upper_rotation.inverted() @ lower_rotation).to_euler("XYZ", elbow.rotation_euler)
    hand.keyframe_insert("rotation_euler", frame=frame)


def animate_character(rig):
    for side in ("L", "R"):
        for key in ("hip", "ankle"):
            rig[f"{key}.{side}"].rotation_mode = "XYZ"
    for f in range(0, FRAMES + 2):
        root_x = START_X + SPEED * f
        w = 2 * math.pi * ((f / CYCLE) % 1.0)          # 0 = left heel strike
        mid = math.cos(w - math.pi * STANCE)            # +1 left mid-stance, -1 right mid-stance
        hip_z = HIP_WALK + BOB * math.cos(2 * (w - math.pi * STANCE))
        # weight shifts over the stance leg, the swing-side hip drops, and the pelvis
        # turns with the forward leg while the shoulders counter-rotate
        pelvis_rot = Euler((0.045 * mid, 0, -0.07 * math.cos(w - 0.15)))
        twist = 0.09 * math.cos(w - 0.25)

        rig["root"].location = (root_x, PATH_Y, 0)
        rig["root"].keyframe_insert("location", frame=f)
        pelvis = rig["pelvis"]
        pelvis.location = (0, 0.016 * math.cos(w - math.pi * STANCE - 0.3), hip_z)
        pelvis.rotation_euler = pelvis_rot
        pelvis.keyframe_insert("location", frame=f)
        pelvis.keyframe_insert("rotation_euler", frame=f)
        p_mat = pelvis_rot.to_matrix()
        p_inv = p_mat.transposed()

        for side, sy, off in (("L", 1, 0.0), ("R", -1, 0.5)):
            fx, fz, pitch = gait_pose(f, off)
            foot = p_inv @ (Vector((fx - root_x, sy * FOOT_Y, fz)) - pelvis.location)
            roll, thigh, shin = leg_ik(Vector((0, sy * HIP_Y, 0)), foot)
            hip, knee, ankle = (rig[f"{k}.{side}"] for k in ("hip", "knee", "ankle"))
            hip_m = Matrix.Rotation(roll, 3, "X") @ Matrix.Rotation(-thigh, 3, "Y")
            hip.rotation_euler = hip_m.to_euler("XYZ", hip.rotation_euler)
            knee.rotation_euler = (0, -(shin - thigh), 0)
            # cancel the leg and pelvis rotation so the foot keeps pointing forward
            ankle_m = (Matrix.Rotation(shin, 3, "Y") @ Matrix.Rotation(-roll, 3, "X")
                       @ p_inv @ Matrix.Rotation(pitch, 3, "Y"))
            ankle.rotation_euler = ankle_m.to_euler("XYZ", ankle.rotation_euler)
            for obj in (hip, knee, ankle):
                obj.keyframe_insert("rotation_euler", frame=f)

        spine = rig["spine"]
        spine.rotation_euler = (-0.75 * pelvis_rot.x, 0.06, twist - pelvis_rot.z)
        spine.keyframe_insert("rotation_euler", frame=f)

        # left arm swings opposite to the left leg, lagging slightly behind the shoulders
        rig["shoulder.L"].rotation_euler = (0.07, 0.27 * math.cos(w - 0.35), 0)
        rig["elbow.L"].rotation_euler = (0, -(0.22 + 0.30 * (0.5 - 0.5 * math.cos(w - 0.5))), 0)
        # right arm carries the coffee: forearm held forward, barely swinging, gentle bob
        phi = 0.20 + 0.04 * math.cos(w - 0.35)
        rig["shoulder.R"].rotation_euler = (-0.07, -phi, 0)
        bend = 1.25 + 0.04 * math.sin(2 * w)
        rig["elbow.R"].rotation_euler = (0, -bend, CARRY_YAW)
        spine_rot = p_mat @ spine.rotation_euler.to_matrix()
        spine_origin = Vector((root_x, PATH_Y, 0)) + pelvis.location + p_mat @ spine.location
        animate_reaction(rig, f, spine_origin, spine_rot, twist)
        for side in ("L", "R"):
            rig[f"shoulder.{side}"].keyframe_insert("rotation_euler", frame=f)
            rig[f"elbow.{side}"].keyframe_insert("rotation_euler", frame=f)


# ---------------------------------------------------------------- camera and bird
def camera_offset(frame):
    start, end = Vector((4.7, -3.6, 1.75)), Vector((4.1, -3.15, 1.62))
    return start.lerp(end, smoothstep(0, FRAMES, frame))


def build_camera(rig):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = LENS
    cam_data.dof.use_dof = True
    cam_data.dof.focus_object = rig["look"]
    cam_data.dof.aperture_fstop = 2.8
    cam = link(bpy.data.objects.new("Camera", cam_data))
    con = cam.constraints.new("TRACK_TO")
    con.target = rig["look"]
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    bpy.context.scene.camera = cam

    # tracking shot with a gentle push-in (3/4 front view)
    for f in range(0, FRAMES + 2):
        cam.location = Vector((START_X + SPEED * f, PATH_Y, 0)) + camera_offset(f)
        cam.keyframe_insert("location", frame=f)


def bird_position(frame):
    # Compose the flight in screen space, then project it to a world-space path.
    t = ((frame - 1) / FPS - BIRD_ENTER) / (BIRD_EXIT - BIRD_ENTER)
    u = max(-0.10, min(1.10, t))
    x = 1.10 - 2.21 * u
    y = 0.84 + 0.05 * math.sin(math.pi * u)
    # stays a little in front of the walker so it never passes straight over the head
    depth = 5.5 - 0.8 * math.sin(math.pi * u)
    offset = camera_offset(frame)
    view = (Vector((0, 0, LOOK_HEIGHT)) - offset).to_track_quat("-Z", "Y")
    half_width = depth * (36 / 2) / LENS
    local = Vector((x * half_width, y * half_width * 9 / 16, -depth))
    return Vector((START_X + SPEED * frame, PATH_Y, 0)) + offset + view @ local


def build_bird():
    feathers = material("BirdFeathers", (0.13, 0.105, 0.081), 0.72)
    dark = material("BirdFlightFeathers", (0.038, 0.027, 0.023), 0.8)
    chest = material("BirdChest", (0.51, 0.40, 0.29), 0.8)
    beak = material("BirdBeak", (0.39, 0.21, 0.07), 0.55)
    root = empty("BirdFlight")
    root.scale = (0.73,) * 3
    root.rotation_mode = "QUATERNION"
    body = MeshBuilder()
    body.blob((0, 0, 0), (0.096, 0.040, 0.043))
    body.blob((0.035, 0, -0.015), (0.052, 0.035, 0.035), mat=2)
    body.blob((0.075, 0, 0.026), (0.040, 0.030, 0.033))
    body.loft([(0.106, 0, 0.021), (0.130, 0, 0.018), (0.144, 0, 0.016)],
              [(0.011, 0.011), (0.008, 0.007), (0.002, 0.002)], ring=12, mats=3)
    for sign in (-1, 1):
        body.blob((0.087, sign * 0.027, 0.033), (0.006, 0.005, 0.006), mat=1)
        for i in range(3):
            body.blob((-0.108, sign * (i + 1) * 0.009, -0.008), (0.073, 0.012, 0.007),
                      mat=1, rot=(0, -0.10, -sign * (i + 1) * 0.17))
    body.build("Bird", [feathers, dark, chest, beak]).parent = root

    wings = []
    for sign in (-1, 1):
        wing = empty("BirdWing", root, (0.019, sign * 0.027, 0.008))
        mesh = MeshBuilder()
        mesh.loft([(0, 0, 0), (-0.015, sign * 0.065, 0.005),
                   (-0.042, sign * 0.126, 0.003), (-0.091, sign * 0.195, -0.008)],
                  [(0.050, 0.014), (0.061, 0.012), (0.047, 0.009), (0.009, 0.004)],
                  ring=16, mats=[0, 0, 1])
        for i in range(5):
            mesh.blob((-0.074 - i * 0.010, sign * (0.163 - i * 0.018), -0.003),
                      (0.049, 0.013, 0.005), mat=1, rot=(0, 0, -sign * (0.81 - i * 0.09)))
        mesh.build("WingFeathers", [feathers, dark]).parent = wing
        wings.append((sign, wing))

    for frame in range(1, FRAMES + 1):
        pos = bird_position(frame)
        heading = bird_position(frame + 0.5) - bird_position(frame - 0.5)
        t = (frame - 1) / FPS
        root.location = pos
        root.rotation_quaternion = heading.to_track_quat("X", "Z")
        root.keyframe_insert("location", frame=frame)
        root.keyframe_insert("rotation_quaternion", frame=frame)
        phase = 2 * math.pi * 3.3 * (t - BIRD_ENTER)
        for sign, wing in wings:
            wing.rotation_euler = (sign * (0.15 + 0.87 * math.sin(phase)), 0,
                                   sign * (0.09 + 0.11 * math.cos(phase)))
            wing.keyframe_insert("rotation_euler", frame=frame)


# ---------------------------------------------------------------- render
def configure_render(scene, preview):
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.fps = FPS
    scene.frame_start = 1
    scene.frame_end = FRAMES
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 50 if preview else 100
    scene.eevee.taa_render_samples = 16 if preview else 32
    scene.render.use_motion_blur = not preview
    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Punchy"
    except TypeError:
        pass


def set_output(scene, video, path):
    img = scene.render.image_settings
    if hasattr(img, "media_type"):
        img.media_type = "VIDEO" if video else "IMAGE"
    if video:
        img.file_format = "FFMPEG"
        ff = scene.render.ffmpeg
        ff.format = "MPEG4"
        ff.codec = "H264"
        ff.constant_rate_factor = "HIGH"
        ff.ffmpeg_preset = "GOOD"
    else:
        img.file_format = "PNG"
    scene.render.filepath = path


def main():
    opts = args()
    scene = reset_scene()
    build_world(scene)
    build_sun()
    build_ground()
    build_crops()
    build_scenery()
    rig = build_character()
    animate_character(rig)
    build_camera(rig)
    build_bird()
    configure_render(scene, opts["preview"])
    scene.frame_set(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "walk_in_field.blend"))

    for frame in opts["still"]:
        scene.frame_set(frame)
        set_output(scene, False, os.path.join(OUT_DIR, f"still_{frame:03d}.png"))
        bpy.ops.render.render(write_still=True)
    if opts["render"]:
        set_output(scene, True, os.path.join(OUT_DIR, "walk_in_field.mp4"))
        bpy.ops.render.render(animation=True)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, "walk_in_field.blend"))


main()
