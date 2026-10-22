RULES = """Module contract (violations are rejected by a static validator):
- Return ONLY JavaScript. One export: `export default function generate(THREE) { ... return root; }`, synchronous.
- THREE exists only as that parameter. Helpers that need it take a parameter named exactly `THREE`. Never alias THREE, never `THREE[expr]`, never reference THREE at top level.
- Forbidden anywhere, even as local variable names: import, require, Math.random, Date, performance, fetch, XMLHttpRequest, WebSocket, window, document, navigator, self, global, globalThis, module, process, eval, Function, setTimeout, setInterval, setImmediate, queueMicrotask, crypto, Proxy, Reflect, WeakRef, Worker, localStorage.
- Allowed THREE members: Group, Object3D, Mesh, InstancedMesh, Line, LineSegments, Points; BufferGeometry, BufferAttribute, Float32BufferAttribute, Uint16BufferAttribute, Uint32BufferAttribute; BoxGeometry, SphereGeometry, CylinderGeometry, CapsuleGeometry, ConeGeometry, TorusGeometry, TorusKnotGeometry, PlaneGeometry, CircleGeometry, RingGeometry, TetrahedronGeometry, OctahedronGeometry, DodecahedronGeometry, IcosahedronGeometry, PolyhedronGeometry, ExtrudeGeometry, LatheGeometry, ShapeGeometry, TubeGeometry, EdgesGeometry; MeshStandardMaterial, MeshPhysicalMaterial, MeshBasicMaterial, LineBasicMaterial, PointsMaterial; DataTexture (procedural data only; geometry needs uv); Vector2, Vector3, Vector4, Matrix3, Matrix4, Quaternion, Euler, Box3, Sphere, Color, MathUtils (not seededRandom/generateUUID); Shape, Path, Curve, CurvePath, CatmullRomCurve3, CubicBezierCurve3, QuadraticBezierCurve3, LineCurve3, EllipseCurve, ArcCurve, SplineCurve; constants such as DoubleSide, FrontSide, SRGBColorSpace, RepeatWrapping, RGBAFormat, UnsignedByteType, LinearFilter, NearestFilter.
- Forbidden THREE APIs: loaders, ShaderMaterial, RawShaderMaterial, MeshLambertMaterial, MeshPhongMaterial, CanvasTexture, morph targets, animation.
- Limits: <=200 draw calls (meshes), <=250,000 vertices, <=50,000 instances, scene depth <=32, <=4 MiB of texture data in total (one 1024x1024 RGBA DataTexture is already 4 MiB; use 256x256 or 512x512), <=50 KB of string+numeric literals, <=5 s, deterministic (no randomness; derive variation from indices with sin/cos hashes).
- Y is up. Build at natural scale; the result is re-centered and scaled into [-0.5, 0.5] automatically."""

JUDGING = """How the result is judged: the module is rendered on a white background (and a mid-gray one) by a camera on the +Z side looking at the origin: from 30 degrees to the right (+X) of +Z, 30 degrees to the left, from about 30 degrees above, and slightly from below; plus the right, back, left and top views. A vision-language model compares each render with the reference photo and prefers the model that looks more like the same object: overall shape and proportions, colors and materials, distinctive parts and details, and a complete, plausible object from every side.
Pose: place the object so that the camera on +Z, slightly above, sees it the way the photo shows it (same side facing the viewer, same upright/lying/profile pose and left-right direction). Model only the object itself, never the floor, background, shadows or reflections."""

CRAFT = """Craft:
- Decompose the object into its real parts. Get the silhouette and proportions right first, then add the distinctive details (bevels, rims, seams, handles, holes, rivets, text-free markings).
- Use LatheGeometry for round profiles, ExtrudeGeometry (with bevel) for flat shapes with outlines, TubeGeometry for bent rods/cables/handles, smooth sphere/capsule segments for organic forms; scale spheres non-uniformly for ellipsoids.
- Use MeshStandardMaterial/MeshPhysicalMaterial with the photo's actual colors (hex), sensible roughness, metalness only for metal. Use small procedural DataTextures for visible patterns (wood grain, fabric, rust, print) when they matter. Glass/clear plastic: MeshPhysicalMaterial with transmission or opacity, but keep it visible with a tint and edges.
- Keep surfaces closed; avoid thin floating planes, z-fighting and parts poking through each other."""

GEN_SYSTEM = ("You are an expert technical artist who rebuilds objects from a single reference photo as procedural Three.js models.\n\n"
              + JUDGING + "\n\n" + CRAFT + "\n\n" + RULES
              + "\n\nDetail: write a thorough module, about 6,000-8,000 tokens of code. Spend the length on what the photo shows: "
    "every distinct part, accurate profiles and proportions, bevels, rims, seams and panel lines, buttons, keys, screws, "
    "holes, feet and hinges, and painted or printed patterns (procedural DataTextures or thin surface geometry). "
    "Use helper functions and loops so the length goes into detail, not repetition.")

GEN_USER = "Reference photo attached. Write the complete Three.js module for this object. Output only the JavaScript."

REFINE_USER = """Attached: (1) the reference photo, (2) a 2x2 sheet of renders of your current module on white: top-left from 30 degrees right of front, top-right from 30 degrees left of front, bottom-left from above the front, bottom-right the right side.
Compare them with the photo. List to yourself the largest visible differences (shape, proportions, pose/orientation, colors, missing or wrong parts), then fix them.
Return the complete improved module (only JavaScript).

Current module:
```js
%s
```"""

REPAIR_USER = """Your module was rejected by the validator or renderer:
%s

Fix the problem and return the complete module (only JavaScript).

Module:
```js
%s
```"""

PICK_SYSTEM = "You are a specialized 3D model evaluation system. Always respond with valid JSON only."

PANEL_USER = """Image 1 is the reference photo of the target object. Images 2 to %d each show one candidate 3D model rendered from the same four front viewpoints (30 degrees right, 30 degrees left, slightly below, from above). Candidate k is image k+1.

For every candidate, judge how faithful a 3D reproduction of the reference it is: the right kind of object, overall shape and proportions, pose, distinctive parts and their counts, colors and materials.

Penalty 0-10 per candidate:
0 = Perfect match to reference
3 = Minor issues (slight shape differences, missing small details)
5 = Moderate issues (wrong style, significant details missing, wrong colors on key parts)
7 = Major issues (wrong category but related)
10 = Completely wrong object

Output JSON only: {"penalties": [<one number per candidate, in order>]}"""
