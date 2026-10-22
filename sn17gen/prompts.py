RULES = """Module contract (violations are rejected by a static validator):
- Return ONLY JavaScript. One export: `export default function generate(THREE) { ... return root; }`, synchronous.
- THREE exists only as that parameter. Helpers that need it take a parameter named exactly `THREE`. Never alias THREE, never `THREE[expr]`, never reference THREE at top level.
- Forbidden anywhere, even as local variable names: import, require, Math.random, Date, performance, fetch, XMLHttpRequest, WebSocket, window, document, navigator, self, global, globalThis, module, process, eval, Function, setTimeout, setInterval, setImmediate, queueMicrotask, crypto, Proxy, Reflect, WeakRef, Worker, localStorage.
- Allowed THREE members: Group, Object3D, Mesh, InstancedMesh, Line, LineSegments, Points; BufferGeometry, BufferAttribute, Float32BufferAttribute, Uint16BufferAttribute, Uint32BufferAttribute; BoxGeometry, SphereGeometry, CylinderGeometry, CapsuleGeometry, ConeGeometry, TorusGeometry, TorusKnotGeometry, PlaneGeometry, CircleGeometry, RingGeometry, TetrahedronGeometry, OctahedronGeometry, DodecahedronGeometry, IcosahedronGeometry, PolyhedronGeometry, ExtrudeGeometry, LatheGeometry, ShapeGeometry, TubeGeometry, EdgesGeometry; MeshStandardMaterial, MeshPhysicalMaterial, MeshBasicMaterial, LineBasicMaterial, PointsMaterial; DataTexture (procedural data only; geometry needs uv); Vector2, Vector3, Vector4, Matrix3, Matrix4, Quaternion, Euler, Box3, Sphere, Color, MathUtils (not seededRandom/generateUUID); Shape, Path, Curve, CurvePath, CatmullRomCurve3, CubicBezierCurve3, QuadraticBezierCurve3, LineCurve3, EllipseCurve, ArcCurve, SplineCurve; constants such as DoubleSide, FrontSide, SRGBColorSpace, RepeatWrapping, RGBAFormat, UnsignedByteType, LinearFilter, NearestFilter.
- Forbidden THREE APIs: loaders, ShaderMaterial, RawShaderMaterial, MeshLambertMaterial, MeshPhongMaterial, CanvasTexture, morph targets, animation.
- Limits: <=200 draw calls (meshes), <=250,000 vertices, <=50,000 instances, scene depth <=32, <=4 MiB of texture data in total (one 1024x1024 RGBA DataTexture is already 4 MiB; use 256x256 or 512x512), <=50 KB of string+numeric literals, <=5 s, deterministic (no randomness; derive variation from indices with sin/cos hashes).
- Y is up. Build at natural scale; the result is re-centered and scaled into [-0.5, 0.5] automatically."""

JUDGING = """How the result is judged: the module is rendered on a white background (and a mid-gray one) by a camera on the +Z side looking at the origin: from 30 degrees to the right (+X) of +Z, 30 degrees to the left, from about 30 degrees above, and slightly from below; plus the right, back, left and top views. The camera has a 49.1 degree field of view at 2.0 units from the origin. A vision-language model compares each render with the reference photo and prefers the model that looks more like the same object: overall shape and proportions, colors and materials, distinctive parts and how many there are, and a complete, plausible object from every side. It marks down missing or extra parts, parts that float free, wrong, faded or darkened colors, and anything that is not the object itself (floor, base, box, frame).
Pose: place the object so that the camera on +Z, slightly above, sees it the way the photo shows it (same side facing the viewer, same upright/lying/profile pose and left-right direction). Keep the photo's pose: if the object lies, leans or sits at an angle in the photo, build it that way in the X-Y plane. An object seen from the side in the photo (vehicle, boat, shoe, knife, pen, animal in profile) has its length along X, not toward the camera. A flat object seen face-on (plate, board, clock, sign, card, tray) shows that face toward +Z, not its edge. Model only the object itself, never the floor, background, shadows or reflections."""

CRAFT = """Craft:
- Decompose the object into its real parts. Get the silhouette and proportions right first, then add the distinctive details (bevels, rims, seams, handles, holes, rivets, markings). Count every repeated part in the photo (legs, spokes, buttons, slats, windows) and build exactly that many. Paint visible labels, logos and printed patterns as simple DataTextures in the photo's colors on the right face.
- Use LatheGeometry for round profiles, ExtrudeGeometry (with bevel) for flat shapes with outlines, TubeGeometry for bent rods/cables/handles, smooth sphere/capsule segments for organic forms; scale spheres non-uniformly for ellipsoids.
- Use MeshStandardMaterial/MeshPhysicalMaterial with the photo's actual colors (hex) and sensible roughness. The judge's lighting turns high metalness nearly black: keep metalness at 0.6 or less and put the metal's shade into the color (silver about #b8b8b8). Keep the photo's hue and saturation; do not wash colors out or darken them. Pure white disappears on the white background (use at most about #e6e6e6) and pure black loses all shape (use about #1c1c1c to #2a2a2a). Use small procedural DataTextures for visible patterns (wood grain, fabric, rust, print) when they matter. Keep parts opaque unless the photo shows them see-through. Glass/clear plastic: transparent with opacity about 0.35-0.6, a visible tint and visible edges or rims; avoid transmission (it renders very slowly).
- The renderer stops slow scenes: stay under about 100,000 vertices, 150 meshes and 10,000 instances; create each DataTexture once (at most 512x512) and reuse it. Put the detail where the cameras see it.
- Wheels turn on their axle: a vehicle seen from the side (length along X, axles along Z) keeps the default TorusGeometry for tyres and needs rotation.x = Math.PI / 2 on CylinderGeometry hubs; a vehicle facing +Z (axles along X) needs rotation.y = Math.PI / 2 on tyres and rotation.z = Math.PI / 2 on hubs.
- Write -(x ** 2), never -x ** 2 (a unary sign directly before ** is a syntax error).
- Keep surfaces closed; avoid thin floating planes, z-fighting and parts poking through each other."""

GEN_SYSTEM = ("You are an expert technical artist who rebuilds objects from a single reference photo as procedural Three.js models.\n\n"
              + JUDGING + "\n\n" + CRAFT + "\n\n" + RULES
              + "\n\nDetail: write a thorough module, about 6,000-8,000 tokens of code. Spend the length on what the photo shows: "
    "every distinct part, accurate profiles and proportions, bevels, rims, seams and panel lines, buttons, keys, screws, "
    "holes, feet and hinges, and painted or printed patterns (procedural DataTextures or thin surface geometry). "
    "Use helper functions and loops so the length goes into detail, not repetition.")

GEN_USER = "Reference photo attached. Write the complete Three.js module for this object. Output only the JavaScript."

GEN_USER_ALT = ("Reference photo attached. Write the complete Three.js module for this object. Start from the proportions: "
                "measure the width, height and depth of every major part as a fraction of the whole object, write them as "
                "named constants, and build every part from them. Output only the JavaScript.")

PLAN_SYSTEM = "You study one reference photo of an object before it is rebuilt as a procedural 3D model. Reply with short plain-text notes. No code."

PLAN_USER = """Reference photo attached. Write notes for rebuilding this object, under 2,000 characters, in this order:
Object: what the object is.
View: which side of the object faces the camera, and the camera elevation in degrees.
Proportions: overall width:height:depth.
Parts: each distinct part with its count, shape and hex colour.
Text: visible text or markings, or none.
Checklist: then 5-7 lines starting with "- ", each one feature a judge comparing a render with this photo would verify."""

PLAN_NOTES = """Notes from a first look at the photo; the photo wins where they disagree:
%s

"""

CHECK_REFINE = """

Checklist from a first look at the photo (the photo wins where they disagree); make sure each item matches:
%s"""

CHECK_PANEL = """Features a faithful model should match (from a first look at the reference; the reference wins where they disagree):
%s

"""

REFINE_USER = """Attached: (1) the reference photo, (2) a 2x2 sheet of renders of your current module on white from the exact four views the judge compares with the photo: top-left from 30 degrees right of front, top-right from 30 degrees left of front, bottom-left from slightly below the front, bottom-right from above the front.
Match the photo's pose, framing and visible side as seen from these views.
Compare them with the photo. Note to yourself what already matches the photo and keep it unchanged; list the largest visible differences (pose/orientation, shape, proportions, part counts, colors and saturation, labels, missing or wrong parts) and fix only those.
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

%sPenalty 0-10 per candidate:
0 = Perfect match to reference
3 = Minor issues (slight shape differences, missing small details)
5 = Moderate issues (wrong style, significant details missing, wrong colors on key parts)
7 = Major issues (wrong category but related)
10 = Completely wrong object

Output JSON only: {"penalties": [<one number per candidate, in order>]}"""

JUDGE_SYSTEM = (
    "You are a specialized 3D model evaluation system.\n"
    "Analyze visual quality and prompt adherence with expert precision.\n"
    "Always respond with valid JSON only."
)

JUDGE_USER = (
    "You see two 3D models rendered from {angle_desc}.\n"
    "The reference image shows the target object.\n\n"
    "Which model is a more faithful 3D reproduction of the reference?\n\n"
    "Penalty 0-10:\n"
    "0 = Perfect match to reference\n"
    "3 = Minor issues (slight shape differences, missing small details)\n"
    "5 = Moderate issues (wrong style, significant details missing)\n"
    "7 = Major issues (wrong category but related, e.g. chair vs stool)\n"
    "10 = Completely wrong object\n\n"
    'Output: {{"penalty_1": <0-10>, "penalty_2": <0-10>, "issues": "<brief>"}}'
)

JUDGE_LABELS = ("Reference image (target object):", "3D model 1:", "3D model 2:")

JUDGE_VIEWS = ((30, 0, "slightly left of the front"), (330, 0, "slightly right of the front"),
               (0, 15, "the front, from slightly below"), (0, -30, "the front, from slightly above"))
