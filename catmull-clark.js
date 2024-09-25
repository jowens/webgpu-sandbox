// inspiration: https://webgpufundamentals.org/webgpu/lessons/webgpu-fundamentals.html

import { Pane } from "https://cdn.jsdelivr.net/npm/tweakpane@4.0.3/dist/tweakpane.min.js";
import {
  vec3,
  mat4,
} from "https://wgpu-matrix.org/dist/3.x/wgpu-matrix.module.js"; // for uniform handling
import {
  makeShaderDataDefinitions,
  makeStructuredView,
} from "https://greggman.github.io/webgpu-utils/dist/1.x/webgpu-utils.module.js";

const adapter = await navigator.gpu?.requestAdapter();
const canTimestamp = adapter.features.has("timestamp-query");
const device = await adapter?.requestDevice({
  requiredFeatures: [...(canTimestamp ? ["timestamp-query"] : [])], // ...: conditional add
});
// const device = await adapter?.requestDevice();
if (!device) {
  fail("Fatal error: Device does not support WebGPU.");
}
const timing_helper = new TimingHelper(device);

// using webgpu-utils to have one struct for uniforms across all kernels
// Seems kind of weird that struct is a WGSL/GPU struct, not a JS/CPU struct,
//   but that seems to be the only option
// the reason I want a struct is so objects can be named and not "uniforms[5]"
// Q: Is this the right way to do things or is it better to have different
//   uniform structures for each kernel?
const uniforms_code = /* wgsl */ `
        struct MyUniforms {
          ROTATE_CAMERA_SPEED: f32,
          TOGGLE_DURATION: f32,
          WIGGLE_MAGNITUDE: f32,
          WIGGLE_SPEED: f32,
          subdiv_level: u32,
          time: f32,
          timestep: f32,
        };
        @group(0) @binding(0) var<uniform> myUniforms: MyUniforms;
      `;
/* why the @group/@binding? gman@:
 * "It's necessary for them to show up in defs.uniforms or defs.storages. You
 *  can use defs.structs to pull out a struct, separately from a group/binding (I think?)"
 */
const uniforms_defs = makeShaderDataDefinitions(uniforms_code);
const uni = makeStructuredView(uniforms_defs.uniforms.myUniforms);

uni.set({
  ROTATE_CAMERA_SPEED: 0.006, // how quickly camera rotates
  TOGGLE_DURATION: 400.0, // number of timesteps between model toggle
  WIGGLE_MAGNITUDE: 0, // 0.002, //0.025, // how much vertices are perturbed
  WIGGLE_SPEED: 0.05, // how quickly perturbations occur
  subdiv_level: 0,
  time: 0.0,
  timestep: 1.0,
});

const models = {
  model: "pyramid",
};

const model_urls = {
  pyramid:
    "https://gist.githubusercontent.com/jowens/fb3a19db8f4c6271cd9b730b77f7d210/raw/311e98007d600dd10a3425be8312139dc442ca5d/square-pyramid.obj",
  teapot_low:
    "https://graphics.cs.utah.edu/courses/cs6620/fall2013/prj05/teapot-low.obj",
};

const pane = new Pane();
pane.addBinding(models, "model", {
  options: { pyramid: "pyramid", teapot_low: "teapot_low" },
});
pane.addBinding(uni.views.ROTATE_CAMERA_SPEED, 0, {
  min: 0,
  max: 1,
  label: "Camera Rotation Speed",
});
pane.addBinding(uni.views.TOGGLE_DURATION, 0, {
  min: 0,
  max: 1000,
  label: "Toggle Duration",
});
pane.addBinding(uni.views.WIGGLE_MAGNITUDE, 0, {
  min: 0,
  max: 0.1,
  label: "Wiggle Magnitude",
});
pane.addBinding(uni.views.WIGGLE_SPEED, 0, {
  min: 0,
  max: 1,
  label: "Wiggle Speed",
});
pane.addBinding(uni.views.subdiv_level, 0, {
  min: 0,
  max: 1,
  step: 1,
  label: "Subdiv Level",
});

const WORKGROUP_SIZE = 64;

const uniforms_buffer = device.createBuffer({
  size: uni.arrayBuffer.byteLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});

const canvas = document.querySelector("canvas");
const context = canvas.getContext("webgpu");
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({
  device: device,
  format: canvasFormat,
});

/** The following tables are precomputed (on the CPU): Niessner 2012:
 * "Feature-adaptive rendering involves a CPU preprocessing step, as well as a
 * GPU runtime component. Input to our algorithm is a base control mesh
 * consisting of vertices and faces, along with optional data consisting
 * of semisharp crease edge tags and hierarchical details. In the CPU
 * preprocessing stage, we use these data to construct tables containing
 * control mesh indices that drive our feature adaptive subdivision process.
 * Since these subdivision tables implicitly encode mesh connectivity, no
 * auxiliary data structures are needed for this purpose. A unique table
 * is constructed for each level of subdivision up to a prescribed maximum,
 * as well as final patch control point index buffers as described in
 * Section 3.2. The base mesh, subdivision tables, and patch index data are
 * uploaded to the GPU, one time only, for subsequent runtime processing.
 * The output of this phase depends only on the topology of the base mesh,
 * crease edges, and hierarchical detail; it is independent of the geometric
 * location of the control points." */

// square pyramid
const objurl1 =
  "https://gist.githubusercontent.com/jowens/ccd142c4d17e6c188c5105a1881561bf/raw/26e58cb754d1dfb8c30c86d33e0c21497c2167e8/square-pyramid.obj";
// diamond
const objurl2 =
  "https://gist.githubusercontent.com/jowens/ebe82add66adfee31fe49579963c515d/raw/2046cff529575615e32a283a9ca2b4e44f3a13d2/diamond.obj";
// teddy
const objurl3 =
  "https://gist.githubusercontent.com/jowens/d49b13c7f847bda5ffc36d2166888b5f/raw/2756e4e3c5be3b2cce35244c961f462411cefaef/teddy.obj";
// al
const objurl4 =
  "https://gist.githubusercontent.com/jowens/360d591b8484958cf1c5b015c96c0958/raw/6390f2a2c720d378d1aa77baba7605c67d40e2e4/al.obj";
// teapot-lower
const objurl5 =
  "https://gist.githubusercontent.com/jowens/508d6d7f70b33010508f3c679abd61ff/raw/0315c1d585a63687034ae4deecb5b49b8d653017/teapot-lower.obj";
// stanford-teapot
const objurl6 =
  "https://gist.githubusercontent.com/jowens/5f7bc872317b5fd5f7d72827967f1c9d/raw/1f846ee3229297520dd855b199d21717e30af91b/stanford-teapot.obj";

const mesh = await urlToMesh(objurl4);
console.log(mesh);

const vertices_size = mesh.level_base_ptr[1].v + mesh.level_count[1].v;
const vertices_object_size = 4; // float4s (but ignore w coord for now)
const normals_object_size = 4; // float4s (but ignore w coord for now)
// float3s were fraught with peril (padding)
const vertices = new Float32Array(vertices_size * vertices_object_size);
// vertex_normals is uninitialized; it's instead set in a kernel
const vertex_normals = new Float32Array(vertices_size * normals_object_size);

/* populate vertices from mesh data structure */
for (let i = 0; i < mesh.level_count[0].v * vertices_object_size; i++) {
  vertices[i] = mesh.vertices[i];
}

// Q: Is a flattened 1D array the right way to represent base faces?
// should it instead be a 2D array, [face][vertex]?
// i am guessing flattened data structures (like this one) are preferred
const base_faces = new Uint32Array(mesh.faces);

// the following is manually generated tri indexes from base_faces
//   and subdiv_1_faces
// TODO: this could totally be generated programmatically from
//   base_faces plus base_face_valence
// prettier-ignore
const triangle_indices = new Uint32Array(mesh.triangles);
const base_triangles_count = mesh.level_count[0].t;
const subdiv_1_triangles_count = mesh.level_count[1].t;
console.assert(
  triangle_indices.length / 3 ==
    base_triangles_count + subdiv_1_triangles_count,
  "triangle count should be sum of base and subdiv_1 triangle counts"
);
const facet_normals = new Float32Array(
  triangle_indices.length * normals_object_size
);

const base_face_valence = new Uint32Array(mesh.face_valence);
// base_face_offset is exclusive_scan('+', base_face_valence)
// TODO: compute that scan in a compute shader
const base_face_offset = new Uint32Array(mesh.face_offset);
const base_faces_count = base_face_valence.length;

const base_edges = new Uint32Array(mesh.edges);
const edges_object_size = 4; // (two faces, two edges)
const base_edges_count = base_edges.length / edges_object_size;

const base_vertex_valence = new Uint32Array(mesh.vertex_valence);
// base_vertex_offset is 2 * exclusive_scan('+', base_vertex_valence)
// TODO: compute that scan in a compute shader
const base_vertex_offset = new Uint32Array(mesh.vertex_offset);
const base_vertex_count = base_vertex_valence.length;
const base_vertex_index = new Uint32Array(mesh.vertex_index);
const base_vertices = new Uint32Array(mesh.base_vertices.flat());

const perturb_input_vertices_module = device.createShaderModule({
  label: "perturb input vertices module",
  code: /* wgsl */ `
                    ${uniforms_code} /* this specifies @group(0) @binding(0) */
                    /* input + output */
                    @group(0) @binding(1) var<storage, read_write> vertices: array<vec3f>;
                    @compute @workgroup_size(${WORKGROUP_SIZE}) fn perturbInputVerticesKernel(
                             @builtin(global_invocation_id) id: vec3u) {
                      let i = id.x;
                      if (i < ${mesh.level_count[0].v}) {
                        let t = myUniforms.time * myUniforms.WIGGLE_SPEED;
                        let stepsize = myUniforms.WIGGLE_MAGNITUDE;
                        let angle_start = f32(i);
                        /* philosophy of animating base vertices:
                         *
                         * - vertex should not move in aggregate over time
                         * - each vertex should move ~differently
                         *
                         * design: each vertex moves in a "random" direction by a fixed amt
                         *         starting direction differs per vertex ("angle_start")
                         *         movements cancel each other out over time
                         */
                        vertices[i] += vec3(stepsize * cos(angle_start + t),
                                            stepsize * sin(angle_start + t),
                                            stepsize * 0.5 * sin(angle_start + t));
                      }
                    }
                  `,
});

/** (1) Calculation of face points
 * Number of faces: base_face_valence.length == base_faces_count
 * for each face: new face point = centroid(vertices of current face)
 * Pseudocode:   (note math operations are on vec3f's)
 * parallel for i in [0 .. base_face_valence.length]:
 *   new_faces[i] = [0,0,0]
 *   for j in [base_face_offset[i] .. base_face_offset[i] + base_face_valence[i]]:
 *     new_faces[i] += vertices[base_faces[j]
 *   new_faces[i] /= base_face_valence[i]
 */
console.log("face pts write_ptr: ", mesh.level_base_ptr[1].f);
const face_points_module = device.createShaderModule({
  label: "face points module",
  code: /* wgsl */ `
                    /* input + output */
                    @group(0) @binding(0) var<storage, read_write> vertices: array<vec3f>;
                            /* input */
                    @group(0) @binding(1) var<storage, read> base_faces: array<u32>;
                    @group(0) @binding(2) var<storage, read> base_face_offset: array<u32>;
                    @group(0) @binding(3) var<storage, read> base_face_valence: array<u32>;

                    /** Niessner 2012:
                      * "The face kernel requires two buffers: one index buffer, whose
                      * entries are the vertex buffer indices for each vertex of the face; a
                      * second buffer stores the valence of the face along with an offset
                      * into the index buffer for the first vertex of each face."
                      *
                      * implementation above: "index buffer" is base_faces
                      *                       "valence of the face" is base_face_valence
                      *                       "offset into the index buffer" is base_face_offset
                      */

                    @compute @workgroup_size(${WORKGROUP_SIZE}) fn facePointsKernel(
                      @builtin(global_invocation_id) id: vec3u) {
                      let i = id.x;
                      if (i < ${mesh.level_count[1].f}) {
                        /* TODO: exit if my index is larger than the size of the input */

                        let out = i + ${mesh.level_base_ptr[1].f};
                        vertices[out] = vec3f(0,0,0);
                        for (var j: u32 = base_face_offset[i]; j < base_face_offset[i] + base_face_valence[i]; j++) {
                          let face_vertex = base_faces[j];
                          vertices[out] += vertices[face_vertex];
                        }
                        vertices[out] /= f32(base_face_valence[i]);
                      }
                      // TODO: decide on vec3f or vec4f and set w if so
                    }
                  `,
});

/** output vertices from face kernel, for debugging:
 * 5 | [-0.6666666865348816, 0, 0.3333333432674408, 0]
 * 6 | [0, -0.6666666865348816, 0.3333333432674408, 0]
 * 7 | [0.6666666865348816, 0, 0.3333333432674408, 0]
 * 8 | [0, 0.6666666865348816, 0.3333333432674408, 0]
 * 9 | [0, 0, 0, 0]
 */

/** (2) Calculation of edge points
 * Number of edges: base_edges.length
 * for each edge: new edge point = average(2 neighboring face points, 2 endpoints of edge)
 * Pseudocode:   (note math operations are on vec3f's)
 * parallel for i in [0 .. ?.length]:
 *   new_edges[i] = 0.25 * ( vertices[edge_id] + vertices[edge_id + 1] +
 *                           vertices[edge_id + 2] + vertices[edge_id + 3])
 */

console.log("edge pts write_ptr: ", mesh.level_base_ptr[1].e);
const edge_points_module = device.createShaderModule({
  label: "edge points module",
  code: /* wgsl */ `
                    /* input + output */
                    @group(0) @binding(0) var<storage, read_write> vertices: array<vec3f>;
                    /* input */
                    @group(0) @binding(1) var<storage, read> base_edges: array<vec4u>;

                    /** "Since a single (non-boundary) edge always has two incident faces and vertices,
                     * the edge kernel needs a buffer for the indices of these entities."
                     *
                     * implementation above: "a buffer for the indices of these entities" is base_edges
                     */

                    @compute @workgroup_size(${WORKGROUP_SIZE}) fn edge_points_kernel(
                      @builtin(global_invocation_id) id: vec3u) {
                        let i = id.x;
                        if (i < ${mesh.level_count[1].e}) {
                          let out = i + ${mesh.level_base_ptr[1].e};
                          let edge_id = i;
                          vertices[out] = vec3f(0,0,0);
                          for (var j: u32 = 0; j < 4; j++) {
                            vertices[out] += vertices[base_edges[edge_id][j]];
                          }
                          vertices[out] *= 0.25;
                        }
                      }
                  `,
});

/** output "edge" vertices from edge kernel, for debugging
 * 10 | -0.4166666865348816, -0.4166666865348816, 0.4166666865348816, 0
 * 11 | -0.4166666865348816, 0.4166666865348816, 0.4166666865348816, 0
 * 12 | -0.6666666865348816, 0, 0.0833333358168602, 0
 * 13 | 0.4166666865348816, -0.4166666865348816, 0.4166666865348816, 0
 * 14 | 0, -0.6666666865348816, 0.0833333358168602, 0
 * 15 | 0.4166666865348816, 0.4166666865348816, 0.4166666865348816, 0
 * 16 | 0.6666666865348816, 0, 0.0833333358168602, 0
 * 17 | 0, 0.6666666865348816, 0.0833333358168602, 0
 */

/** (3) Calculation of vertex points
 * This is more involved. References:
 * - https://www.rorydriscoll.com/2008/08/01/catmull-clark-subdivision-the-basics/
 * - https://en.wikipedia.org/wiki/Catmull%E2%80%93Clark_subdivision_surface
 * Big picture:
 * - n is valence of this point
 * - F is the average of all neighboring faces (new face points)
 * - Ve is the average of the other endpoint of all incident edges
 *   - The actual math is "midpoint of all incident edges", but one end of all
 *     those edges is just V (below), so we lump that contribution into the V term
 *   - F and Ve are just listed in the base_vertices table
 * - V is this vertex
 *   - Output is (F + Ve + (n-2) V) / n
 * - If F and Ve points are f_0, f_1, Ve_0, ...:
 *   - Output is [(f_0 + f_1 + ... + Ve_0 + Ve_1 + ...) / n _ (n-2) V] / n
 * Number of vertex points: base_vertex_valence.length
 * Pseudocode:   (note math operations are on vec3f's)
 * parallel for i in [0 .. base_vertex_valence.length]:
 *   new_vertex[i] = [0,0,0]
 *   valence = base_vertex_valence[i]
 *   for j in [base_vertex_offset[i] .. base_vertex_offset[i] + base_vertex_valence[i]]:
 *     new_vertex[i] += vertices[base_vertices[j]]
 *   new_vertex[i] /= base_vertex_valence[i]
 *   new_vertex[i] += (n-2) * base_vertex_index[i]
 *   new_vertex[i] /= base_vertex_valence[i]
 */

console.log("vertex pts write_ptr: ", mesh.level_base_ptr[1].v);
const vertex_points_module = device.createShaderModule({
  label: "vertex points module",
  code: /* wgsl */ `
                    /* input + output */
                    @group(0) @binding(0) var<storage, read_write> vertices: array<vec3f>;
                    /* input */
                    @group(0) @binding(1) var<storage, read> base_vertices: array<u32>;
                    @group(0) @binding(2) var<storage, read> base_vertex_offset: array<u32>;
                    @group(0) @binding(3) var<storage, read> base_vertex_valence: array<u32>;
                    @group(0) @binding(4) var<storage, read> base_vertex_index: array<u32>;

                    /** "We use an index buffer containing the indices of the incident edge and
                     * vertex points."
                     *
                     * implementation above: "a buffer for the indices of these entities" is base_vertices
                     */

                    @compute @workgroup_size(${WORKGROUP_SIZE}) fn vertex_points_kernel(
                      @builtin(global_invocation_id) id: vec3u) {
                        let i = id.x;
                        if (i < ${mesh.level_count[1].v}) {
                          let out = i + ${mesh.level_base_ptr[1].v};
                          let valence = base_vertex_valence[i];
                          vertices[out] = vec3f(0,0,0);
                          for (var j: u32 = base_vertex_offset[i]; j < base_vertex_offset[i] + 2 * base_vertex_valence[i]; j++) {
                            let base_vertex = base_vertices[j];
                            vertices[out] += vertices[base_vertex];
                          }
                          vertices[out] /= f32(valence);
                          vertices[out] += f32(valence - 2) * vertices[base_vertex_index[i]];
                          vertices[out] /= f32(valence);
                          // TODO: decide on vec3f or vec4f and set w if so
                      }
                    }
                  `,
});

/** output vertices from vertex kernel, for debugging
 * 18 | -3.725290298461914e-9, 0, 0.5833333134651184, 0
 * 19 | -0.40740740299224854, 0.40740740299224854, 0.18518519401550293, 0
 * 20 | -0.40740740299224854, -0.40740740299224854, 0.18518519401550293, 0
 * 21 | 0.40740740299224854, -0.40740740299224854, 0.18518519401550293, 0
 * 22 | 0.40740740299224854, 0.40740740299224854, 0.18518519401550293, 0
 */

const facet_normals_module = device.createShaderModule({
  label: "compute facet normals module",
  code: /* wgsl */ `
                    /* output */
                    @group(0) @binding(0) var<storage, read_write> facet_normals: array<vec3f>;
                    /* input */
                    @group(0) @binding(1) var<storage, read> vertices: array<vec3f>;
                    @group(0) @binding(2) var<storage, read> triangle_indices: array<u32>;

                     /** Algorithm:
                      * For tri in all triangles:
                      *   Fetch all 3 vertices of tri
                      *   Compute normalize(cross(v1-v0, v2-v0))
                      *   For each vertex in tri:
                      *     Atomically add it to vertex_normals[vertex]
                      *     /* Can't do this! No f32 atomics */
                      * For vertex in all vertices:
                      *   Normalize vertex_normals[vertex]
                      *
                      * OK, so we can't do this approach w/o f32 atomics
                      * So we will instead convert this scatter to gather
                      * This is wasteful; every vertex will walk the entire
                      *   index array looking for matches.
                      * Could alternately build a mapping of {vtx->facet}
                      *
                      * (1) For tri in all triangles:
                      *   Fetch all 3 vertices of tri
                      *   Compute normalize(cross(v1-v0, v2-v0))
                      *   Store that vector as a facet normal
                      * (2) For vertex in all vertices:
                      *   normal[vertex] = (0,0,0)
                      *   For tri in all triangles:
                      *     // note expensive doubly-nested loop!
                      *     if my vertex is in that triangle:
                      *       normal[vertex] += facet_normal[tri]
                      *   normalize(normal[vertex])
                      */
                    @compute @workgroup_size(${WORKGROUP_SIZE}) fn facet_normals_kernel(
                      @builtin(global_invocation_id) id: vec3u) {
                        let tri = id.x;
                        if (tri < arrayLength(&facet_normals)) {
                          /* note triangle_indices is u32 not vec3, do math accordingly */
                          let v0: vec3f = vertices[triangle_indices[tri * 3]];
                          let v1: vec3f = vertices[triangle_indices[tri * 3 + 1]];
                          let v2: vec3f = vertices[triangle_indices[tri * 3 + 2]];
                          facet_normals[tri] = normalize(cross(v1-v0, v2-v0));
                        }
                      }
                  `,
});

const vertex_normals_module = device.createShaderModule({
  label: "compute vertex normals module",
  code: /* wgsl */ `
                    /* output */
                    @group(0) @binding(0) var<storage, read_write> vertex_normals: array<vec3f>;
                    /* input */
                    @group(0) @binding(1) var<storage, read> facet_normals: array<vec3f>;
                    @group(0) @binding(2) var<storage, read> triangle_indices: array<u32>;

                    /* see facet_normals_module for algorithm */

                    @compute @workgroup_size(${WORKGROUP_SIZE}) fn vertex_normals_kernel(
                      @builtin(global_invocation_id) id: vec3u) {
                        let vtx = id.x;
                        if (vtx < arrayLength(&vertex_normals)) {
                          vertex_normals[vtx] = vec3f(0, 0, 0);
                          /* note triangle_indices is u32 not vec3, do math accordingly */
                          for (var tri: u32 = 0; tri < arrayLength(&triangle_indices) / 3; tri++) {
                            for (var tri_vtx: u32 = 0; tri_vtx < 3; tri_vtx++) { /* unroll */
                              if (vtx == triangle_indices[tri * 3 + tri_vtx]) {
                                vertex_normals[vtx] += facet_normals[tri];
                              }
                            }
                          }
                          vertex_normals[vtx] = normalize(vertex_normals[vtx]);
                        }
                    }
                  `,
});

const render_module = device.createShaderModule({
  label: "render module",
  code: /* wgsl */ `
                    struct VertexInput {
                      @location(0) pos: vec4f,
                      @location(1) vertex_normals: vec3f,
                      @builtin(vertex_index) vertex_index: u32,
                    };

                    struct VertexOutput {
                      @builtin(position) pos: vec4f,
                      @location(0) color: vec4f,
                    };

                    // https://webgpu.github.io/webgpu-samples/?sample=rotatingCube#basic.vert.wgsl
                    struct Uniforms {
                      modelViewProjectionMatrix : mat4x4f,
                    }
                    @binding(0) @group(0) var<uniform> uniforms : Uniforms;

                    @vertex
                    fn vertex_main(@location(0) pos: vec4f,
                                  @location(1) norm: vec3f,
                                  @builtin(vertex_index) vertex_index: u32) -> VertexOutput {
                      var output: VertexOutput;
                      output.pos = uniforms.modelViewProjectionMatrix * pos;
                      output.color = vec4f( // this generates 64 different colors
                        0.35 + select(0, 0.6, (vertex_index & 1) != 0) - select(0, 0.3, (vertex_index & 8) != 0),
                        0.35 + select(0, 0.6, (vertex_index & 2) != 0) - select(0, 0.3, (vertex_index & 16) != 0),
                        0.35 + select(0, 0.6, (vertex_index & 4) != 0) - select(0, 0.3, (vertex_index & 32) != 0),
                        0.75 /* partial transparency might aid debugging */);
                      /* let's try "lighting", in model space */
                      /* this is just a dot product with the infinite white light at (1,1,1) */
                      /* it's just choosing the normal vector as the color, scaled to [0,1] */
                      // output.color = vec4f(norm.x, norm.y, norm.z, 0.75);
                      output.color = vec4f(0.5*(norm.x+1), 0.5*(norm.y+1), 0.5*(norm.z+1), 0.75);
                      return output;
                    }

                    @fragment
                    fn fragment_main(input: VertexOutput) -> @location(0) vec4f {
                      return input.color;
                    }
                  `,
});

const perturb_pipeline = device.createComputePipeline({
  label: "perturb input vertices compute pipeline",
  layout: "auto",
  compute: {
    module: perturb_input_vertices_module,
  },
});

const face_pipeline = device.createComputePipeline({
  label: "face points compute pipeline",
  layout: "auto",
  compute: {
    module: face_points_module,
  },
});

const edge_pipeline = device.createComputePipeline({
  label: "edge points compute pipeline",
  layout: "auto",
  compute: {
    module: edge_points_module,
  },
});

const vertex_pipeline = device.createComputePipeline({
  label: "vertex points compute pipeline",
  layout: "auto",
  compute: {
    module: vertex_points_module,
  },
});

const facet_normals_pipeline = device.createComputePipeline({
  label: "facet normals compute pipeline",
  layout: "auto",
  compute: {
    module: facet_normals_module,
  },
});

const vertex_normals_pipeline = device.createComputePipeline({
  label: "vertex normals compute pipeline",
  layout: "auto",
  compute: {
    module: vertex_normals_module,
  },
});

const depth_texture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: "depth24plus",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

const render_pipeline = device.createRenderPipeline({
  label: "render pipeline",
  layout: "auto",
  vertex: {
    module: render_module,
    entryPoint: "vertex_main",
    buffers: [
      {
        // Buffer 0
        arrayStride: 16,
        attributes: [
          {
            shaderLocation: 0, // position
            format: "float32x3",
            offset: 0,
          },
          {
            shaderLocation: 1, // normals
            format: "float32x3",
            offset: 0,
          },
        ],
      },
      // could add more buffers here
    ],
  },
  fragment: {
    module: render_module,
    entryPoint: "fragment_main",
    targets: [
      {
        format: canvasFormat,
      },
    ],
  },
  depthStencil: {
    depthWriteEnabled: true,
    depthCompare: "less",
    format: "depth24plus",
  },
});

// create buffers on the GPU to hold data
// read-only inputs:
const base_faces_buffer = device.createBuffer({
  label: "base faces buffer",
  size: base_faces.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(base_faces_buffer, 0, base_faces);

const base_edges_buffer = device.createBuffer({
  label: "base edges buffer",
  size: base_edges.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(base_edges_buffer, 0, base_edges);

const base_face_offset_buffer = device.createBuffer({
  label: "base face offset",
  size: base_face_offset.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(base_face_offset_buffer, 0, base_face_offset);

const base_face_valence_buffer = device.createBuffer({
  label: "base face valence",
  size: base_face_valence.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(base_face_valence_buffer, 0, base_face_valence);

const base_vertices_buffer = device.createBuffer({
  label: "base vertices buffer",
  size: base_vertices.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(base_vertices_buffer, 0, base_vertices);

const base_vertex_offset_buffer = device.createBuffer({
  label: "base vertex offset buffer",
  size: base_vertex_offset.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(base_vertex_offset_buffer, 0, base_vertex_offset);

const base_vertex_valence_buffer = device.createBuffer({
  label: "base vertex valence buffer",
  size: base_vertex_valence.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(base_vertex_valence_buffer, 0, base_vertex_valence);

const base_vertex_index_buffer = device.createBuffer({
  label: "base vertex index buffer",
  size: base_vertex_index.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(base_vertex_index_buffer, 0, base_vertex_index);

const triangle_indices_buffer = device.createBuffer({
  label: "triangle indices buffer",
  size: triangle_indices.byteLength,
  usage:
    GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(triangle_indices_buffer, 0, triangle_indices);

const mvx_length = 4 * 16; /* float32 4x4 matrix */
const mvx_buffer = device.createBuffer({
  label: "modelview + transformation matrix uniform buffer",
  size: mvx_length,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
// write happens at the start of every frame

// vertex buffer is both input and output
const vertices_buffer = device.createBuffer({
  label: "vertex buffer",
  size: vertices.byteLength,
  usage:
    GPUBufferUsage.STORAGE |
    GPUBufferUsage.VERTEX |
    GPUBufferUsage.COPY_DST |
    GPUBufferUsage.COPY_SRC,
});
device.queue.writeBuffer(vertices_buffer, 0, vertices);

const facet_normals_buffer = device.createBuffer({
  label: "facet normals buffer",
  size: facet_normals.byteLength,
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
device.queue.writeBuffer(facet_normals_buffer, 0, facet_normals);

const vertex_normals_buffer = device.createBuffer({
  label: "vertex normals buffer",
  size: vertex_normals.byteLength,
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
device.queue.writeBuffer(vertex_normals_buffer, 0, vertex_normals);

/** and the mappable output buffers (I believe that "mappable" is the only way to read from GPU->CPU) */
const mappable_vertices_result_buffer = device.createBuffer({
  label: "mappable vertices result buffer",
  size: vertices.byteLength,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});
const mappable_facet_normals_result_buffer = device.createBuffer({
  label: "mappable facet normals result buffer",
  size: facet_normals.byteLength,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});
const mappable_vertex_normals_result_buffer = device.createBuffer({
  label: "mappable vertex normals result buffer",
  size: vertex_normals.byteLength,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

/** Set up bindGroups per compute kernel to tell the shader which buffers to use */
const perturb_bind_group = device.createBindGroup({
  label: "bindGroup for perturb input vertices kernel",
  layout: perturb_pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: uniforms_buffer } },
    { binding: 1, resource: { buffer: vertices_buffer } },
  ],
});

const face_bind_group = device.createBindGroup({
  label: "bindGroup for face kernel",
  layout: face_pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: vertices_buffer } },
    { binding: 1, resource: { buffer: base_faces_buffer } },
    { binding: 2, resource: { buffer: base_face_offset_buffer } },
    { binding: 3, resource: { buffer: base_face_valence_buffer } },
  ],
});

const edge_bind_group = device.createBindGroup({
  label: "bindGroup for edge kernel",
  layout: edge_pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: vertices_buffer } },
    { binding: 1, resource: { buffer: base_edges_buffer } },
  ],
});

const vertex_bind_group = device.createBindGroup({
  label: "bindGroup for vertex kernel",
  layout: vertex_pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: vertices_buffer } },
    { binding: 1, resource: { buffer: base_vertices_buffer } },
    { binding: 2, resource: { buffer: base_vertex_offset_buffer } },
    { binding: 3, resource: { buffer: base_vertex_valence_buffer } },
    { binding: 4, resource: { buffer: base_vertex_index_buffer } },
  ],
});

const facet_normals_bind_group = device.createBindGroup({
  label: "bindGroup for computing facet normals",
  layout: facet_normals_pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: facet_normals_buffer } },
    { binding: 1, resource: { buffer: vertices_buffer } },
    { binding: 2, resource: { buffer: triangle_indices_buffer } },
  ],
});

const vertex_normals_bind_group = device.createBindGroup({
  label: "bindGroup for computing vertex normals",
  layout: vertex_normals_pipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: vertex_normals_buffer } },
    { binding: 1, resource: { buffer: facet_normals_buffer } },
    { binding: 2, resource: { buffer: triangle_indices_buffer } },
  ],
});

const render_bind_group = device.createBindGroup({
  label: "bindGroup for rendering kernel",
  layout: render_pipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: mvx_buffer } }],
});

const aspect = canvas.width / canvas.height;
const projection_matrix = mat4.perspective((2 * Math.PI) / 5, aspect, 1, 100.0);
const model_view_projection_matrix = mat4.create();

function get_transformation_matrix() {
  /* this view matrix simply does some time-dependent rotation */
  /* of course adding camera control would be better */
  const view_matrix = mat4.identity();
  let now = uni.views.time[0] * uni.views.ROTATE_CAMERA_SPEED[0];
  mat4.translate(view_matrix, vec3.fromValues(0, 0, -3), view_matrix);
  mat4.rotateZ(view_matrix, now, view_matrix);
  mat4.rotateY(view_matrix, now, view_matrix);
  mat4.rotateX(view_matrix, now, view_matrix);
  mat4.multiply(projection_matrix, view_matrix, model_view_projection_matrix);
  return model_view_projection_matrix;
}

/** there are a TON of things in this frame() call that
 * can probably be moved outside the call
 *
 * I need to know what those are!
 */
async function frame() {
  /**
   * Definitely there's two things that need to go CPU->GPU every frame
   *
   * (1) Uniforms, since they can be altered by the user at runtime
   * in the pane (also time is here)
   * (2) Transformation matrix, since it changes every frame
   */
  device.queue.writeBuffer(uniforms_buffer, 0, uni.arrayBuffer);

  const transformation_matrix = get_transformation_matrix();
  device.queue.writeBuffer(
    mvx_buffer,
    0,
    transformation_matrix.buffer,
    transformation_matrix.byteOffset,
    transformation_matrix.byteLength
  );

  // Encode commands to do the computation
  const encoder = device.createCommandEncoder({
    label:
      "overall computation (perturb, face, edge, vertex, normals) + graphics encoder",
  });

  const perturb_pass = encoder.beginComputePass({
    label: "perturb input vertices kernel compute pass",
  });
  perturb_pass.setPipeline(perturb_pipeline);
  perturb_pass.setBindGroup(0, perturb_bind_group);
  perturb_pass.dispatchWorkgroups(
    Math.ceil(mesh.level_count[0].v / WORKGROUP_SIZE)
  );
  perturb_pass.end();

  const face_pass = encoder.beginComputePass({
    label: "face kernel compute pass",
  });
  face_pass.setPipeline(face_pipeline);
  face_pass.setBindGroup(0, face_bind_group);
  face_pass.dispatchWorkgroups(
    Math.ceil(mesh.level_count[1].f / WORKGROUP_SIZE)
  );
  face_pass.end();

  const edge_pass = encoder.beginComputePass({
    label: "edge kernel compute pass",
  });
  edge_pass.setPipeline(edge_pipeline);
  edge_pass.setBindGroup(0, edge_bind_group);
  edge_pass.dispatchWorkgroups(
    Math.ceil(mesh.level_count[1].e / WORKGROUP_SIZE)
  );
  edge_pass.end();

  const vertex_pass = encoder.beginComputePass({
    label: "vertex kernel compute pass",
  });
  vertex_pass.setPipeline(vertex_pipeline);
  vertex_pass.setBindGroup(0, vertex_bind_group);
  vertex_pass.dispatchWorkgroups(
    Math.ceil(mesh.level_count[1].v / WORKGROUP_SIZE)
  );
  vertex_pass.end();

  const facet_normals_pass = timing_helper.beginComputePass(encoder, {
    label: "facet normals compute pass",
  });
  facet_normals_pass.setPipeline(facet_normals_pipeline);
  facet_normals_pass.setBindGroup(0, facet_normals_bind_group);
  facet_normals_pass.dispatchWorkgroups(
    Math.ceil((mesh.level_count[0].t + mesh.level_count[1].t) / WORKGROUP_SIZE)
  );
  facet_normals_pass.end();

  const vertex_normals_pass = encoder.beginComputePass({
    label: "vertex normals compute pass",
  });
  vertex_normals_pass.setPipeline(vertex_normals_pipeline);
  vertex_normals_pass.setBindGroup(0, vertex_normals_bind_group);
  vertex_normals_pass.dispatchWorkgroups(
    Math.ceil(vertices_size / WORKGROUP_SIZE)
  );
  vertex_normals_pass.end();

  // Encode a command to copy the results to a mappable buffer.
  // this is (from, to)
  encoder.copyBufferToBuffer(
    vertices_buffer,
    0,
    mappable_vertices_result_buffer,
    0,
    mappable_vertices_result_buffer.size
  );
  encoder.copyBufferToBuffer(
    facet_normals_buffer,
    0,
    mappable_facet_normals_result_buffer,
    0,
    mappable_facet_normals_result_buffer.size
  );
  encoder.copyBufferToBuffer(
    vertex_normals_buffer,
    0,
    mappable_vertex_normals_result_buffer,
    0,
    mappable_vertex_normals_result_buffer.size
  );

  const render_pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0.4, a: 1.0 },
        storeOp: "store",
      },
    ],
    depthStencilAttachment: {
      view: depth_texture.createView(),

      depthClearValue: 1.0,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });

  // Now render those tris.
  render_pass.setPipeline(render_pipeline);
  render_pass.setBindGroup(0, render_bind_group);
  render_pass.setVertexBuffer(0, vertices_buffer);
  let start_idx = -1;
  let end_idx = -1;
  const now = uni.views.time[0];

  render_pass.setIndexBuffer(triangle_indices_buffer, "uint32");
  // next line switches every TOGGLE_DURATION frames
  // switch ((now / uni.views.TOGGLE_DURATION) & 1) {
  // instead switch explicitly on subdiv_level
  // clearly this math can be much much simpler
  switch (uni.views.subdiv_level[0]) {
    case 0 /* draws tris [0, base_triangles_count) */:
      start_idx = 0;
      end_idx = mesh.level_count[0].t * 3;
      break;
    case 1 /* draws tris [base_triangles_count, base + subdiv counts) */:
      start_idx = mesh.level_count[0].t * 3;
      end_idx = (mesh.level_count[0].t + mesh.level_count[1].t) * 3;
      break;
  }
  render_pass.drawIndexed(
    end_idx - start_idx /* count */,
    1 /* instance */,
    start_idx /* start */
  );

  // End the render pass and submit the command buffer
  render_pass.end();

  // Finish encoding and submit the commands
  const command_buffer = encoder.finish();
  device.queue.submit([command_buffer]);

  // Read the results
  await mappable_vertices_result_buffer.mapAsync(GPUMapMode.READ);
  const vertices_result = new Float32Array(
    mappable_vertices_result_buffer.getMappedRange().slice()
  );
  mappable_vertices_result_buffer.unmap();
  await mappable_facet_normals_result_buffer.mapAsync(GPUMapMode.READ);
  const facet_normals_result = new Float32Array(
    mappable_facet_normals_result_buffer.getMappedRange().slice()
  );
  mappable_facet_normals_result_buffer.unmap();
  await mappable_vertex_normals_result_buffer.mapAsync(GPUMapMode.READ);
  const vertex_normals_result = new Float32Array(
    mappable_vertex_normals_result_buffer.getMappedRange().slice()
  );
  mappable_vertex_normals_result_buffer.unmap();

  /* is this correct for getting timing info? */
  timing_helper.getResult().then((res) => {
    // console.log("timing helper result", res);
  });

  // console.log("vertex buffer", vertices_result);
  // console.log("time", uni.views.time[0]);
  uni.views.time[0] = uni.views.time[0] + uni.views.timestep[0];
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

function fail(msg) {
  // eslint-disable-next-line no-alert
  alert(msg);
}
