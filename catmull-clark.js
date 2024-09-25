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
const timingHelper = new TimingHelper(device);

// we can set runtime params from the input URL
const urlParams = new URL(window.location.href).searchParams;
const debug = urlParams.get("debug"); // string or undefined
// if we want more:
//   Object.fromEntries(new URL(window.location.href).searchParams.entries());
// if url is 'https://foo.com/bar.html?abc=123&def=456&xyz=banana` then params is
// { abc: '123', def: '456', xyz: 'banana' }   // notice they are strings, not numbers.

// using webgpu-utils to have one struct for uniforms across all kernels
// Seems kind of weird that struct is a WGSL/GPU struct, not a JS/CPU struct,
//   but that seems to be the only option
// the reason I want a struct is so objects can be named and not "uniforms[5]"
// Q: Is this the right way to do things or is it better to have different
//   uniform structures for each kernel?
const uniformsCode = /* wgsl */ `
        const MAX_LEVEL = 10;
        struct Level {
          f: u32, e: u32, v: u32, t: u32,
        };
        struct MyUniforms {
          ROTATE_CAMERA_SPEED: f32,
          TOGGLE_DURATION: f32,
          WIGGLE_MAGNITUDE: f32,
          WIGGLE_SPEED: f32,
          subdivLevel: u32,
          @align(16) levelCount: array<Level, MAX_LEVEL>,
          levelBasePtr: array<Level, MAX_LEVEL>,
          time: f32,
          timestep: f32,
        };
        @group(0) @binding(0) var<uniform> myUniforms: MyUniforms;
      `;
/* why the @group/@binding? gman@:
 * "It's necessary for them to show up in defs.uniforms or defs.storages. You
 *  can use defs.structs to pull out a struct, separately from a group/binding (I think?)"
 */
const uniformsDefs = makeShaderDataDefinitions(uniformsCode);
const uni = makeStructuredView(uniformsDefs.uniforms.myUniforms);

uni.set({
  ROTATE_CAMERA_SPEED: 0.006, // how quickly camera rotates
  TOGGLE_DURATION: 400.0, // number of timesteps between model toggle
  WIGGLE_MAGNITUDE: 0, // 0.002, //0.025, // how much vertices are perturbed
  WIGGLE_SPEED: 0.05, // how quickly perturbations occur
  subdivLevel: 0,
  time: 0.0,
  timestep: 1.0,
});

const models = {
  model: "pyramid",
};

const modelUrls = {
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
  max: 0.02,
  label: "Wiggle Magnitude",
});
pane.addBinding(uni.views.WIGGLE_SPEED, 0, {
  min: 0,
  max: 1,
  label: "Wiggle Speed",
});
pane.addBinding(uni.views.subdivLevel, 0, {
  min: 0,
  max: 1,
  step: 1,
  label: "Subdiv Level",
});

const WORKGROUP_SIZE = 64;

const uniformsBuffer = device.createBuffer({
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

const mesh = await urlToMesh(objurl3);
console.log(mesh);

const verticesSize = mesh.levelBasePtr[1].v + mesh.levelCount[1].v;
const verticesObjectSize = 4; // float4s (but ignore w coord for now)
const normalsObjectSize = 4; // float4s (but ignore w coord for now)
// float3s were fraught with peril (padding)
const vertices = new Float32Array(verticesSize * verticesObjectSize);
// vertexNormals is uninitialized; it's instead set in a kernel
const vertexNormals = new Float32Array(verticesSize * normalsObjectSize);

/* populate vertices from mesh data structure */
for (let i = 0; i < mesh.levelCount[0].v * verticesObjectSize; i++) {
  vertices[i] = mesh.vertices[i];
}

// Q: Is a flattened 1D array the right way to represent base faces?
// should it instead be a 2D array, [face][vertex]?
// i am guessing flattened data structures (like this one) are preferred
const baseFaces = new Uint32Array(mesh.faces);
const triangleIndices = new Uint32Array(mesh.triangles);
const facetNormals = new Float32Array(
  triangleIndices.length * normalsObjectSize // normal per tri
);
const baseFaceValence = new Uint32Array(mesh.faceValence);
// baseFaceOffset is exclusive_scan('+', baseFaceValence)
// TODO: compute that scan in a compute shader
const baseFaceOffset = new Uint32Array(mesh.faceOffset);
const baseEdges = new Uint32Array(mesh.edges);
const baseVertexValence = new Uint32Array(mesh.vertexValence);
// baseVertexOffset is 2 * exclusive_scan('+', baseVertexValence)
// TODO: compute that scan in a compute shader
const baseVertexOffset = new Uint32Array(mesh.vertexOffset);
const baseVertexIndex = new Uint32Array(mesh.vertexIndex);
const baseVertices = new Uint32Array(mesh.baseVertices.flat());

const perturbInputVerticesModule = device.createShaderModule({
  label: "perturb input vertices module",
  code: /* wgsl */ `
                    ${uniformsCode} /* this specifies @group(0) @binding(0) */
                    /* input + output */
                    @group(0) @binding(1) var<storage, read_write> vertices: array<vec3f>;
                    @compute @workgroup_size(${WORKGROUP_SIZE}) fn perturbInputVerticesKernel(
                             @builtin(global_invocation_id) id: vec3u) {
                      let i = id.x;
                      if (i < arrayLength(&vertices)) {
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
 * Number of faces: baseFaceValence.length == baseFacesCount
 * for each face: new face point = centroid(vertices of current face)
 * Pseudocode:   (note math operations are on vec3f's)
 * parallel for i in [0 .. baseFaceValence.length]:
 *   newFaces[i] = [0,0,0]
 *   for j in [baseFaceOffset[i] .. baseFaceOffset[i] + baseFaceValence[i]]:
 *     newFaces[i] += vertices[baseFaces[j]
 *   newFaces[i] /= baseFaceValence[i]
 */
console.log("face pts write_ptr: ", mesh.levelBasePtr[1].f);
const facePointsModule = device.createShaderModule({
  label: "face points module",
  code: /* wgsl */ `
                    /* input + output */
                    @group(0) @binding(0) var<storage, read_write> vertices: array<vec3f>;
                            /* input */
                    @group(0) @binding(1) var<storage, read> baseFaces: array<u32>;
                    @group(0) @binding(2) var<storage, read> baseFaceOffset: array<u32>;
                    @group(0) @binding(3) var<storage, read> baseFaceValence: array<u32>;

                    /** Niessner 2012:
                      * "The face kernel requires two buffers: one index buffer, whose
                      * entries are the vertex buffer indices for each vertex of the face; a
                      * second buffer stores the valence of the face along with an offset
                      * into the index buffer for the first vertex of each face."
                      *
                      * implementation above: "index buffer" is baseFaces
                      *                       "valence of the face" is baseFaceValence
                      *                       "offset into the index buffer" is baseFaceOffset
                      */

                    @compute @workgroup_size(${WORKGROUP_SIZE}) fn facePointsKernel(
                      @builtin(global_invocation_id) id: vec3u) {
                      let i = id.x;
                      if (i < ${mesh.levelCount[1].f}) {
                        /* TODO: exit if my index is larger than the size of the input */

                        let out = i + ${mesh.levelBasePtr[1].f};
                        vertices[out] = vec3f(0,0,0);
                        for (var j: u32 = baseFaceOffset[i]; j < baseFaceOffset[i] + baseFaceValence[i]; j++) {
                          let faceVertex = baseFaces[j];
                          vertices[out] += vertices[faceVertex];
                        }
                        vertices[out] /= f32(baseFaceValence[i]);
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
 * Number of edges: baseEdges.length
 * for each edge: new edge point = average(2 neighboring face points, 2 endpoints of edge)
 * Pseudocode:   (note math operations are on vec3f's)
 * parallel for i in [0 .. ?.length]:
 *   newEdges[i] = 0.25 * ( vertices[edgeID] + vertices[edgeID + 1] +
 *                           vertices[edgeID + 2] + vertices[edgeID + 3])
 */

console.log("edge pts write_ptr: ", mesh.levelBasePtr[1].e);
const edgePointsModule = device.createShaderModule({
  label: "edge points module",
  code: /* wgsl */ `
                    /* input + output */
                    @group(0) @binding(0) var<storage, read_write> vertices: array<vec3f>;
                    /* input */
                    @group(0) @binding(1) var<storage, read> baseEdges: array<vec4u>;

                    /** "Since a single (non-boundary) edge always has two incident faces and vertices,
                     * the edge kernel needs a buffer for the indices of these entities."
                     *
                     * implementation above: "a buffer for the indices of these entities" is baseEdges
                     */

                    @compute @workgroup_size(${WORKGROUP_SIZE}) fn edgePointsKernel(
                      @builtin(global_invocation_id) id: vec3u) {
                        let i = id.x;
                        if (i < ${mesh.levelCount[1].e}) {
                          let out = i + ${mesh.levelBasePtr[1].e};
                          let edgeID = i;
                          vertices[out] = vec3f(0,0,0);
                          for (var j: u32 = 0; j < 4; j++) {
                            vertices[out] += vertices[baseEdges[edgeID][j]];
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
 *   - F and Ve are just listed in the baseVertices table
 * - V is this vertex
 *   - Output is (F + Ve + (n-2) V) / n
 * - If F and Ve points are f_0, f1, Ve_0, ...:
 *   - Output is [(f_0 + f1 + ... + Ve_0 + Ve1 + ...) / n _ (n-2) V] / n
 * Number of vertex points: baseVertexValence.length
 * Pseudocode:   (note math operations are on vec3f's)
 * parallel for i in [0 .. baseVertexValence.length]:
 *   newVertex[i] = [0,0,0]
 *   valence = baseVertexValence[i]
 *   for j in [baseVertexOffset[i] .. baseVertexOffset[i] + baseVertexValence[i]]:
 *     newVertex[i] += vertices[baseVertices[j]]
 *   newVertex[i] /= baseVertexValence[i]
 *   newVertex[i] += (n-2) * baseVertexIndex[i]
 *   newVertex[i] /= baseVertexValence[i]
 */

console.log("vertex pts write_ptr: ", mesh.levelBasePtr[1].v);
const vertexPointsModule = device.createShaderModule({
  label: "vertex points module",
  code: /* wgsl */ `
                    /* input + output */
                    @group(0) @binding(0) var<storage, read_write> vertices: array<vec3f>;
                    /* input */
                    @group(0) @binding(1) var<storage, read> baseVertices: array<u32>;
                    @group(0) @binding(2) var<storage, read> baseVertexOffset: array<u32>;
                    @group(0) @binding(3) var<storage, read> baseVertexValence: array<u32>;
                    @group(0) @binding(4) var<storage, read> baseVertexIndex: array<u32>;

                    /** "We use an index buffer containing the indices of the incident edge and
                     * vertex points."
                     *
                     * implementation above: "a buffer for the indices of these entities" is baseVertices
                     */

                    @compute @workgroup_size(${WORKGROUP_SIZE}) fn vertexPointsKernel(
                      @builtin(global_invocation_id) id: vec3u) {
                        let i = id.x;
                        if (i < ${mesh.levelCount[1].v}) {
                          let out = i + ${mesh.levelBasePtr[1].v};
                          let valence = baseVertexValence[i];
                          vertices[out] = vec3f(0,0,0);
                          for (var j: u32 = baseVertexOffset[i]; j < baseVertexOffset[i] + 2 * baseVertexValence[i]; j++) {
                            let baseVertex = baseVertices[j];
                            vertices[out] += vertices[baseVertex];
                          }
                          vertices[out] /= f32(valence);
                          vertices[out] += f32(valence - 2) * vertices[baseVertexIndex[i]];
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

const facetNormalsModule = device.createShaderModule({
  label: "compute facet normals module",
  code: /* wgsl */ `
                    /* output */
                    @group(0) @binding(0) var<storage, read_write> facetNormals: array<vec3f>;
                    /* input */
                    @group(0) @binding(1) var<storage, read> vertices: array<vec3f>;
                    @group(0) @binding(2) var<storage, read> triangleIndices: array<u32>;

                     /** Algorithm:
                      * For tri in all triangles:
                      *   Fetch all 3 vertices of tri
                      *   Compute normalize(cross(v1-v0, v2-v0))
                      *   For each vertex in tri:
                      *     Atomically add it to vertexNormals[vertex]
                      *     /* Can't do this! No f32 atomics */
                      * For vertex in all vertices:
                      *   Normalize vertexNormals[vertex]
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
                    @compute @workgroup_size(${WORKGROUP_SIZE}) fn facetNormalsKernel(
                      @builtin(global_invocation_id) id: vec3u) {
                        let tri = id.x;
                        if (tri < arrayLength(&facetNormals)) {
                          /* note triangleIndices is u32 not vec3, do math accordingly */
                          let v0: vec3f = vertices[triangleIndices[tri * 3]];
                          let v1: vec3f = vertices[triangleIndices[tri * 3 + 1]];
                          let v2: vec3f = vertices[triangleIndices[tri * 3 + 2]];
                          facetNormals[tri] = normalize(cross(v1-v0, v2-v0));
                        }
                      }
                  `,
});

const vertexNormalsModule = device.createShaderModule({
  label: "compute vertex normals module",
  code: /* wgsl */ `
                    /* output */
                    @group(0) @binding(0) var<storage, read_write> vertexNormals: array<vec3f>;
                    /* input */
                    @group(0) @binding(1) var<storage, read> facetNormals: array<vec3f>;
                    @group(0) @binding(2) var<storage, read> triangleIndices: array<u32>;

                    /* see facetNormalsModule for algorithm */

                    @compute @workgroup_size(${WORKGROUP_SIZE}) fn vertexNormalsKernel(
                      @builtin(global_invocation_id) id: vec3u) {
                        let vtx = id.x;
                        if (vtx < arrayLength(&vertexNormals)) {
                          vertexNormals[vtx] = vec3f(0, 0, 0);
                          /* note triangleIndices is u32 not vec3, do math accordingly */
                          for (var tri: u32 = 0; tri < arrayLength(&triangleIndices) / 3; tri++) {
                            for (var triVtx: u32 = 0; triVtx < 3; triVtx++) { /* unroll */
                              if (vtx == triangleIndices[tri * 3 + triVtx]) {
                                vertexNormals[vtx] += facetNormals[tri];
                              }
                            }
                          }
                          vertexNormals[vtx] = normalize(vertexNormals[vtx]);
                        }
                    }
                  `,
});

const renderModule = device.createShaderModule({
  label: "render module",
  code: /* wgsl */ `
                    struct VertexInput {
                      @location(0) pos: vec4f,
                      @location(1) vertexNormals: vec3f,
                      @builtin(vertex_index) vertexIndex: u32,
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
                    fn vertexMain(@location(0) pos: vec4f,
                                  @location(1) norm: vec3f,
                                  @builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
                      var output: VertexOutput;
                      output.pos = uniforms.modelViewProjectionMatrix * pos;
                      output.color = vec4f( // this generates 64 different colors
                        0.35 + select(0, 0.6, (vertexIndex & 1) != 0) - select(0, 0.3, (vertexIndex & 8) != 0),
                        0.35 + select(0, 0.6, (vertexIndex & 2) != 0) - select(0, 0.3, (vertexIndex & 16) != 0),
                        0.35 + select(0, 0.6, (vertexIndex & 4) != 0) - select(0, 0.3, (vertexIndex & 32) != 0),
                        0.75 /* partial transparency might aid debugging */);
                      /* let's try "lighting", in model space */
                      /* this is just a dot product with the infinite white light at (1,1,1) */
                      /* it's just choosing the normal vector as the color, scaled to [0,1] */
                      // output.color = vec4f(norm.x, norm.y, norm.z, 0.75);
                      output.color = vec4f(0.5*(norm.x+1), 0.5*(norm.y+1), 0.5*(norm.z+1), 0.75);
                      return output;
                    }

                    @fragment
                    fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
                      return input.color;
                    }
                  `,
});

const perturbPipeline = device.createComputePipeline({
  label: "perturb input vertices compute pipeline",
  layout: "auto",
  compute: {
    module: perturbInputVerticesModule,
  },
});

const facePipeline = device.createComputePipeline({
  label: "face points compute pipeline",
  layout: "auto",
  compute: {
    module: facePointsModule,
  },
});

const edgePipeline = device.createComputePipeline({
  label: "edge points compute pipeline",
  layout: "auto",
  compute: {
    module: edgePointsModule,
  },
});

const vertexPipeline = device.createComputePipeline({
  label: "vertex points compute pipeline",
  layout: "auto",
  compute: {
    module: vertexPointsModule,
  },
});

const facetNormalsPipeline = device.createComputePipeline({
  label: "facet normals compute pipeline",
  layout: "auto",
  compute: {
    module: facetNormalsModule,
  },
});

const vertexNormalsPipeline = device.createComputePipeline({
  label: "vertex normals compute pipeline",
  layout: "auto",
  compute: {
    module: vertexNormalsModule,
  },
});

const depthTexture = device.createTexture({
  size: [canvas.width, canvas.height],
  format: "depth24plus",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

const renderPipeline = device.createRenderPipeline({
  label: "render pipeline",
  layout: "auto",
  vertex: {
    module: renderModule,
    entryPoint: "vertexMain",
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
    module: renderModule,
    entryPoint: "fragmentMain",
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
const baseFacesBuffer = device.createBuffer({
  label: "base faces buffer",
  size: baseFaces.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(baseFacesBuffer, 0, baseFaces);

const baseEdgesBuffer = device.createBuffer({
  label: "base edges buffer",
  size: baseEdges.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(baseEdgesBuffer, 0, baseEdges);

const baseFaceOffsetBuffer = device.createBuffer({
  label: "base face offset",
  size: baseFaceOffset.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(baseFaceOffsetBuffer, 0, baseFaceOffset);

const baseFaceValenceBuffer = device.createBuffer({
  label: "base face valence",
  size: baseFaceValence.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(baseFaceValenceBuffer, 0, baseFaceValence);

const baseVerticesBuffer = device.createBuffer({
  label: "base vertices buffer",
  size: baseVertices.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(baseVerticesBuffer, 0, baseVertices);

const baseVertexOffsetBuffer = device.createBuffer({
  label: "base vertex offset buffer",
  size: baseVertexOffset.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(baseVertexOffsetBuffer, 0, baseVertexOffset);

const baseVertexValenceBuffer = device.createBuffer({
  label: "base vertex valence buffer",
  size: baseVertexValence.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(baseVertexValenceBuffer, 0, baseVertexValence);

const baseVertexIndexBuffer = device.createBuffer({
  label: "base vertex index buffer",
  size: baseVertexIndex.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(baseVertexIndexBuffer, 0, baseVertexIndex);

const triangleIndicesBuffer = device.createBuffer({
  label: "triangle indices buffer",
  size: triangleIndices.byteLength,
  usage:
    GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
});
device.queue.writeBuffer(triangleIndicesBuffer, 0, triangleIndices);

const mvxLength = 4 * 16; /* float32 4x4 matrix */
const mvxBuffer = device.createBuffer({
  label: "modelview + transformation matrix uniform buffer",
  size: mvxLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
// write happens at the start of every frame

// vertex buffer is both input and output
const verticesBuffer = device.createBuffer({
  label: "vertex buffer",
  size: vertices.byteLength,
  usage:
    GPUBufferUsage.STORAGE |
    GPUBufferUsage.VERTEX |
    GPUBufferUsage.COPY_DST |
    GPUBufferUsage.COPY_SRC,
});
device.queue.writeBuffer(verticesBuffer, 0, vertices);

const facetNormalsBuffer = device.createBuffer({
  label: "facet normals buffer",
  size: facetNormals.byteLength,
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
device.queue.writeBuffer(facetNormalsBuffer, 0, facetNormals);

const vertexNormalsBuffer = device.createBuffer({
  label: "vertex normals buffer",
  size: vertexNormals.byteLength,
  usage:
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
});
device.queue.writeBuffer(vertexNormalsBuffer, 0, vertexNormals);

/** and the mappable output buffers (I believe that "mappable" is the only way to read from GPU->CPU) */
const mappableVerticesResultBuffer = device.createBuffer({
  label: "mappable vertices result buffer",
  size: vertices.byteLength,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});
const mappableFacetNormalsResultBuffer = device.createBuffer({
  label: "mappable facet normals result buffer",
  size: facetNormals.byteLength,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});
const mappableVertexNormalsResultBuffer = device.createBuffer({
  label: "mappable vertex normals result buffer",
  size: vertexNormals.byteLength,
  usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
});

/** Set up bindGroups per compute kernel to tell the shader which buffers to use */
const perturbBindGroup = device.createBindGroup({
  label: "bindGroup for perturb input vertices kernel",
  layout: perturbPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: uniformsBuffer } },
    {
      binding: 1,
      resource: {
        buffer: verticesBuffer,
        offset: 0,
        size: mesh.levelBasePtr[1].v * verticesObjectSize * 4,
        // TODO: Can I compute this size better?
      },
    },
  ],
});

const faceBindGroup = device.createBindGroup({
  label: "bindGroup for face kernel",
  layout: facePipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: verticesBuffer } },
    { binding: 1, resource: { buffer: baseFacesBuffer } },
    { binding: 2, resource: { buffer: baseFaceOffsetBuffer } },
    { binding: 3, resource: { buffer: baseFaceValenceBuffer } },
  ],
});

const edgeBindGroup = device.createBindGroup({
  label: "bindGroup for edge kernel",
  layout: edgePipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: verticesBuffer } },
    { binding: 1, resource: { buffer: baseEdgesBuffer } },
  ],
});

const vertexBindGroup = device.createBindGroup({
  label: "bindGroup for vertex kernel",
  layout: vertexPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: verticesBuffer } },
    { binding: 1, resource: { buffer: baseVerticesBuffer } },
    { binding: 2, resource: { buffer: baseVertexOffsetBuffer } },
    { binding: 3, resource: { buffer: baseVertexValenceBuffer } },
    { binding: 4, resource: { buffer: baseVertexIndexBuffer } },
  ],
});

const facetNormalsBindGroup = device.createBindGroup({
  label: "bindGroup for computing facet normals",
  layout: facetNormalsPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: facetNormalsBuffer } },
    { binding: 1, resource: { buffer: verticesBuffer } },
    { binding: 2, resource: { buffer: triangleIndicesBuffer } },
  ],
});

const vertexNormalsBindGroup = device.createBindGroup({
  label: "bindGroup for computing vertex normals",
  layout: vertexNormalsPipeline.getBindGroupLayout(0),
  entries: [
    { binding: 0, resource: { buffer: vertexNormalsBuffer } },
    { binding: 1, resource: { buffer: facetNormalsBuffer } },
    { binding: 2, resource: { buffer: triangleIndicesBuffer } },
  ],
});

const renderBindGroup = device.createBindGroup({
  label: "bindGroup for rendering kernel",
  layout: renderPipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: mvxBuffer } }],
});

const aspect = canvas.width / canvas.height;
const projectionMatrix = mat4.perspective((2 * Math.PI) / 5, aspect, 1, 100.0);
const modelviewProjectionMatrix = mat4.create();

function getTransformationMatrix() {
  /* this view matrix simply does some time-dependent rotation */
  /* of course adding camera control would be better */
  const viewMatrix = mat4.identity();
  let now = uni.views.time[0] * uni.views.ROTATE_CAMERA_SPEED[0];
  mat4.translate(viewMatrix, vec3.fromValues(0, 0, -3), viewMatrix);
  mat4.rotateZ(viewMatrix, now, viewMatrix);
  mat4.rotateY(viewMatrix, now, viewMatrix);
  mat4.rotateX(viewMatrix, now, viewMatrix);
  mat4.multiply(projectionMatrix, viewMatrix, modelviewProjectionMatrix);
  return modelviewProjectionMatrix;
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
  device.queue.writeBuffer(uniformsBuffer, 0, uni.arrayBuffer);

  const transformationMatrix = getTransformationMatrix();
  device.queue.writeBuffer(
    mvxBuffer,
    0,
    transformationMatrix.buffer,
    transformationMatrix.byteOffset,
    transformationMatrix.byteLength
  );

  // Encode commands to do the computation
  const encoder = device.createCommandEncoder({
    label:
      "overall computation (perturb, face, edge, vertex, normals) + graphics encoder",
  });

  const computePass = timingHelper.beginComputePass(encoder, {
    label: "compute pass, all compute kernels",
  });
  computePass.setPipeline(perturbPipeline);
  computePass.setBindGroup(0, perturbBindGroup);
  computePass.dispatchWorkgroups(
    Math.ceil(mesh.levelCount[0].v / WORKGROUP_SIZE)
  );

  computePass.setPipeline(facePipeline);
  computePass.setBindGroup(0, faceBindGroup);
  computePass.dispatchWorkgroups(
    Math.ceil(mesh.levelCount[1].f / WORKGROUP_SIZE)
  );

  computePass.setPipeline(edgePipeline);
  computePass.setBindGroup(0, edgeBindGroup);
  computePass.dispatchWorkgroups(
    Math.ceil(mesh.levelCount[1].e / WORKGROUP_SIZE)
  );

  computePass.setPipeline(vertexPipeline);
  computePass.setBindGroup(0, vertexBindGroup);
  computePass.dispatchWorkgroups(
    Math.ceil(mesh.levelCount[1].v / WORKGROUP_SIZE)
  );

  computePass.setPipeline(facetNormalsPipeline);
  computePass.setBindGroup(0, facetNormalsBindGroup);
  computePass.dispatchWorkgroups(
    Math.ceil((mesh.levelCount[0].t + mesh.levelCount[1].t) / WORKGROUP_SIZE)
  );

  computePass.setPipeline(vertexNormalsPipeline);
  computePass.setBindGroup(0, vertexNormalsBindGroup);
  computePass.dispatchWorkgroups(Math.ceil(verticesSize / WORKGROUP_SIZE));
  computePass.end();

  // Encode a command to copy the results to a mappable buffer.
  // this is (from, to)
  encoder.copyBufferToBuffer(
    verticesBuffer,
    0,
    mappableVerticesResultBuffer,
    0,
    mappableVerticesResultBuffer.size
  );
  encoder.copyBufferToBuffer(
    facetNormalsBuffer,
    0,
    mappableFacetNormalsResultBuffer,
    0,
    mappableFacetNormalsResultBuffer.size
  );
  encoder.copyBufferToBuffer(
    vertexNormalsBuffer,
    0,
    mappableVertexNormalsResultBuffer,
    0,
    mappableVertexNormalsResultBuffer.size
  );

  const renderPass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: { r: 0, g: 0, b: 0.4, a: 1.0 },
        storeOp: "store",
      },
    ],
    depthStencilAttachment: {
      view: depthTexture.createView(),

      depthClearValue: 1.0,
      depthLoadOp: "clear",
      depthStoreOp: "store",
    },
  });

  // Now render those tris.
  renderPass.setPipeline(renderPipeline);
  renderPass.setBindGroup(0, renderBindGroup);
  renderPass.setVertexBuffer(0, verticesBuffer);
  let startIdx = -1;
  let endIdx = -1;
  const now = uni.views.time[0];

  renderPass.setIndexBuffer(triangleIndicesBuffer, "uint32");
  // next line switches every TOGGLE_DURATION frames
  // switch ((now / uni.views.TOGGLE_DURATION) & 1) {
  // instead switch explicitly on subdivLevel
  // clearly this math can be much much simpler
  switch (uni.views.subdivLevel[0]) {
    case 0 /* draws tris [0, baseTrianglesCount) */:
      startIdx = 0;
      endIdx = mesh.levelCount[0].t * 3;
      break;
    case 1 /* draws tris [baseTrianglesCount, base + subdiv counts) */:
      startIdx = mesh.levelCount[0].t * 3;
      endIdx = (mesh.levelCount[0].t + mesh.levelCount[1].t) * 3;
      break;
  }
  renderPass.drawIndexed(
    endIdx - startIdx /* count */,
    1 /* instance */,
    startIdx /* start */
  );

  // End the render pass and submit the command buffer
  renderPass.end();

  // Finish encoding and submit the commands
  const commandBuffer = encoder.finish();
  device.queue.submit([commandBuffer]);

  // Read the results
  if (debug) {
    await mappableVerticesResultBuffer.mapAsync(GPUMapMode.READ);
    const verticesResult = new Float32Array(
      mappableVerticesResultBuffer.getMappedRange().slice()
    );
    mappableVerticesResultBuffer.unmap();
    await mappableFacetNormalsResultBuffer.mapAsync(GPUMapMode.READ);
    const facetNormalsResult = new Float32Array(
      mappableFacetNormalsResultBuffer.getMappedRange().slice()
    );
    mappableFacetNormalsResultBuffer.unmap();
    await mappableVertexNormalsResultBuffer.mapAsync(GPUMapMode.READ);
    const vertexNormalsResult = new Float32Array(
      mappableVertexNormalsResultBuffer.getMappedRange().slice()
    );
    mappableVertexNormalsResultBuffer.unmap();
    console.log("vertex buffer", verticesResult);
  }

  /* is this correct for getting timing info? */
  timingHelper.getResult().then((res) => {
    // console.log("timing helper result", res);
  });

  uni.views.time[0] = uni.views.time[0] + uni.views.timestep[0];
  // console.log("time", uni.views.time[0]);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

function fail(msg) {
  // eslint-disable-next-line no-alert
  alert(msg);
}
