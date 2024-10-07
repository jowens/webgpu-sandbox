import { Pane } from "https://cdn.jsdelivr.net/npm/tweakpane@4.0.3/dist/tweakpane.min.js";
import {
  vec3,
  mat4,
} from "https://wgpu-matrix.org/dist/3.x/wgpu-matrix.module.js"; // for uniform handling
import {
  makeShaderDataDefinitions,
  makeStructuredView,
} from "https://greggman.github.io/webgpu-utils/dist/1.x/webgpu-utils.module.js";

// We can set runtime params from the input URL!
const urlParams = new URL(window.location.href).searchParams;
const debug = urlParams.get("debug"); // string or undefined
let frameCount = urlParams.get("frameCount");
frameCount = frameCount == undefined ? -1 : parseInt(frameCount, 10);
const separateComputePasses = urlParams.get("separateComputePasses");
const timingEnabled = urlParams.get("timing");

const adapter = await navigator.gpu?.requestAdapter();
const canTimestamp = adapter.features.has("timestamp-query");
const device = await adapter?.requestDevice({
  requiredFeatures: [
    ...(canTimestamp && timingEnabled ? ["timestamp-query"] : []),
  ], // ...: conditional add
});
if (!device) {
  fail("Fatal error: Device does not support WebGPU.");
}

// if we want more:
//   Object.fromEntries(new URL(window.location.href).searchParams.entries());
// if url is 'https://foo.com/bar.html?abc=123&def=456&xyz=banana` then params is
// { abc: '123', def: '456', xyz: 'banana' }   // notice they are strings, not numbers.

// using webgpu-utils to have one struct for uniforms across all kernels
// design goal here: have one single place to record params, even at the cost
//   of manual copying
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
    level: u32, // used during compute kernels
    @align(16) levelCount: array<Level, MAX_LEVEL>,
    @align(16) levelBasePtr: array<Level, MAX_LEVEL>,
    time: f32,
    timestep: f32,
  };
  @group(0) @binding(0) var<uniform> myUniforms: MyUniforms;`;
/* why the @group/@binding? gman@:
 * "It's necessary for them to show up in defs.uniforms or defs.storages. You
 *  can use defs.structs to pull out a struct, separately from a group/binding (I think?)"
 * "As your comment mentions you can use uniforms_defs.structs.MyUniforms"
 */
const uniformsDefs = makeShaderDataDefinitions(uniformsCode);
const uni = makeStructuredView(uniformsDefs.uniforms.myUniforms);

uni.set({
  ROTATE_CAMERA_SPEED: 0.006, // how quickly camera rotates
  TOGGLE_DURATION: 400.0, // number of timesteps between model toggle
  WIGGLE_MAGNITUDE: 0, // 0.002, //0.025, // how much vertices are perturbed
  WIGGLE_SPEED: 0.05, // how quickly perturbations occur
  subdivLevel: urlParams.get("subdivLevel")
    ? parseInt(urlParams.get("subdivLevel"), 10)
    : 0,
  level: 0,
  time: 0.0,
  timestep: 1.0,
});

const paneParams = {
  model: urlParams.get("model") ? urlParams.get("model") : "square_pyramid", // default starting point
  shading: "smooth",
};

const modelToURL = {
  square_pyramid:
    "https://gist.githubusercontent.com/jowens/ccd142c4d17e6c188c5105a1881561bf/raw/26e58cb754d1dfb8c30c86d33e0c21497c2167e8/square-pyramid.obj",
  diamond:
    "https://gist.githubusercontent.com/jowens/ebe82add66adfee31fe49579963c515d/raw/2046cff529575615e32a283a9ca2b4e44f3a13d2/diamond.obj",
  teddy:
    "https://gist.githubusercontent.com/jowens/d49b13c7f847bda5ffc36d2166888b5f/raw/2756e4e3c5be3b2cce35244c961f462411cefaef/teddy.obj",
  al: "https://gist.githubusercontent.com/jowens/360d591b8484958cf1c5b015c96c0958/raw/6390f2a2c720d378d1aa77baba7605c67d40e2e4/al.obj",
  teapot_lowres:
    "https://gist.githubusercontent.com/jowens/508d6d7f70b33010508f3c679abd61ff/raw/0315c1d585a63687034ae4deecb5b49b8d653017/teapot-lower.obj",
  stanford_teapot:
    "https://gist.githubusercontent.com/jowens/5f7bc872317b5fd5f7d72827967f1c9d/raw/1f846ee3229297520dd855b199d21717e30af91b/stanford-teapot.obj",
  ogre: "http://localhost:8000/meshes/ogre.obj",
};

const WORKGROUP_SIZE = 256;

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

const perturbInputVerticesModule = device.createShaderModule({
  label: "perturb input vertices module",
  code: /* wgsl */ `
    ${uniformsCode} /* this specifies @group(0) @binding(0) */
    /* input + output */
    @group(0) @binding(1) var<storage, read_write> vertices: array<vec3f>;
    @compute @workgroup_size(${WORKGROUP_SIZE}) fn perturbInputVerticesKernel(
             @builtin(global_invocation_id) id: vec3u) {
      let i = id.x;
      if (i < arrayLength(&vertices)) { // only call on base vertices, but that is
                                        // enforced by the binding
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
    }`,
});

/** (1) Calculation of face points
 * Number of faces: faceValence.length == facesCount
 * for each face: new face point = centroid(vertices of current face)
 * Pseudocode:   (note math operations are on vec3f's)
 * parallel for i in [0 .. faceValence.length]:
 *   newFaces[i] = [0,0,0]
 *   for j in [faceOffset[i] .. faceOffset[i] + faceValence[i]]:
 *     newFaces[i] += vertices[faces[j]
 *   newFaces[i] /= faceValence[i]
 */

const facePointsModule = device.createShaderModule({
  label: "face points module",
  code: /* wgsl */ `
    ${uniformsCode} /* this specifies @group(0) @binding(0) */
    /* input + output */
    @group(0) @binding(1) var<storage, read_write> vertices: array<vec3f>;
            /* input */
    @group(0) @binding(2) var<storage, read> faces: array<u32>;
    @group(0) @binding(3) var<storage, read> faceOffset: array<u32>;
    @group(0) @binding(4) var<storage, read> faceValence: array<u32>;
    @group(0) @binding(5) var<storage, read> faceOffsetPtr: array<u32>;
    /** Niessner 2012:
      * "The face kernel requires two buffers: one index buffer, whose
      * entries are the vertex buffer indices for each vertex of the face; a
      * second buffer stores the valence of the face along with an offset
      * into the index buffer for the first vertex of each face."
      *
      * implementation above: "index buffer" is faces
      *                       "valence of the face" is faceValence
      *                       "offset into the index buffer" is faceOffset
      */
    @compute @workgroup_size(${WORKGROUP_SIZE}) fn facePointsKernel(
      @builtin(global_invocation_id) id: vec3u) {
      let i = id.x; /* [0, number of faces) */
      if (i < myUniforms.levelCount[myUniforms.level].f) {
        // faceOffsetPtr[level] points to the first element in level's faceOffset
        let in = i + faceOffsetPtr[myUniforms.level];
        let out = i + myUniforms.levelBasePtr[myUniforms.level].f;
        vertices[out] = vec3f(0,0,0);
        for (var j: u32 = faceOffset[in]; j < faceOffset[in] + faceValence[in]; j++) {
          let faceVertex = faces[j];
          vertices[out] += vertices[faceVertex];
        }
        vertices[out] /= f32(faceValence[in]);
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
 * Number of edges: edges.length
 * for each edge: new edge point = average(2 neighboring face points, 2 endpoints of edge)
 * Pseudocode:   (note math operations are on vec3f's)
 * parallel for i in [0 .. ?.length]:
 *   newEdges[i] = 0.25 * ( vertices[edgeID] + vertices[edgeID + 1] +
 *                           vertices[edgeID + 2] + vertices[edgeID + 3])
 */

const edgePointsModule = device.createShaderModule({
  label: "edge points module",
  code: /* wgsl */ `
    ${uniformsCode} /* this specifies @group(0) @binding(0) */
    /* input + output */
    @group(0) @binding(1) var<storage, read_write> vertices: array<vec3f>;
    /* input */
    @group(0) @binding(2) var<storage, read> edges: array<vec4u>;
    @group(0) @binding(3) var<storage, read> edgeOffsetPtr: array<u32>;

    /** "Since a single (non-boundary) edge always has two incident faces and vertices,
     * the edge kernel needs a buffer for the indices of these entities."
     *
     * implementation above: "a buffer for the indices of these entities" is edges
     */

    @compute @workgroup_size(${WORKGROUP_SIZE}) fn edgePointsKernel(
      @builtin(global_invocation_id) id: vec3u) {
        let i = id.x;
        if (i < myUniforms.levelCount[myUniforms.level].e) {
          /* edgeID is the index into the edges data structure */
          let edgeID = i + edgeOffsetPtr[myUniforms.level];
          let out = i + myUniforms.levelBasePtr[myUniforms.level].e;
          vertices[out] = vec3f(0,0,0);
          for (var j: u32 = 0; j < 4; j++) {
            vertices[out] += vertices[edges[edgeID][j]];
          }
          vertices[out] *= 0.25;
        }
      }`,
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
 *   - F and Ve are just listed in the vertexNeighbors table
 * - V is this vertex
 *   - Output is (F + Ve + (n-2) V) / n
 * - If F and Ve points are f_0, f1, Ve_0, ...:
 *   - Output is [(f_0 + f1 + ... + Ve_0 + Ve1 + ...) / n _ (n-2) V] / n
 * Number of vertex points: vertexValence.length
 * Pseudocode:   (note math operations are on vec3f's)
 * parallel for i in [0 .. vertexValence.length]:
 *   newVertex[i] = [0,0,0]
 *   valence = vertexValence[i]
 *   for j in [vertexOffset[i] .. vertexOffset[i] + vertexValence[i]]:
 *     newVertex[i] += vertices[vertexNeighbors[j]]
 *   newVertex[i] /= vertexValence[i]
 *   newVertex[i] += (n-2) * vertexIndex[i]
 *   newVertex[i] /= vertexValence[i]
 */

const vertexPointsModule = device.createShaderModule({
  label: "vertex points module",
  code: /* wgsl */ `
    ${uniformsCode} /* this specifies @group(0) @binding(0) */
    /* input + output */
    @group(0) @binding(1) var<storage, read_write> vertices: array<vec3f>;
    /* input */
    @group(0) @binding(2) var<storage, read> vertexNeighbors: array<u32>;
    @group(0) @binding(3) var<storage, read> vertexOffset: array<u32>;
    @group(0) @binding(4) var<storage, read> vertexValence: array<u32>;
    @group(0) @binding(5) var<storage, read> vertexIndex: array<u32>;
    @group(0) @binding(6) var<storage, read> vertexOffsetPtr: array<u32>;

    /** "We use an index buffer containing the indices of the incident edge and
     * vertex points."
     *
     * implementation above: "a buffer for the indices of these entities" is vertexNeighbors
     */

    @compute @workgroup_size(${WORKGROUP_SIZE}) fn vertexPointsKernel(
      @builtin(global_invocation_id) id: vec3u) {
        let i = id.x;
        if (i < myUniforms.levelCount[myUniforms.level].v) {
          let in = i + vertexOffsetPtr[myUniforms.level];
          let out = i + myUniforms.levelBasePtr[myUniforms.level].v;
          let valence = vertexValence[in];
          vertices[out] = vec3f(0,0,0);
          for (var j: u32 = vertexOffset[in]; j < vertexOffset[in] + 2 * vertexValence[in]; j++) {
            let vertex = vertexNeighbors[j];
            vertices[out] += vertices[vertex];
          }
          vertices[out] /= f32(valence);
          vertices[out] += f32(valence - 2) * vertices[vertexIndex[in]];
          vertices[out] /= f32(valence);
          // TODO: decide on vec3f or vec4f and set w if so
      }
    }`,
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
    ${uniformsCode} /* this specifies @group(0) @binding(0) */
    /* output */
    @group(0) @binding(1) var<storage, read_write> facetNormals: array<vec3f>;
    /* input */
    @group(0) @binding(2) var<storage, read> vertices: array<vec3f>;
    @group(0) @binding(3) var<storage, read> triangleIndices: array<u32>;

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
      * This is wasteful (O(n^2)); every vertex will walk the entire
      *   index array looking for matches.
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
      *
      * That has terrible peformance.
      * So, at model load time, build a mapping of {vtx->triangles},
      *   with valence and offset information
      *
      * for vertex in all vertices:
      *   normal = (0,0,0)
      *   for triangle in triangle_neighbors[vertex]:
      *     normal += triangle_normal
      *   return normalize(normal)
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
      }`,
});

const vertexNormalsON2Module = device.createShaderModule({
  label: "compute vertex normals (O(n^2)) module",
  code: /* wgsl */ `
    ${uniformsCode} /* this specifies @group(0) @binding(0) */
    /* output */
    @group(0) @binding(1) var<storage, read_write> vertexNormals: array<vec3f>;
    /* input */
    @group(0) @binding(2) var<storage, read> facetNormals: array<vec3f>;
    @group(0) @binding(3) var<storage, read> triangleIndices: array<u32>;

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
    }`,
});

const vertexNormalsModule = device.createShaderModule({
  label: "compute vertex normals (O(n)) module",
  code: /* wgsl */ `
    ${uniformsCode} /* this specifies @group(0) @binding(0) */
    /* output */
    @group(0) @binding(1) var<storage, read_write> vertexNormals: array<vec3f>;
    /* input */
    @group(0) @binding(2) var<storage, read> facetNormals: array<vec3f>;
    @group(0) @binding(3) var<storage, read> vertexToTriangles: array<u32>;
    @group(0) @binding(4) var<storage, read> vertexToTrianglesOffset: array<u32>;
    @group(0) @binding(5) var<storage, read> vertexToTrianglesValence: array<u32>;

    /* see facetNormalsModule for algorithm */

    @compute @workgroup_size(${WORKGROUP_SIZE}) fn vertexNormalsKernel(
      @builtin(global_invocation_id) id: vec3u) {
        let vtx = id.x;
        if (vtx < arrayLength(&vertexNormals)) {
          vertexNormals[vtx] = vec3f(0, 0, 0);
          for (var neighbor: u32 = vertexToTrianglesOffset[vtx]; neighbor < vertexToTrianglesOffset[vtx] + vertexToTrianglesValence[vtx]; neighbor++) {
            vertexNormals[vtx] += facetNormals[vertexToTriangles[neighbor]];
            }
          vertexNormals[vtx] = normalize(vertexNormals[vtx]);
        }
    }`,
});

function renderCode(shadingType = "") {
  // or shadingType = @interpolate(flat)
  return /* wgsl */ `
  struct VertexInput {
    @location(0) pos: vec4f,
    @location(1) vertexNormals: vec3f,
    @builtin(vertex_index) vertexIndex: u32,
  };

  struct VertexOutput {
    @builtin(position) pos: vec4f,
    @location(0) ${shadingType} color: vec4f,
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
  }`;
}

const renderModules = [];
renderModules["smooth"] = device.createShaderModule({
  label: "render module",
  code: renderCode(),
});
renderModules["flat"] = device.createShaderModule({
  label: "render module",
  code: renderCode("@interpolate(flat)"),
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

const vertexNormalsON2Pipeline = device.createComputePipeline({
  label: "vertex normals O(N^2) compute pipeline",
  layout: "auto",
  compute: {
    module: vertexNormalsON2Module,
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

const renderPipelines = [];
["smooth", "flat"].forEach(
  (keyword) =>
    (renderPipelines[keyword] = device.createRenderPipeline({
      label: "render pipeline",
      layout: "auto",
      vertex: {
        module: renderModules[keyword],
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
        module: renderModules[keyword],
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
    }))
);

// create buffers on the GPU to hold data

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

async function loadMesh(url) {
  const mesh = await urlToMesh(url);
  console.log(mesh);
  return mesh;
}

let mesh = await loadMesh(modelToURL[paneParams.model]);

/**
 * class GPUContext holds all data that is relevant to the GPU side
 * It is constructed from a CPU-side (JS) Mesh data structure
 * It is a class because we rebuild it whenever we load a new mesh
 * It holds all GPU buffers and bind groups, both of which are
 *   rebuilt whenever a new Mesh is loaded
 */
class GPUContext {
  createGPUBuffers() {
    // read-only inputs:
    this.facesBuffer = device.createBuffer({
      label: "base faces buffer",
      size: mesh.faces.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.edgesBuffer = device.createBuffer({
      label: "base edges buffer",
      size: mesh.edges.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.faceOffsetBuffer = device.createBuffer({
      label: "base face offset",
      size: mesh.faceOffset.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.faceValenceBuffer = device.createBuffer({
      label: "face valence",
      size: mesh.faceValence.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.faceOffsetPtrBuffer = device.createBuffer({
      label: "face offset",
      size: mesh.faceOffsetPtr.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.edgeOffsetPtrBuffer = device.createBuffer({
      label: "edge offset",
      size: mesh.edgeOffsetPtr.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.vertexOffsetPtrBuffer = device.createBuffer({
      label: "vertex offset",
      size: mesh.vertexOffsetPtr.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.vertexNeighborsBuffer = device.createBuffer({
      label: "vertex neighbors buffer",
      size: mesh.vertexNeighbors.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.vertexOffsetBuffer = device.createBuffer({
      label: "vertex offset buffer",
      size: mesh.vertexOffset.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.vertexValenceBuffer = device.createBuffer({
      label: "vertex valence buffer",
      size: mesh.vertexValence.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.vertexIndexBuffer = device.createBuffer({
      label: "vertex index buffer",
      size: mesh.vertexIndex.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.triangleIndicesBuffer = device.createBuffer({
      label: "triangle indices buffer",
      size: mesh.triangles.byteLength,
      usage:
        GPUBufferUsage.INDEX | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.vertexToTrianglesBuffer = device.createBuffer({
      label: "vertex to triangles buffer",
      size: mesh.vertexToTriangles.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.vertexToTrianglesOffsetBuffer = device.createBuffer({
      label: "vertex to triangles offset buffer",
      size: mesh.vertexToTrianglesOffset.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.vertexToTrianglesValenceBuffer = device.createBuffer({
      label: "vertex to triangles valence buffer",
      size: mesh.vertexToTrianglesValence.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    // vertex buffer is both input and output
    this.verticesBuffer = device.createBuffer({
      label: "vertex buffer",
      size: mesh.vertices.byteLength,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.VERTEX |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });

    this.facetNormalsBuffer = device.createBuffer({
      label: "facet normals buffer",
      size: mesh.facetNormals.byteLength,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });

    this.vertexNormalsBuffer = device.createBuffer({
      label: "vertex normals buffer",
      size: mesh.vertexNormals.byteLength,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    });

    /** and the mappable output buffers (I believe that "mappable" is the only way to read from GPU->CPU) */
    this.mappableVerticesResultBuffer = device.createBuffer({
      label: "mappable vertices result buffer",
      size: mesh.vertices.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.mappableFacetNormalsResultBuffer = device.createBuffer({
      label: "mappable facet normals result buffer",
      size: mesh.facetNormals.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    this.mappableVertexNormalsResultBuffer = device.createBuffer({
      label: "mappable vertex normals result buffer",
      size: mesh.vertexNormals.byteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  createBindGroups() {
    /** Set up bindGroups per compute kernel to tell the shader which buffers to use */
    /** I had hoped to do all verticesBuffer bindings as slices as below,
     * but buffer alignment restrictions appear to make this impossible.
     *   Offset (25568) of [Buffer "vertex buffer"] does not satisfy the minimum
     *   BufferBindingType::Storage alignment (256).
     * This particular bindGroup is OK because it's always at the beginning of
     * verticesBuffer.
     */
    // TODO compute this using sizeof() so it's not hardcoded
    const bytesPerVertex = mesh.vertexSize * 4;
    this.perturbBindGroup = device.createBindGroup({
      label: "bindGroup for perturb input vertices kernel",
      layout: perturbPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformsBuffer } },
        {
          binding: 1,
          resource: {
            buffer: this.verticesBuffer,
            offset: 0,
            size: mesh.levelCount[0].v * bytesPerVertex,
          },
        },
      ],
    });

    this.faceBindGroup = device.createBindGroup({
      label: `bindGroup for face kernel`,
      layout: facePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformsBuffer } },
        { binding: 1, resource: { buffer: this.verticesBuffer } },
        { binding: 2, resource: { buffer: this.facesBuffer } },
        { binding: 3, resource: { buffer: this.faceOffsetBuffer } },
        { binding: 4, resource: { buffer: this.faceValenceBuffer } },
        { binding: 5, resource: { buffer: this.faceOffsetPtrBuffer } },
      ],
    });

    this.edgeBindGroup = device.createBindGroup({
      label: "bindGroup for edge kernel",
      layout: edgePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformsBuffer } },
        { binding: 1, resource: { buffer: this.verticesBuffer } },
        { binding: 2, resource: { buffer: this.edgesBuffer } },
        { binding: 3, resource: { buffer: this.edgeOffsetPtrBuffer } },
      ],
    });

    this.vertexBindGroup = device.createBindGroup({
      label: "bindGroup for vertex kernel",
      layout: vertexPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniformsBuffer } },
        { binding: 1, resource: { buffer: this.verticesBuffer } },
        { binding: 2, resource: { buffer: this.vertexNeighborsBuffer } },
        { binding: 3, resource: { buffer: this.vertexOffsetBuffer } },
        { binding: 4, resource: { buffer: this.vertexValenceBuffer } },
        { binding: 5, resource: { buffer: this.vertexIndexBuffer } },
        { binding: 6, resource: { buffer: this.vertexOffsetPtrBuffer } },
      ],
    });

    this.facetNormalsBindGroup = device.createBindGroup({
      label: "bindGroup for computing facet normals",
      layout: facetNormalsPipeline.getBindGroupLayout(0),
      entries: [
        // { binding: 0, resource: { buffer: uniformsBuffer } },
        { binding: 1, resource: { buffer: this.facetNormalsBuffer } },
        { binding: 2, resource: { buffer: this.verticesBuffer } },
        { binding: 3, resource: { buffer: this.triangleIndicesBuffer } },
      ],
    });

    this.vertexNormalsBindGroup = device.createBindGroup({
      label: "bindGroup for computing vertex normals",
      layout: vertexNormalsPipeline.getBindGroupLayout(0),
      entries: [
        // { binding: 0, resource: { buffer: uniformsBuffer } },
        { binding: 1, resource: { buffer: this.vertexNormalsBuffer } },
        { binding: 2, resource: { buffer: this.facetNormalsBuffer } },
        { binding: 3, resource: { buffer: this.vertexToTrianglesBuffer } },
        {
          binding: 4,
          resource: { buffer: this.vertexToTrianglesOffsetBuffer },
        },
        {
          binding: 5,
          resource: { buffer: this.vertexToTrianglesValenceBuffer },
        },
      ],
    });

    this.vertexNormalsON2BindGroup = device.createBindGroup({
      label: "bindGroup for computing vertex normals",
      layout: vertexNormalsON2Pipeline.getBindGroupLayout(0),
      entries: [
        // { binding: 0, resource: { buffer: uniformsBuffer } },
        { binding: 1, resource: { buffer: this.vertexNormalsBuffer } },
        { binding: 2, resource: { buffer: this.facetNormalsBuffer } },
        { binding: 3, resource: { buffer: this.triangleIndicesBuffer } },
      ],
    });

    this.renderBindGroups = [];
    ["smooth", "flat"].forEach(
      (keyword) =>
        (this.renderBindGroups[keyword] = device.createBindGroup({
          label: "bindGroup for rendering kernel",
          layout: renderPipelines[keyword].getBindGroupLayout(0),
          entries: [{ binding: 0, resource: { buffer: mvxBuffer } }],
        }))
    );
  }

  writeToGPUBuffers() {
    device.queue.writeBuffer(this.facesBuffer, 0, mesh.faces);
    device.queue.writeBuffer(this.edgesBuffer, 0, mesh.edges);
    device.queue.writeBuffer(this.faceOffsetBuffer, 0, mesh.faceOffset);
    device.queue.writeBuffer(this.faceOffsetPtrBuffer, 0, mesh.faceOffsetPtr);
    device.queue.writeBuffer(this.edgeOffsetPtrBuffer, 0, mesh.edgeOffsetPtr);
    device.queue.writeBuffer(
      this.vertexOffsetPtrBuffer,
      0,
      mesh.vertexOffsetPtr
    );
    device.queue.writeBuffer(this.faceValenceBuffer, 0, mesh.faceValence);
    device.queue.writeBuffer(
      this.vertexNeighborsBuffer,
      0,
      mesh.vertexNeighbors
    );
    device.queue.writeBuffer(this.vertexOffsetBuffer, 0, mesh.vertexOffset);
    device.queue.writeBuffer(this.vertexValenceBuffer, 0, mesh.vertexValence);
    device.queue.writeBuffer(this.vertexIndexBuffer, 0, mesh.vertexIndex);
    device.queue.writeBuffer(this.triangleIndicesBuffer, 0, mesh.triangles);
    device.queue.writeBuffer(this.verticesBuffer, 0, mesh.vertices);
    device.queue.writeBuffer(
      this.vertexToTrianglesBuffer,
      0,
      mesh.vertexToTriangles
    );
    device.queue.writeBuffer(
      this.vertexToTrianglesValenceBuffer,
      0,
      mesh.vertexToTrianglesValence
    );
    device.queue.writeBuffer(
      this.vertexToTrianglesOffsetBuffer,
      0,
      mesh.vertexToTrianglesOffset
    );
    device.queue.writeBuffer(this.facetNormalsBuffer, 0, mesh.facetNormals);
    device.queue.writeBuffer(this.vertexNormalsBuffer, 0, mesh.vertexNormals);
    uni.set({ levelCount: mesh.levelCount, levelBasePtr: mesh.levelBasePtr });
  }
  destroyGPUBuffers() {
    this.facesBuffer.destroy();
    this.edgesBuffer.destroy();
    this.faceOffsetBuffer.destroy();
    this.faceOffsetPtrBuffer.destroy();
    this.edgeOffsetPtrBuffer.destroy();
    this.vertexOffsetPtrBuffer.destroy();
    this.faceValenceBuffer.destroy();
    this.vertexNeighborsBuffer.destroy();
    this.vertexOffsetBuffer.destroy();
    this.vertexValenceBuffer.destroy();
    this.vertexIndexBuffer.destroy();
    this.triangleIndicesBuffer.destroy();
    this.verticesBuffer.destroy();
    this.vertexToTrianglesBuffer.destroy();
    this.vertexToTrianglesValenceBuffer.destroy();
    this.vertexToTrianglesOffsetBuffer.destroy();
    this.facetNormalsBuffer.destroy();
    this.vertexNormalsBuffer.destroy();
    this.mappableVerticesResultBuffer.destroy();
    this.mappableVertexNormalsResultBuffer.destroy();
    this.mappableVertexNormalsResultBuffer.destroy();
  }
  constructor(mesh) {
    this.createGPUBuffers();
    this.writeToGPUBuffers();
    this.createBindGroups();
  }
}

const pane = new Pane();
pane
  .addBinding(paneParams, "model", {
    options: {
      // what it shows : what it returns
      "Square Pyramid": "square_pyramid",
      Diamond: "diamond",
      Teddy: "teddy",
      Al: "al",
      "Teapot (Low Res)": "teapot_lowres",
      "Stanford Teapot": "stanford_teapot",
      Ogre: "ogre",
    },
  })
  .on("change", async (ev) => {
    mesh = await loadMesh(modelToURL[ev.value]);
    ctx.destroyGPUBuffers();
    ctx = new GPUContext(mesh);
    frame();
  });
pane.addBinding(paneParams, "shading", {
  options: {
    // what it shows : what it returns
    Smooth: "smooth",
    Flat: "flat",
  },
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
  max: mesh.maxLevel,
  step: 1,
  label: "Subdiv Level",
});

const mvxLength = 4 * 16; /* float32 4x4 matrix */
const mvxBuffer = device.createBuffer({
  label: "modelview + transformation matrix uniform buffer",
  size: mvxLength,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
// write happens at the start of every frame

let ctx = new GPUContext(mesh);

async function frame() {
  /**
   * Definitely there's two things that need to go CPU->GPU every frame
   *
   * (1) Uniforms, since they can be altered by the user at runtime
   *     in the pane (also time is here)
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

  function passBoundary(
    separateComputePasses,
    timingHelper,
    computePasses,
    encoder
  ) {
    if (separateComputePasses) {
      computePasses.at(-1).end();
      computePasses.push(
        timingHelper.beginComputePass(encoder, {
          label: `compute pass ${
            computePasses.length - 1
          }, all compute kernels`,
        })
      );
    }
  }

  // Encode commands to do the computation
  const encoder = device.createCommandEncoder({
    label:
      "overall computation (perturb, face, edge, vertex, normals) + graphics encoder",
  });

  const kernels = separateComputePasses ? uni.views.subdivLevel[0] * 3 + 3 : 1;

  const timingHelper = new TimingHelper(device, kernels);
  const computePasses = [];
  computePasses.push(
    timingHelper.beginComputePass(encoder, {
      label: `compute pass ${computePasses.length - 1}, all compute kernels`,
    })
  );
  computePasses.at(-1).setPipeline(perturbPipeline);
  computePasses.at(-1).setBindGroup(0, ctx.perturbBindGroup);
  computePasses
    .at(-1)
    .dispatchWorkgroups(Math.ceil(mesh.levelCount[0].v / WORKGROUP_SIZE));
  passBoundary(separateComputePasses, timingHelper, computePasses, encoder);

  /** The face, edge, and vertex kernels run once per level */
  for (var level = 1; level <= uni.views.subdivLevel[0]; level++) {
    // update the level on the CPU ...
    uni.views.level[0] = level;
    // ... and GPU. The following kernels need to know the level because
    // they use it to select input and output pointers within the GPU-side
    // data structures
    device.queue.writeBuffer(uniformsBuffer, 0, uni.arrayBuffer);

    computePasses.at(-1).setPipeline(facePipeline);
    computePasses.at(-1).setBindGroup(0, ctx.faceBindGroup);
    computePasses
      .at(-1)
      .dispatchWorkgroups(Math.ceil(mesh.levelCount[level].f / WORKGROUP_SIZE));
    passBoundary(separateComputePasses, timingHelper, computePasses, encoder);

    computePasses.at(-1).setPipeline(edgePipeline);
    computePasses.at(-1).setBindGroup(0, ctx.edgeBindGroup);
    computePasses
      .at(-1)
      .dispatchWorkgroups(Math.ceil(mesh.levelCount[level].e / WORKGROUP_SIZE));
    passBoundary(separateComputePasses, timingHelper, computePasses, encoder);

    computePasses.at(-1).setPipeline(vertexPipeline);
    computePasses.at(-1).setBindGroup(0, ctx.vertexBindGroup);
    computePasses
      .at(-1)
      .dispatchWorkgroups(Math.ceil(mesh.levelCount[level].v / WORKGROUP_SIZE));
    passBoundary(separateComputePasses, timingHelper, computePasses, encoder);
  }

  computePasses.at(-1).setPipeline(facetNormalsPipeline);
  computePasses.at(-1).setBindGroup(0, ctx.facetNormalsBindGroup);
  computePasses
    .at(-1)
    .dispatchWorkgroups(Math.ceil(mesh.facetNormals.length / WORKGROUP_SIZE));

  passBoundary(separateComputePasses, timingHelper, computePasses, encoder);
  computePasses.at(-1).setPipeline(vertexNormalsPipeline);
  computePasses.at(-1).setBindGroup(0, ctx.vertexNormalsBindGroup);
  computePasses
    .at(-1)
    .dispatchWorkgroups(Math.ceil(mesh.vertexNormals.length / WORKGROUP_SIZE));
  computePasses.at(-1).end();

  // Encode a command to copy the results to a mappable buffer.
  // this is (from, to)
  if (debug) {
    encoder.copyBufferToBuffer(
      ctx.verticesBuffer,
      0,
      ctx.mappableVerticesResultBuffer,
      0,
      ctx.mappableVerticesResultBuffer.size
    );
    encoder.copyBufferToBuffer(
      ctx.facetNormalsBuffer,
      0,
      ctx.mappableFacetNormalsResultBuffer,
      0,
      ctx.mappableFacetNormalsResultBuffer.size
    );
    encoder.copyBufferToBuffer(
      ctx.vertexNormalsBuffer,
      0,
      ctx.mappableVertexNormalsResultBuffer,
      0,
      ctx.mappableVertexNormalsResultBuffer.size
    );
  }

  const renderPass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: context.getCurrentTexture().createView(),
        loadOp: "clear",
        clearValue: [0, 0, 0.4, 1.0],
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
  renderPass.setPipeline(renderPipelines[paneParams.shading]);
  renderPass.setBindGroup(0, ctx.renderBindGroups[paneParams.shading]);
  renderPass.setVertexBuffer(0, ctx.verticesBuffer);
  const now = uni.views.time[0];

  renderPass.setIndexBuffer(ctx.triangleIndicesBuffer, "uint32");
  // next line switches every TOGGLE_DURATION frames
  // switch ((now / uni.views.TOGGLE_DURATION) & 1) {
  // instead just compute based on subdivLevel
  renderPass.drawIndexed(
    mesh.levelCount[uni.views.subdivLevel[0]].t * 3 /* count */,
    1 /* instance */,
    mesh.levelBasePtr[uni.views.subdivLevel[0]].t * 3 /* start */
  );

  // End the render pass and submit the command buffer
  renderPass.end();

  // Finish encoding and submit the commands
  const commandBuffer = encoder.finish();
  device.queue.submit([commandBuffer]);

  // Read the results
  if (debug) {
    await ctx.mappableVerticesResultBuffer.mapAsync(GPUMapMode.READ);
    const verticesResult = new Float32Array(
      ctx.mappableVerticesResultBuffer.getMappedRange().slice()
    );
    ctx.mappableVerticesResultBuffer.unmap();
    await ctx.mappableFacetNormalsResultBuffer.mapAsync(GPUMapMode.READ);
    const facetNormalsResult = new Float32Array(
      ctx.mappableFacetNormalsResultBuffer.getMappedRange().slice()
    );
    ctx.mappableFacetNormalsResultBuffer.unmap();
    await ctx.mappableVertexNormalsResultBuffer.mapAsync(GPUMapMode.READ);
    const vertexNormalsResult = new Float32Array(
      ctx.mappableVertexNormalsResultBuffer.getMappedRange().slice()
    );
    ctx.mappableVertexNormalsResultBuffer.unmap();
    console.log("vertex buffer", verticesResult);
  }

  /* is this correct for getting timing info? */
  timingHelper.getResult().then((res) => {
    // console.log("Compute pass time:", res, "ns");
  });

  uni.views.time[0] = uni.views.time[0] + uni.views.timestep[0];
  // console.log("time", uni.views.time[0]);
  if (frameCount == 0) {
    return;
  } else if (frameCount > 0) {
    frameCount--;
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

function fail(msg) {
  // eslint-disable-next-line no-alert
  alert(msg);
}
