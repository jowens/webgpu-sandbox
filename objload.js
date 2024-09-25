const fileContents = `
# square pyramid from Niessner et al. (2012)
v 0 0 1
v -1 1 0
v -1 -1 0
v 1 -1 0
v 1 1 0
f 1 2 3
f 1 3 4
f 1 4 5
f 1 5 2
f 5 4 3 2
`;
// 2 = V - E + F = 5 - 8 + 5

const objFile = new OBJFile(fileContents);
const parsedObj = objFile.parse();

function edgeToKey(e1, e2) {
  return [e1, e2].join();
}

// generic class to encapsulate counts/offsets per level
class Level {
  constructor(f, e, v, t) {
    this.f = f;
    this.e = e;
    this.v = v;
    this.t = t;
  }
}

class SubdivMesh {
  constructor(verticesIn, facesIn) {
    /* everything prefixed with "this." is a data structure that will go to the GPU */
    /* everything else is internal-only and will not be externally visible */
    this.vertices = []; // why can't I do new Float32Array?
    this.verticesReset = [];
    this.faces = []; // indexed per vertex
    this.triangles = [];
    this.faceValence = [];
    this.faceOffset = [];
    this.edges = [];
    this.baseVertices = [];
    this.vertexOffset = [];
    this.vertexValence = [];
    this.vertexIndex = [];
    const vertexSize = 4; // # elements per vertex
    const initialVertexCount = verticesIn.length;
    this.levelCount = [new Level(0, 0, initialVertexCount, 0)];
    this.levelBasePtr = [new Level(0, 0, 0, -1)];
    this.scaleInput = true;
    this.largestInput = 0.0;
    const level = 1; // will loop through levels later
    // OBJ stores faces in CCW order
    // The OBJ (or .OBJ) file format stores vertices in a counterclockwise order by default. This means that if the vertices are ordered counterclockwise around a face, both the face and the normal will point toward the viewer. If the vertices are ordered clockwise, both will point away from the viewer.
    // this means that going in declaration order means face is on the LEFT
    // base_edges:
    // v0 f0 v1 f1
    // walking from v0 to v1 means f0 is on right and f1 is on left

    // vertexNeighborsMap: v -> [v0 f0 v1 f1 ...]
    //   where v# is the other end of an edge connecting v->v#
    //   and f# is the face to the left of v->v#
    const vertexNeighborsMap = new Map();
    for (let i = 0; i < this.levelCount[0].v; i++) {
      this.vertices.push(
        verticesIn[i].x,
        verticesIn[i].y,
        verticesIn[i].z,
        1.0
      );
      this.largestInput = Math.abs(
        Math.max(
          Math.abs(verticesIn[i].x),
          Math.abs(verticesIn[i].y),
          Math.abs(verticesIn[i].z),
          this.largestInput
        )
      );
      vertexNeighborsMap.set(i, []);
    }
    if (this.scaleInput) {
      for (let i = 0; i < this.levelCount[0].v; i++) {
        this.vertices[i * vertexSize + 0] /= this.largestInput;
        this.vertices[i * vertexSize + 1] /= this.largestInput;
        this.vertices[i * vertexSize + 2] /= this.largestInput;
      }
    }
    /* cleanup, get rid of negative zeroes */
    this.vertices = this.vertices.map((num) => (num == -0 ? 0 : num));
    this.verticesReset = this.vertices.slice();

    /** calculating these offsets and lengths is a pain,
     * because we really have to walk the entire input data
     * structure just to get the counts
     *
     * also we will have different ways to do this between
     * the initial read data structure from the file and
     * our internal data structure
     *
     * one possibility will be to convert the .obj data
     * structure to ours at the outset, and then have uniform
     * handling as we loop
     *
     * another possibility is to have loops at the beginning
     * of the level calculation that just compute all the
     * counts/offsets
     */
    this.levelCount.push(
      new Level(
        facesIn.length,
        facesIn.length + initialVertexCount - 2,
        initialVertexCount,
        0
      )
    );
    this.levelBasePtr.push(
      new Level(
        initialVertexCount,
        initialVertexCount + this.levelCount[level].f,
        initialVertexCount +
          this.levelCount[level].f +
          this.levelCount[level].e,
        -1
      )
    );

    console.log("Counts:   ", this.levelCount);
    console.log("Base ptr: ", this.levelBasePtr);

    let edgePointID = this.levelBasePtr[1].e;
    // edgeToFace: [v_start,v_end] -> face on left
    const edgeToFace = new Map();
    // edgePointID: [v_start,v_end] -> edgePointID
    const edgeToEdgeID = new Map();

    /* how many edges are there? Could compute in two ways:
     * - Euler characteristic E = V + F - 2 (manifold only)
     * - Walk through faces, count edges
     */

    for (
      let i = 0, faceOffset = 0, fPointsPtr = this.levelBasePtr[level].f;
      i < facesIn.length;
      i++, fPointsPtr++
    ) {
      // i indexes the face from the input file
      // faceOffset indexes individual vertices within the faces
      //   in the input file
      // fPointsPtr indexes into the output vertex array
      this.faceOffset.push(faceOffset);
      this.faceValence.push(facesIn[i].vertices.length);
      faceOffset += facesIn[i].vertices.length;
      const thisFacePtr = this.faces.length;
      for (let j = 0; j < facesIn[i].vertices.length; j++) {
        // here is where we do the 1-indexed to 0-indexed conversion
        //   (the vertexIndex - 1 below)
        // if we wanted normals or texture coordinates from the obj
        // file, here's where we'd record them
        this.faces.push(facesIn[i].vertices[j].vertexIndex - 1);
      }
      switch (facesIn[i].vertices.length) {
        case 3: // face is a triangle
          this.triangles.push(
            this.faces.at(-3),
            this.faces.at(-2),
            this.faces.at(-1)
          );
          this.levelCount[0].t += 1;
          break;
        case 4: // face is a quad
          this.triangles.push(
            this.faces.at(-4),
            this.faces.at(-3),
            this.faces.at(-2),
            this.faces.at(-4),
            this.faces.at(-2),
            this.faces.at(-1)
          );
          this.levelCount[0].t += 2;
          break;
        default:
          console.log(
            `ERROR: Face ${i} has valence ${facesIn[i].vertices.length}, can only support 3 and 4 currently`
          );
          break;
      }
      // same loop through vertices in this face,
      // but record edges this time
      const vBase = thisFacePtr;
      for (let j = 0; j < facesIn[i].vertices.length; j++) {
        const start = vBase + j;
        let end = start + 1;
        if (end >= this.faces.length) {
          // wraparound, last vertex <-> first one
          end = vBase;
        }
        const edge = edgeToKey(this.faces[start], this.faces[end]);
        const edgeRev = edgeToKey(this.faces[end], this.faces[start]);
        if (edgeToFace.has(edge)) {
          console.log(`ERROR: edge ${edge} already in edgeToFace`);
        }
        edgeToFace.set(edge, fPointsPtr);
        /**  in a manifold mesh, each edge will be set twice, so it's
         *   OK if it's already set; but if it sets here, it better set twice */
        if (edgeToEdgeID.has(edge) ^ edgeToEdgeID.has(edgeRev)) {
          console.log(
            `ERROR: Inconsistent edges in edgeToEdgeID: ${edge}, ${edgeRev}`
          );
        } else if (!edgeToEdgeID.has(edge)) {
          edgeToEdgeID.set(edge, edgePointID);
          edgeToEdgeID.set(edgeRev, edgePointID);
          edgePointID++;
        }
      }
    }

    // all faces have been ingested, let's subdivide!
    // XXX WRONG probably want to set vBase smarter than 0
    for (
      let vBase = 0, i = 0, fPointsPtr = this.levelBasePtr[level].f;
      i < facesIn.length;
      vBase += facesIn[i].vertices.length, i++, fPointsPtr++
    ) {
      // to make nomenclature easier, let's have tiny functions v and e
      // they have to be arrow functions to inherit "this" from the surrounding scope
      const v = (idx) => {
        return this.levelBasePtr[level].v + this.faces[vBase + idx];
      };
      const e = (v0, v1) => {
        return edgeToEdgeID.get(
          edgeToKey(this.faces[vBase + v0], this.faces[vBase + v1])
        );
      };
      switch (facesIn[i].vertices.length) {
        case 3: // triangle
          // build quads and triangles!
          // prettier-ignore
          this.faces.push( // three quads
            fPointsPtr, e(0, 2), v(0), e(0, 1),
            fPointsPtr, e(0, 1), v(1), e(1, 2),
            fPointsPtr, e(1, 2), v(2), e(0, 2)
          );
          // prettier-ignore
          this.triangles.push(
            /** TODO: There is a right way to subdivide quads->tris
             * need to compare both diagonals & pick the better one
             * especially if this is concave */
            fPointsPtr, e(0, 2), v(0),
            fPointsPtr, v(0), e(0, 1),
            fPointsPtr, e(0, 1), v(1),
            fPointsPtr, v(1), e(1, 2),
            fPointsPtr, e(1, 2), v(2),
            fPointsPtr, v(2), e(0, 2),
          );
          this.levelCount[level].t += 6;
          break;
        case 4: // quad
          //  `Subdividing quad with vertices ${this.faces[vBase]}, ${
          //    this.faces[vBase + 1]
          //  }, ${this.faces[vBase + 2]}, ${this.faces[vBase + 3]}`
          // prettier-ignore
          this.faces.push( // four quads
            /* TODO: Am I picking the right quads to subdivide? It's an octagon */
            fPointsPtr, v(0), e(0, 1), v(1),
            fPointsPtr, v(1), e(1, 2), v(2),
            fPointsPtr, v(2), e(2, 3), v(3),
            fPointsPtr, v(3), e(3, 0), v(0),
          );
          // prettier-ignore
          this.triangles.push(
            /** TODO: There is a right way to subdivide quads->tris
             * need to compare both diagonals & pick the better one
             * especially if this is concave */
            /* TODO: Am I picking the right quads to subdivide? It's an octagon */
            fPointsPtr, v(0), e(0, 1),
            fPointsPtr, e(0, 1), v(1),
            fPointsPtr, v(1), e(1, 2),
            fPointsPtr, e(1, 2), v(2),
            fPointsPtr, v(2), e(2, 3),
            fPointsPtr, e(2, 3), v(3),
            fPointsPtr, v(3), e(3, 0),
            fPointsPtr, e(3, 0), v(0),
          );
          this.levelCount[level].t += 8;
          break;
        default:
          console.log(
            "ERROR: Mesh data structure currently only supports faces of valence 3 or 4"
          );
          break;
      }
    }
    // now we have a map (edgeToFace) full of {edge -> face}
    //   and a map (edgeToEdgeID) full of {edge -> edgeID}
    // we iterate over edgeToEdgeID because its entry order
    //   is the canonical order
    edgeToEdgeID.forEach((edgeID, edge) => {
      const v = edge.split(",").map((n) => parseInt(n, 10));
      const reverseEdge = edgeToKey(v[1], v[0]);
      if (
        !edgeToFace.has(reverseEdge) ||
        !edgeToFace.has(edge) ||
        !edgeToEdgeID.has(reverseEdge)
      ) {
        // if we have a manifold mesh, every edge has two faces, one
        //   in each direction of the edge
        // let's assert that
        console.log("ERROR: non-manifold surface");
        if (!edgeToFace.has(edge)) {
          console.log("  ", edge, " not in edgeToFace (edge)");
        }
        if (!edgeToFace.has(reverseEdge)) {
          console.log("  ", reverseEdge, " not in edgeToFace (reverseEdge)");
        }
        if (!edgeToEdgeID.has(reverseEdge)) {
          console.log("  ", reverseEdge, " not in edgeToEdgeID (reverseEdge)");
        }
      }
      const f = [edgeToFace.get(edge), edgeToFace.get(reverseEdge)];
      // push into edge array: v[0], f[0], v[1], f[1]
      // iterating through edgeToEdgeID ensures a consistent order
      if (f[1] > f[0]) {
        // in Niessner, all edges have f[1] > f[0]
        this.edges.push(v[0], f[0], v[1], f[1]);
        vertexNeighborsMap.get(v[0]).push(v[1], f[1]);
        vertexNeighborsMap.get(v[1]).push(v[0], f[0]);
      }
    });

    // now we populate baseVertices
    var vertexOffset = 0;
    vertexNeighborsMap.forEach((neighbors, vertex) => {
      this.vertexIndex.push(vertex);
      this.baseVertices.push(neighbors);
      this.vertexValence.push(neighbors.length / 2);
      this.vertexOffset.push(vertexOffset);
      vertexOffset += neighbors.length;
    });
  }
}
// const mesh = new SubdivMesh(
//   parsedObj.models[0].vertices,
//  parsedObj.models[0].faces
//);

async function urlToMesh(url) {
  const response = await fetch(url);
  const objtext = await response.text();
  const objFile = new OBJFile(objtext);
  const parsedObj = objFile.parse();
  console.log(parsedObj);
  const mesh = new SubdivMesh(
    parsedObj.models[0].vertices,
    parsedObj.models[0].faces
  );
  // 2 = V - E + F
  return mesh;
}
