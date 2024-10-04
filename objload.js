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
  exclusive_prefix_sum(inList) {
    const outList = [];
    var sum = 0;
    for (var i = 0; i < inList.length; i++) {
      outList.push(sum);
      sum += inList[i];
    }
    return outList;
  }

  constructor(verticesIn, facesIn) {
    /* everything prefixed with "this." is a data structure that will go to the GPU */
    /* everything else is internal-only and will not be externally visible */
    /* strategy: build everything up as JS objects and then lower to Arrays on return */
    this.vertices = [];
    this.verticesReset = [];
    const facesInternal = []; // indexed by [level][face]
    facesInternal[0] = [];
    this.faces = []; // indexed per vertex
    this.triangles = [];
    this.faceValence = [];
    this.faceOffset = [];
    // next three: we will never access xOffsetPtr[0]
    this.faceOffsetPtr = [-1]; // indexed by level, points into faceOffset
    this.edgeOffsetPtr = [-1]; // indexed by level, points into edges
    this.vertexOffsetPtr = [-1]; // indexed by level, points into vertexOffset
    this.edges = [];
    this.vertexNeighbors = []; // array of arrays, flattened before sending to GPU
    this.vertexOffset = [];
    this.vertexValence = [];
    this.vertexIndex = [];
    this.vertexToTriangles = new Array(); // indexed by vertex number, has list of tri neighbors
    this.vertexToTrianglesOffset = [];
    this.vertexToTrianglesValence = [];
    this.vertexSize = 4; // # elements per vertex (ignore w coord for now)
    this.normalSize = 4; // float4s (ignore w coord for now)

    /** levelCount[L].x is the starting index into the vertices array for level L, point type x
     * levelBasePtr[L].x is the number of point type x in level L
     * levelBasePtr ~= exclusive-sum-scan(levelCount), mostly
     * now: populate level 0 of levelCount and levelBasePtr
     * assumes manifold surface!
     */
    this.levelCount = [
      new Level(
        facesIn.length,
        facesIn.length + verticesIn.length - 2, // <- manifold assumption
        verticesIn.length,
        0 // triangle starting point, will increment later
      ),
    ];
    // the only points in the vertex buffer @ level 0 are vertices, so all offsets are 0
    this.levelBasePtr = [new Level(0, 0, 0, 0)];

    this.scaleInput = true; // if true, scales output into [-1,1]^3 box
    this.largestInput = 0.0;
    this.maxLevel = 4; // valid levels are <= maxLevel

    /** OBJ stores faces in CCW order
     * The OBJ (or .OBJ) file format stores vertices in a counterclockwise order by default.
     *   This means that if the vertices are ordered counterclockwise around a face, both the face
     *   and the normal will point toward the viewer. If the vertices are ordered clockwise,
     *   both will point away from the viewer.
     * This means that going in declaration order means face is on the LEFT
     * If base_edges is
     *   v0 f0 v1 f1
     * then walking from v0 to v1 means f0 is on right and f1 is on left.
     */

    /** vertexNeighborsMap[level]: v -> [v0 f0 v1 f1 ...]
     *   where v# is the other end of an edge connecting v->v#
     *   and f# is the face to the left of v->v#
     */
    const vertexNeighborsMap = [new Map()];
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
    }
    if (this.scaleInput) {
      for (let i = 0; i < this.levelCount[0].v; i++) {
        this.vertices[i * this.vertexSize + 0] /= this.largestInput;
        this.vertices[i * this.vertexSize + 1] /= this.largestInput;
        this.vertices[i * this.vertexSize + 2] /= this.largestInput;
      }
    }
    /* cleanup, get rid of negative zeroes */
    this.vertices = this.vertices.map((num) => (num == -0 ? 0 : num));
    this.verticesReset = this.vertices.slice();

    /* Now do initial processing of faces from .obj */
    // convert 1-indexed to 0-indexed, copy into facesInternal
    this.faceOffsetPtr.push(this.faceOffset.length); // should be zero
    for (let i = 0; i < facesIn.length; i++) {
      facesInternal[0].push(
        facesIn[i].vertices.map((vtx) => vtx.vertexIndex - 1)
      );
      // every time we push to faces, also push to face{Offset, Valence}
      this.faceOffset.push(
        this.faceOffset.length == 0
          ? 0
          : this.faceOffset.at(-1) + this.faceValence.at(-1)
      );
      this.faceValence.push(facesInternal[0][i].length);
      // if we had normals or texture coords, deal with them here
      this.faces.push(...facesInternal[0][i]);
      /**
       * Now turn those faces into triangles; support arbitrary valence.
       * There are probably smarter ways to go face->triangles
       *   e.g., look at convexity
       */
      const valence = facesInternal[0][i].length;
      /* also triangulate this face and append it to triangles */
      for (let j = 2 - valence; j != 0; j++) {
        this.triangles.push(
          this.faces.at(-valence),
          this.faces.at(j - 1),
          this.faces.at(j)
          /* triangles: (-3, -2, -1)
           * quads: (-4, -3, -2) (-4, -2, -1) */
        );
        const triID = this.triangles.length / 3 - 1;
        for (let k = -3; k < 0; k++) {
          const vtxID = this.triangles.at(k);
          if (this.vertexToTriangles[vtxID] == undefined) {
            this.vertexToTriangles[vtxID] = [];
          }
          this.vertexToTriangles[vtxID].push(triID);
        }
      }
      this.levelCount[0].t += valence - 2;
    }

    /** This is the main loop, building up the data structures incrementally by level.
     * In each loop we will:
     * - Push and populate a new level{Count,BasePtr}[level]
     * - Loop over faces, build up data structures per edge
     * - Populate vertexNeighborsMap: vertex -> its neighbors
     * - Loop over faces again, subdivide them
     * - Loop over edges, generate indexes of neighboring vertices/faces
     * - Loop over vertexNeighborsMap, generate vertex data structures
     * - Lengthen vertices array (but don't fill it, it's empty beyond the
     *     initial vertices and those will be generated by the GPU)
     */
    for (var level = 1; level <= this.maxLevel; level++) {
      // f, e, v, t
      this.levelCount.push(
        new Level(facesInternal[level - 1].length, -1, -1, 0)
      );
      this.levelBasePtr.push(
        new Level(
          this.levelCount[level - 1].v + this.levelBasePtr[level - 1].v,
          this.levelCount[level - 1].v +
            this.levelBasePtr[level - 1].v +
            this.levelCount[level].f,
          -1,
          this.levelCount[level - 1].t + this.levelBasePtr[level - 1].t
        )
      );

      let edgePointID = this.levelBasePtr[level].e;
      // edgeToFace: [v_start,v_end] -> face on left
      const edgeToFace = new Map();
      // edgePointID: [v_start,v_end] -> edgePointID
      const edgeToEdgeID = new Map();
      // vertexNeighborsMap: list of neighbors per vertex
      vertexNeighborsMap.push(new Map());

      /* utility function to generate a string from a (e1, e2) tuple */
      /* this is used as a key in a Map */
      function edgeToKey(e1, e2) {
        return [e1, e2].join();
      }

      /** face loop: goal is to make a list of edges
       * input is faces from previous level (facesInternal)
       * outputs are edgeToFace, edgeToEdgeID (both are Maps)
       */
      facesInternal[level] = [];
      const seenVertices = new Set();
      for (
        let i = 0, facePointID = this.levelBasePtr[level].f;
        i < this.levelCount[level].f;
        i++, facePointID++
      ) {
        // loop through vertices in this face, recording edges
        const faceLen = facesInternal[level - 1][i].length;
        for (let j = 0; j < faceLen; j++) {
          let start = j;
          let end = start + 1;
          if (end >= faceLen) {
            // wraparound, last vertex <-> first one
            end = 0;
          }
          const startVtx = facesInternal[level - 1][i][start];
          const endVtx = facesInternal[level - 1][i][end];
          seenVertices.add(startVtx, endVtx);
          const edge = edgeToKey(startVtx, endVtx);
          const edgeRev = edgeToKey(endVtx, startVtx);
          if (edgeToFace.has(edge)) {
            console.log(`ERROR: edge ${edge} already in edgeToFace`);
          }
          edgeToFace.set(edge, facePointID);
          /**  in a manifold mesh, each edge will be set twice, so it's
           *   OK if it's already set; but if it sets here, it better be set twice */
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
      Array.from(seenVertices)
        .sort((a, b) => a - b) // numerical sort
        .forEach((v) => vertexNeighborsMap[level].set(v, []));

      // now we know how many edges there are, so finish level{Count,BasePtr}
      this.levelCount[level].e = edgeToEdgeID.size / 2;
      this.levelCount[level].v = seenVertices.size;
      this.levelBasePtr[level].v =
        this.levelBasePtr[level].e + this.levelCount[level].e;

      // all faces have been ingested, let's subdivide!
      // loop through all faces in previous level
      this.faceOffsetPtr.push(this.faceOffset.length);
      for (
        let i = 0, fPointsPtr = this.levelBasePtr[level].f;
        i < facesInternal[level - 1].length;
        i++, fPointsPtr++
      ) {
        // to make nomenclature easier, let's have tiny functions v and e
        // they have to be arrow functions to inherit "this" from the surrounding scope
        // v says "given vertex idx, what will be its v point?"
        const v = (idx) => {
          /** Complicated logic!
           * - Source is something in the face region of level-1
           * - So subtract off that bias
           * - Then map to the vertex region of level
           */
          return (
            this.levelBasePtr[level].v -
            this.levelBasePtr[level - 1].f +
            facesInternal[level - 1][i][idx]
          );
        };
        // e says "given two vertices, return the edge that connects them"
        const e = (v0, v1) => {
          const key = edgeToKey(
            facesInternal[level - 1][i][v0],
            facesInternal[level - 1][i][v1]
          );
          if (!edgeToEdgeID.has(key)) {
            console.log(
              `ERROR [l=${level}]: edgeToEdgeID (${v0}, ${v1}) does not have key ${key} `
            );
            console.log(edgeToEdgeID);
          }
          return edgeToEdgeID.get(key);
        };
        /* now we do the subdivision, push both quads and triangles */
        const mod = (n, d) => {
          // wraps negative numbers sensibly
          return ((n % d) + d) % d;
        };
        const valence = facesInternal[level - 1][i].length;
        /** this looks more complicated than it is
         * for quads (e.g.) the first 2 faces are:
         *   fPointsPtr, e(3,0), v(0), v(0,1)
         *   fPointsPtr, e(1,0), v(1), v(1,2)
         */
        for (let j = 0; j < valence; j++) {
          facesInternal[level].push([
            /* one quad (per vertex of input face) */
            fPointsPtr,
            e(mod(j - 1, valence), j),
            v(j),
            e(j, mod(j + 1, valence)),
          ]);
          this.faces.push(...facesInternal[level].at(-1)); // same as above
          this.faceOffset.push(
            this.faceOffset.at(-1) + this.faceValence.at(-1)
          );
          this.faceValence.push(facesInternal[level].at(-1).length);
          this.triangles.push(
            /** there exists likely a smarter way of subdividing the quad:
             * what if (e.g.) it's non-convex? we should measure
             * both diagonals before deciding */
            /* first tri of above quad */
            fPointsPtr,
            e(mod(j - 1, valence), j),
            v(j),
            /* second tri of above quad */
            fPointsPtr,
            v(j),
            e(j, mod(j + 1, valence))
          );
          var triID = this.triangles.length / 3 - 2;
          for (let k = -6; k < -3; k++) {
            const vtxID = this.triangles.at(k);
            if (this.vertexToTriangles[vtxID] == undefined) {
              this.vertexToTriangles[vtxID] = [];
            }
            this.vertexToTriangles[vtxID].push(triID);
          }
          triID++;
          for (let k = -3; k < 0; k++) {
            const vtxID = this.triangles.at(k);
            if (this.vertexToTriangles[vtxID] == undefined) {
              this.vertexToTriangles[vtxID] = [];
            }
            this.vertexToTriangles[vtxID].push(triID);
          }
        }
        this.levelCount[level].t += valence * 2;
      }
      // now we have a map (edgeToFace) full of {edge -> face}
      //   and a map (edgeToEdgeID) full of {edge -> edgeID}
      // we iterate over edgeToEdgeID because its entry order
      //   is the canonical order
      // edgeOffsetPtr[level]: starting point for edges for this level
      this.edgeOffsetPtr.push(this.edges.length / 4);
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
            console.log(
              "  ",
              reverseEdge,
              " not in edgeToEdgeID (reverseEdge)"
            );
          }
        }
        const f = [edgeToFace.get(edge), edgeToFace.get(reverseEdge)];
        // push into edge array: v[0], f[0], v[1], f[1]
        // iterating through edgeToEdgeID ensures a consistent order
        if (f[1] > f[0]) {
          // in Niessner, all edges have f[1] > f[0]
          this.edges.push(v[0], f[0], v[1], f[1]);
          vertexNeighborsMap[level].get(v[0]).push(v[1], f[1]);
          vertexNeighborsMap[level].get(v[1]).push(v[0], f[0]);
        }
      });

      // now we populate vertexNeighbors
      this.vertexOffsetPtr.push(this.vertexNeighbors.length); // # of vertices
      var vertexOffset = this.vertexNeighbors.flat().length; // # of neighbors
      vertexNeighborsMap[level].forEach((neighbors, vertex) => {
        this.vertexIndex.push(vertex);
        this.vertexNeighbors.push(neighbors);
        this.vertexValence.push(neighbors.length / 2);
        this.vertexOffset.push(vertexOffset);
        vertexOffset += neighbors.length;
      });

      this.verticesSize = this.levelBasePtr[level].v + this.levelCount[level].v;
      // this seems weird, but I can just lengthen an array by setting its length?
      this.vertices.length = this.verticesSize * this.vertexSize;
    }

    // Now wrap everything in JS arrays and return from constructor
    this.vertices = new Float32Array(this.vertices);
    this.faces = new Uint32Array(this.faces);
    this.edges = new Uint32Array(this.edges);
    this.triangles = new Uint32Array(this.triangles);
    this.faceValence = new Uint32Array(this.faceValence);
    this.faceOffset = new Uint32Array(this.faceOffset);
    this.faceOffsetPtr = new Uint32Array(this.faceOffsetPtr);
    this.edgeOffsetPtr = new Uint32Array(this.edgeOffsetPtr);
    this.vertexOffsetPtr = new Uint32Array(this.vertexOffsetPtr);
    this.vertexValence = new Uint32Array(this.vertexValence);
    this.vertexOffset = new Uint32Array(this.vertexOffset);
    this.vertexIndex = new Uint32Array(this.vertexIndex);
    this.vertexNeighbors = new Uint32Array(this.vertexNeighbors.flat());
    this.vertexToTrianglesValence = this.vertexToTriangles.map(
      (tris) => tris.length
    );
    this.vertexToTrianglesOffset = new Uint32Array(
      this.exclusive_prefix_sum(this.vertexToTrianglesValence)
    );
    this.vertexToTrianglesValence = new Uint32Array(
      this.vertexToTrianglesValence
    );
    this.vertexToTriangles = new Uint32Array(this.vertexToTriangles.flat());
    /* the next two arrays are empty and will be filled by the GPU */
    this.vertexNormals = new Float32Array(this.verticesSize * this.normalSize);
    this.facetNormals = new Float32Array(
      (this.triangles.length * this.normalSize) / 3
    );
  }
}

// entry point: urlToMesh(url) -> parsed Mesh
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
