// const fileContents =
//   "v 0 0 0 \n" + "v 0 1 0 \n" + "v 1 0 0 \n" + "f 1 2 3";
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
  f = -1;
  e = -1;
  v = -1;
  constructor(f, e, v) {
    this.f = f;
    this.e = e;
    this.v = v;
  }
}

class SubdivMesh {
  constructor(verticesIn, facesIn) {
    this.vertices = []; // why can't I do new Float32Array?
    this.faces = []; // indexed per vertex
    this.triangles = [];
    this.face_valence = [];
    this.face_offset = [];
    this.edges = [];
    this.base_vertices = [];
    this.vertex_offset = [];
    this.vertex_valence = [];
    this.vertex_index = [];
    this.points_count = [];
    const vertex_size = 4; // # elements per vertex
    const initial_vertex_count = verticesIn.length;
    this.level_count = [new Level(initial_vertex_count, 0, 0)];
    this.level_base_ptr = [
      new Level(0, initial_vertex_count, initial_vertex_count),
      new Level(initial_vertex_count, initial_vertex_count + facesIn.length),
    ];
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
    for (let i = 0; i < this.level_count[0].v; i++) {
      this.vertices.push(
        verticesIn[i].x,
        verticesIn[i].y,
        verticesIn[i].z,
        1.0
      );
      vertexNeighborsMap.set(i, []);
    }

    this.level_count.push(new Level(facesIn.length));

    let edgePointID = this.level_base_ptr[1].e;
    // edgeToFace: [v_start,v_end] -> face on left
    const edgeToFace = new Map();
    // edgePointID: [v_start,v_end] -> edgePointID
    const edgeToEdgeID = new Map();
    for (
      let i = 0, face_offset = 0, f_points_ptr = this.level_base_ptr[level].f;
      i < facesIn.length;
      i++, f_points_ptr++
    ) {
      // i indexes the face from the input file
      // face_offset indexes individual vertices within the faces
      //   in the input file
      // f_points_ptr indexes into the output vertex array
      this.face_offset.push(face_offset);
      this.face_valence.push(facesIn[i].vertices.length);
      face_offset += facesIn[i].vertices.length;
      const thisFacePtr = this.faces.length;
      for (let j = 0; j < facesIn[i].vertices.length; j++) {
        // here is where we do the 1-indexed to 0-indexed conversion
        //   (the vertexIndex - 1 below)
        // if we wanted normals or texture coordinates from the obj
        // file, here's where we'd record them
        this.faces.push(facesIn[i].vertices[j].vertexIndex - 1);
      }
      // same loop through vertices in this face,
      // but record edges this time
      const v_base = thisFacePtr;
      for (let j = 0; j < facesIn[i].vertices.length; j++) {
        const start = v_base + j;
        let end = start + 1;
        if (end >= this.faces.length) {
          // wraparound, last vertex <-> first one
          end = v_base;
        }
        const edge = edgeToKey(this.faces[start], this.faces[end]);
        const edgeRev = edgeToKey(this.faces[end], this.faces[start]);
        if (edgeToFace.has(edge)) {
          console.log(`ERROR: edge ${edge} already in edgeToFace`);
        }
        edgeToFace.set(edge, f_points_ptr);
        /* in a manifold mesh, each edge will be set twice, so it's
        OK if it's already set; but if it sets here, it better set twice */
        if (edgeToEdgeID.has(edge) ^ edgeToEdgeID.has(edgeRev)) {
          console.log(
            `Inconsistent edges in edgeToEdgeID: ${edge}, ${edgeRev}`
          );
        } else if (!edgeToEdgeID.has(edge)) {
          edgeToEdgeID.set(edge, edgePointID);
          edgeToEdgeID.set(edgeRev, edgePointID);
          edgePointID++;
        }
      }
      switch (facesIn[i].vertices.length) {
        case 3: // triangle
          console.log(
            `Subdividing tri with vertices ${this.faces[v_base]}, ${
              this.faces[v_base + 1]
            }, ${this.faces[v_base + 2]}\n
            Face point = ${f_points_ptr}\n
            Edge ${this.faces[v_base]}->${
              this.faces[v_base + 1]
            } is ${edgeToEdgeID.get(
              edgeToKey(this.faces[v_base], this.faces[v_base + 1])
            )}\n
            Edge ${this.faces[v_base + 1]}->${
              this.faces[v_base + 2]
            } is ${edgeToEdgeID.get(
              edgeToKey(this.faces[v_base + 1], this.faces[v_base + 2])
            )}\n
            Edge ${this.faces[v_base + 2]}->${
              this.faces[v_base]
            } is ${edgeToEdgeID.get(
              edgeToKey(this.faces[v_base + 2], this.faces[v_base])
            )}\n
            Output quads:\n
            ${f_points_ptr}, x, ${this.faces[v_base]}, y\n
            ${f_points_ptr}, x, ${this.faces[v_base + 1]}, y\n
            ${f_points_ptr}, x, ${this.faces[v_base + 2]}, y\n
            `
          );
          break;
        case 4: // quad
          console.log(
            `Subdividing quad with vertices ${this.faces[v_base]}, ${
              this.faces[v_base + 1]
            }, ${this.faces[v_base + 2]}, ${this.faces[v_base + 3]}`
          );
          break;
        default:
          console.log(
            "ERROR: Mesh data structure currently only supports faces of valence 3 or 4"
          );
          break;
      }
    }
    // now we have a map (edgeToFace) full of {edge -> face}
    // and a map (edgeToEdgeID) full of {edge -> edgeID}
    edgeToFace.forEach((face, edge) => {
      const v = edge.split(",").map((n) => parseInt(n, 10));
      const reverseEdge = edgeToKey(v[1], v[0]);
      if (!edgeToFace.has(reverseEdge)) {
        // if we have a manifold mesh, every edge has two faces, one
        //   in each direction of the edge
        // let's assert that
        console.log(
          `ERROR: non-manifold surface, ${reverseEdge} not in edgeToFace`
        );
      }
      const f = [edgeToFace.get(reverseEdge), face];
      // push into edge array: v[0], f[0], v[1], f[1]
      // order is arbitrary but edgeToEdgeID should take care of that
      if (f[1] > f[0]) {
        // in Niessner, all edges have f[1] > f[0]
        this.edges.push(v[0], f[0], v[1], f[1]);
        vertexNeighborsMap.get(v[0]).push(v[1], f[1]);
        vertexNeighborsMap.get(v[1]).push(v[0], f[0]);
      }
    });

    // now we populate base_vertices
    var vertex_offset = 0;
    vertexNeighborsMap.forEach((neighbors, vertex) => {
      this.vertex_index.push(vertex);
      this.base_vertices.push(neighbors);
      this.vertex_valence.push(neighbors.length / 2);
      this.vertex_offset.push(vertex_offset);
      vertex_offset += neighbors.length;
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
