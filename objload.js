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

const objFile = new OBJFile(fileContents);
const parsedObj = objFile.parse();

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
    this.vertices_ptr = [];

    const vertex_size = 4; // # elements per vertex

    // OBJ stores faces in CCW order
    // The OBJ (or .OBJ) file format stores vertices in a counterclockwise order by default. This means that if the vertices are ordered counterclockwise around a face, both the face and the normal will point toward the viewer. If the vertices are ordered clockwise, both will point away from the viewer.
    // this means that going in declaration order means face is on the LEFT
    // base_edges:
    // v0 f0 v1 f1
    // walking from v0 to v1 means f0 is on right and f1 is on left

    const vertexNeighborsMap = new Map();
    for (let i = 0; i < verticesIn.length; i++) {
      this.vertices.push(
        verticesIn[i].x,
        verticesIn[i].y,
        verticesIn[i].z,
        1.0
      );
      vertexNeighborsMap.set(i, []);
    }

    // this.vertices_ptr = number of base vertices
    this.vertices_ptr.push(this.vertices.length / vertex_size);
    const edgeMap = new Map();
    for (
      let i = 0, face_offset = 0, f_points_ptr = this.vertices_ptr[0];
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
      for (
        let j = 0, v_base = thisFacePtr;
        j < facesIn[i].vertices.length;
        j++
      ) {
        const start = v_base + j;
        let end = start + 1;
        if (end >= this.faces.length) {
          // wraparound, last vertex <-> first one
          end = v_base;
        }
        const edge = [this.faces[start], this.faces[end]].join();
        if (edgeMap.has(edge)) {
          console.log(`ERROR: edge ${edge} already in edgeMap`);
        }
        edgeMap.set(edge, f_points_ptr);
      }
    }
    // now we have a map (edgeMap) full of {edge -> face}
    edgeMap.forEach((face, edge) => {
      const v = edge.split(",").map((n) => parseInt(n, 10));
      const reverseEdge = [v[1], v[0]].join();
      if (!edgeMap.has(reverseEdge)) {
        // if we have a manifold mesh, every edge has two faces, one
        //   in each direction of the edge
        // let's assert that
        console.log(
          `ERROR: non-manifold surface, ${reverseEdge} not in edgeMap`
        );
      }
      const f = [edgeMap.get(reverseEdge), face];
      // push into edge array: v[0], f[0], v[1], f[1]
      // order might be a problem: it'll be in pushed-in order
      //   not worrying about order of edge points in vertex array for now
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
const mesh = new SubdivMesh(
  parsedObj.models[0].vertices,
  parsedObj.models[0].faces
);

async function urlToMesh(url) {
  const response = await fetch(url);
  const objtext = await response.text();
  const objFile = new OBJFile(objtext);
  const parsedObj = objFile.parse();
  const mesh = new SubdivMesh(
    parsedObj.models[0].vertices,
    parsedObj.models[0].faces
  );
  return mesh;
}
