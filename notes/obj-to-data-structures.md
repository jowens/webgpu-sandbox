# How to go from a parsed .obj file to the starting data structures for subdivision

Here's the data structures we need to derive:

```
vertices_size         # how many vertices in level-0 + level-1
                      # this is the entire vertex buffer
base_vertex_positions # given in .obj
base_faces            # given in .obj
subdiv_1_faces        # must be generated
triangle_indices      # f(base_faces, base_face_valence)
                      #   and subdiv_1_{faces, valence}
subdiv_1_triangles_count
base_face_valence     # f(base_faces)
base_face_offset      # exclusive-scan(+, base_faces)
base_edges            # f(?)
base_vertex_valence   # f(?)
base_vertex_offset    # 2 * exclusive-scan(+, base_vertex_vlnc)
base_vertex_index     # f(?)
base_vertices         # f(?)
```

## Proposed data structure changes in subdivider code

`offset` and `valence` can probably be put into one data structure that covers all vertices (and is the size of the vertex buffer). `valence` could be done as a segmented scan if necessary.

Be better at explicitly exposing hierarchy (many data structures will be indexed internally by hierarchy level)

## Q: When to convert from 1-based indexing (in .obj) vs. 0-based indexing (in subdivision code)?

Probably after parsing (don't change the parser), then we can update the parser / get a different parser and use that unchanged.
