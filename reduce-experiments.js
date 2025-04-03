/**
 * Input: Pointer to memory
 * Output: Local (function) value in a register for every thread
 * Action: Load one word per thread at its gid, reduce over all threads, return reduction
 */
const reduceWGSL1 = /* wgsl */ `
enable subgroups;
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<storage, read> in: array<u32>;

var<workgroup> wg_temp: array<atomic<u32>, 1>;

fn reduceWorkgroup(input: ptr<storage, array<u32>, read>,
                   gid: vec3u,
                  ) -> u32 {
  atomicAdd(&wg_temp[0], input[gid.x]);
  workgroupBarrier();
  return atomicLoad(&wg_temp[0]);
}

@compute @workgroup_size(128) fn reduceKernel(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_index) lidx: u32,
  @builtin(workgroup_id) wgid: vec3u) {
  let r = reduceWorkgroup(&in, gid);
  if (lidx == 0) {
    out[wgid.x] = r;
  }
}`;

export const reduceWGSL = reduceWGSL1;
