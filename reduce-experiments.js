/**
 * Input: Pointer to memory
 * Output: Local (function) value in a register for every thread
 * Action: Load one word per thread at its gid, reduce over all threads, return reduction
 */
const reduceWGSL1 = ({ args }) => {
  return /* wgsl */ `
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
};

const reduceWGSL2 = (args) => {
  return /* wgsl */ `
enable subgroups;
@group(0) @binding(0) var<storage, read_write> out: array<u32>;
@group(0) @binding(1) var<storage, read> in: array<u32>;

const BLOCK_DIM: u32 = ${args.workgroupSize};
const TEMP_WG_MEM_SIZE = BLOCK_DIM / ${args.MIN_SUBGROUP_SIZE};
var<workgroup> wg_temp: array<u32, TEMP_WG_MEM_SIZE>;

fn reduceWorkgroup(input: ptr<storage, array<u32>, read>,
                   gid: vec3u,
                   lidx: u32,
                   sgid: u32,
                   sgsz: u32
                  ) -> u32 {
  let sid = lidx / sgsz;
  let lane_log = u32(countTrailingZeros(sgsz)); /* log_2(sgsz) */
  let local_spine: u32 = BLOCK_DIM >> lane_log; /* BLOCK_DIM / subgroup size; how
                                                 * many partial reductions in this tile? */
  let aligned_size_base = 1u << ((u32(countTrailingZeros(local_spine)) + lane_log - 1u) / lane_log * lane_log);
    /* fix for aligned_size_base == 1 (needed when subgroup_size == BLOCK_DIM) */
  let aligned_size = select(aligned_size_base, BLOCK_DIM, aligned_size_base == 1);

  let t_red = in[gid.x];
  let s_red = subgroupAdd(t_red);
  if (sgid == 0u) {
    wg_temp[sid] = s_red;
  }
  workgroupBarrier();
  var f_red: u32 = 0;

  var offset = 0u;
  var top_offset = 0u;
  let lane_pred = sgid == sgsz - 1u;
  if (sgsz > aligned_size) {
    /* don't enter the loop */
    f_red = wg_temp[lidx + top_offset];
  } else {
    for (var j = sgsz; j <= aligned_size; j <<= lane_log) {
      let step = local_spine >> offset;
      let pred = lidx < step;
      f_red = subgroupAdd(select(0,
                                 wg_temp[lidx + top_offset],
                                 pred));
      if (pred && lane_pred) {
        wg_temp[sid + step + top_offset] = f_red;
      }
      workgroupBarrier();
      top_offset += step;
      offset += lane_log;
    }
  }
  return f_red;
}

@compute @workgroup_size(BLOCK_DIM) fn reduceKernel(
  @builtin(global_invocation_id) gid: vec3u,
  @builtin(local_invocation_index) lidx: u32,
  @builtin(workgroup_id) wgid: vec3u,
  @builtin(subgroup_invocation_id) sgid: u32,
  @builtin(subgroup_size) sgsz: u32) {
  let r = reduceWorkgroup(&in, gid, lidx, sgid, sgsz);
  if (lidx == 0) {
    out[wgid.x] = r;
  }
}`;
};

export const reduceWGSL = reduceWGSL2;
