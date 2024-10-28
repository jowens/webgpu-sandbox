"use strict";
const { create, globals } = require("../../src/dawn-build/dawn.node");
Object.assign(globalThis, globals); // Provides constants like GPUBufferUsage.MAP_READ
let navigator = { gpu: create([]) };

// begin TimingHelper code
function assert(cond, msg = "") {
  if (!cond) {
    throw new Error(msg);
  }
}

class TimingHelper {
  #canTimestamp;
  #device;
  #querySet;
  #resolveBuffer;
  #resultBuffer;
  #resultBuffers = [];
  // state can be 'free', 'need resolve', 'wait for result'
  #state = "free";

  constructor(device) {
    this.#device = device;
    this.#canTimestamp = device.features.has("timestamp-query");
    if (this.#canTimestamp) {
      this.#querySet = device.createQuerySet({
        type: "timestamp",
        count: 2,
      });
      this.#resolveBuffer = device.createBuffer({
        size: this.#querySet.count * 8,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
    }
  }

  #beginTimestampPass(encoder, fnName, descriptor) {
    if (this.#canTimestamp) {
      assert(this.#state === "free", "state not free");
      this.#state = "need resolve";

      const pass = encoder[fnName]({
        ...descriptor,
        ...{
          timestampWrites: {
            querySet: this.#querySet,
            beginningOfPassWriteIndex: 0,
            endOfPassWriteIndex: 1,
          },
        },
      });

      const resolve = () => this.#resolveTiming(encoder);
      pass.end = (function (origFn) {
        return function () {
          origFn.call(this);
          resolve();
        };
      })(pass.end);

      return pass;
    } else {
      return encoder[fnName](descriptor);
    }
  }

  beginRenderPass(encoder, descriptor = {}) {
    return this.#beginTimestampPass(encoder, "beginRenderPass", descriptor);
  }

  beginComputePass(encoder, descriptor = {}) {
    return this.#beginTimestampPass(encoder, "beginComputePass", descriptor);
  }

  #resolveTiming(encoder) {
    if (!this.#canTimestamp) {
      return;
    }
    assert(this.#state === "need resolve", "must call addTimestampToPass");
    this.#state = "wait for result";

    this.#resultBuffer =
      this.#resultBuffers.pop() ||
      this.#device.createBuffer({
        size: this.#resolveBuffer.size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });

    encoder.resolveQuerySet(
      this.#querySet,
      0,
      this.#querySet.count,
      this.#resolveBuffer,
      0
    );
    encoder.copyBufferToBuffer(
      this.#resolveBuffer,
      0,
      this.#resultBuffer,
      0,
      this.#resultBuffer.size
    );
  }

  async getResult() {
    if (!this.#canTimestamp) {
      return 0;
    }
    assert(this.#state === "wait for result", "must call resolveTiming");
    this.#state = "free";

    const resultBuffer = this.#resultBuffer;
    await resultBuffer.mapAsync(GPUMapMode.READ);
    const times = new BigInt64Array(resultBuffer.getMappedRange());
    console.log(
      "Returned timestamps in result buffer:",
      Number(times[1]),
      Number(times[0])
    );
    const duration = Number(times[1] - times[0]);
    resultBuffer.unmap();
    this.#resultBuffers.push(resultBuffer);
    return duration;
  }
}
// end TimingHelper code

async function main() {
  const adapter = await navigator.gpu?.requestAdapter();
  const canTimestamp = adapter.features.has("timestamp-query");
  const device = await adapter?.requestDevice({
    requiredFeatures: [...(canTimestamp ? ["timestamp-query"] : [])],
  });

  if (!device) {
    fail("Fatal error: Device does not support WebGPU.");
  }

  if (!canTimestamp) {
    fail(
      'Fatal error: Device does not support WebGPU timestamp query (`adapter.features.has("timestamp-query")` is false).'
    );
  }

  const workgroupSize = 64;
  const memsrcSize = 2 ** 24;

  const workgroupCount = memsrcSize / workgroupSize;
  const dispatchGeometry = [workgroupCount, 1];
  while (
    dispatchGeometry[0] > adapter.limits.maxComputeWorkgroupsPerDimension
  ) {
    dispatchGeometry[0] /= 2;
    dispatchGeometry[1] *= 2;
  }
  console.log(`workgroup count: ${workgroupCount}
      workgroup size: ${workgroupSize}
      maxComputeWGPerDim: ${adapter.limits.maxComputeWorkgroupsPerDimension}
      dispatchGeometry: ${dispatchGeometry}`);

  const memsrc = new Uint32Array(memsrcSize);
  for (let i = 0; i < memsrc.length; i++) {
    memsrc[i] = i;
  }

  const memcpyModule = device.createShaderModule({
    label: "copy large chunk of memory from memSrc to memDest",
    code: /* wgsl */ `
                          /* output */
                          @group(0) @binding(0) var<storage, read_write> memDest: array<u32>;
                          /* input */
                          @group(0) @binding(1) var<storage, read> memSrc: array<u32>;

                          @compute @workgroup_size(${workgroupSize}) fn memcpyKernel(
                            @builtin(global_invocation_id) id: vec3u,
                            @builtin(num_workgroups) nwg: vec3u,
                            @builtin(workgroup_id) wgid: vec3u) {
                              let i = id.y * nwg.x * ${workgroupSize} + id.x;
                              memDest[i] = memSrc[i] + 1;
                          }
                        `,
  });

  const memcpyPipeline = device.createComputePipeline({
    label: "memcpy compute pipeline",
    layout: "auto",
    compute: {
      module: memcpyModule,
    },
  });

  // create buffers on the GPU to hold data
  // read-only inputs:
  const memsrcBuffer = device.createBuffer({
    label: "memory source buffer",
    size: memsrc.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(memsrcBuffer, 0, memsrc);

  const memdestBuffer = device.createBuffer({
    label: "memory destination buffer",
    size: memsrc.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const mappableMemdstBuffer = device.createBuffer({
    label: "mappable memory destination buffer",
    size: memsrc.byteLength,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  /** Set up bindGroups per compute kernel to tell the shader which buffers to use */
  const memcpyBindGroup = device.createBindGroup({
    label: "bindGroup for memcpy kernel",
    layout: memcpyPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: memdestBuffer } },
      { binding: 1, resource: { buffer: memsrcBuffer } },
    ],
  });

  const timingHelper = new TimingHelper(device);
  for (var i = 0; i < 250; i++) {
    const encoder = device.createCommandEncoder({
      label: "memcpy encoder",
    });
    const memcpyPass = encoder.beginComputePass(encoder, {
      label: "memcpy compute pass",
    });
    memcpyPass.setPipeline(memcpyPipeline);
    memcpyPass.setBindGroup(0, memcpyBindGroup);
    // TODO handle not evenly divisible by wgSize
    memcpyPass.dispatchWorkgroups(...dispatchGeometry);
    memcpyPass.end();

    // Encode a command to copy the results to a mappable buffer.
    // this is (from, to)
    encoder.copyBufferToBuffer(
      memdestBuffer,
      0,
      mappableMemdstBuffer,
      0,
      mappableMemdstBuffer.size
    );

    // Finish encoding and submit the commands
    const command_buffer = encoder.finish();
    device.queue.submit([command_buffer]);

    // Read the results
    await mappableMemdstBuffer.mapAsync(GPUMapMode.READ);
    const memdest = new Uint32Array(
      mappableMemdstBuffer.getMappedRange().slice()
    );
    mappableMemdstBuffer.unmap();
    let errors = 0;
    for (let i = 0; i < memdest.length; i++) {
      if (memsrc[i] + 1 != memdest[i]) {
        if (errors < 5) {
          console.log(
            `Error ${errors}: i=${i}, src=${memsrc[i]}, dest=${memdest[i]}`
          );
        }
        errors++;
      }
    }
    if (errors > 0) {
      console.log(`${i} | Memdest size: ${memdest.length} | Errors: ${errors}`);
    } else {
      console.log(`${i} | Memdest size: ${memdest.length} | No errors!`);
    }
  }
}
main();

function fail(msg) {
  // eslint-disable-next-line no-alert
  alert(msg);
}
