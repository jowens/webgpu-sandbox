async function main(navigator) {
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

  /** Next, timestamp support */
  const querySet = device.createQuerySet({
    type: "timestamp",
    count: 2,
  });

  const timestampWrites = {
    querySet: querySet,
    beginningOfPassWriteIndex: 0, // Write timestamp in index 0 when pass begins.
    endOfPassWriteIndex: 1, // Write timestamp in index 1 when pass ends.
  };

  const encoder = device.createCommandEncoder({
    label: "memcpy encoder",
  });

  const resolveBuffer = device.createBuffer({
    size: 2 * 8, // querySet.count * 8,
    usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
  });

  const resultBuffer = device.createBuffer({
    size: resolveBuffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const memcpyPassDescriptor = {
    label: "memcpy compute pass",
    timestampWrites: {
      querySet: querySet,
      beginningOfPassWriteIndex: 0,
      endOfPassWriteIndex: 1,
    },
  };
  const memcpyPass = encoder.beginComputePass({
    timestampWrites: timestampWrites,
  }); // memcpyPassDescriptor); //timestampWrites);

  memcpyPass.setPipeline(memcpyPipeline);
  memcpyPass.setBindGroup(0, memcpyBindGroup);
  // TODO handle not evenly divisible by wgSize
  memcpyPass.dispatchWorkgroups(...dispatchGeometry);
  memcpyPass.end();

  encoder.resolveQuerySet(
    querySet,
    0,
    /* querySet.count */ 2,
    resolveBuffer,
    0
  );

  if (resultBuffer.mapState === "unmapped") {
    encoder.copyBufferToBuffer(
      resolveBuffer,
      0,
      resultBuffer,
      0,
      resultBuffer.size
    );
  }

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
  // await device.queue.onSubmittedWorkDone();

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
    console.log(`Memdest size: ${memdest.length} | Errors: ${errors}`);
  } else {
    console.log(`Memdest size: ${memdest.length} | No errors!`);
  }

  if (canTimestamp && resultBuffer.mapState === "unmapped") {
    resultBuffer.mapAsync(GPUMapMode.READ).then(() => {
      const times = new BigInt64Array(resultBuffer.getMappedRange());
      let ns = Number(times[1] - times[0]);
      console.log("Timing raw data:", times[0], times[1], ns);
      resultBuffer.unmap();
      let bytesTransferred = 2 * memdest.byteLength;
      console.log(
        `Timing result: ${ns} ns; transferred ${bytesTransferred} bytes; bandwidth = ${
          bytesTransferred / ns
        } GB/s`
      );
    });
  }

  function fail(msg) {
    // eslint-disable-next-line no-alert
    alert(msg);
  }
}
main(navigator);
