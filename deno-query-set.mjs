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

  const querySet = device.createQuerySet({
    type: "timestamp",
    count: 2,
  });

  console.log(querySet);

  function fail(msg) {
    // eslint-disable-next-line no-alert
    alert(msg);
  }
}

await main(navigator);
