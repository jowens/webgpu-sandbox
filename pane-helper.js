// by gman@google.com, 17 Sept 2024
// bridges gap between webgpu-utils's uniform handling
//   and tweakpane's parameter setting

// Sample code: https://jsgist.org/?src=c0184b52a049995d0fb038df3b8663f4

// Creates a factory for an object that gets/sets elements of an array
// based on accessors (like x, y, z or r, g, b)
function makeNamedIndexProxyMaker(indices) {
  const keys = indices.split("");
  return function (target) {
    const o = {};
    Object.defineProperties(
      o,
      Object.fromEntries(
        keys.map((key, i) => {
          return [
            key,
            {
              enumerable: true,
              get: () => target[i],
              set: (v) => (target[i] = v),
            },
          ];
        })
      )
    );
    return o;
  };
}

// Make a function that adds a binding to a pane for an array of values
// where the values are accessed by accessor or x, y, z, or r, g, b
// because tweakpane wants that (T_T)
function makeAccessorHelper(indices) {
  const make = makeNamedIndexProxyMaker(indices);
  return function (pane, params, property, options = {}) {
    const localParams = { value: make(params[property]) };
    window.l = localParams;
    return pane.addBinding(localParams, "value", {
      label: property,
      ...options,
      color: { type: "float" },
    });
  };
}

const rgbHelper = makeAccessorHelper("rgb");
const xyzHelper = makeAccessorHelper("xyz");

// how to set one value
//  pane.addBinding(fsUniformValues.views.mix, '0', { min: 0, max: 1, label: 'mix' });
//  rgbHelper(pane, fsUniformValues.views, 'color');
//  xyzHelper(pane, fsUniformValues.views, 'lightDirection');
