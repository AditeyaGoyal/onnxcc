#pragma once

#include <iosfwd>
#include "onnxcc/ir/graph.h"

namespace onnxcc {

// Prints a human-readable representation of a model Graph to the provided stream.
// Formats:
//   - Header block: graph name, ir_version, opset_version, inputs, outputs
//   - Nodes: [idx] op_type (name): in0, in1 -> out0 (skipping absent optional inputs)
//   - Initializers: dtype, shape, byte size per tensor, and total byte size
void print_graph(const Graph& graph, std::ostream& out);

}  // namespace onnxcc