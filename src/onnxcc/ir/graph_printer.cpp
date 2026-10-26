#include "onnxcc/ir/graph_printer.h"

#include <cstddef>
#include <exception>
#include <ostream>
#include <string>
#include <vector>

#include "onnxcc/ir/datatype.h"
#include "onnxcc/ir/graph.h"
#include "onnxcc/ir/node.h"
#include "onnxcc/ir/tensor.h"
#include "onnxcc/ir/tensor_shape.h"

namespace onnxcc {

namespace {

// Formats a comma-separated list of tensor names with their shapes and dtypes for the header block
void print_tensor_summary(const std::vector<Tensor>& tensors, std::ostream& out) {
    bool first = true;
    for (const Tensor& t : tensors) {
        if (!first) {
            out << ", ";
        }
        out << t.name;
        if (t.dtype != DataType::UNDEFINED) {
            out << " (" << dtype_name(t.dtype) << " " << t.shape.to_string() << ")";
        }
        first = false;
    }
}

// Computes the size in bytes of a tensor's data. If data is stored in memory,
// uses data.size(); otherwise computes from shape and dtype if defined.
// If shape is dynamic or calculation overflows, returns 0 as size cannot be determined.
std::size_t get_tensor_byte_size(const Tensor& t) {
    if (!t.data.empty()) {
        return t.data.size();
    }
    if (t.dtype != DataType::UNDEFINED && !t.shape.is_dynamic()) {
        try {
            return t.shape.num_bytes(t.dtype);
        } catch (const std::exception&) {
            // Shape overflow or undefined size returns 0
            return 0;
        }
    }
    return 0;
}

}  // namespace

void print_graph(const Graph& graph, std::ostream& out) {
    // header block
    out << "graph: " << (graph.name.empty() ? "<unnamed>" : graph.name) << "\n";
    out << "ir_version: " << graph.ir_version << "\n";
    out << "opset: " << graph.opset_version << "\n";

    out << "inputs: ";
    if (graph.inputs.empty()) {
        out << "<none>";
    } else {
        print_tensor_summary(graph.inputs, out);
    }
    out << "\n";

    out << "outputs: ";
    if (graph.outputs.empty()) {
        out << "<none>";
    } else {
        print_tensor_summary(graph.outputs, out);
    }
    out << "\n";

    // nodes: [idx] op_type (name): in0, in1 -> out0
    out << "nodes (" << graph.nodes.size() << "):\n";
    std::size_t node_idx = 0;
    for (const Node& node : graph.nodes) {
        out << "[" << node_idx << "] " << node.op_type;
        if (!node.name.empty()) {
            out << " (" << node.name << ")";
        }
        out << ":";

        // empty name represents an absent optional input (node.h), skip to avoid phantom edges
        bool first_input = true;
        for (const std::string& input_name : node.inputs) {
            if (input_name.empty()) {
                continue;
            }
            if (!first_input) {
                out << ", ";
            } else {
                out << " ";
            }
            out << input_name;
            first_input = false;
        }

        out << " ->";

        bool first_output = true;
        for (const std::string& output_name : node.outputs) {
            if (output_name.empty()) {
                continue;
            }
            if (!first_output) {
                out << ", ";
            } else {
                out << " ";
            }
            out << output_name;
            first_output = false;
        }

        out << "\n";
        ++node_idx;
    }

    // initializers with dtype, shape, byte size and a total
    out << "initializers (" << graph.initializers.size() << "):\n";
    std::size_t total_bytes = 0;
    for (const Tensor& init : graph.initializers) {
        const std::size_t bytes = get_tensor_byte_size(init);
        total_bytes += bytes;

        out << init.name << ": "
            << dtype_name(init.dtype) << " "
            << init.shape.to_string() << " ("
            << bytes << " bytes)\n";
    }
    out << "total initializer size: " << total_bytes << " bytes\n";
}

}  // namespace onnxcc