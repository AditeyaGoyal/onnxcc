#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "onnxcc/ir/datatype.h"
#include "onnxcc/ir/graph.h"
#include "onnxcc/ir/graph_printer.h"
#include "onnxcc/ir/node.h"
#include "onnxcc/ir/tensor.h"

namespace onnxcc {
namespace {

Tensor make_dummy_tensor(const std::string& name, DataType dt,
                         std::vector<std::int64_t> dims, std::size_t num_bytes = 0) {
    Tensor t;
    t.name = name;
    t.dtype = dt;
    t.shape.dims = std::move(dims);
    t.is_initializer = (num_bytes > 0);
    if (num_bytes > 0) {
        t.data.resize(num_bytes, 0);
    }
    return t;
}

Graph make_test_graph() {
    Graph g;
    g.name = "test_network";
    g.ir_version = 8;
    g.opset_version = 13;

    g.inputs.push_back(make_dummy_tensor("input", DataType::FLOAT32, {1, 1, 28, 28}));
    g.outputs.push_back(make_dummy_tensor("output", DataType::FLOAT32, {1, 10}));

    // Initializers
    g.initializers.push_back(make_dummy_tensor("conv1_w", DataType::FLOAT32, {6, 1, 5, 5}, 600));
    g.initializers.push_back(make_dummy_tensor("conv1_b", DataType::FLOAT32, {6}, 24));

    // Node 0: Conv with 3 inputs (input, weight, bias)
    Node conv;
    conv.name = "conv1";
    conv.op_type = "Conv";
    conv.inputs = {"input", "conv1_w", "conv1_b"};
    conv.outputs = {"conv1_out"};
    g.nodes.push_back(std::move(conv));

    // Node 1: Conv with absent optional bias: {"conv1_out", "conv2_w", ""}
    Node conv_no_bias;
    conv_no_bias.name = "conv2";
    conv_no_bias.op_type = "Conv";
    conv_no_bias.inputs = {"conv1_out", "conv2_w", ""};
    conv_no_bias.outputs = {"conv2_out"};
    g.nodes.push_back(std::move(conv_no_bias));

    return g;
}

}  // namespace

TEST(GraphPrinter, PrintsHeaderBlockAccurately) {
    const Graph g = make_test_graph();
    std::ostringstream oss;
    print_graph(g, oss);
    const std::string out = oss.str();

    EXPECT_NE(out.find("graph: test_network\n"), std::string::npos);
    EXPECT_NE(out.find("ir_version: 8\n"), std::string::npos);
    EXPECT_NE(out.find("opset: 13\n"), std::string::npos);
    EXPECT_NE(out.find("inputs: input (FLOAT32 [1, 1, 28, 28])"), std::string::npos);
    EXPECT_NE(out.find("outputs: output (FLOAT32 [1, 10])"), std::string::npos);
}

TEST(GraphPrinter, PrintsNodeLineFormat) {
    const Graph g = make_test_graph();
    std::ostringstream oss;
    print_graph(g, oss);
    const std::string out = oss.str();

    // Verify format: [idx] op_type (name): in0, in1 -> out0
    EXPECT_NE(out.find("[0] Conv (conv1): input, conv1_w, conv1_b -> conv1_out\n"),
              std::string::npos);
}

TEST(GraphPrinter, SkipsEmptyStringInputsWithoutPhantomEdges) {
    const Graph g = make_test_graph();
    std::ostringstream oss;
    print_graph(g, oss);
    const std::string out = oss.str();

    // Absent bias must NOT produce ", " or phantom edge before "->"
    EXPECT_NE(out.find("[1] Conv (conv2): conv1_out, conv2_w -> conv2_out\n"),
              std::string::npos);
    EXPECT_EQ(out.find("conv2_w,  ->"), std::string::npos);
    EXPECT_EQ(out.find("conv2_w, ->"), std::string::npos);
}

TEST(GraphPrinter, PrintsInitializersAndTotalBytes) {
    const Graph g = make_test_graph();
    std::ostringstream oss;
    print_graph(g, oss);
    const std::string out = oss.str();

    EXPECT_NE(out.find("conv1_w: FLOAT32 [6, 1, 5, 5] (600 bytes)"), std::string::npos);
    EXPECT_NE(out.find("conv1_b: FLOAT32 [6] (24 bytes)"), std::string::npos);
    EXPECT_NE(out.find("total initializer size: 624 bytes\n"), std::string::npos);
}

TEST(GraphPrinter, EmptyGraphRendersCleanly) {
    const Graph empty_graph;
    std::ostringstream oss;
    print_graph(empty_graph, oss);
    const std::string out = oss.str();

    EXPECT_NE(out.find("graph: <unnamed>\n"), std::string::npos);
    EXPECT_NE(out.find("inputs: <none>\n"), std::string::npos);
    EXPECT_NE(out.find("outputs: <none>\n"), std::string::npos);
    EXPECT_NE(out.find("nodes (0):\n"), std::string::npos);
    EXPECT_NE(out.find("initializers (0):\n"), std::string::npos);
    EXPECT_NE(out.find("total initializer size: 0 bytes\n"), std::string::npos);
}

}  // namespace onnxcc