#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "onnxcc/ir/datatype.h"
#include "onnxcc/ir/graph.h"
#include "onnxcc/ir/graph_printer.h"
#include "onnxcc/ir/node.h"
#include "onnxcc/ir/status.h"
#include "onnxcc/ir/tensor.h"
#include "onnxcc/parser/parser.h"

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

    // Node 2: Constant node with no inputs to test stray space formatting
    Node constant_node;
    constant_node.name = "c1";
    constant_node.op_type = "Constant";
    constant_node.outputs = {"const_out"};
    g.nodes.push_back(std::move(constant_node));

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
}

TEST(GraphPrinter, NodeWithNoInputsHasNoStraySpace) {
    const Graph g = make_test_graph();
    std::ostringstream oss;
    print_graph(g, oss);
    const std::string out = oss.str();

    // Node without inputs must print "Constant (c1): -> const_out" (no double space)
    EXPECT_NE(out.find("[2] Constant (c1): -> const_out\n"), std::string::npos);
}

TEST(GraphPrinter, NodeWithEmptyNameOmitsParentheses) {
    Graph g;
    Node n;
    n.op_type = "Relu";
    n.inputs = {"x"};
    n.outputs = {"y"};
    g.nodes.push_back(std::move(n));

    std::ostringstream oss;
    print_graph(g, oss);
    const std::string out = oss.str();

    EXPECT_NE(out.find("[0] Relu: x -> y\n"), std::string::npos);
    EXPECT_EQ(out.find("Relu ():"), std::string::npos);
}

TEST(GraphPrinter, InitializerFallbackToShapeWhenDataEmpty) {
    Graph g;
    // {2, 3} FLOAT32 has 6 elements * 4 bytes = 24 bytes
    Tensor t;
    t.name = "w_shape_only";
    t.dtype = DataType::FLOAT32;
    t.shape.dims = {2, 3};
    t.is_initializer = true;
    g.initializers.push_back(std::move(t));

    std::ostringstream oss;
    print_graph(g, oss);
    const std::string out = oss.str();

    EXPECT_NE(out.find("w_shape_only: FLOAT32 [2, 3] (24 bytes)"), std::string::npos);
    EXPECT_NE(out.find("total initializer size: 24 bytes\n"), std::string::npos);
}

TEST(GraphPrinter, DynamicDimensionRendersAsQuestionMark) {
    Graph g;
    g.inputs.push_back(make_dummy_tensor("dynamic_input", DataType::FLOAT32, {-1, 10}));

    std::ostringstream oss;
    print_graph(g, oss);
    const std::string out = oss.str();

    EXPECT_NE(out.find("dynamic_input (FLOAT32 [?, 10])"), std::string::npos);
}

TEST(GraphPrinter, DynamicInitializerRendersUnknownSize) {
    Graph g;
    Tensor t;
    t.name = "dynamic_w";
    t.dtype = DataType::FLOAT32;
    t.shape.dims = {-1, 10};
    t.is_initializer = true;
    g.initializers.push_back(std::move(t));

    std::ostringstream oss;
    print_graph(g, oss);
    const std::string out = oss.str();

    EXPECT_NE(out.find("dynamic_w: FLOAT32 [?, 10] (unknown)"), std::string::npos);
    EXPECT_NE(out.find("total initializer size: unknown\n"), std::string::npos);
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

// ── ONNX Fixture Test (Checklist item) ──
TEST(GraphPrinter, ParsesAndPrintsRealMlpModel) {
    // Model path passed via ONNXCC_FIXTURE_DIR CMake definition
    std::filesystem::path fixture_path = std::filesystem::path(ONNXCC_FIXTURE_DIR) / "mlp.onnx";
    if (!std::filesystem::exists(fixture_path)) {
        fixture_path = std::filesystem::path(ONNXCC_FIXTURE_DIR) / "tiny_mlp.onnx";
    }

    ASSERT_TRUE(std::filesystem::exists(fixture_path))
        << "Required ONNX fixture missing: " << fixture_path;

    Graph graph;
    const Status status = parser::parse_file(fixture_path.string(), graph);
    ASSERT_TRUE(status.ok()) << status.message();

    // Verify known ONNX facts
    EXPECT_EQ(graph.nodes.size(), 6u);
    EXPECT_EQ(graph.initializers.size(), 4u);

    std::ostringstream oss;
    print_graph(graph, oss);
    const std::string out = oss.str();

    // Verify nodes and initializers in printed output
    EXPECT_NE(out.find("nodes (6):"), std::string::npos);
    EXPECT_NE(out.find("matmul_1"), std::string::npos);
    EXPECT_NE(out.find("matmul_2"), std::string::npos);
    EXPECT_NE(out.find("initializers (4):"), std::string::npos);
    EXPECT_NE(out.find("w1: FLOAT32 [4, 8] (128 bytes)"), std::string::npos);
    EXPECT_NE(out.find("b1: FLOAT32 [8] (32 bytes)"), std::string::npos);
    EXPECT_NE(out.find("w2: FLOAT32 [8, 2] (64 bytes)"), std::string::npos);
    EXPECT_NE(out.find("b2: FLOAT32 [2] (8 bytes)"), std::string::npos);
    EXPECT_NE(out.find("total initializer size: 232 bytes"), std::string::npos);
}

}  // namespace onnxcc
