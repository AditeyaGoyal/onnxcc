#include <gtest/gtest.h>

#include <compare>
#include <cstdint>
#include <type_traits>
#include <variant>

#include "onnxcc/ir/attribute.h"
#include "onnxcc/ir/datatype.h"
#include "onnxcc/ir/graph.h"
#include "onnxcc/ir/graph_printer.h"
#include "onnxcc/ir/node.h"
#include "onnxcc/ir/status.h"
#include "onnxcc/ir/tensor.h"
#include "onnxcc/ir/tensor_shape.h"
#include "onnxcc/memory/arena.h"
#include "onnxcc/memory/tensor_view.h"
#include "onnxcc/operators/op_compiler.h"
#include "onnxcc/operators/op_kernel.h"
#include "onnxcc/operators/op_registry.h"

// tests to check the headers coexist in one translation unit
namespace onnxcc {

static_assert(std::is_same_v<std::underlying_type_t<DataType>, std::int32_t>);
static_assert(static_cast<std::int32_t>(DataType::UNDEFINED) == 0);
static_assert(static_cast<std::int32_t>(DataType::FLOAT32) == 1);
static_assert(static_cast<std::int32_t>(DataType::INT32) == 6);
static_assert(static_cast<std::int32_t>(DataType::INT64) == 7);
static_assert(static_cast<std::int32_t>(DataType::BOOL) == 9);
static_assert(static_cast<std::int32_t>(DataType::FLOAT64) == 11);

static_assert(dtype_size(DataType::FLOAT32) == 4);
static_assert(dtype_size(DataType::INT32) == 4);
static_assert(dtype_size(DataType::INT64) == 8);
static_assert(dtype_size(DataType::BOOL) == 1);
static_assert(dtype_size(DataType::FLOAT64) == 8);

// the defaulted operator<=> is what gives tests ==, != and ordered container keys
static_assert(std::three_way_comparable<TensorShape>);

// six alternatives, and attribute.cpp must instantiate each functions for all of them
static_assert(std::variant_size_v<Attribute> == 6);

// tensor must stay copyable because it is a variant alternative. A non-copyable Tensor
// makes Attribute, Node and vector<Node> non-copyable
static_assert(std::is_copy_constructible_v<Tensor>);
static_assert(std::is_copy_constructible_v<Node>);

static_assert(!std::is_default_constructible_v<Status>);

static_assert(!std::is_copy_constructible_v<Graph>);
static_assert(std::is_move_constructible_v<Graph>);
static_assert(!std::is_copy_constructible_v<MemoryArena>);

// both are held as unique_ptr to base. Deleting through the base without a virtual destructor is undefined behaviour
static_assert(std::has_virtual_destructor_v<OpKernel>);
static_assert(std::has_virtual_destructor_v<OpCompiler>);
static_assert(std::is_abstract_v<OpKernel>);
static_assert(std::is_abstract_v<OpCompiler>);

}  // namespace onnxcc

TEST(HeadersCompile, EveryHeaderIsIncludable) {
    SUCCEED();
}
