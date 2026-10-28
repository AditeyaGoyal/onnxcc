#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#include "onnxcc/cli/cli.h"
#include "onnxcc/cli/exit_codes.h"

namespace {

TEST(Cli, HelpGoesToStdout) {
    std::ostringstream out;
    std::ostringstream err;
    const char* argv[] = {"onnxcc", "--help"};

    const int code = onnxcc::cli::run(2, argv, out, err);

    EXPECT_EQ(onnxcc::cli::kExitOk, code);
    EXPECT_TRUE(err.str().empty());
    EXPECT_NE(std::string::npos, out.str().find("dump"));
}

TEST(Cli, UsageOnErrorGoesToStderr) {
    std::ostringstream out;
    std::ostringstream err;
    const char* argv[] = {"onnxcc"};

    const int code = onnxcc::cli::run(1, argv, out, err);

    EXPECT_EQ(onnxcc::cli::kExitUsage, code);
    EXPECT_TRUE(out.str().empty());
    EXPECT_NE(std::string::npos, err.str().find("usage"));
}

TEST(Cli, MissingModelFileFailsWithoutBeingAUsageError) {
    std::ostringstream out;
    std::ostringstream err;
    const char* argv[] = {"onnxcc", "dump", "--model", "no_such_model_12345.onnx"};

    const int code = onnxcc::cli::run(4, argv, out, err);

    EXPECT_EQ(onnxcc::cli::kExitFailure, code);
    EXPECT_TRUE(out.str().empty());
    EXPECT_NE(std::string::npos, err.str().find("no_such_model_12345.onnx"));
}

// fixtures are generated, not committed, so this makes its own file
TEST(Cli, ExistingModelFileSucceeds) {
    const std::filesystem::path model =
        std::filesystem::temp_directory_path() / "onnxcc_cli_test_model.onnx";
    std::ofstream(model) << "pretend onnx bytes";

    std::ostringstream out;
    std::ostringstream err;
    const std::string path = model.string();
    const char* argv[] = {"onnxcc", "dump", "--model", path.c_str(), "--verbose"};

    const int code = onnxcc::cli::run(5, argv, out, err);

    EXPECT_EQ(onnxcc::cli::kExitOk, code);
    EXPECT_TRUE(err.str().empty());
    EXPECT_NE(std::string::npos, out.str().find(path));
    EXPECT_NE(std::string::npos, out.str().find("size:"));

    std::filesystem::remove(model);
}

TEST(Cli, DirectoryIsRejected) {
    const std::string dir = std::filesystem::temp_directory_path().string();

    std::ostringstream out;
    std::ostringstream err;
    const char* argv[] = {"onnxcc", "dump", "--model", dir.c_str()};

    const int code = onnxcc::cli::run(4, argv, out, err);

    EXPECT_EQ(onnxcc::cli::kExitFailure, code);
    EXPECT_TRUE(out.str().empty());
}

TEST(Cli, DumpShowGraphPrintsParsedGraph) {
    std::filesystem::path fixture_path = std::filesystem::path(ONNXCC_FIXTURE_DIR) / "mlp.onnx";
    if (!std::filesystem::exists(fixture_path)) {
        fixture_path = std::filesystem::path(ONNXCC_FIXTURE_DIR) / "tiny_mlp.onnx";
    }

    ASSERT_TRUE(std::filesystem::exists(fixture_path))
        << "Required ONNX fixture missing: " << fixture_path;

    std::ostringstream out;
    std::ostringstream err;
    const std::string path = fixture_path.string();
    const char* argv[] = {"onnxcc", "dump", "--model", path.c_str(), "--show-graph"};

    const int code = onnxcc::cli::run(5, argv, out, err);

    EXPECT_EQ(onnxcc::cli::kExitOk, code);
    EXPECT_TRUE(err.str().empty());
    EXPECT_NE(out.str().find("nodes (6):"), std::string::npos);
    EXPECT_NE(out.str().find("initializers (4):"), std::string::npos);
    EXPECT_NE(out.str().find("total initializer size: 232 bytes"), std::string::npos);
}

}  // namespace
