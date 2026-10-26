#include "onnxcc/cli/dump_command.h"

#include <filesystem>
#include <ostream>
#include <system_error>

#include "onnxcc/cli/exit_codes.h"
#include "onnxcc/ir/graph.h"
#include "onnxcc/ir/graph_printer.h"
#include "onnxcc/ir/status.h"
#include "onnxcc/parser/parser.h"
#include "onnxcc/version.h"

namespace onnxcc::cli {

    int run_dump(const DumpOptions& options, std::ostream& out, std::ostream& err) {
        const std::filesystem::path model(options.model);

        // non-throwing overloads, so an unreadable parent dir reports instead of aborting
        std::error_code ec;
        if (!std::filesystem::exists(model, ec) || ec) {
            err << "onnxcc dump: error: model file '" << options.model << "' does not exist\n";
            return kExitFailure;
        }
        // exists() is also true for dirs, devices and fifos (named pipes, where a read waits
        // for whoever writes), so a fifo would hang the parser instead of failing
        if (!std::filesystem::is_regular_file(model, ec) || ec) {
            err << "onnxcc dump: error: '" << options.model << "' is not a regular file\n";
            return kExitFailure;
        }

        out << "model: " << options.model << "\n";

        if (options.verbose) {
            const std::uintmax_t size = std::filesystem::file_size(model, ec);
            out << "onnxcc: " << get_version() << " (" << get_version_codename() << ")\n";
            if (!ec) {
                out << "size: " << size << " bytes\n";
            }
        }

        if (options.show_graph) {
            Graph graph;
            const Status parse_status = parser::parse_file(options.model, graph);
            if (!parse_status.ok()) {
                err << "onnxcc dump: error: " << parse_status.message() << "\n";
                return kExitFailure;
            }
            print_graph(graph, out);
        }

        return kExitOk;
    }
}
