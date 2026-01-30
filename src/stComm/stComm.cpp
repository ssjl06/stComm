#include "stComm/stComm.h"
#include <sstream>

namespace stComm {

const char* getVersion() {
    static std::string version;
    if (version.empty()) {
        std::ostringstream oss;
        oss << VERSION_MAJOR << "." << VERSION_MINOR << "." << VERSION_PATCH;
        version = oss.str();
    }
    return version.c_str();
}

} // namespace stComm
