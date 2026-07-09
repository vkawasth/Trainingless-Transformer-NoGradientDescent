#pragma once
// py_bridge.hpp
// Embeds a CPython interpreter via pybind11 and exposes the geometry compiler
// surface as C++ methods that run on the TaskQueue worker thread.
//
// ============================ THE GIL, PRECISELY ============================
// CPython has ONE global interpreter lock. Only the thread holding it may touch
// any Python object. Our architecture has (at least) two threads:
//     main thread          -- constructs things, calls .get() on futures
//     TaskQueue worker      -- actually executes the ops (which call Python)
//
// pybind11's rules we rely on:
//   * py::scoped_interpreter starts the interpreter ON THE THREAD that creates
//     it, and that thread initially HOLDS the GIL.
//   * To let ANOTHER thread (the worker) ever run Python, the thread that holds
//     the GIL must RELEASE it. We do that once, right after init, by keeping a
//     py::gil_scoped_release alive on the main thread for the bridge's lifetime.
//   * Every task body that touches Python must acquire the GIL first via
//     py::gil_scoped_acquire (RAII: acquires in ctor, releases in dtor). This
//     is re-entrant-safe and pairs with the release above.
//
// Consequences / gotchas an interviewer probes:
//   * If you forget the main-thread release, the worker BLOCKS FOREVER trying
//     to acquire a GIL the main thread never let go of -> deadlock.
//   * Holding the GIL across a long torch op serializes everything. That's OK
//     here (ops are meant to be serialized) but is why real inference servers
//     release the GIL inside C++ compute and re-acquire only for Python calls.
//   * The interpreter must be torn down on the SAME thread that created it, and
//     only after the worker has stopped calling Python. Ordering in the driver
//     matters: shut the queue down BEFORE the interpreter dies.
// ===========================================================================

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <string>
#include <memory>
#include <stdexcept>

namespace py = pybind11;

namespace bridge {

// Owns the interpreter and a handle to a GeometryCompiler instance.
// One PyBridge per process (CPython is a singleton).
class PyBridge {
    // interp_ is a unique_ptr (not a default-initialized member) so we can
    // configure a PyConfig BEFORE the interpreter starts. A plain
    // `scoped_interpreter interp_{}` would call Py_Initialize() too early,
    // before we could point it at the venv.
    std::unique_ptr<py::scoped_interpreter> interp_;
    std::unique_ptr<py::gil_scoped_release> main_release_;  // let workers run
    py::object compiler_;                      // the GeometryCompiler instance

public:
    // module_name: e.g. "geo_compiler_surface"; extra_path: dir containing it.
    // venv_python: absolute path to the venv's interpreter, e.g.
    //   /Users/.../fact_env/bin/python3.14
    // Given that executable, CPython reads the adjacent pyvenv.cfg and wires up
    // BOTH the base stdlib (so `encodings` resolves) and the venv site-packages
    // (so torch/numpy resolve) with correct sys.prefix/base_prefix. No
    // PYTHONHOME/PYTHONPATH needed, from any shell or working directory.
    explicit PyBridge(const std::string& module_name,
                      const std::string& extra_path,
                      const std::string& venv_python,
                      bool use_real_corpus = false) {
        // ---- configure the interpreter to be venv-aware ----
        PyConfig config;
        PyConfig_InitPythonConfig(&config);
        // Let site.py run (it reads pyvenv.cfg and adds site-packages).
        config.site_import = 1;

        if (!venv_python.empty()) {
            PyStatus st = PyConfig_SetBytesString(
                &config, &config.executable, venv_python.c_str());
            if (PyStatus_Exception(st)) {
                PyConfig_Clear(&config);
                throw std::runtime_error("PyConfig_SetBytesString(executable) failed");
            }
        }

        // scoped_interpreter takes ownership of running Py_InitializeFromConfig
        // with our config. It clears the config afterwards.
        interp_ = std::make_unique<py::scoped_interpreter>(&config);
        // (from here we HOLD the GIL on this thread)

        {
            // Make the surface module importable and build one instance.
            py::module_ sys = py::module_::import("sys");
            sys.attr("path").attr("insert")(0, extra_path);
            py::module_ mod = py::module_::import(module_name.c_str());
            compiler_ = mod.attr("GeometryCompiler")(
                py::arg("seed") = 99,
                py::arg("use_real_corpus") = use_real_corpus);
        }
        // Release the GIL from the main thread so the worker can acquire it.
        // Kept alive for the bridge's lifetime.
        main_release_ = std::make_unique<py::gil_scoped_release>();
    }

    ~PyBridge() {
        // Re-acquire the GIL on the main thread before compiler_ / interpreter
        // are destroyed, so Python object teardown is legal.
        main_release_.reset();          // re-acquires GIL (dtor of release)
        compiler_ = py::object();       // drop the instance under the GIL
        interp_.reset();                // interpreter dtor last, on this thread
    }

    PyBridge(const PyBridge&) = delete;
    PyBridge& operator=(const PyBridge&) = delete;

    // -------- ops: each acquires the GIL, calls Python, returns a scalar -----
    // These are meant to be called FROM WITHIN a task on the worker thread.

    double eval_val(int n = 12) {
        py::gil_scoped_acquire gil;
        return compiler_.attr("eval_val")(py::arg("n") = n).cast<double>();
    }

    double train_steps(int k = 10) {
        py::gil_scoped_acquire gil;
        return compiler_.attr("train_steps")(py::arg("k") = k).cast<double>();
    }

    double gluing_defect(int n = 6) {
        py::gil_scoped_acquire gil;
        return compiler_.attr("gluing_defect")(py::arg("n") = n).cast<double>();
    }

    // ---- geometry-compiler phases (each mutates the model, returns val) ----
    double saddle() {
        py::gil_scoped_acquire gil;
        return compiler_.attr("saddle")().cast<double>();
    }

    double mfpump(int seed = 0) {
        py::gil_scoped_acquire gil;
        return compiler_.attr("mfpump")(py::arg("seed") = seed).cast<double>();
    }

    double lanczos() {
        py::gil_scoped_acquire gil;
        return compiler_.attr("lanczos")().cast<double>();
    }

    double phi_clean() {
        py::gil_scoped_acquire gil;
        return (double)compiler_.attr("phi_clean")().cast<long>();
    }

    double basin_settle(int max_steps = 150) {
        py::gil_scoped_acquire gil;
        return compiler_.attr("basin_settle")(py::arg("max_steps") = max_steps).cast<double>();
    }
    double tau_retry() {
        py::gil_scoped_acquire gil;
        return compiler_.attr("tau_retry")().cast<double>();
    }
    double snapper_jump() {
        py::gil_scoped_acquire gil;
        return compiler_.attr("snapper_jump")().cast<double>();
    }
    double topogate() {
        py::gil_scoped_acquire gil;
        return compiler_.attr("topogate")().cast<double>();
    }
    double align_lm() {
        py::gil_scoped_acquire gil;
        return compiler_.attr("align_lm")().cast<double>();
    }
    double k0_split() {
        py::gil_scoped_acquire gil;
        return compiler_.attr("k0_split")().cast<double>();
    }
    double joint_ce() {
        py::gil_scoped_acquire gil;
        return compiler_.attr("joint_ce")().cast<double>();
    }

    double mem_allocated_mib() {
        py::gil_scoped_acquire gil;
        return compiler_.attr("mem_allocated_mib")().cast<double>();
    }

    long num_params() {
        py::gil_scoped_acquire gil;
        return compiler_.attr("num_params")().cast<long>();
    }

    long step_count() {
        py::gil_scoped_acquire gil;
        return compiler_.attr("step_count")().cast<long>();
    }
};

} // namespace bridge
