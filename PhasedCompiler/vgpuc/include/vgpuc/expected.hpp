#pragma once
// Prefer std::expected (C++23). If the standard library predates it, this
// header static_asserts so the user knows to upgrade -- we deliberately do NOT
// hand-roll a replacement here, because the whole point of the project is to
// practice the real std::expected monadic interface (and_then/transform/or_else).
#include <version>

#if defined(__cpp_lib_expected) && __cpp_lib_expected >= 202202L
  #include <expected>
  namespace vgpuc {
      template <class T, class E> using Expected = std::expected<T, E>;
      template <class E>          using Unexpected = std::unexpected<E>;
      using std::unexpect;
  }
#else
  #error "This project requires <expected> (C++23). Build with a libc++/libstdc++ that provides std::expected, e.g. LLVM 16+ / GCC 13+ with -std=c++23."
#endif
