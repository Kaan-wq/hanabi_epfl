// Minimal test assertions for engine tests; avoids a test-framework dependency.

#ifndef HANABI_TESTS_CHECK_H_
#define HANABI_TESTS_CHECK_H_

#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace hanabi_test {

struct TestCase {
  std::string name;
  std::function<void()> body;
};

inline std::vector<TestCase>& Registry() {
  static std::vector<TestCase> tests;
  return tests;
}

inline int& FailureCount() {
  static int failures = 0;
  return failures;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> body) {
    Registry().push_back({name, std::move(body)});
  }
};

// Runs every registered test; returns a process exit code. A test that throws fails.
inline int RunAll() {
  for (const auto& test : Registry()) {
    const int before = FailureCount();
    try {
      test.body();
    } catch (const std::exception& e) {
      ++FailureCount();
      std::cerr << test.name << " threw: " << e.what() << '\n';
    }
    std::cout << (FailureCount() == before ? "[ OK ] " : "[FAIL] ") << test.name << '\n';
  }
  std::cout << Registry().size() << " tests, " << FailureCount() << " failed checks\n";
  return FailureCount() == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace hanabi_test

// NOLINTBEGIN(bugprone-macro-parentheses): `name` is an identifier.
#define HANABI_TEST(name)                                             \
  static void name();                                                 \
  static const hanabi_test::Registrar name##_registrar(#name, &name); \
  static void name()
// NOLINTEND(bugprone-macro-parentheses)

#define CHECK(condition)                                                              \
  do {                                                                                \
    if (!(condition)) {                                                               \
      ++hanabi_test::FailureCount();                                                  \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK(" #condition ") failed\n"; \
    }                                                                                 \
  } while (false)

#define CHECK_EQ(a, b)                                                                   \
  do {                                                                                   \
    if (!((a) == (b))) {                                                                 \
      ++hanabi_test::FailureCount();                                                     \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK_EQ(" #a ", " #b ") failed\n"; \
    }                                                                                    \
  } while (false)

#endif  // HANABI_TESTS_CHECK_H_
