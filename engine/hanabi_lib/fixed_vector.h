// A vector with inline storage for at most N elements, so copying it never
// allocates. Hands use it (at most kMaxHandSize cards): states, observations
// and re-determinization copy hands constantly. It offers the subset of the
// std::vector interface hands need; growing past N aborts.

#ifndef HANABI_LIB_FIXED_VECTOR_H_
#define HANABI_LIB_FIXED_VECTOR_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

#include "util.h"

namespace hanabi_learning_env {

template <typename T, size_t N>
class FixedVector {
 public:
  using value_type = T;
  using iterator = T*;
  using const_iterator = const T*;

  FixedVector() = default;

  size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }
  static constexpr size_t capacity() { return N; }

  T& operator[](size_t index) { return items_[index]; }
  const T& operator[](size_t index) const { return items_[index]; }
  T& back() { return items_[size_ - 1]; }
  const T& back() const { return items_[size_ - 1]; }

  T* data() { return items_.data(); }
  const T* data() const { return items_.data(); }
  iterator begin() { return items_.data(); }
  iterator end() { return items_.data() + size_; }
  const_iterator begin() const { return items_.data(); }
  const_iterator end() const { return items_.data() + size_; }

  void clear() { size_ = 0; }
  void push_back(const T& value) {
    REQUIRE(size_ < N);
    items_[size_++] = value;
  }
  // Sets the size to `count`, filling new elements with `value`.
  void resize(size_t count, const T& value) {
    REQUIRE(count <= N);
    std::fill(items_.begin() + size_, items_.begin() + std::max(count, size_), value);
    size_ = count;
  }
  iterator insert(const_iterator position, const T& value) {
    REQUIRE(size_ < N);
    const size_t index = position - begin();
    std::move_backward(begin() + index, end(), end() + 1);
    items_[index] = value;
    ++size_;
    return begin() + index;
  }
  iterator erase(const_iterator position) {
    const size_t index = position - begin();
    std::move(begin() + index + 1, end(), begin() + index);
    --size_;
    return begin() + index;
  }

  // A std::vector copy, for callers that keep or pass on the elements.
  // Explicit, so a hand is never copied by accident.
  explicit operator std::vector<T>() const { return {begin(), end()}; }

  bool operator==(const FixedVector& other) const {
    return std::equal(begin(), end(), other.begin(), other.end());
  }

 private:
  std::array<T, N> items_{};
  size_t size_ = 0;
};

}  // namespace hanabi_learning_env

#endif  // HANABI_LIB_FIXED_VECTOR_H_
