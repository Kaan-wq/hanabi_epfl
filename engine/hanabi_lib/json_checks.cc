#include "json_checks.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace hanabi_learning_env {

namespace {

using json = nlohmann::json;

// Messages are only built on failure: these checks run on every deserialization.
[[noreturn]] void Fail(const char* key, const char* problem) {
  throw std::invalid_argument(std::string("'") + key + "' " + problem);
}

// Converts an integer JSON value to int, checking the range without overflow.
int ToInt(const json& value, int min_value, int max_value, const char* key) {
  if (!value.is_number_integer()) {
    Fail(key, "must contain integers");
  }
  if (value.is_number_unsigned() &&
      value.get<uint64_t>() > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    Fail(key, "is out of range");
  }
  const int64_t number = value.get<int64_t>();
  if (number < min_value || number > max_value) {
    Fail(key, "is out of range");
  }
  return static_cast<int>(number);
}

}  // namespace

const json& JsonField(const json& j, const char* key) {
  if (!j.is_object()) {
    Fail(key, "is missing: expected an object");
  }
  const auto it = j.find(key);
  if (it == j.end()) {
    Fail(key, "is missing");
  }
  return *it;
}

int JsonInt(const json& j, const char* key, int min_value, int max_value) {
  return ToInt(JsonField(j, key), min_value, max_value, key);
}

bool JsonBool(const json& j, const char* key) {
  const json& value = JsonField(j, key);
  if (!value.is_boolean()) {
    Fail(key, "must be a boolean");
  }
  return value.get<bool>();
}

std::string JsonString(const json& j, const char* key) {
  const json& value = JsonField(j, key);
  if (!value.is_string()) {
    Fail(key, "must be a string");
  }
  return value.get<std::string>();
}

const json& JsonArray(const json& j, const char* key) {
  const json& value = JsonField(j, key);
  if (!value.is_array()) {
    Fail(key, "must be an array");
  }
  return value;
}

std::vector<int> JsonIntArray(const json& j, const char* key, int min_value, int max_value) {
  const json& array = JsonArray(j, key);
  std::vector<int> values;
  values.reserve(array.size());
  for (const json& value : array) {
    values.push_back(ToInt(value, min_value, max_value, key));
  }
  return values;
}

std::vector<bool> JsonBoolArray(const json& j, const char* key) {
  const json& array = JsonArray(j, key);
  std::vector<bool> values;
  values.reserve(array.size());
  for (const json& value : array) {
    if (!value.is_boolean()) {
      Fail(key, "must contain booleans");
    }
    values.push_back(value.get<bool>());
  }
  return values;
}

void CheckRange(int value, int min_value, int max_value, const char* what) {
  if (value < min_value || value > max_value) {
    throw std::invalid_argument(std::string(what) + " is out of range");
  }
}

}  // namespace hanabi_learning_env
