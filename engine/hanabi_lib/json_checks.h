// Checked field reads for the engine's fromJSON functions.
//
// Each function throws std::invalid_argument naming the field when it is
// missing, has the wrong JSON type, or is out of range, so malformed input is
// rejected (and the C API returns false) instead of being read as garbage.

#ifndef HANABI_LIB_JSON_CHECKS_H_
#define HANABI_LIB_JSON_CHECKS_H_

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace hanabi_learning_env {

// Returns j[key]. Throws if j is not an object or has no such key.
const nlohmann::json& JsonField(const nlohmann::json& j, const char* key);

// Returns j[key] as an integer in [min_value, max_value].
int JsonInt(const nlohmann::json& j, const char* key, int min_value, int max_value);

bool JsonBool(const nlohmann::json& j, const char* key);

std::string JsonString(const nlohmann::json& j, const char* key);

// Returns j[key], which must be an array.
const nlohmann::json& JsonArray(const nlohmann::json& j, const char* key);

// Returns j[key] as an array of integers, each in [min_value, max_value].
std::vector<int> JsonIntArray(const nlohmann::json& j, const char* key, int min_value,
                              int max_value);

// Returns j[key] as an array of booleans.
std::vector<bool> JsonBoolArray(const nlohmann::json& j, const char* key);

// Throws if value is not in [min_value, max_value]; `what` names it in the message.
void CheckRange(int value, int min_value, int max_value, const char* what);

}  // namespace hanabi_learning_env

#endif  // HANABI_LIB_JSON_CHECKS_H_
